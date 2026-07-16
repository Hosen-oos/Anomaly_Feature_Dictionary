"""M4 字典权重的数据驱动更新 + 逐类型阈值标定（设计架构 §4 M4、实验四）。

权重更新：对每类 k，比较该类攻击样本与正常样本在各信号位置的尾部放大 T 的均值差，
差值大 = 该位置对区分此类攻击有判别力 → 归一化为数据驱动权重，与机理先验权重按
prior_ratio 混合（设计说先验:数据=1:1 起步）。

阈值标定：权重更新后，在验证集上对每类扫 match_threshold，取 F1 最优（设计 §4 M4：
"τ_k 逐类型在验证集上按 F1 最优选取"），替代拍脑袋的 0.55 统一先验。
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from mlusd.match.dictionary import AttackDictionary
from mlusd.match.matcher import match_one
from mlusd.types import N_ANGLES, N_LAYERS, TxContext, VALID_POSITIONS


def signal_tensors(det, contexts: list[TxContext]):
    """对每个 ctx 算 (T 矩阵, mask)。det 需已 fit（提取器+校准器）。"""
    out = []
    for ctx in contexts:
        S, mask = det._raw_matrix(ctx)
        _, T, _ = det.calibrator.transform(S, mask)
        out.append((T, mask))
    return out


def _mean_T(tensors) -> np.ndarray:
    """各位置 T 的 nan 感知均值（缺失位置不计入）。"""
    acc = np.zeros((N_LAYERS, N_ANGLES)); cnt = np.zeros((N_LAYERS, N_ANGLES))
    for T, _ in tensors:
        for (l, j) in VALID_POSITIONS:
            v = T[l - 1, j - 1]
            if np.isfinite(v):
                acc[l - 1, j - 1] += v; cnt[l - 1, j - 1] += 1
    return np.where(cnt > 0, acc / np.maximum(cnt, 1), 0.0)


def update_dictionary_weights(dicts: list[AttackDictionary],
                              attack_tensors_by_type: dict,
                              normal_tensors,
                              prior_ratio: float = 0.5) -> None:
    """就地更新每个字典的 weights（先验 + 数据驱动，按 prior_ratio 混合）。"""
    normal_mean = _mean_T(normal_tensors)
    for d in dicts:
        atk = attack_tensors_by_type.get(d.attack_type)
        if not atk:
            continue
        attack_mean = _mean_T(atk)
        diff = np.zeros((N_LAYERS, N_ANGLES))
        for (l, j) in VALID_POSITIONS:
            diff[l - 1, j - 1] = max(0.0, attack_mean[l - 1, j - 1]
                                     - normal_mean[l - 1, j - 1])
        data_w = diff / diff.sum() if diff.sum() > 0 else np.zeros_like(diff)
        prior = d.weights.copy()
        prior_w = prior / prior.sum() if prior.sum() > 0 else prior
        d.weights = prior_ratio * prior_w + (1 - prior_ratio) * data_w


def tune_thresholds(dicts: list[AttackDictionary],
                    attack_tensors_by_type: dict,
                    normal_tensors,
                    alpha_fp: float = 0.005,
                    min_threshold: float = 0.05) -> dict[str, float]:
    """逐类型把 match_threshold 标定到正常匹配分的 (1-alpha_fp) 分位数（就地写回）。

    设计取舍：KNOWN 分类要**高精度**（说"是flash_loan"就得对）——因判定是跨类型
    argmax，若按裸 F1 调阈值会把阈值压太低、多类低阈值叠加导致正常样本被大量误判
    KNOWN。改为控制每类正常误报率 ≤ alpha_fp，牺牲部分召回换精度。返回 {type: τ}。
    """
    tuned = {}
    for d in dicts:
        neg_scores = [match_one(d, T, m).final_score for T, m in normal_tensors]
        if neg_scores:
            tau = float(np.quantile(neg_scores, 1.0 - alpha_fp))
        else:
            tau = d.match_threshold
        d.match_threshold = max(tau, min_threshold)
        tuned[d.attack_type] = d.match_threshold
    return tuned
