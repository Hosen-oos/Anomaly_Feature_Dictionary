"""M4 已知类型信号匹配（报告 4.5.2.2）。

覆盖率   c_k    = Σ_l r_k[l]·m[l] / Σ_l r_k[l]
基础分   base_k = Σ W_k[l,j]·T[l,j] / Σ W_k[l,j]      （只在可用且非 NaN 的位置求和）
最终分   final_k = base_k · c_k
贡献度   contrib_k[l,j] = W_k[l,j]·T[l,j] / Σ W·T

字典在两个粒度上实例化。位级权重 W_k 作用于 8 个格值，坐标与层×视角对应；参数级对比
权重（contrastive.py）作用于参数分位数，保留位内 max 聚合所舍弃的细节，用于分辨机理
相近的类型。调用方通过 base_override 传入参数级分 score_k 时，它替代 base_k 进入
final_k = base_k·c_k；无论走哪个粒度，覆盖率缩放与贡献度分解都不变——贡献度必须定义在
层×视角坐标系上才可解释，故始终由 W_k·T 计算。
"""
from __future__ import annotations

import numpy as np

from mlusd.match.dictionary import AttackDictionary
from mlusd.types import Contribution, MatchResult, VALID_POSITIONS


def match_one(d: AttackDictionary, T: np.ndarray,
              mask: tuple[int, int, int, int],
              base_override: float | None = None) -> MatchResult:
    req = d.layer_requirements
    coverage = float((req * np.asarray(mask)).sum() / req.sum())

    num = 0.0
    den = 0.0
    weighted: list[tuple[int, int, float]] = []
    for (l, j) in VALID_POSITIONS:
        w = d.weights[l - 1, j - 1]
        if w <= 0 or not mask[l - 1]:
            continue
        t = T[l - 1, j - 1]
        if not np.isfinite(t):
            continue
        num += w * t
        den += w
        weighted.append((l, j, w * t))

    base = num / den if den > 0 else 0.0
    if base_override is not None:      # 参数级 score_k 替代位级 base_k
        base = float(base_override)
    total_wt = sum(v for _, _, v in weighted)
    contributions = [
        Contribution(layer=l, angle=j, value=(v / total_wt if total_wt > 0 else 0.0))
        for l, j, v in sorted(weighted, key=lambda x: -x[2])
    ]
    return MatchResult(
        attack_type=d.attack_type,
        base_score=float(base),
        coverage=coverage,
        final_score=float(base * coverage),
        match_threshold=d.match_threshold,
        coverage_threshold=d.coverage_threshold,
        contributions=contributions,
    )


def match_all(dicts: list[AttackDictionary], T: np.ndarray,
              mask: tuple[int, int, int, int],
              base_override: dict[str, float] | None = None) -> list[MatchResult]:
    """按最终匹配分数降序返回全部类型的匹配结果。

    base_override: {攻击类型: score_k}。未在其中出现的类型退回位级 base_k。
    """
    results = [
        match_one(d, T, mask,
                  None if base_override is None else base_override.get(d.attack_type))
        for d in dicts
    ]
    return sorted(results, key=lambda r: -r.final_score)
