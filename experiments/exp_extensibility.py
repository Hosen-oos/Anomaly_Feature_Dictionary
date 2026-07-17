"""用扩充数据（phishing/sandwich 各 400）做：
(1) 稳定 per-type 开放集 AUROC（对比 20 样本的噪声版）；
(2) 可扩展性曲线：用 N=20/50/100/200/300 个 phishing 样本增量构建其字典（仅更新该类
    权重+阈值，不重训任何模型），测 phishing 的 KNOWN 分类 P/R/F1 vs N。

注意：fit 用非邻域 d_cal 以与 d_known_ext（非邻域）一致，避免 L1 图特征错配。
"""
from __future__ import annotations

import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mlusd.dataset.build import load_contexts                    # noqa: E402
from mlusd.match.dictionary import load_dictionaries              # noqa: E402
from mlusd.match.matcher import match_one                         # noqa: E402
from mlusd.match.weight_update import (                           # noqa: E402
    signal_tensors, tune_thresholds, update_dictionary_weights)
from mlusd.pipeline import Detector                               # noqa: E402
from mlusd.signals.factory import default_extractors              # noqa: E402
from mlusd.types import Verdict                                   # noqa: E402

DICT = ROOT / "configs/dictionaries"


def _ubar(det, ctxs):
    out = []
    for c in ctxs:
        S, m = det._raw_matrix(c)
        Q, _, g = det.calibrator.transform(S, m)
        out.append(det.openset.ubar(Q, m, g))
    return np.asarray(out)


def _auroc(det, attacks, normals):
    y = np.r_[np.ones(len(attacks)), np.zeros(len(normals))]
    s = np.r_[_ubar(det, attacks), _ubar(det, normals)]
    return roc_auc_score(y, s)


def main():
    dcal = load_contexts(ROOT / "data/splits/d_cal.pkl.gz")      # 非邻域
    ext = load_contexts(ROOT / "data/splits/d_known_ext.pkl.gz")
    random.Random(0).shuffle(dcal)
    fit_norm = dcal[:8000]
    val_norm = dcal[8000:10000]
    test_norm = dcal[10000:12000]

    by = defaultdict(list)
    for c in ext:
        by[c.latent.get("attack_type")].append(c)

    det = Detector(default_extractors(), load_dictionaries(DICT),
                   alpha=0.01, min_group_size=150).fit(fit_norm)

    # (1) 稳定 AUROC
    print("=== 稳定开放集 AUROC（扩充后，n=400/类）===")
    for t in sorted(by):
        print(f"  {t:<12} AUROC={_auroc(det, by[t], test_norm):.3f}  (n={len(by[t])})")

    # (2) 可扩展性曲线（以 phishing 为新增类）
    ph = by["phishing"][:]
    random.Random(1).shuffle(ph)
    tr, te = ph[:280], ph[280:]           # 280 训练池 / 120 测试
    norm_val_T = signal_tensors(det, val_norm)
    ph_test_T = signal_tensors(det, te)
    norm_test_T = signal_tensors(det, test_norm[:1000])
    print("\n=== 可扩展性：phishing 字典用 N 样本增量构建（不重训模型）===")
    print(f"{'N':>5}{'KNOWN召回':>10}{'精度':>8}{'F1':>8}")
    for N in [20, 50, 100, 200, 280]:
        dicts = load_dictionaries(DICT)
        det.dictionaries = dicts
        det._req = {d.attack_type: d.layer_requirements for d in dicts}
        atk_T = {"phishing": signal_tensors(det, tr[:N])}
        update_dictionary_weights(dicts, atk_T, signal_tensors(det, fit_norm[:2000]), 0.5)
        tune_thresholds(dicts, atk_T, norm_val_T, alpha_fp=0.005)
        phd = next(d for d in dicts if d.attack_type == "phishing")
        tp = sum(match_one(phd, T, m).final_score >= phd.match_threshold for T, m in ph_test_T)
        fp = sum(match_one(phd, T, m).final_score >= phd.match_threshold for T, m in norm_test_T)
        rec = tp / len(ph_test_T)
        prec = tp / (tp + fp) if tp + fp else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        print(f"{N:>5}{rec:>10.2f}{prec:>8.2f}{f1:>8.2f}")


if __name__ == "__main__":
    main()
