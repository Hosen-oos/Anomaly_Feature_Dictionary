"""实验二（开放集检测）+ 实验三（消融），统一用稳健指标 AUROC(攻击 vs 正常, 用 Ū 排序)。

Ū（开放集分数）与字典无关，天然是"类型无关的整体异常度"，故：
- 实验二 = 每类攻击 vs 正常的 AUROC（即"该类作为未知时能否被 Ū 检出"，等价 LOTO 的检测侧）
  + 真未知 D_open 的 AUROC；
- 实验三 = 各消融变体下同一 AUROC 的变化。
AUROC 在 111 攻击 vs 数千正常上很稳，不受 KNOWN 小样本噪声影响。

    python -m experiments.exp23_openset_ablation
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
from mlusd.pipeline import Detector                               # noqa: E402
from mlusd.signals.factory import ablation_extractors, default_extractors  # noqa: E402

DICT = ROOT / "configs/dictionaries"


def _ubar_scores(det, ctxs):
    out = []
    for c in ctxs:
        S, m = det._raw_matrix(c)
        Q, _, g = det.calibrator.transform(S, m)
        out.append(det.openset.ubar(Q, m, g))
    return np.asarray(out)


def _fit(det_kwargs, extractors, fit_norm):
    det = Detector(extractors, load_dictionaries(DICT), alpha=0.01, **det_kwargs)
    det.fit(fit_norm)
    return det


def _auroc(det, attacks, normals):
    sa = _ubar_scores(det, attacks)
    sn = _ubar_scores(det, normals)
    y = np.r_[np.ones(len(sa)), np.zeros(len(sn))]
    s = np.r_[sa, sn]
    return roc_auc_score(y, s)


def main():
    dcal = load_contexts(ROOT / "data/splits/d_cal_nbr.pkl.gz")
    dknown = load_contexts(ROOT / "data/splits/d_known_nbr.pkl.gz")
    dopen = None
    p = ROOT / "data/splits/d_open_nbr.pkl.gz"
    if p.exists():
        dopen = load_contexts(p)
    random.Random(0).shuffle(dcal)
    fit_norm = dcal[:8000]
    test_norm = dcal[8000:10000]
    by_type = defaultdict(list)
    for c in dknown:
        by_type[c.latent.get("attack_type")].append(c)

    # ---------- 实验二：全模型的开放集检测 AUROC ----------
    full = _fit(dict(min_group_size=150), default_extractors(), fit_norm)
    print("=== 实验二：开放集检测 AUROC (Ū 排序, 攻击 vs 正常) ===")
    overall = _auroc(full, dknown, test_norm)
    print(f"  全部攻击           AUROC = {overall:.3f}  (n_attack={len(dknown)}, n_normal={len(test_norm)})")
    for t in sorted(by_type):
        a = _auroc(full, by_type[t], test_norm)
        print(f"  {t:<20} AUROC = {a:.3f}  (n={len(by_type[t])})")
    if dopen:
        print(f"  真未知 D_open       AUROC = {_auroc(full, dopen, test_norm):.3f}  (n={len(dopen)})")

    # ---------- 实验三：消融（同一 overall AUROC 的变化）----------
    print("\n=== 实验三：消融（overall AUROC，越低说明该设计越重要）===")
    print(f"  {'完整模型':<28} {overall:.3f}")
    variants = [
        ("−分组校准(全体共用ECDF)", dict(min_group_size=150, single_group=True), default_extractors()),
        ("Fisher→max",             dict(min_group_size=150, openset_mode="max"), default_extractors()),
        ("Fisher→mean",            dict(min_group_size=150, openset_mode="mean"), default_extractors()),
        ("格内聚合 max→mean(稀释)", dict(min_group_size=150), ablation_extractors("mean")),
    ]
    for name, kw, exts in variants:
        det = _fit(kw, exts, fit_norm)
        a = _auroc(det, dknown, test_norm)
        print(f"  {name:<28} {a:.3f}   (Δ={a-overall:+.3f})")


if __name__ == "__main__":
    main()
