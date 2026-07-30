"""PTXPHISH 评测：样本量扩大后 phishing 检测是否改善 + 硬负样本校准的作用。

三组对比（检测 AUROC，钓鱼 vs 正常）：
  A. 研究一种子 phishing（n≈20/400）+ 随机正常校准        ← 现状基线
  B. PTXPHISH 钓鱼（大样本）+ 随机正常校准                 ← 只换评测集，看样本量效应
  C. PTXPHISH 钓鱼 + **硬负样本(Benign KOL/Dev)混入校准集**  ← 看硬负样本是否让校准更锐利
另报"钓鱼 vs 硬负样本"的 AUROC——这是最难也最有意义的设定（形似钓鱼的合法交易）。
"""
from __future__ import annotations

import random
import sys
import warnings
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mlusd.dataset.build import load_contexts               # noqa: E402
from mlusd.match.dictionary import load_dictionaries         # noqa: E402
from mlusd.pipeline import Detector                          # noqa: E402
from mlusd.signals.factory import default_extractors         # noqa: E402


def fit_det(fit_norm):
    return Detector(default_extractors(), load_dictionaries(ROOT / "configs/dictionaries"),
                    alpha=0.01, min_group_size=150,
                    openset_aggregator="learned").fit(fit_norm)


def scores(det, ctxs):
    out = []
    for c in ctxs:
        S, m = det._raw_matrix(c)
        Q, _, g = det.calibrator.transform(S, m)
        out.append(det.openset.raw_score(Q, m)
                   if hasattr(det.openset, "raw_score") else det.openset.ubar(Q, m, g))
    return np.asarray(out)


def auroc(sa, sn):
    if len(sa) == 0 or len(sn) == 0:
        return float("nan")
    return roc_auc_score(np.r_[np.ones(len(sa)), np.zeros(len(sn))], np.r_[sa, sn])


def main():
    dcal = load_contexts(ROOT / "data/splits/d_cal.pkl.gz")     # 非邻域版，与 PTX 采集一致
    random.Random(0).shuffle(dcal)
    ptx_a = load_contexts(ROOT / "data/splits/d_phish_ptx.pkl.gz")
    ptx_b = load_contexts(ROOT / "data/splits/d_benign_ptx.pkl.gz")
    seed_ph = [c for c in load_contexts(ROOT / "data/splits/d_known_ext.pkl.gz")
               if c.latent.get("attack_type") == "phishing"]
    print(f"PTX 钓鱼 {len(ptx_a)} | PTX 硬负 {len(ptx_b)} | 研究一 phishing {len(seed_ph)}")

    fit_norm, test_norm = dcal[:8000], dcal[8000:10000]

    # A/B：随机正常校准
    det = fit_det(fit_norm)
    sn = scores(det, test_norm)
    print("\n=== 随机正常交易作校准集 ===")
    print(f"  A 研究一 phishing vs 正常      AUROC = {auroc(scores(det, seed_ph), sn):.3f}")
    print(f"  B PTXPHISH 钓鱼 vs 正常        AUROC = {auroc(scores(det, ptx_a), sn):.3f}")
    print(f"  B' PTXPHISH 钓鱼 vs 硬负样本   AUROC = {auroc(scores(det, ptx_a), scores(det, ptx_b)):.3f}"
          "   ← 最难设定（形似钓鱼的合法交易）")

    # C：硬负样本混入校准集
    n_hard = len(ptx_b) // 2
    det_h = fit_det(fit_norm[:8000 - n_hard] + ptx_b[:n_hard])
    sn_h = scores(det_h, test_norm)
    print(f"\n=== 混入 {n_hard} 笔硬负样本作校准集 ===")
    print(f"  C PTXPHISH 钓鱼 vs 正常        AUROC = {auroc(scores(det_h, ptx_a), sn_h):.3f}")
    print(f"  C' PTXPHISH 钓鱼 vs 硬负样本   AUROC = "
          f"{auroc(scores(det_h, ptx_a), scores(det_h, ptx_b[n_hard:])):.3f}")

    # 分子类型（手法级）
    from collections import defaultdict
    by = defaultdict(list)
    for c in ptx_a:
        by[c.latent.get("subtype", "?")].append(c)
    print("\n=== 分手法子类型 AUROC（vs 正常，n≥15）===")
    sa_all = scores(det, ptx_a)
    for t, cs in sorted(by.items(), key=lambda x: -len(x[1])):
        if len(cs) >= 15:
            print(f"  {t:<34} {auroc(scores(det, cs), sn):.3f}  (n={len(cs)})")


if __name__ == "__main__":
    main()
