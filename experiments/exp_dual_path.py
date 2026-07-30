"""两路校准并联：单侧路径（擅长"大即异常"）+ 折叠双侧路径（擅长"小即异常"），
各自转成全局分位数后取 max。与格内参数池的"非稀释 max 聚合"是同一母题。

误报控制：两个 (1-α/2) 拒绝域的并集 ≈ α，故各路用 α/2。
"""
from __future__ import annotations

import random
import sys
import warnings
from collections import defaultdict
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


def raw(det, ctxs):
    out = []
    for c in ctxs:
        S, m = det._raw_matrix(c)
        Q, _, g = det.calibrator.transform(S, m)
        out.append(det.openset.raw_score(Q, m))
    return np.asarray(out)


def to_pct(ref_sorted, vals):
    """相对正常参考分布的全局分位数（使两路可比）。"""
    return np.searchsorted(ref_sorted, vals, side="left") / (len(ref_sorted) + 1)


def auroc(sa, sn):
    if len(sa) == 0 or len(sn) == 0:
        return float("nan")
    return roc_auc_score(np.r_[np.ones(len(sa)), np.zeros(len(sn))], np.r_[sa, sn])


def main():
    dcal = load_contexts(ROOT / "data/splits/d_cal.pkl.gz")
    random.Random(0).shuffle(dcal)
    fit_norm, test_norm = dcal[:8000], dcal[8000:10000]
    ptx_a = load_contexts(ROOT / "data/splits/d_phish_ptx.pkl.gz")
    ptx_b = load_contexts(ROOT / "data/splits/d_benign_ptx.pkl.gz")
    dknown = load_contexts(ROOT / "data/splits/d_known.pkl.gz")
    by_sub = defaultdict(list)
    for c in ptx_a:
        by_sub[c.latent.get("subtype", "?")].append(c)
    by_type = defaultdict(list)
    for c in dknown:
        by_type[c.latent.get("attack_type")].append(c)

    dicts = load_dictionaries(ROOT / "configs/dictionaries")
    paths = {}
    for name, kw in [("single", dict()), ("folded", dict(two_sided=True))]:
        det = Detector(default_extractors(), dicts, alpha=0.005, min_group_size=150,
                       openset_aggregator="learned", **kw).fit(fit_norm)
        ref = np.sort(raw(det, fit_norm[:4000]))     # 正常参考分布
        paths[name] = (det, ref)

    groups = {"PTX钓鱼整体": ptx_a, "PTX硬负(对照)": ptx_b}
    for t, cs in by_sub.items():
        if len(cs) >= 15:
            groups["  " + t] = cs
    for t, cs in sorted(by_type.items()):
        groups["[六类]" + t] = cs

    # 各路 + 并联的分数
    def score_set(ctxs):
        p = {n: to_pct(ref, raw(det, ctxs)) for n, (det, ref) in paths.items()}
        p["dual_max"] = np.maximum(p["single"], p["folded"])
        return p

    sn = score_set(test_norm)
    print(f"{'指标':<38}{'单侧':>9}{'折叠':>9}{'并联max':>10}")
    for label, cs in groups.items():
        sa = score_set(cs)
        row = "".join(f"{auroc(sa[k], sn[k]):>9.3f}" if k != "dual_max"
                      else f"{auroc(sa[k], sn[k]):>10.3f}"
                      for k in ["single", "folded", "dual_max"])
        print(f"{label:<38}{row}")

    # 并联的误报率（α/2 各路 → 期望 ≈1%）
    thr = {n: np.quantile(sn[n], 0.995) for n in ["single", "folded"]}
    fp = np.mean((sn["single"] >= thr["single"]) | (sn["folded"] >= thr["folded"]))
    print(f"\n并联误报率（各路 α/2=0.5%）= {fp*100:.2f}%  (目标 ≈1%)")


if __name__ == "__main__":
    main()
