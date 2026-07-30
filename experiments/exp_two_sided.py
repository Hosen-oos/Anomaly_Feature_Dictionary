"""双侧校准消融：能否修复"异常地简单"类型（payable/地址投毒钓鱼 AUROC 0.178）
且不伤害原有类型。单侧 vs 双侧，同数据同协议对比。
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
    dcal = load_contexts(ROOT / "data/splits/d_cal.pkl.gz")
    random.Random(0).shuffle(dcal)
    fit_norm, test_norm = dcal[:8000], dcal[8000:10000]
    ptx_a = load_contexts(ROOT / "data/splits/d_phish_ptx.pkl.gz")
    ptx_b = load_contexts(ROOT / "data/splits/d_benign_ptx.pkl.gz")
    dknown = load_contexts(ROOT / "data/splits/d_known.pkl.gz")   # 非邻域版保持一致
    by_sub = defaultdict(list)
    for c in ptx_a:
        by_sub[c.latent.get("subtype", "?")].append(c)
    by_type = defaultdict(list)
    for c in dknown:
        by_type[c.latent.get("attack_type")].append(c)

    res = {}
    configs = [("单侧", dict()), ("双侧折叠", dict(two_sided=True)),
               ("双尾特征", dict(dual_tail=True))]
    for name, kw in configs:
        det = Detector(default_extractors(), load_dictionaries(ROOT / "configs/dictionaries"),
                       alpha=0.01, min_group_size=150, openset_aggregator="learned",
                       **kw).fit(fit_norm)
        sn = scores(det, test_norm)
        r = {"PTX钓鱼整体": auroc(scores(det, ptx_a), sn),
             "PTX钓鱼vs硬负": auroc(scores(det, ptx_a), scores(det, ptx_b))}
        for t, cs in by_sub.items():
            if len(cs) >= 15:
                r["  " + t] = auroc(scores(det, cs), sn)
        for t, cs in sorted(by_type.items()):
            r["[六类]" + t] = auroc(scores(det, cs), sn)
        res[name] = r

    names = [n for n, _ in configs]
    print(f"{'指标':<38}" + "".join(f"{n:>10}" for n in names))
    for k in res[names[0]]:
        print(f"{k:<38}" + "".join(f"{res[n][k]:>10.3f}" for n in names))


if __name__ == "__main__":
    main()
