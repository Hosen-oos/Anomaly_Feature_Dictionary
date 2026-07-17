"""验证学习型聚合器：AUROC 提升 + FP 控制仍在（对比 fisher）。"""
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


def ubars(det, ctxs):
    """返回 (决策用组内分位ubar, 排序用原始分score, 组). fisher 无 raw_score 时用 ubar 兜底。"""
    out = []
    for c in ctxs:
        S, m = det._raw_matrix(c)
        Q, _, g = det.calibrator.transform(S, m)
        ub = det.openset.ubar(Q, m, g)
        rs = det.openset.raw_score(Q, m) if hasattr(det.openset, "raw_score") else ub
        out.append((ub, rs, "".join(map(str, m))))
    return out


def main():
    dcal = load_contexts(ROOT / "data/splits/d_cal_nbr.pkl.gz")
    dknown = load_contexts(ROOT / "data/splits/d_known_nbr.pkl.gz")
    random.Random(0).shuffle(dcal)
    fit_norm, test_norm = dcal[:8000], dcal[8000:10000]

    for agg in ["fisher", "learned"]:
        det = Detector(default_extractors(), load_dictionaries(ROOT / "configs/dictionaries"),
                       alpha=0.01, min_group_size=150, openset_aggregator=agg).fit(fit_norm)
        a = ubars(det, dknown)
        n = ubars(det, test_norm)
        y = np.r_[np.ones(len(a)), np.zeros(len(n))]
        # AUROC 用排序分（learned=原始分, fisher=Ū）；FP 用决策分位 ubar
        auroc_rank = roc_auc_score(y, np.r_[[x[1] for x in a], [x[1] for x in n]])
        auroc_ubar = roc_auc_score(y, np.r_[[x[0] for x in a], [x[0] for x in n]])
        tau = det.openset.threshold
        fp = defaultdict(lambda: [0, 0])
        for ub, _, k in n:
            fp[k][1] += 1
            if ub >= tau:
                fp[k][0] += 1
        fpstr = "  ".join(f"{k}:{v[0]/max(v[1],1)*100:.1f}%" for k, v in sorted(fp.items()))
        print(f"[{agg:7}] AUROC(排序分)={auroc_rank:.3f}  AUROC(Ū)={auroc_ubar:.3f}  各组FP: {fpstr}")


if __name__ == "__main__":
    main()
