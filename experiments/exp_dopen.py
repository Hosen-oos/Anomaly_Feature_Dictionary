"""终极开放集测试：D_open 真未知（重入/访问控制等，字典中完全没有）能否被检出为异常。
report 检测 AUROC + 在 α=1% 阈值下的检出率（判为 UNKNOWN）。
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
from mlusd.types import Verdict                              # noqa: E402


def main():
    dcal = load_contexts(ROOT / "data/splits/d_cal_nbr.pkl.gz")
    dknown = load_contexts(ROOT / "data/splits/d_known_nbr.pkl.gz")
    dopen = load_contexts(ROOT / "data/splits/d_open.pkl.gz")
    random.Random(0).shuffle(dcal)
    fit_norm, test_norm = dcal[:8000], dcal[8000:10000]
    det = Detector(default_extractors(), load_dictionaries(ROOT / "configs/dictionaries"),
                   alpha=0.01, min_group_size=150, openset_aggregator="learned").fit(fit_norm)

    def rawscore(ctxs):
        out = []
        for c in ctxs:
            S, m = det._raw_matrix(c)
            Q, _, g = det.calibrator.transform(S, m)
            out.append(det.openset.raw_score(Q, m))
        return np.asarray(out)

    sn = rawscore(test_norm)
    so = rawscore(dopen)
    sk = rawscore(dknown)
    print("=== 终极开放集：D_open 真未知（字典中完全没有的攻击类型）===")
    print(f"  D_open 检测 AUROC（真未知 vs 正常）= {roc_auc_score(np.r_[np.ones(len(so)), np.zeros(len(sn))], np.r_[so, sn]):.3f}  (n={len(dopen)})")
    print(f"  参照：D_known（六类）AUROC          = {roc_auc_score(np.r_[np.ones(len(sk)), np.zeros(len(sn))], np.r_[sk, sn]):.3f}")

    # α=1% 阈值下的检出率（判为 UNKNOWN）
    vo = [det.detect(c).verdict for c in dopen]
    flagged = sum(v in (Verdict.UNKNOWN, Verdict.KNOWN, Verdict.INSUFFICIENT) for v in vo)
    print(f"  D_open 在 α=1% 下检出率（判为异常）= {flagged}/{len(dopen)} = {flagged/len(dopen)*100:.0f}%")
    from collections import Counter
    print(f"  判定分布: {dict(Counter(v.value for v in vo))}")


if __name__ == "__main__":
    main()
