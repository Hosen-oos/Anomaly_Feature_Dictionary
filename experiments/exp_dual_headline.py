"""在主设定（d_cal_nbr / d_known_nbr，头条数字 0.790 的那套）上验证两路并联，
并与最强基线 IsolationForest(0.779) 对比。
"""
from __future__ import annotations

import random
import sys
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mlusd.baselines.features import feature_matrix          # noqa: E402
from mlusd.dataset.build import load_contexts                # noqa: E402
from mlusd.match.dictionary import load_dictionaries          # noqa: E402
from mlusd.pipeline import Detector, DualPathDetector         # noqa: E402
from mlusd.signals.factory import default_extractors          # noqa: E402


def auroc(sa, sn):
    return roc_auc_score(np.r_[np.ones(len(sa)), np.zeros(len(sn))], np.r_[sa, sn])


def main():
    dcal = load_contexts(ROOT / "data/splits/d_cal_nbr.pkl.gz")
    dknown = load_contexts(ROOT / "data/splits/d_known_nbr.pkl.gz")
    p = ROOT / "data/splits/d_open.pkl.gz"
    dopen = load_contexts(p) if p.exists() else None
    random.Random(0).shuffle(dcal)
    fit_norm, test_norm = dcal[:8000], dcal[8000:10000]
    by = defaultdict(list)
    for c in dknown:
        by[c.latent.get("attack_type")].append(c)
    dicts = load_dictionaries(ROOT / "configs/dictionaries")

    # 单侧（现默认）
    det1 = Detector(default_extractors(), dicts, alpha=0.01, min_group_size=150,
                    openset_aggregator="learned").fit(fit_norm)
    def s1(cs):
        out = []
        for c in cs:
            S, m = det1._raw_matrix(c)
            Q, _, _ = det1.calibrator.transform(S, m)
            out.append(det1.openset.raw_score(Q, m))
        return np.asarray(out)

    # 两路并联
    det2 = DualPathDetector(default_extractors, dicts, alpha=0.01,
                            min_group_size=150).fit(fit_norm)
    def s2(cs):
        return np.asarray([det2.detect_score(c) for c in cs])

    n1, n2 = s1(test_norm), s2(test_norm)
    labels = ["整体"] + sorted(by) + (["D_open真未知"] if dopen else [])
    sets = {"整体": dknown, **{t: by[t] for t in sorted(by)}}
    if dopen:
        sets["D_open真未知"] = dopen

    print(f"{'指标':<22}{'单侧(现默认)':>14}{'两路并联':>11}{'Δ':>9}")
    macro1, macro2 = [], []
    for k in labels:
        a, b = auroc(s1(sets[k]), n1), auroc(s2(sets[k]), n2)
        print(f"{k:<22}{a:>14.3f}{b:>11.3f}{b-a:>+9.3f}")
        if k not in ("整体", "D_open真未知"):
            macro1.append(a); macro2.append(b)
    print(f"{'六类宏平均':<22}{np.mean(macro1):>14.3f}{np.mean(macro2):>11.3f}"
          f"{np.mean(macro2)-np.mean(macro1):>+9.3f}")

    # 基线参照
    Xn = feature_matrix(fit_norm)
    ifb = IsolationForest(n_estimators=200, random_state=0).fit(Xn)
    ba = auroc(-ifb.score_samples(feature_matrix(dknown)),
               -ifb.score_samples(feature_matrix(test_norm)))
    print(f"\n参照基线 IsolationForest(扁平25维) 整体 AUROC = {ba:.3f}")

    # 并联误报率
    fp = np.mean(n2 >= np.quantile(n2, 1 - 0.01))
    print(f"并联在 α=1% 阈值下的留出误报率 = {fp*100:.2f}%")


if __name__ == "__main__":
    main()
