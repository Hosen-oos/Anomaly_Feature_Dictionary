"""验证：双路表示集成进 Detector 后（dual_repr=True 为默认），主表与显著性是否兑现。

对比 dual_repr=False（原 8 维格值）与 True（双路 max），并对 IForest 做配对显著性检验。
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
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mlusd.baselines.features import feature_matrix          # noqa: E402
from mlusd.dataset.build import load_contexts                # noqa: E402
from mlusd.match.dictionary import load_dictionaries          # noqa: E402
from mlusd.pipeline import Detector                           # noqa: E402
from mlusd.signals.factory import default_extractors          # noqa: E402

N_BOOT = 1500


def auroc(sa, sn):
    return roc_auc_score(np.r_[np.ones(len(sa)), np.zeros(len(sn))], np.r_[sa, sn])


def paired(sa1, sn1, sa2, sn2, n=N_BOOT, seed=0):
    rng = np.random.default_rng(seed)
    d = np.empty(n)
    for b in range(n):
        ia = rng.integers(0, len(sa1), len(sa1))
        inn = rng.integers(0, len(sn1), len(sn1))
        d[b] = auroc(sa1[ia], sn1[inn]) - auroc(sa2[ia], sn2[inn])
    return d.mean(), np.percentile(d, 2.5), np.percentile(d, 97.5)


def main():
    dcal = load_contexts(ROOT / "data/splits/d_cal_nbr_blk_l4_sw.pkl.gz")
    dknown = load_contexts(ROOT / "data/splits/d_known_nbr_blk_l4_sw.pkl.gz")
    p = ROOT / "data/splits/d_open_l4.pkl.gz"
    dopen = load_contexts(p) if p.exists() else None
    random.Random(0).shuffle(dcal)
    fit_norm, test_norm = dcal[:8000], dcal[8000:10000]
    by = defaultdict(list)
    for c in dknown:
        by[c.latent.get("attack_type")].append(c)
    dicts = load_dictionaries(ROOT / "configs/dictionaries")

    dets = {}
    for tag, dr in [("单路(8维格值)", False), ("双路(默认)", True)]:
        d = Detector(default_extractors(), dicts, alpha=0.01, min_group_size=150,
                     openset_aggregator="learned", dual_repr=dr).fit(fit_norm)
        dets[tag] = d
        print(f"{tag}: 参数维度 {len(d._param_names)}")

    def sc(det, cs):
        out = []
        for c in cs:
            S, m = det._raw_matrix(c)
            Q, _, _ = det.calibrator.transform(S, m)
            pv = det._param_vec(c) if det._param_names else None
            out.append(det.openset.raw_score(Q, m, pv))
        return np.asarray(out)

    Xn = feature_matrix(fit_norm)
    scaler = StandardScaler().fit(Xn)
    ifm = IsolationForest(n_estimators=300, random_state=0).fit(scaler.transform(Xn))
    base = lambda cs: -ifm.score_samples(scaler.transform(feature_matrix(cs)))  # noqa: E731

    n1 = sc(dets["单路(8维格值)"], test_norm)
    n2 = sc(dets["双路(默认)"], test_norm)
    nb = base(test_norm)

    targets = [("整体六类", dknown)] + [(t, by[t]) for t in sorted(by)]
    if dopen:
        targets.append(("★D_open真未知", dopen))
    print(f"\n{'目标':<18}{'n':>5}{'单路':>9}{'双路':>9}{'IForest':>9}"
          f"{'双路−IForest [95%CI]':>28}")
    for name, cs in targets:
        a1, a2, ab = sc(dets["单路(8维格值)"], cs), sc(dets["双路(默认)"], cs), base(cs)
        v1, v2, vb = auroc(a1, n1), auroc(a2, n2), auroc(ab, nb)
        m, lo, hi = paired(a2, n2, ab, nb)
        star = "*" if (lo > 0 or hi < 0) else " "
        print(f"{name:<18}{len(cs):>5}{v1:>9.3f}{v2:>9.3f}{vb:>9.3f}"
              f"{f'{m:+.3f} [{lo:+.2f},{hi:+.2f}]':>26}{star}")
    print("  * 表示差异显著（配对自助 95%CI 不含 0）")


if __name__ == "__main__":
    main()
