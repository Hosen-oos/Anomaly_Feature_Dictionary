"""决定性诊断：校准到底有没有检测价值？
对同一个 IsolationForest 聚合器，比较喂入不同的 8 维表示：
  原始信号 S8（无校准） vs 校准分位 Q8 vs 尾部放大 T8 vs Fisher-Ū（我们的聚合）
以及扁平特征（25维，上界参照）。若 Q8/T8+IForest 追平或超过扁平，则校准有用，只需换聚合器。
"""
from __future__ import annotations

import random
import sys
import warnings
from pathlib import Path

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mlusd.baselines.features import feature_matrix              # noqa: E402
from mlusd.dataset.build import load_contexts                    # noqa: E402
from mlusd.match.dictionary import load_dictionaries              # noqa: E402
from mlusd.pipeline import Detector                               # noqa: E402
from mlusd.signals.factory import default_extractors              # noqa: E402
from mlusd.types import VALID_POSITIONS                           # noqa: E402


def auroc(sa, sn):
    return roc_auc_score(np.r_[np.ones(len(sa)), np.zeros(len(sn))], np.r_[sa, sn])


def main():
    dcal = load_contexts(ROOT / "data/splits/d_cal_nbr.pkl.gz")
    dknown = load_contexts(ROOT / "data/splits/d_known_nbr.pkl.gz")
    random.Random(0).shuffle(dcal)
    fit_norm, test_norm = dcal[:8000], dcal[8000:10000]
    det = Detector(default_extractors(), load_dictionaries(ROOT / "configs/dictionaries"),
                   alpha=0.01, min_group_size=150).fit(fit_norm)

    def mats(ctxs):
        s8, q8, t8, ub = [], [], [], []
        for c in ctxs:
            S, m = det._raw_matrix(c)
            Q, T, g = det.calibrator.transform(S, m)
            s8.append([S[l-1, j-1] if np.isfinite(S[l-1, j-1]) else 0.0 for (l, j) in VALID_POSITIONS])
            q8.append([Q[l-1, j-1] if np.isfinite(Q[l-1, j-1]) else 0.0 for (l, j) in VALID_POSITIONS])
            t8.append([T[l-1, j-1] if np.isfinite(T[l-1, j-1]) else 0.0 for (l, j) in VALID_POSITIONS])
            ub.append(det.openset.ubar(Q, m, g))
        return np.asarray(s8), np.asarray(q8), np.asarray(t8), np.asarray(ub)

    print("提取表示...")
    s8n, q8n, t8n, _ = mats(fit_norm)
    s8tn, q8tn, t8tn, ubtn = mats(test_norm)
    s8a, q8a, t8a, uba = mats(dknown)
    Xn, Xtn, Xa = feature_matrix(fit_norm), feature_matrix(test_norm), feature_matrix(dknown)

    def ifauroc(tr, ta, tn):
        m = IsolationForest(n_estimators=200, random_state=0).fit(tr)
        return auroc(-m.score_samples(ta), -m.score_samples(tn))

    print("\n=== 同一 IForest 聚合器，不同 8 维表示（+ 扁平上界）===")
    print(f"  扁平25维 + IForest        AUROC = {ifauroc(Xn, Xa, Xtn):.3f}  (上界参照)")
    print(f"  原始S8   + IForest        AUROC = {ifauroc(s8n, s8a, s8tn):.3f}")
    print(f"  校准Q8   + IForest        AUROC = {ifauroc(q8n, q8a, q8tn):.3f}")
    print(f"  尾部T8   + IForest        AUROC = {ifauroc(t8n, t8a, t8tn):.3f}")
    print(f"  我们的 Fisher-Ū           AUROC = {auroc(uba, ubtn):.3f}")
    # 校准表示 + 扁平拼接（校准是否补充扁平）
    print(f"  扁平25 ⊕ 校准Q8 + IForest AUROC = {ifauroc(np.c_[Xn, q8n], np.c_[Xa, q8a], np.c_[Xtn, q8tn]):.3f}")


if __name__ == "__main__":
    main()
