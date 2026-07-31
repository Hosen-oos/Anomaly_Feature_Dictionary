"""补齐关键对照：基线方法在 D_open 真未知上的表现。

此前只报了本框架在 D_open 上的 0.928，未与基线比较——若 IsolationForest 等通用方法
同样能检出真未知，则"开放集检测"不构成差异化优势。本实验补上这一对照。
"""
from __future__ import annotations

import random
import sys
import warnings
from pathlib import Path

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mlusd.baselines.features import feature_matrix          # noqa: E402
from mlusd.dataset.build import load_contexts                # noqa: E402
from mlusd.match.dictionary import load_dictionaries          # noqa: E402
from mlusd.pipeline import Detector                           # noqa: E402
from mlusd.signals.factory import default_extractors          # noqa: E402


def auroc(sa, sn):
    return roc_auc_score(np.r_[np.ones(len(sa)), np.zeros(len(sn))], np.r_[sa, sn])


def run(name, cal, known, opn, n_fit, n_test):
    dcal = load_contexts(ROOT / cal)
    dknown = load_contexts(ROOT / known)
    dopen = load_contexts(ROOT / opn)
    random.Random(0).shuffle(dcal)
    fit_norm, test_norm = dcal[:n_fit], dcal[n_fit:n_fit + n_test]

    det = Detector(default_extractors(), load_dictionaries(ROOT / "configs/dictionaries"),
                   alpha=0.01, min_group_size=150,
                   openset_aggregator="learned").fit(fit_norm)
    def ours(cs):
        out = []
        for c in cs:
            S, m = det._raw_matrix(c)
            Q, _, _ = det.calibrator.transform(S, m)
            out.append(det.openset.raw_score(Q, m))
        return np.asarray(out)

    Xn = feature_matrix(fit_norm)
    sc = StandardScaler().fit(Xn)
    Zn, Zt = sc.transform(Xn), sc.transform(feature_matrix(test_norm))
    Zk, Zo = sc.transform(feature_matrix(dknown)), sc.transform(feature_matrix(dopen))
    ifm = IsolationForest(n_estimators=200, random_state=0).fit(Zn)
    lof = LocalOutlierFactor(n_neighbors=20, novelty=True).fit(Zn)

    print(f"\n=== {name} ===")
    print(f"{'方法':<28}{'已知六类':>10}{'D_open真未知':>14}")
    print(f"{'★本框架':<28}{auroc(ours(dknown), ours(test_norm)):>10.3f}"
          f"{auroc(ours(dopen), ours(test_norm)):>14.3f}")
    print(f"{'IsolationForest(扁平)':<28}{auroc(-ifm.score_samples(Zk), -ifm.score_samples(Zt)):>10.3f}"
          f"{auroc(-ifm.score_samples(Zo), -ifm.score_samples(Zt)):>14.3f}")
    print(f"{'LOF(扁平)':<28}{auroc(-lof.decision_function(Zk), -lof.decision_function(Zt)):>10.3f}"
          f"{auroc(-lof.decision_function(Zo), -lof.decision_function(Zt)):>14.3f}")


def main():
    run("配置 A 全特征 (BigQuery)", "data/splits/d_cal_nbr.pkl.gz",
        "data/splits/d_known_nbr.pkl.gz", "data/splits/d_open.pkl.gz", 8000, 2000)
    run("配置 B 管线对齐 (RPC)", "data/splits/d_cal_rpc.pkl.gz",
        "data/splits/d_known_rpc.pkl.gz", "data/splits/d_open_rpc.pkl.gz", 4000, 1900)


if __name__ == "__main__":
    main()
