"""基线对比（A 刊必需）。同数据同任务同指标(AUROC 攻击 vs 正常)。

无监督组（都只在正常上训练，与本框架开放集设定一致，公平对比）：
  IsolationForest / OneClassSVM / LOF（扁平特征）· 原始8信号+IForest（无校准，隔离校准贡献）
  · 本框架（Ū，多层分组校准）
监督组（用攻击标签，参考上界）：RandomForest / HistGradientBoosting（扁平特征）
"""
from __future__ import annotations

import random
import sys
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.ensemble import (
    HistGradientBoostingClassifier, IsolationForest, RandomForestClassifier)
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mlusd.baselines.features import feature_matrix              # noqa: E402
from mlusd.dataset.build import load_contexts                    # noqa: E402
from mlusd.match.dictionary import load_dictionaries              # noqa: E402
from mlusd.pipeline import Detector                               # noqa: E402
from mlusd.signals.factory import default_extractors              # noqa: E402
from mlusd.types import VALID_POSITIONS                           # noqa: E402


def auroc(scores_attack, scores_normal):
    y = np.r_[np.ones(len(scores_attack)), np.zeros(len(scores_normal))]
    return roc_auc_score(y, np.r_[scores_attack, scores_normal])


def main():
    dcal = load_contexts(ROOT / "data/splits/d_cal_nbr.pkl.gz")
    dknown = load_contexts(ROOT / "data/splits/d_known_nbr.pkl.gz")
    random.Random(0).shuffle(dcal)
    fit_norm, test_norm = dcal[:8000], dcal[8000:10000]
    by = defaultdict(list)
    for c in dknown:
        by[c.latent.get("attack_type")].append(c)

    print("提特征...")
    Xn = feature_matrix(fit_norm)
    Xtn = feature_matrix(test_norm)
    Xa = feature_matrix(dknown)
    sc = StandardScaler().fit(Xn)
    Zn, Ztn, Za = sc.transform(Xn), sc.transform(Xtn), sc.transform(Xa)

    # 本框架（学习型聚合器）+ 原始8信号
    det = Detector(default_extractors(), load_dictionaries(ROOT / "configs/dictionaries"),
                   alpha=0.01, min_group_size=150, openset_aggregator="learned").fit(fit_norm)

    def raw8(ctxs):
        out = []
        for c in ctxs:
            S, m = det._raw_matrix(c)
            out.append([S[l - 1, j - 1] if np.isfinite(S[l - 1, j - 1]) else 0.0
                        for (l, j) in VALID_POSITIONS])
        return np.asarray(out)

    def ubar(ctxs):     # 用学习型聚合器的全局排序分（检测力，与全局基线公平比）
        out = []
        for c in ctxs:
            S, m = det._raw_matrix(c)
            Q, _, g = det.calibrator.transform(S, m)
            out.append(det.openset.raw_score(Q, m))
        return np.asarray(out)

    R8n, R8tn, R8a = raw8(fit_norm), raw8(test_norm), raw8(dknown)

    print("\n=== 无监督异常检测 AUROC（攻击 vs 正常，均只在正常上训练）===")
    rows = {}
    # IsolationForest / OCSVM / LOF on flat features
    ifm = IsolationForest(n_estimators=200, random_state=0).fit(Zn)
    rows["IsolationForest(扁平)"] = (-ifm.score_samples(Za), -ifm.score_samples(Ztn))
    ocs = OneClassSVM(gamma="scale", nu=0.05).fit(Zn)
    rows["OneClassSVM(扁平)"] = (-ocs.decision_function(Za), -ocs.decision_function(Ztn))
    lof = LocalOutlierFactor(n_neighbors=20, novelty=True).fit(Zn)
    rows["LOF(扁平)"] = (-lof.decision_function(Za), -lof.decision_function(Ztn))
    # 原始8信号 + IForest（无校准，隔离校准贡献）
    if8 = IsolationForest(n_estimators=200, random_state=0).fit(R8n)
    rows["原始8信号+IForest(无校准)"] = (-if8.score_samples(R8a), -if8.score_samples(R8tn))
    # 本框架
    rows["★本框架(Ū 多层分组校准)"] = (ubar(dknown), ubar(test_norm))

    for name, (sa, sn) in rows.items():
        print(f"  {name:<28} AUROC = {auroc(sa, sn):.3f}")

    # 分类型 AUROC（无监督）
    print("\n=== 分类型 AUROC（本框架 vs 最强扁平基线 IForest）===")
    ours_a = {t: ubar(by[t]) for t in by}
    ours_n = ubar(test_norm)
    if_a = {t: -ifm.score_samples(sc.transform(feature_matrix(by[t]))) for t in by}
    if_n = -ifm.score_samples(Ztn)
    print(f"  {'类型':<20}{'本框架':>8}{'IForest':>9}")
    for t in sorted(by):
        print(f"  {t:<20}{auroc(ours_a[t], ours_n):>8.3f}{auroc(if_a[t], if_n):>9.3f}")

    # 监督组（参考上界）
    print("\n=== 监督分类 AUROC（用攻击标签，train/test 6/4，参考上界）===")
    idx = list(range(len(dknown))); random.Random(1).shuffle(idx)
    k = int(len(idx) * 0.6)
    atr, ate = [dknown[i] for i in idx[:k]], [dknown[i] for i in idx[k:]]
    Xtr = np.vstack([feature_matrix(atr), Xn[:4000]])
    ytr = np.r_[np.ones(len(atr)), np.zeros(4000)]
    Xte_a, Xte_n = feature_matrix(ate), Xtn
    for name, clf in [("RandomForest", RandomForestClassifier(n_estimators=200, random_state=0)),
                      ("HistGradientBoosting", HistGradientBoostingClassifier(random_state=0))]:
        clf.fit(sc.transform(Xtr), ytr)
        sa = clf.predict_proba(sc.transform(Xte_a))[:, 1]
        sn = clf.predict_proba(Xtn)[:, 1]
        print(f"  {name:<28} AUROC = {auroc(sa, sn):.3f}")


if __name__ == "__main__":
    main()
