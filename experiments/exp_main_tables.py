"""主实验表（两套内部一致的配置并列报告）。

配置 A 全特征（BigQuery）：d_cal_nbr + d_known(111) + d_open —— 有 L3 与真实邻域图
配置 B 管线对齐（RPC）  ：d_cal_rpc + d_known_rpc(215) + d_open_rpc —— 无 L3/邻域，
                          但攻击样本近 2 倍且与正常集同管线（无采集偽影）

两套各自内部一致，不可跨配置直接比较；并列报告可分离"特征完整性"与"样本量"的影响。
输出：检测 AUROC（整体/分类型/真未知）、消融、基线对照。

    python -m experiments.exp_main_tables
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

from mlusd.baselines.features import feature_matrix                       # noqa: E402
from mlusd.dataset.build import load_contexts                             # noqa: E402
from mlusd.match.dictionary import load_dictionaries                       # noqa: E402
from mlusd.pipeline import Detector                                        # noqa: E402
from mlusd.signals.factory import ablation_extractors, default_extractors   # noqa: E402
from mlusd.types import VALID_POSITIONS                                    # noqa: E402


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


def run_config(name, cal_path, known_path, open_path, n_fit, n_test):
    print(f"\n{'='*66}\n配置 {name}\n{'='*66}")
    dcal = load_contexts(ROOT / cal_path)
    dknown = load_contexts(ROOT / known_path)
    p = ROOT / open_path
    dopen = load_contexts(p) if p.exists() else None
    random.Random(0).shuffle(dcal)
    fit_norm, test_norm = dcal[:n_fit], dcal[n_fit:n_fit + n_test]
    by = defaultdict(list)
    for c in dknown:
        by[c.latent.get("attack_type")].append(c)
    print(f"正常 fit {len(fit_norm)} / test {len(test_norm)} | 攻击 {len(dknown)} | "
          f"真未知 {len(dopen) if dopen else 0}")
    print("攻击分布:", {t: len(v) for t, v in sorted(by.items())})

    dicts = load_dictionaries(ROOT / "configs/dictionaries")
    det = Detector(default_extractors(), dicts, alpha=0.01, min_group_size=150,
                   openset_aggregator="learned").fit(fit_norm)
    sn = scores(det, test_norm)
    overall = auroc(scores(det, dknown), sn)

    print("\n--- 检测 AUROC ---")
    print(f"  {'整体':<22}{overall:.3f}")
    macro = []
    for t in sorted(by):
        a = auroc(scores(det, by[t]), sn)
        macro.append(a)
        print(f"  {t:<22}{a:.3f}  (n={len(by[t])})")
    print(f"  {'六类宏平均':<22}{np.mean(macro):.3f}")
    if dopen:
        print(f"  {'D_open 真未知':<22}{auroc(scores(det, dopen), sn):.3f}  (n={len(dopen)})")

    print("\n--- 消融（整体 AUROC）---")
    print(f"  {'完整模型':<26}{overall:.3f}")
    for tag, kw, exts in [
        ("Fisher 聚合(替学习)", dict(openset_mode="fisher", openset_aggregator="fisher"),
         default_extractors()),
        ("−量值化参数", dict(openset_aggregator="learned"), default_extractors(magnitude=False)),
        ("格内聚合 max→均值", dict(openset_aggregator="learned"), ablation_extractors("mean")),
        ("−分组校准", dict(openset_aggregator="learned", single_group=True),
         default_extractors()),
    ]:
        d = Detector(exts, dicts, alpha=0.01, min_group_size=150, **kw).fit(fit_norm)
        a = auroc(scores(d, dknown), scores(d, test_norm))
        print(f"  {tag:<26}{a:.3f}   (Δ={a-overall:+.3f})")

    # −校准（原始 8 信号 + IForest）
    def raw8(cs):
        out = []
        for c in cs:
            S, m = det._raw_matrix(c)
            out.append([S[l-1, j-1] if np.isfinite(S[l-1, j-1]) else 0.0
                        for (l, j) in VALID_POSITIONS])
        return np.asarray(out)
    m8 = IsolationForest(n_estimators=200, random_state=0).fit(raw8(fit_norm))
    a = auroc(-m8.score_samples(raw8(dknown)), -m8.score_samples(raw8(test_norm)))
    print(f"  {'−校准(原始S8+IForest)':<26}{a:.3f}   (Δ={a-overall:+.3f})")

    print("\n--- 基线对照 ---")
    Xn = feature_matrix(fit_norm)
    mb = IsolationForest(n_estimators=200, random_state=0).fit(Xn)
    ab = auroc(-mb.score_samples(feature_matrix(dknown)),
               -mb.score_samples(feature_matrix(test_norm)))
    print(f"  {'IsolationForest(扁平)':<26}{ab:.3f}   (本框架 {overall:.3f}, "
          f"Δ={overall-ab:+.3f})")


def main():
    run_config("A 全特征 (BigQuery: 有 L3 + 邻域图)",
               "data/splits/d_cal_nbr.pkl.gz", "data/splits/d_known_nbr.pkl.gz",
               "data/splits/d_open.pkl.gz", 8000, 2000)
    run_config("B 管线对齐 (RPC: 无 L3/邻域, 攻击样本 2x)",
               "data/splits/d_cal_rpc.pkl.gz", "data/splits/d_known_rpc.pkl.gz",
               "data/splits/d_open_rpc.pkl.gz", 4000, 1900)


if __name__ == "__main__":
    main()
