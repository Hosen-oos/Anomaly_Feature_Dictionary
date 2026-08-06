"""主结果的置信区间与显著性检验。

动机：分类型攻击样本仅 n≈20，单个样本进出即造成 1/20=0.05 的 AUROC 变化——
主表所有分类型数字都建立在这个粒度上却无误差棒，是最大的可攻击点。本实验补上：

1) 自助置信区间（分层重采样攻击与正常两侧）
2) 本框架 vs 基线的**配对**自助检验（同一批样本上比较，控制样本波动）
   —— 重点验证核心卖点 D_open（真未知）的优势是否显著
3) 同时在 d_known_rpc（每类 n 更大：flash 48/rug 45/ponzi 42）上复算，
   看结论是否随样本量稳定

    python -m experiments.exp_significance
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

N_BOOT = 2000


def auroc(sa, sn):
    if len(sa) == 0 or len(sn) == 0:
        return float("nan")
    return roc_auc_score(np.r_[np.ones(len(sa)), np.zeros(len(sn))], np.r_[sa, sn])


def boot_ci(sa, sn, n_boot=N_BOOT, seed=0):
    """分层自助：攻击侧与正常侧各自有放回重采样。"""
    rng = np.random.default_rng(seed)
    vals = np.empty(n_boot)
    for b in range(n_boot):
        ia = rng.integers(0, len(sa), len(sa))
        inn = rng.integers(0, len(sn), len(sn))
        vals[b] = auroc(sa[ia], sn[inn])
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def paired_boot(sa1, sn1, sa2, sn2, n_boot=N_BOOT, seed=0):
    """配对自助检验：两方法在**同一批重采样样本**上比较，控制样本波动。
    返回 (Δ均值, Δ的95%CI, P(Δ>0))。"""
    rng = np.random.default_rng(seed)
    d = np.empty(n_boot)
    for b in range(n_boot):
        ia = rng.integers(0, len(sa1), len(sa1))
        inn = rng.integers(0, len(sn1), len(sn1))
        d[b] = auroc(sa1[ia], sn1[inn]) - auroc(sa2[ia], sn2[inn])
    return float(d.mean()), (float(np.percentile(d, 2.5)),
                             float(np.percentile(d, 97.5))), float((d > 0).mean())


def run(tag, cal_p, known_p, open_p, n_fit, n_test):
    dcal = load_contexts(ROOT / cal_p)
    dknown = load_contexts(ROOT / known_p)
    p = ROOT / open_p
    dopen = load_contexts(p) if p.exists() else None
    random.Random(0).shuffle(dcal)
    fit_norm, test_norm = dcal[:n_fit], dcal[n_fit:n_fit + n_test]
    by = defaultdict(list)
    for c in dknown:
        by[c.latent.get("attack_type")].append(c)

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
    ifm = IsolationForest(n_estimators=200, random_state=0).fit(sc.transform(Xn))
    base = lambda cs: -ifm.score_samples(sc.transform(feature_matrix(cs)))  # noqa: E731

    on, bn = ours(test_norm), base(test_norm)
    print(f"\n{'='*72}\n{tag}\n{'='*72}")
    print(f"{'目标':<20}{'n':>5}{'本框架 [95%CI]':>26}{'IForest':>10}"
          f"{'Δ [95%CI]':>24}{'P(Δ>0)':>9}")

    targets = [("整体六类", dknown)] + [(t, by[t]) for t in sorted(by)]
    if dopen:
        targets.append(("★D_open真未知", dopen))
    for name, cs in targets:
        oa, ba = ours(cs), base(cs)
        a, (lo, hi) = auroc(oa, on), boot_ci(oa, on)
        b = auroc(ba, bn)
        dm, (dlo, dhi), pgt = paired_boot(oa, on, ba, bn)
        star = "*" if (dlo > 0 or dhi < 0) else " "   # CI 不含 0 即显著
        print(f"{name:<20}{len(cs):>5}{f'{a:.3f} [{lo:.2f},{hi:.2f}]':>26}{b:>10.3f}"
              f"{f'{dm:+.3f} [{dlo:+.2f},{dhi:+.2f}]':>24}{pgt:>8.2f}{star}")
    print("  * 表示 Δ 的 95% 自助 CI 不含 0（差异显著）")


def main():
    run("配置 A 全特征（BigQuery，每类 n≈20）",
        "data/splits/d_cal_nbr.pkl.gz", "data/splits/d_known_nbr.pkl.gz",
        "data/splits/d_open.pkl.gz", 8000, 2000)
    run("配置 B 管线对齐（RPC，每类 n 更大：flash48/rug45/ponzi42）",
        "data/splits/d_cal_rpc.pkl.gz", "data/splits/d_known_rpc.pkl.gz",
        "data/splits/d_open_rpc.pkl.gz", 4000, 1900)


if __name__ == "__main__":
    main()
