"""假设检验：开放集检测的瓶颈是"只看 8 个聚合格值"。

诊断依据：
- sandwich 的 opposite_same_pool 命中 82.5% vs 正常 1.7%（48 倍富集），但进入 L1-j2 格后
  与 fan_in/cycle/benford/total_flow_mag 等一起取 max，被更极端的参数掩盖；
- 同一信号在 M4（per-param 对比式字典）让 sandwich 归因达 10/10，
  在 M5（8 格聚合）检测只 +0.014；
- 有监督实验：8 维校准信号 0.987 vs 25 维扁平特征 0.996 —— 压缩有损。

本实验让开放集检测器直接消费**全部参数分位数**（约 40+ 维，掩码感知），
与现行 8 维格值方案对比，重点看 sandwich / rug_pull 是否回升。
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

from mlusd.baselines.features import feature_matrix              # noqa: E402
from mlusd.dataset.build import load_contexts                    # noqa: E402
from mlusd.match.contrastive import collect_param_quantiles       # noqa: E402
from mlusd.match.dictionary import load_dictionaries              # noqa: E402
from mlusd.pipeline import Detector                               # noqa: E402
from mlusd.signals.factory import default_extractors              # noqa: E402
from mlusd.types import VALID_POSITIONS                           # noqa: E402


def auroc(sa, sn):
    return roc_auc_score(np.r_[np.ones(len(sa)), np.zeros(len(sn))], np.r_[sa, sn])


def boot_ci(sa, sn, n=1000, seed=0):
    rng = np.random.default_rng(seed)
    v = [auroc(sa[rng.integers(0, len(sa), len(sa))],
               sn[rng.integers(0, len(sn), len(sn))]) for _ in range(n)]
    return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


def main():
    dcal = load_contexts(ROOT / "data/splits/d_cal_nbr_blk_l4_sw.pkl.gz")
    dknown = load_contexts(ROOT / "data/splits/d_known_nbr_blk_l4_sw.pkl.gz")
    random.Random(0).shuffle(dcal)
    fit_norm, test_norm = dcal[:8000], dcal[8000:10000]
    by = defaultdict(list)
    for c in dknown:
        by[c.latent.get("attack_type")].append(c)

    det = Detector(default_extractors(), load_dictionaries(ROOT / "configs/dictionaries"),
                   alpha=0.01, min_group_size=150,
                   openset_aggregator="learned").fit(fit_norm)

    # ---- 表示 1：现行 8 维格值（经学习聚合器）----
    def cells8(cs):
        out = []
        for c in cs:
            S, m = det._raw_matrix(c)
            Q, _, _ = det.calibrator.transform(S, m)
            out.append(det.openset.raw_score(Q, m))
        return np.asarray(out)

    # ---- 表示 2：全部参数分位数（掩码感知）----
    print("收集参数分位数...")
    fit_pq = [collect_param_quantiles(det, c)[0] for c in fit_norm]
    names = sorted({k for d in fit_pq for k in d})
    print(f"参数维度 {len(names)}（对比：8 维格值 / 25 维扁平特征）")

    def pvec(qs_list, ctxs):
        X = np.full((len(qs_list), len(names) + 4), 0.5)   # 缺失填中性 0.5
        for i, (qs, c) in enumerate(zip(qs_list, ctxs)):
            for j, k in enumerate(names):
                if k in qs:
                    X[i, j] = qs[k]
            X[i, len(names):] = c.availability             # 附掩码，让模型知道缺了什么
        return X

    Xf = pvec(fit_pq, fit_norm)
    m_pp = IsolationForest(n_estimators=300, random_state=0).fit(Xf)

    def params_score(cs):
        qs = [collect_param_quantiles(det, c)[0] for c in cs]
        return -m_pp.score_samples(pvec(qs, cs))

    # ---- 表示 3：扁平手工特征（IForest 基线，参照）----
    m_flat = IsolationForest(n_estimators=300, random_state=0).fit(feature_matrix(fit_norm))
    flat_score = lambda cs: -m_flat.score_samples(feature_matrix(cs))  # noqa: E731

    reps = {"现行(8维格值)": cells8, "本实验(全参数分位数)": params_score,
            "IForest(25维扁平)": flat_score}
    sn = {k: f(test_norm) for k, f in reps.items()}

    print(f"\n{'目标':<20}{'n':>5}" + "".join(f"{k:>24}" for k in reps))
    targets = [("整体六类", dknown)] + [(t, by[t]) for t in sorted(by)]
    for name, cs in targets:
        row = f"{name:<20}{len(cs):>5}"
        for k, f in reps.items():
            sa = f(cs)
            a = auroc(sa, sn[k])
            lo, hi = boot_ci(sa, sn[k])
            row += f"{f'{a:.3f} [{lo:.2f},{hi:.2f}]':>24}"
        print(row)


if __name__ == "__main__":
    main()
