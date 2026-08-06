"""8 维格值与 47 维参数的合并方案（两者各有所长，需兼得）。

已测得的权衡：
  8 维格值   —— max 聚合等于特征选择，利于**信号集中**的类型（phishing 0.697）
  47 维参数 —— 保留全部粒度，利于**信号关系型/分散**的类型（sandwich 0.955、ponzi 0.836）
  互有胜负：sandwich +0.179 但 phishing −0.166

三种合并策略对比：
  A 拼接（55 维）           —— 让 IForest 自行取用
  B 双路取 max（全局分位数） —— 与格内参数池的非稀释 max 同一母题
  C 双路取均值             —— 折中对照
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
    p = ROOT / "data/splits/d_open_l4.pkl.gz"
    dopen = load_contexts(p) if p.exists() else None
    random.Random(0).shuffle(dcal)
    fit_norm, test_norm = dcal[:8000], dcal[8000:10000]
    by = defaultdict(list)
    for c in dknown:
        by[c.latent.get("attack_type")].append(c)

    det = Detector(default_extractors(), load_dictionaries(ROOT / "configs/dictionaries"),
                   alpha=0.01, min_group_size=150,
                   openset_aggregator="learned").fit(fit_norm)

    def cellvec(cs):
        out = []
        for c in cs:
            S, m = det._raw_matrix(c)
            Q, _, _ = det.calibrator.transform(S, m)
            out.append([Q[l-1, j-1] if (m[l-1] and np.isfinite(Q[l-1, j-1])) else 0.0
                        for (l, j) in VALID_POSITIONS])
        return np.asarray(out)

    print("收集参数分位数...")
    fit_pq = [collect_param_quantiles(det, c)[0] for c in fit_norm]
    names = sorted({k for d in fit_pq for k in d})

    def paramvec(cs, pq=None):
        pq = pq or [collect_param_quantiles(det, c)[0] for c in cs]
        X = np.full((len(cs), len(names) + 4), 0.5)
        for i, (qs, c) in enumerate(zip(pq, cs)):
            for j, k in enumerate(names):
                if k in qs:
                    X[i, j] = qs[k]
            X[i, len(names):] = c.availability
        return X

    Xc_fit, Xp_fit = cellvec(fit_norm), paramvec(fit_norm, fit_pq)
    print(f"格值 {Xc_fit.shape[1]} 维 | 参数 {Xp_fit.shape[1]} 维 | 拼接 {Xc_fit.shape[1]+Xp_fit.shape[1]} 维")

    m_cell = IsolationForest(n_estimators=300, random_state=0).fit(Xc_fit)
    m_par = IsolationForest(n_estimators=300, random_state=0).fit(Xp_fit)
    m_cat = IsolationForest(n_estimators=300, random_state=0).fit(np.c_[Xc_fit, Xp_fit])
    # 双路取 max/均值需先各自转全局分位数（同尺度可比）
    ref_c = np.sort(-m_cell.score_samples(Xc_fit[:4000]))
    ref_p = np.sort(-m_par.score_samples(Xp_fit[:4000]))

    def pct(ref, v):
        return np.searchsorted(ref, v, side="left") / (len(ref) + 1)

    def scores(cs):
        Xc, Xp = cellvec(cs), paramvec(cs)
        sc = -m_cell.score_samples(Xc)
        sp = -m_par.score_samples(Xp)
        pc, pp = pct(ref_c, sc), pct(ref_p, sp)
        return {"8维格值": sc, "47维参数": sp,
                "A 拼接55维": -m_cat.score_samples(np.c_[Xc, Xp]),
                "B 双路max": np.maximum(pc, pp),
                "C 双路均值": (pc + pp) / 2}

    sn = scores(test_norm)
    keys = list(sn)
    targets = [("整体六类", dknown)] + [(t, by[t]) for t in sorted(by)]
    if dopen:
        targets.append(("★D_open真未知", dopen))

    print(f"\n{'目标':<18}{'n':>5}" + "".join(f"{k:>21}" for k in keys))
    for name, cs in targets:
        sa = scores(cs)
        row = f"{name:<18}{len(cs):>5}"
        for k in keys:
            a = auroc(sa[k], sn[k])
            lo, hi = boot_ci(sa[k], sn[k])
            row += f"{f'{a:.3f} [{lo:.2f},{hi:.2f}]':>21}"
        print(row)


if __name__ == "__main__":
    main()
