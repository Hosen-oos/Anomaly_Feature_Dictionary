"""假设：秩校准在极端尾部饱和，是 rug_pull 落后 IForest 的原因。

诊断依据：rug_pull 的签名是**极端原始计数**（act_transfer 75.95 vs 正常 0.75，z=70；
n_logs 104.9 vs 1.57；n_calls 205 vs 3.84）。分位数把 76/200/1000 都映射到 ~0.999，
量级信息被压掉；IForest 用原始计数则能区分。

修法：在**参数路**（双路表示的辅助路）补几维**未经分位数变换的 log 量级**通道——
分位数保证跨类型可比（主路），log 量级保留极端尾部分辨率（辅路）。
本实验检验该修法能否收窄 rug_pull 与 IForest 的差距。
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
from mlusd.match.dictionary import load_dictionaries              # noqa: E402
from mlusd.pipeline import Detector                               # noqa: E402
from mlusd.signals.factory import default_extractors              # noqa: E402
from mlusd.signals.l2_semantic import lift_actions                # noqa: E402


def mag_channels(ctx) -> list[float]:
    """未饱和的 log 量级通道（不做分位数变换）。"""
    n_logs = len(ctx.event_logs or [])
    n_calls = len(ctx.trace.calls) if ctx.trace else 0
    acts = lift_actions(ctx)
    n_tr = sum(1 for a in acts if a.kind == "transfer")
    return [np.log1p(n_logs), np.log1p(n_calls), np.log1p(len(acts)), np.log1p(n_tr),
            float(ctx.trace.max_depth) if ctx.trace else 0.0]


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
                   openset_aggregator="learned", dual_repr=True).fit(fit_norm)

    def base_pv(cs):
        return np.vstack([det._param_vec(c) for c in cs])

    def mag_pv(cs):
        return np.vstack([np.r_[det._param_vec(c), mag_channels(c)] for c in cs])

    # 两种参数路：原始（纯分位数） vs 补量级通道
    variants = {}
    for tag, fn in [("参数路(纯分位数)", base_pv), ("参数路+量级通道", mag_pv)]:
        Xf = fn(fit_norm)
        m = IsolationForest(n_estimators=300, random_state=0).fit(Xf)
        ref = np.sort(-m.score_samples(Xf[:4000]))
        variants[tag] = (m, ref, fn)

    # 格值路（与双路 max 组合用）
    def cellscore(cs):
        out = []
        for c in cs:
            S, mk = det._raw_matrix(c)
            Q, _, _ = det.calibrator.transform(S, mk)
            out.append(-det.openset._model.score_samples(
                det.openset._vec(Q, mk).reshape(1, -1))[0])
        return np.asarray(out)
    ref_c = np.sort(cellscore(fit_norm[:4000]))
    pct = lambda ref, v: np.searchsorted(ref, v, side="left") / (len(ref) + 1)  # noqa: E731

    def dual(cs, tag):
        m, ref, fn = variants[tag]
        return np.maximum(pct(ref_c, cellscore(cs)),
                          pct(ref, -m.score_samples(fn(cs))))

    m_flat = IsolationForest(n_estimators=300, random_state=0).fit(feature_matrix(fit_norm))
    flat = lambda cs: -m_flat.score_samples(feature_matrix(cs))  # noqa: E731

    reps = {"双路(现默认)": lambda cs: dual(cs, "参数路(纯分位数)"),
            "双路+量级通道": lambda cs: dual(cs, "参数路+量级通道"),
            "IForest(扁平)": flat}
    sn = {k: f(test_norm) for k, f in reps.items()}

    targets = [("整体六类", dknown)] + [(t, by[t]) for t in sorted(by)]
    if dopen:
        targets.append(("★D_open真未知", dopen))
    print(f"\n{'目标':<18}{'n':>5}" + "".join(f"{k:>23}" for k in reps))
    for name, cs in targets:
        row = f"{name:<18}{len(cs):>5}"
        for k, f in reps.items():
            sa = f(cs)
            a = auroc(sa, sn[k]); lo, hi = boot_ci(sa, sn[k])
            row += f"{f'{a:.3f} [{lo:.2f},{hi:.2f}]':>23}"
        print(row)


if __name__ == "__main__":
    main()
