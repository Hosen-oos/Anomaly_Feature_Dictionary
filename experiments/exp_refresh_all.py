"""用最终配置（学习聚合器 + 量值化）重刷全部实验表，保证自洽。

实验二：每类开放集检测 AUROC（学习聚合器原始排序分）+ 整体。
实验三：消融——各设计成分对整体 AUROC 的贡献（学习vs Fisher、量值化、校准、分组、格内聚合）。
输出为论文可直接引用的表。
"""
from __future__ import annotations

import random
import sys
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mlusd.baselines.features import feature_matrix              # noqa: E402
from mlusd.dataset.build import load_contexts                    # noqa: E402
from mlusd.match.dictionary import load_dictionaries              # noqa: E402
from mlusd.pipeline import Detector                               # noqa: E402
from mlusd.signals.factory import ablation_extractors, default_extractors  # noqa: E402
from mlusd.types import VALID_POSITIONS                           # noqa: E402

DICT = ROOT / "configs/dictionaries"


def score_fn(det):
    """检测排序分：learned 用原始分，fisher 用 Ū。"""
    def f(ctxs):
        out = []
        for c in ctxs:
            S, m = det._raw_matrix(c)
            Q, _, g = det.calibrator.transform(S, m)
            out.append(det.openset.raw_score(Q, m) if hasattr(det.openset, "raw_score")
                       else det.openset.ubar(Q, m, g))
        return np.asarray(out)
    return f


def auroc(sa, sn):
    return roc_auc_score(np.r_[np.ones(len(sa)), np.zeros(len(sn))], np.r_[sa, sn])


def fit(exts, fit_norm, **kw):
    return Detector(exts, load_dictionaries(DICT), alpha=0.01, min_group_size=150, **kw).fit(fit_norm)


def main():
    dcal = load_contexts(ROOT / "data/splits/d_cal_nbr.pkl.gz")
    dknown = load_contexts(ROOT / "data/splits/d_known_nbr.pkl.gz")
    random.Random(0).shuffle(dcal)
    fit_norm, test_norm = dcal[:8000], dcal[8000:10000]
    by = defaultdict(list)
    for c in dknown:
        by[c.latent.get("attack_type")].append(c)

    full = fit(default_extractors(), fit_norm, openset_aggregator="learned")
    sf = score_fn(full)
    sn = sf(test_norm)
    overall = auroc(sf(dknown), sn)

    print("=== 实验二：开放集检测 AUROC（最终配置：学习聚合+量值化）===")
    print(f"  {'整体':<20} {overall:.3f}")
    for t in sorted(by):
        print(f"  {t:<20} {auroc(sf(by[t]), sn):.3f}  (n={len(by[t])})")

    print("\n=== 实验三：消融（整体 AUROC，Δ 相对完整模型）===")
    print(f"  {'完整模型(学习+量值化)':<28} {overall:.3f}")
    # −量值化
    d = fit(default_extractors(magnitude=False), fit_norm, openset_aggregator="learned")
    a = auroc(score_fn(d)(dknown), score_fn(d)(test_norm))
    print(f"  {'−量值化参数':<28} {a:.3f}   (Δ={a-overall:+.3f})")
    # Fisher 聚合替代学习
    d = fit(default_extractors(), fit_norm, openset_aggregator="fisher")
    a = auroc(score_fn(d)(dknown), score_fn(d)(test_norm))
    print(f"  {'Fisher 聚合(替学习)':<28} {a:.3f}   (Δ={a-overall:+.3f})")
    # 格内聚合 max→mean
    d = fit(ablation_extractors("mean"), fit_norm, openset_aggregator="learned")
    a = auroc(score_fn(d)(dknown), score_fn(d)(test_norm))
    print(f"  {'格内聚合 max→均值':<28} {a:.3f}   (Δ={a-overall:+.3f})")
    # −校准（原始 S8 + 学习聚合的对照：直接把 IForest 喂原始 S8）
    from sklearn.ensemble import IsolationForest
    def raw8(ctxs):
        out = []
        for c in ctxs:
            S, m = full._raw_matrix(c)
            out.append([S[l-1, j-1] if np.isfinite(S[l-1, j-1]) else 0.0 for (l, j) in VALID_POSITIONS])
        return np.asarray(out)
    ifm = IsolationForest(n_estimators=200, random_state=0).fit(raw8(fit_norm))
    a = auroc(-ifm.score_samples(raw8(dknown)), -ifm.score_samples(raw8(test_norm)))
    print(f"  {'−校准(原始S8+IForest)':<28} {a:.3f}   (Δ={a-overall:+.3f})")

    print("\n=== 参照基线 ===")
    from sklearn.ensemble import IsolationForest as IF
    Xn = feature_matrix(fit_norm)
    ifb = IF(n_estimators=200, random_state=0).fit(Xn)
    print(f"  IsolationForest(扁平25维)     {auroc(-ifb.score_samples(feature_matrix(dknown)), -ifb.score_samples(feature_matrix(test_norm))):.3f}")


if __name__ == "__main__":
    main()
