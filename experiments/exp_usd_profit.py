"""锚定美元利润特征的判别力检验（是否值得保留）。

用 WETH/稳定币/WBTC 作价值锚定，无需外部价格源。检验：
1) 各攻击类型上的覆盖率（多少笔能算出锚定利润）与量级
2) 加入后对检测 AUROC 的影响（配置 A）
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

from mlusd.dataset.build import load_contexts               # noqa: E402
from mlusd.match.dictionary import load_dictionaries         # noqa: E402
from mlusd.pipeline import Detector                          # noqa: E402
from mlusd.signals.factory import default_extractors         # noqa: E402
from mlusd.signals.profit import attribution_params          # noqa: E402


def scores(det, cs):
    out = []
    for c in cs:
        S, m = det._raw_matrix(c)
        Q, _, _ = det.calibrator.transform(S, m)
        out.append(det.openset.raw_score(Q, m))
    return np.asarray(out)


def auroc(sa, sn):
    return roc_auc_score(np.r_[np.ones(len(sa)), np.zeros(len(sn))], np.r_[sa, sn])


def main():
    dcal = load_contexts(ROOT / "data/splits/d_cal_nbr.pkl.gz")
    dknown = load_contexts(ROOT / "data/splits/d_known_nbr.pkl.gz")
    random.Random(0).shuffle(dcal)
    by = defaultdict(list)
    for c in dknown:
        by[c.latent.get("attack_type")].append(c)

    print("=== 锚定美元利润的覆盖率与量级 ===")
    print(f"{'类型':<22}{'n':>5}{'可算出利润':>12}{'中位USD':>14}{'≥10万占比':>11}")
    for t in sorted(by) + ["(正常)"]:
        cs = by[t] if t != "(正常)" else dcal[:300]
        vals, large = [], 0
        for c in cs:
            p = attribution_params(c)
            if "usd_profit_mag" in p and p["usd_profit_mag"] > 0:
                vals.append(np.expm1(p["usd_profit_mag"]))
                large += p.get("large_usd_profit", 0)
        cov = len(vals) / len(cs) if cs else 0
        med = np.median(vals) if vals else 0
        print(f"{t:<22}{len(cs):>5}{cov*100:>11.0f}%{med:>14,.0f}"
              f"{(large/len(cs)*100 if cs else 0):>10.0f}%")

    # 对检测的影响：开/关 usd 特征
    print("\n=== 对检测 AUROC 的影响 ===")
    import mlusd.signals.profit as P
    fit_norm, test_norm = dcal[:8000], dcal[8000:10000]
    orig = P.ANCHORS
    res = {}
    for tag, anchors in [("含锚定USD利润", orig), ("不含(消融)", {})]:
        P.ANCHORS = anchors
        det = Detector(default_extractors(), load_dictionaries(ROOT / "configs/dictionaries"),
                       alpha=0.01, min_group_size=150,
                       openset_aggregator="learned").fit(fit_norm)
        sn = scores(det, test_norm)
        row = {"整体": auroc(scores(det, dknown), sn)}
        for t in sorted(by):
            row[t] = auroc(scores(det, by[t]), sn)
        res[tag] = row
    P.ANCHORS = orig

    keys = list(res["含锚定USD利润"])
    print(f"{'指标':<22}{'含USD':>10}{'不含':>10}{'Δ':>9}")
    for k in keys:
        a, b = res["含锚定USD利润"][k], res["不含(消融)"][k]
        print(f"{k:<22}{a:>10.3f}{b:>10.3f}{a-b:>+9.3f}")


if __name__ == "__main__":
    main()
