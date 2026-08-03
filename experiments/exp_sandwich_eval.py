"""强三明治判定（文献共识条件）的效果评测。

对比：含强判定信号 vs 剥离。重点看 sandwich 类是否改善，并报告各条件的命中率——
若完整模式命中过低，需诚实说明数据集只含攻击者单笔（而非完整三笔）这一限制。
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

KEYS = ["sw_in_swap_block", "sw_opposite_same_pool", "sw_amount_linked",
        "sw_victim_between", "sw_full_pattern"]


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
    dcal = load_contexts(ROOT / "data/splits/d_cal_nbr_blk_l4_sw.pkl.gz")
    dknown = load_contexts(ROOT / "data/splits/d_known_nbr_blk_l4_sw.pkl.gz")
    random.Random(0).shuffle(dcal)
    by = defaultdict(list)
    for c in dknown:
        by[c.latent.get("attack_type")].append(c)

    print("=== 强三明治各条件命中率 ===")
    print(f"{'类型':<20}{'n':>5}" + "".join(f"{k.replace('sw_',''):>17}" for k in KEYS))
    for t in sorted(by) + ["(正常)"]:
        cs = by[t] if t != "(正常)" else dcal[:3000]
        vals = []
        for k in KEYS:
            v = [c.latent.get("sandwich_ctx", {}).get(k, 0.0) for c in cs]
            vals.append(np.mean(v) if v else 0.0)
        print(f"{t:<20}{len(cs):>5}" + "".join(f"{v:>17.3f}" for v in vals))

    fit_norm, test_norm = dcal[:8000], dcal[8000:10000]
    print("\n=== 对检测 AUROC 的影响 ===")
    res = {}
    for tag, strip in [("含强三明治判定", False), ("剥离(消融)", True)]:
        if strip:
            for c in dcal + dknown:
                c.latent.pop("sandwich_ctx", None)
        det = Detector(default_extractors(), load_dictionaries(ROOT / "configs/dictionaries"),
                       alpha=0.01, min_group_size=150,
                       openset_aggregator="learned").fit(fit_norm)
        sn = scores(det, test_norm)
        row = {"整体": auroc(scores(det, dknown), sn)}
        for t in sorted(by):
            row[t] = auroc(scores(det, by[t]), sn)
        res[tag] = row

    print(f"{'指标':<22}{'含强判定':>11}{'剥离':>10}{'Δ':>9}")
    for k in res["含强三明治判定"]:
        a, b = res["含强三明治判定"][k], res["剥离(消融)"][k]
        print(f"{k:<22}{a:>11.3f}{b:>10.3f}{a-b:>+9.3f}")


if __name__ == "__main__":
    main()
