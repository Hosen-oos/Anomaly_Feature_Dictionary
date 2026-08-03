"""L4 广义化评测：补充合约结构性信号后，L4 层是否真正可用、对检测是否有贡献。

对比：含 L4 结构信号 vs 剥离（模拟此前 L4 几乎不可用的状态）。
"""
from __future__ import annotations

import random
import sys
import warnings
from collections import Counter, defaultdict
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
    dcal = load_contexts(ROOT / "data/splits/d_cal_nbr_blk_l4.pkl.gz")
    dknown = load_contexts(ROOT / "data/splits/d_known_nbr_blk_l4.pkl.gz")
    dopen = load_contexts(ROOT / "data/splits/d_open_l4.pkl.gz")
    random.Random(0).shuffle(dcal)
    print("可用性组（攻击）:", dict(Counter("".join(map(str, c.availability)) for c in dknown)))
    print("可用性组（正常）:", dict(Counter("".join(map(str, c.availability)) for c in dcal[:3000])))

    by = defaultdict(list)
    for c in dknown:
        by[c.latent.get("attack_type")].append(c)

    # 合约元信息在各类型上的分布
    print("\n=== 合约结构性信号分布 ===")
    print(f"{'类型':<22}{'有元信息':>10}{'中位部署天数':>14}{'微型字节码':>12}")
    for t in sorted(by) + ["(正常)"]:
        cs = by[t] if t != "(正常)" else dcal[:2000]
        metas = [c.offchain.contract_meta for c in cs
                 if c.offchain and c.offchain.contract_meta]
        ages = [m["age_days"] for m in metas if "age_days" in m]
        tiny = sum(1 for m in metas if m.get("bytecode_len", 1e9) < 500)
        print(f"{t:<22}{len(metas)/len(cs)*100:>9.0f}%"
              f"{(np.median(ages) if ages else 0):>14.0f}"
              f"{(tiny/len(metas)*100 if metas else 0):>11.0f}%")

    fit_norm, test_norm = dcal[:8000], dcal[8000:10000]
    print("\n=== 对检测 AUROC 的影响 ===")
    res = {}
    for tag, strip in [("含 L4 结构信号", False), ("剥离 L4(消融)", True)]:
        if strip:
            for c in dcal + dknown + dopen:
                if c.offchain:
                    c.offchain.contract_meta = {}
        det = Detector(default_extractors(), load_dictionaries(ROOT / "configs/dictionaries"),
                       alpha=0.01, min_group_size=150,
                       openset_aggregator="learned").fit(fit_norm)
        sn = scores(det, test_norm)
        row = {"整体": auroc(scores(det, dknown), sn),
               "D_open真未知": auroc(scores(det, dopen), sn)}
        for t in sorted(by):
            row[t] = auroc(scores(det, by[t]), sn)
        res[tag] = row

    print(f"{'指标':<22}{'含L4':>10}{'剥离':>10}{'Δ':>9}")
    for k in res["含 L4 结构信号"]:
        a, b = res["含 L4 结构信号"][k], res["剥离 L4(消融)"][k]
        print(f"{k:<22}{a:>10.3f}{b:>10.3f}{a-b:>+9.3f}")


if __name__ == "__main__":
    main()
