"""跨交易上下文对检测的影响（重点看 sandwich 是否改善）。

sandwich 的前后夹击结构本不在单笔交易内——此前 0.745 的表现受限于此。
本实验用补充了同区块相邻交易结构的数据集，对比开/关该组参数。
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
    dcal = load_contexts(ROOT / "data/splits/d_cal_nbr_blk.pkl.gz")
    dknown = load_contexts(ROOT / "data/splits/d_known_nbr_blk.pkl.gz")
    random.Random(0).shuffle(dcal)
    by = defaultdict(list)
    for c in dknown:
        by[c.latent.get("attack_type")].append(c)

    # 先看结构信号在各类型上的分布（判断方向是否合理）
    print("=== 跨交易结构信号分布 ===")
    print(f"{'类型':<22}{'n':>5}{'同发送方夹击':>13}{'紧邻同目标':>12}{'窗口同目标均值':>15}")
    for t in sorted(by) + ["(正常)"]:
        cs = by[t] if t != "(正常)" else dcal[:2000]
        bcs = [c.latent.get("block_ctx", {}) for c in cs]
        ss = sum(1 for b in bcs if b.get("same_sender_around"))
        aj = sum(1 for b in bcs if b.get("adjacent_same_target"))
        st = np.mean([b.get("same_target_around", 0) for b in bcs]) if bcs else 0
        print(f"{t:<22}{len(cs):>5}{ss/len(cs)*100:>12.0f}%{aj/len(cs)*100:>11.0f}%{st:>15.2f}")

    # 开/关对比
    fit_norm, test_norm = dcal[:8000], dcal[8000:10000]
    print("\n=== 对检测 AUROC 的影响 ===")
    res = {}
    for tag, strip in [("含跨交易上下文", False), ("不含(消融)", True)]:
        if strip:
            for c in dcal + dknown:
                c.latent.pop("block_ctx", None)
        det = Detector(default_extractors(), load_dictionaries(ROOT / "configs/dictionaries"),
                       alpha=0.01, min_group_size=150,
                       openset_aggregator="learned").fit(fit_norm)
        sn = scores(det, test_norm)
        row = {"整体": auroc(scores(det, dknown), sn)}
        for t in sorted(by):
            row[t] = auroc(scores(det, by[t]), sn)
        res[tag] = row

    print(f"{'指标':<22}{'含上下文':>11}{'不含':>10}{'Δ':>9}")
    for k in res["含跨交易上下文"]:
        a, b = res["含跨交易上下文"][k], res["不含(消融)"][k]
        print(f"{k:<22}{a:>11.3f}{b:>10.3f}{a-b:>+9.3f}")


if __name__ == "__main__":
    main()
