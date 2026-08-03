"""可解释性实验：贡献度 top-k 是否命中该攻击类型的机理层级（设计 §8 实验四）。

金标准来自攻击机理分析（与字典 YAML 的权重设计一致）：每类攻击的判别信息**应当**
主要落在特定层。若系统给出的贡献度 top-k 命中这些层，说明解释不是事后编造，
而是与机理一致——这是"结构化可解释性"的量化证据。

对照：随机基线（从该交易的可用信号中随机取 k 个）——排除"碰巧命中"。
"""
from __future__ import annotations

import random
import sys
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mlusd.dataset.build import load_contexts               # noqa: E402
from mlusd.match.dictionary import load_dictionaries         # noqa: E402
from mlusd.pipeline import Detector                          # noqa: E402
from mlusd.signals.factory import default_extractors         # noqa: E402
from mlusd.types import VALID_POSITIONS                      # noqa: E402

# 机理金标准：每类攻击的判别信息应落在哪些 (层,角度)
# 依据攻击机理（与字典权重设计一致），非事后拟合
GOLD: dict[str, set[tuple[int, int]]] = {
    # 闪电贷：借入-操作-归还闭环（L2经济）+ 深调用轨迹（L3）
    "flash_loan": {(2, 2), (3, 1), (3, 3)},
    # 价格操控：连环 swap 造成储备偏移（L2经济）+ 执行轨迹
    "price_manipulation": {(2, 2), (3, 1), (3, 3)},
    # Rug Pull：流动性撤出（L2经济）+ 链下项目情报（L4）
    "rug_pull": {(2, 2), (4, 3)},
    # 钓鱼：授权-转账不一致（L2信息差异）+ 资金归集（L1经济）+ 链下标签
    "phishing": {(2, 3), (1, 2), (4, 3)},
    # 庞氏：多来源扇入聚集（L1经济）
    "ponzi": {(1, 2), (1, 1)},
    # 三明治：区块内前后夹击结构（L1）+ 兑换经济（L2经济）
    "sandwich": {(1, 2), (1, 1), (2, 2)},
}


def main():
    k_list = [1, 3]
    dcal = load_contexts(ROOT / "data/splits/d_cal_nbr.pkl.gz")
    dknown = load_contexts(ROOT / "data/splits/d_known_nbr.pkl.gz")
    random.Random(0).shuffle(dcal)
    det = Detector(default_extractors(), load_dictionaries(ROOT / "configs/dictionaries"),
                   alpha=0.01, min_group_size=150,
                   openset_aggregator="learned").fit(dcal[:8000])

    by = defaultdict(list)
    for c in dknown:
        by[c.latent.get("attack_type")].append(c)

    print("=== 可解释性：贡献度排序命中机理层级 ===")
    print("随机基线 = |金标准∩可用| / |可用|（top-1 的期望命中率），排除'碰巧命中'")
    print(f"{'攻击类型':<20}{'n':>4}{'top-1':>8}{'随机':>7}{'提升':>7}"
          f"{'top-3':>8}{'随机3':>7}{'金标准':>22}")
    acc = {"t1": [], "r1": [], "t3": [], "r3": []}
    for t in sorted(by):
        gold = GOLD.get(t, set())
        h1 = h3 = 0.0
        e1 = e3 = 0.0          # 随机期望（解析计算，避免抽样噪声）
        n = 0
        for c in by[t]:
            S, m = det._raw_matrix(c)
            Q, T, _ = det.calibrator.transform(S, m)
            avail = [(l, j) for (l, j) in VALID_POSITIONS
                     if m[l - 1] and np.isfinite(T[l - 1, j - 1])]
            if not avail:
                continue
            n += 1
            ranked = sorted(avail, key=lambda p: -T[p[0] - 1, p[1] - 1])
            g_av = [p for p in avail if p in gold]
            h1 += 1.0 if ranked[0] in gold else 0.0
            h3 += 1.0 if any(p in gold for p in ranked[:3]) else 0.0
            # 随机期望：top-1 = |g|/|a|；top-3 = 1 - C(a-g,3)/C(a,3)
            a, g = len(avail), len(g_av)
            e1 += g / a
            k = min(3, a)
            from math import comb
            e3 += 1.0 - (comb(a - g, k) / comb(a, k) if a - g >= k else 0.0)
        if n == 0:
            continue
        r1, rr1, r3, rr3 = h1 / n, e1 / n, h3 / n, e3 / n
        for key, v in zip(["t1", "r1", "t3", "r3"], [r1, rr1, r3, rr3]):
            acc[key].append(v)
        gs = ",".join(f"L{l}j{j}" for l, j in sorted(gold))
        print(f"{t:<20}{n:>4}{r1:>8.2f}{rr1:>7.2f}{r1-rr1:>+7.2f}"
              f"{r3:>8.2f}{rr3:>7.2f}   {gs:<22}")

    m1, mr1 = np.mean(acc["t1"]), np.mean(acc["r1"])
    m3, mr3 = np.mean(acc["t3"]), np.mean(acc["r3"])
    print(f"\n{'宏平均':<20}{'':>4}{m1:>8.2f}{mr1:>7.2f}{m1-mr1:>+7.2f}{m3:>8.2f}{mr3:>7.2f}")
    print(f"\ntop-1 相对随机提升: {m1-mr1:+.2f}（{(m1/mr1-1)*100:+.0f}%）")


if __name__ == "__main__":
    main()
