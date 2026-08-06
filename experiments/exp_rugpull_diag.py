"""rug_pull 诊断：为何仍显著落后 IForest（0.946 vs 0.982），且 remove_liquidity 为 0%？

三个待查问题：
1. rug_pull 交易究竟发出哪些事件？remove_liquidity(Burn) 真的没有，还是解码没覆盖？
2. IForest 用的 25 维扁平特征里，哪几维在 rug_pull 上最具判别力（我们缺了什么）？
3. 未识别事件的 topic0 top 列表——是否有该补的签名。
"""
from __future__ import annotations

import random
import sys
import warnings
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mlusd.baselines.features import FEATURE_NAMES, feature_matrix   # noqa: E402
from mlusd.dataset.build import load_contexts                        # noqa: E402
from mlusd.signals.l2_semantic import _event_name, lift_actions      # noqa: E402


def main():
    dcal = load_contexts(ROOT / "data/splits/d_cal_nbr_blk_l4_sw.pkl.gz")
    dknown = load_contexts(ROOT / "data/splits/d_known_nbr_blk_l4_sw.pkl.gz")
    random.Random(0).shuffle(dcal)
    by = defaultdict(list)
    for c in dknown:
        by[c.latent.get("attack_type")].append(c)
    rug = by["rug_pull"]
    print(f"rug_pull 样本 {len(rug)} 笔\n")

    # 1) 事件与动作分布
    ev, act, unk = Counter(), Counter(), Counter()
    for c in rug:
        for lg in (c.event_logs or []):
            n = _event_name(lg)
            ev[n] += 1
            if n.startswith("Unknown_"):
                unk[n[8:]] += 1
        for a in lift_actions(c):
            act[a.kind] += 1
    print("已识别事件:", dict(ev.most_common(10)))
    print("提升出的动作:", dict(act.most_common(10)))
    print("未识别 topic0 前缀 top8:", unk.most_common(8))

    # 含 Burn / remove_liquidity 的比例
    n_burn = sum(1 for c in rug
                 if any(_event_name(l) == "Burn" for l in (c.event_logs or [])))
    n_rm = sum(1 for c in rug
               if any(a.kind == "remove_liquidity" for a in lift_actions(c)))
    print(f"\n含 Burn 事件: {n_burn}/{len(rug)}   含 remove_liquidity 动作: {n_rm}/{len(rug)}")

    # 2) 扁平特征上 rug_pull vs 正常的差异（找我们缺的判别维）
    Xr = feature_matrix(rug)
    Xn = feature_matrix(dcal[:2000])
    mu_r, mu_n = Xr.mean(axis=0), Xn.mean(axis=0)
    sd = Xn.std(axis=0) + 1e-9
    z = (mu_r - mu_n) / sd
    order = np.argsort(-np.abs(z))[:10]
    print("\n=== 扁平特征上 rug_pull 与正常差异最大的 10 维（标准化差）===")
    print(f"{'特征':<20}{'rug均值':>14}{'正常均值':>14}{'z':>9}")
    for i in order:
        print(f"{FEATURE_NAMES[i]:<20}{mu_r[i]:>14.2f}{mu_n[i]:>14.2f}{z[i]:>9.2f}")


if __name__ == "__main__":
    main()
