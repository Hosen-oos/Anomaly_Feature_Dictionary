"""检查利润归因特征能否分离 flash_loan vs price_manipulation（本轮核心目标）。
直接看各类型上这些特征的均值，以及两两可分性。
"""
from __future__ import annotations

import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mlusd.dataset.build import load_contexts               # noqa: E402
from mlusd.signals.profit import attribution_params         # noqa: E402

KEYS = ["passthrough_max", "drain_imbalance", "profit_concentration",
        "loss_concentration", "sender_is_winner", "n_tokens_norm", "sender_roundtrip",
        "victim_is_pool", "winner_is_pool", "pool_loss_share"]


def main():
    dknown = load_contexts(ROOT / "data/splits/d_known_nbr.pkl.gz")
    by = defaultdict(list)
    for c in dknown:
        by[c.latent.get("attack_type")].append(c)
    dcal = load_contexts(ROOT / "data/splits/d_cal_nbr.pkl.gz")
    random.Random(0).shuffle(dcal)
    by["(正常)"] = dcal[:300]

    types = [t for t in ["flash_loan", "price_manipulation", "rug_pull",
                         "phishing", "ponzi", "sandwich", "(正常)"] if t in by]
    prof = {}
    for t in types:
        acc = defaultdict(list)
        for c in by[t]:
            p = attribution_params(c)
            for k in KEYS:
                if k in p:
                    acc[k].append(p[k])
        prof[t] = {k: float(np.mean(v)) if v else 0.0 for k, v in acc.items()}

    print("=== 利润归因特征均值 ===")
    print(f"{'特征':<22}" + "".join(f"{t[:10]:>12}" for t in types))
    for k in KEYS:
        print(f"{k:<22}" + "".join(f"{prof[t].get(k,0):>12.2f}" for t in types))

    print("\n=== flash_loan vs price_manipulation 可分性（差值绝对值排序）===")
    fl, pm = prof.get("flash_loan", {}), prof.get("price_manipulation", {})
    diffs = sorted(((abs(fl.get(k,0)-pm.get(k,0)), k) for k in KEYS), reverse=True)
    for d, k in diffs:
        print(f"  {k:<24} |Δ|={d:.3f}   flash={fl.get(k,0):.2f}  price={pm.get(k,0):.2f}")


if __name__ == "__main__":
    main()
