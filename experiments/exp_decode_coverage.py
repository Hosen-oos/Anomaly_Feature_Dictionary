"""验证扩充解码覆盖的效果：各类型的动作识别率 + 未识别事件占比。"""
from __future__ import annotations

import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mlusd.dataset.build import load_contexts               # noqa: E402
from mlusd.signals.l2_semantic import _event_name, lift_actions  # noqa: E402

ACTS = ["flashloan", "borrow", "repay", "swap", "remove_liquidity",
        "add_liquidity", "withdraw", "approve"]


def main():
    dknown = load_contexts(ROOT / "data/splits/d_known_nbr.pkl.gz")
    by = defaultdict(list)
    for c in dknown:
        by[c.latent.get("attack_type")].append(c)

    print("=== 动作出现率（%）——扩充解码后 ===")
    print(f"{'类型':<20}" + "".join(f"{a[:9]:>11}" for a in ACTS))
    for t in sorted(by):
        rates = []
        for a in ACTS:
            n = sum(1 for c in by[t] if any(x.kind == a for x in lift_actions(c)))
            rates.append(100.0 * n / len(by[t]))
        print(f"{t:<20}" + "".join(f"{r:>11.0f}" for r in rates))

    print("\n=== 事件解码覆盖率 ===")
    tot_known = tot_all = 0
    unknown_top = Counter()
    for t in sorted(by):
        k = a = 0
        for c in by[t]:
            for lg in (c.event_logs or []):
                a += 1
                if _event_name(lg).startswith("Unknown_"):
                    unknown_top[_event_name(lg)[8:]] += 1
                else:
                    k += 1
        tot_known += k; tot_all += a
        print(f"  {t:<20} {k}/{a} = {100.0*k/max(a,1):.0f}%")
    print(f"  {'总计':<20} {tot_known}/{tot_all} = {100.0*tot_known/max(tot_all,1):.0f}%")
    print(f"\n仍未识别的 top-8 topic0: {unknown_top.most_common(8)}")


if __name__ == "__main__":
    main()
