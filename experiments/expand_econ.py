"""经济类攻击补齐到研究一种子池上限（每类 ~50），走 Alchemy 免费 RPC。

为何用 RPC 而非 BigQuery：新增 140 笔散布在 92 个日期，BigQuery 按日期分区
需 ~368 GB；而 RPC 每笔 ~52 CU，140 笔仅 ~7k CU。代价是无 trace（免费档不支持
debug_traceTransaction），L3 层将不可用——但经济类的主要信号在 L2 语义层，
且可用性掩码会如实反映缺层（异构可用性设计本就承接这种情况）。

    export ALCHEMY_URL=...
    python -m experiments.expand_econ
"""
from __future__ import annotations

import os
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mlusd.collect.cache import DiskCache                    # noqa: E402
from mlusd.collect.context import build_context               # noqa: E402
from mlusd.collect.sources import JsonRpcSource               # noqa: E402
from mlusd.dataset.build import load_contexts, save_contexts  # noqa: E402
from mlusd.dataset.seeds import (                             # noqa: E402
    DEFAULT_SEED_DIR, STRONG_SEED, load_seeds, select_per_type)

ECON = ["flash_loan", "price_manipulation", "ponzi", "rug_pull"]


def main():
    url = os.environ.get("ALCHEMY_URL")
    if not url:
        sys.exit("请设置 ALCHEMY_URL")

    have = load_contexts(ROOT / "data/splits/d_known.pkl.gz")
    have_h = {c.tx_hash for c in have}
    seeds = load_seeds(DEFAULT_SEED_DIR / STRONG_SEED)
    sel = select_per_type([s for s in seeds if s.attack_type in ECON], 60)
    new = [s for s in sel if s.tx_hash not in have_h]
    print(f"经济类补齐：新增 {len(new)} 笔 {dict(Counter(s.attack_type for s in new))}")

    src = JsonRpcSource(url, cache=DiskCache(ROOT / "data/cache"),
                        fetch_call_trace=False)
    t0 = time.time()
    def warm(h):
        src.get_transaction(h); src.get_receipt(h)
    with ThreadPoolExecutor(max_workers=12) as ex:
        list(ex.map(warm, [s.tx_hash for s in new]))

    type_of = {s.tx_hash: s.attack_type for s in new}
    added = []
    for s in new:
        ctx = build_context(s.tx_hash, src)
        if ctx is not None:
            ctx.latent["attack_type"] = type_of[s.tx_hash]
            added.append(ctx)
    print(f"采集成功 {len(added)}/{len(new)}，耗时 {time.time()-t0:.0f}s")

    merged = have + added
    save_contexts(merged, ROOT / "data/splits/d_known_full.pkl.gz")
    print(f"\nD_known_full: {len(merged)} 笔 -> d_known_full.pkl.gz")
    print("类型分布:", dict(Counter(c.latent.get("attack_type") for c in merged)))
    print("可用性组:", dict(Counter("".join(map(str, c.availability)) for c in merged)))


if __name__ == "__main__":
    main()
