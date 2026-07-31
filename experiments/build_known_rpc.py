"""把全部攻击样本与真未知集用 RPC 重采，建立"管线对齐"配置。

背景：d_known_full 现为混合采集（111 笔 BigQuery 有 trace + 104 笔 RPC 无 trace），
直接评测会让 RPC 部分被系统性低估（缺失 L3 填 0 → 显得更正常）。故统一重采为 RPC 形态，
与 d_cal_rpc 完全对齐。代价是无 L3 与真实邻域图，收益是攻击样本量 111→215 且无偽影。

    export ALCHEMY_URL=...
    python -m experiments.build_known_rpc
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


def recollect(src, ctxs, tag):
    hashes = [c.tx_hash for c in ctxs]
    meta = {c.tx_hash: dict(c.latent) for c in ctxs}
    def warm(h):
        src.get_transaction(h); src.get_receipt(h)
    with ThreadPoolExecutor(max_workers=12) as ex:
        list(ex.map(warm, hashes))
    out = []
    for h in hashes:
        c = build_context(h, src)
        if c is not None:
            c.latent.update(meta.get(h, {}))
            out.append(c)
    print(f"  {tag}: {len(out)}/{len(hashes)}")
    return out


def main():
    url = os.environ.get("ALCHEMY_URL")
    if not url:
        sys.exit("请设置 ALCHEMY_URL")
    src = JsonRpcSource(url, cache=DiskCache(ROOT / "data/cache"),
                        fetch_call_trace=False)
    t0 = time.time()

    print("RPC 重采（管线对齐）:")
    known = load_contexts(ROOT / "data/splits/d_known_full.pkl.gz")
    k = recollect(src, known, "d_known_full -> d_known_rpc")
    save_contexts(k, ROOT / "data/splits/d_known_rpc.pkl.gz")

    p = ROOT / "data/splits/d_open.pkl.gz"
    if p.exists():
        o = recollect(src, load_contexts(p), "d_open -> d_open_rpc")
        save_contexts(o, ROOT / "data/splits/d_open_rpc.pkl.gz")

    print(f"\n耗时 {time.time()-t0:.0f}s")
    print("攻击类型:", dict(Counter(c.latent.get("attack_type") for c in k)))
    print("可用性组:", dict(Counter("".join(map(str, c.availability)) for c in k)))
    print("有 trace:", sum(1 for c in k if c.trace is not None), "/", len(k))


if __name__ == "__main__":
    main()
