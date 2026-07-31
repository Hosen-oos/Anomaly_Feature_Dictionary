"""用 RPC 重采正常交易，消除跨采集管线的偽影。

问题（实测）：PTXPHISH 走 Alchemy 免费档 RPC（无 debug_trace → 0% 有 trace），
D_cal 走 BigQuery（100% 有 trace）。单凭"是否有 trace"即可完美分离两者
（单特征 AUROC=1.000），使任何 RPC-vs-BigQuery 的比较都被污染。

修复：取 D_cal 的交易哈希用同一套 RPC 重采一遍，得到管线匹配的正常对照集
（同样无 trace）。此后 PTXPHISH 的所有对比都在同管线下进行。

    export ALCHEMY_URL=...
    python -m experiments.build_cal_rpc --n 6000
"""
from __future__ import annotations

import argparse
import os
import random
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=6000)
    ap.add_argument("--workers", type=int, default=20)
    args = ap.parse_args()
    url = os.environ.get("ALCHEMY_URL")
    if not url:
        sys.exit("请设置 ALCHEMY_URL")

    dcal = load_contexts(ROOT / "data/splits/d_cal.pkl.gz")
    random.Random(0).shuffle(dcal)
    hashes = [c.tx_hash for c in dcal[:args.n]]
    print(f"从 D_cal 取 {len(hashes)} 个正常交易哈希，用 RPC 重采（管线对齐）")

    src = JsonRpcSource(url, cache=DiskCache(ROOT / "data/cache"),
                        fetch_call_trace=False)
    done = [0]
    def warm(h):
        src.get_transaction(h); src.get_receipt(h)
        done[0] += 1
        if done[0] % 500 == 0:
            print(f"  ...{done[0]}/{len(hashes)}", flush=True)
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        list(ex.map(warm, hashes))

    out = []
    for h in hashes:
        ctx = build_context(h, src)
        if ctx is not None:
            out.append(ctx)
    save_contexts(out, ROOT / "data/splits/d_cal_rpc.pkl.gz")
    print(f"\nd_cal_rpc: {len(out)} 笔，耗时 {time.time()-t0:.0f}s")
    print("可用性组:", dict(Counter("".join(map(str, c.availability)) for c in out)))
    print("有 trace 比例:", sum(1 for c in out if c.trace is not None), "/", len(out))


if __name__ == "__main__":
    main()
