"""用 Alchemy 免费档 RPC 采集 PTXPHISH 交易（tx + receipt），构建 phishing 扩容集与硬负样本集。

为何走 RPC 而非 BigQuery：PTXPHISH 只给哈希不给日期，BigQuery 按日期分区，无日期则需全表
扫描（极贵）；而 eth_getTransactionByHash/Receipt 在免费档可用且便宜（各 ~26 CU，
30M CU/月额度足够全量 18.5k×2 次调用 ≈ 1M CU）。L1 邻域图缺失时自动降级为交易内转账图。

    export ALCHEMY_URL=https://eth-mainnet.g.alchemy.com/v2/<KEY>
    python -m experiments.build_ptxphish --n-attack 1000 --n-benign 1000   # 试点
    python -m experiments.build_ptxphish --all                            # 全量
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

from mlusd.collect.cache import DiskCache                        # noqa: E402
from mlusd.collect.context import build_context                   # noqa: E402
from mlusd.collect.sources import JsonRpcSource                   # noqa: E402
from mlusd.dataset.build import save_contexts                     # noqa: E402
from mlusd.dataset.ptxphish import attacks, benign, load_ptxphish, subtype_counts  # noqa: E402


def collect(hashes, src, workers=8):
    """并发预热缓存（tx+receipt），再逐个组装 context。"""
    done = [0]
    def warm(h):
        src.get_transaction(h)
        src.get_receipt(h)
        done[0] += 1
        if done[0] % 200 == 0:
            print(f"    ...已取 {done[0]}/{len(hashes)}", flush=True)
    with ThreadPoolExecutor(max_workers=workers) as ex:
        list(ex.map(warm, hashes))
    out = []
    for h in hashes:
        ctx = build_context(h, src)
        if ctx is not None:
            out.append(ctx)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-attack", type=int, default=1000)
    ap.add_argument("--n-benign", type=int, default=1000)
    ap.add_argument("--all", action="store_true", help="全量采集")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    url = os.environ.get("ALCHEMY_URL")
    if not url:
        sys.exit("请设置环境变量 ALCHEMY_URL")

    seeds = load_ptxphish()
    atk, ben = attacks(seeds), benign(seeds)
    print(f"PTXPHISH: 攻击 {len(atk)} | 良性 {len(ben)}")
    print("攻击子类型:", subtype_counts(atk))

    if not args.all:
        random.Random(0).shuffle(atk)
        random.Random(0).shuffle(ben)
        atk, ben = atk[:args.n_attack], ben[:args.n_benign]
    print(f"\n本次采集: 攻击 {len(atk)} | 良性 {len(ben)}")

    # 免费档无 debug_traceTransaction，关掉以免每笔都白等一次失败请求
    src = JsonRpcSource(url, cache=DiskCache(ROOT / "data/cache"),
                        fetch_call_trace=False)
    sub_of = {s.tx_hash: s.subtype for s in atk}

    t0 = time.time()
    print("\n[攻击] 采集中...")
    ctx_a = collect([s.tx_hash for s in atk], src, args.workers)
    for c in ctx_a:
        c.latent["attack_type"] = "phishing"
        c.latent["subtype"] = sub_of.get(c.tx_hash, "")
    save_contexts(ctx_a, ROOT / "data/splits/d_phish_ptx.pkl.gz")
    print(f"  -> {len(ctx_a)} 笔 d_phish_ptx.pkl.gz")

    print("\n[良性硬负样本] 采集中...")
    ctx_b = collect([s.tx_hash for s in ben], src, args.workers)
    save_contexts(ctx_b, ROOT / "data/splits/d_benign_ptx.pkl.gz")
    print(f"  -> {len(ctx_b)} 笔 d_benign_ptx.pkl.gz")

    print(f"\n总耗时 {time.time()-t0:.0f}s")
    print("攻击可用性组:", dict(Counter("".join(map(str, c.availability)) for c in ctx_a)))
    print("良性可用性组:", dict(Counter("".join(map(str, c.availability)) for c in ctx_b)))


if __name__ == "__main__":
    main()
