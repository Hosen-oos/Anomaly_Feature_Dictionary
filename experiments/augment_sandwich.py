"""为数据集补充**强三明治判定**信号（文献共识条件）。

此前的跨交易上下文只用"同区块 + 同 to 地址"（最弱条件），sandwich 仅 +0.017。
本脚本取同区块的 Swap 事件，按文献条件判定：同一流动性池 + 方向相反 +
金额链接 + 中间夹第三方 swap（受害者）。结果写入 latent["sandwich_ctx"]。

    python -m experiments.augment_sandwich --datasets d_known_nbr_blk --dry-run
    python -m experiments.augment_sandwich --datasets d_known_nbr_blk d_cal_nbr_blk
"""
from __future__ import annotations

import argparse
import datetime
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mlusd.collect.bigquery import BigQuerySource            # noqa: E402
from mlusd.dataset.build import load_contexts, save_contexts  # noqa: E402
from mlusd.signals.sandwich import sandwich_signals           # noqa: E402

PROJECT = "project-b471d110-9146-4221-872"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", required=True)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--max-per-set", type=int, default=12000)
    args = ap.parse_args()

    loaded, blocks = {}, set()
    for name in args.datasets:
        cs = load_contexts(ROOT / f"data/splits/{name}.pkl.gz")[:args.max_per_set]
        loaded[name] = cs
        blocks |= {c.block_number for c in cs if c.block_number}
        print(f"  {name}: {len(cs)} 笔")
    bq = BigQuerySource(project=PROJECT)
    dates = {datetime.datetime.utcfromtimestamp(c.timestamp).strftime("%Y-%m-%d")
             for cs in loaded.values() for c in cs if c.timestamp}
    if not dates:
        dates = set(bq.dates_for_blocks(sorted(blocks)).values())
    print(f"区块 {len(blocks)} 个 / 日期 {len(dates)} 个")

    if args.dry_run:
        gb = bq.gb(bq.block_swaps(sorted(blocks), sorted(dates), dry_run=True))
        print(f"干跑：logs 表扫描 ~{gb:.1f} GB")
        return

    # 原始 Swap 日志落盘缓存——判定逻辑若有修正可离线重算，避免重复付 BigQuery 费用
    import gzip
    import json
    cache_p = ROOT / "data/cache/block_swaps.json.gz"
    if cache_p.exists():
        with gzip.open(cache_p, "rt", encoding="utf-8") as f:
            swaps = {int(k): v for k, v in json.load(f).items()}
        print(f"命中缓存：{len(swaps)} 个区块的 Swap 事件（未重复查询 BigQuery）")
    else:
        # 分批查询：一次拉 8000+ 区块的全部 Swap 日志结果集过大，易 SSL 中断；
        # 分批同时降低单次负载并让失败只影响一批（分区裁剪相同，总扫描字节不变）
        blist = sorted(blocks)
        batch = 2000
        swaps = {}
        for i in range(0, len(blist), batch):
            part = blist[i:i + batch]
            got = bq.block_swaps(part, sorted(dates))
            swaps.update(got)
            print(f"  批次 {i//batch+1}/{(len(blist)+batch-1)//batch}: "
                  f"+{len(got)} 区块（累计 {len(swaps)}）", flush=True)
        cache_p.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(cache_p, "wt", encoding="utf-8") as f:
            json.dump({str(k): v for k, v in swaps.items()}, f)
        print(f"取到 {len(swaps)} 个区块的 Swap 事件（已缓存）")
    for name, cs in loaded.items():
        n = 0
        full = Counter()
        for c in cs:
            logs = swaps.get(c.block_number)
            if not logs:
                continue
            sig = sandwich_signals(c, logs)
            if sig:
                c.latent["sandwich_ctx"] = sig
                n += 1
                full[sig.get("sw_full_pattern", 0.0)] += 1
        save_contexts(cs, ROOT / f"data/splits/{name}_sw.pkl.gz")
        print(f"  {name}: 补充 {n}/{len(cs)} -> {name}_sw.pkl.gz  "
              f"完整三明治模式命中={full.get(1.0, 0)}")


if __name__ == "__main__":
    main()
