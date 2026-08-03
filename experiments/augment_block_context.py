"""跨交易上下文：为数据集补充"同区块相邻交易"结构（三明治检测的必要信息）。

动机：三明治攻击的前后夹击结构**不在单笔交易内**——受害交易前后各有一笔同一攻击者、
同一交易对的 swap。这是框架此前 sandwich 表现受限（0.745）的架构性原因。

提取三个结构信号写入 latent["block_ctx"]，由 L1-j2 参数池消费：
  same_sender_around   前后窗口内是否有同一发送方的交易（夹击者特征）
  adjacent_same_target 紧邻的前一笔/后一笔是否指向同一合约（同一交易对）
  same_target_around   窗口内指向同一合约的交易数

    python -m experiments.augment_block_context --datasets d_known_nbr --dry-run
    python -m experiments.augment_block_context --datasets d_known_nbr d_cal_nbr
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mlusd.collect.bigquery import BigQuerySource            # noqa: E402
from mlusd.dataset.build import load_contexts, save_contexts  # noqa: E402

PROJECT = "project-b471d110-9146-4221-872"
WINDOW = 3      # 前后各看 3 笔


def compute(ctx, txs: list[dict]) -> dict:
    """由同区块交易列表算出该交易的跨交易结构特征。"""
    idx = {t["index"]: t for t in txs}
    me = idx.get(ctx.tx_index)
    if me is None:
        return {}
    lo, hi = ctx.tx_index - WINDOW, ctx.tx_index + WINDOW
    around = [t for t in txs if lo <= t["index"] <= hi and t["index"] != ctx.tx_index]
    same_sender = any((t["from"] or "").lower() == (ctx.from_address or "").lower()
                      for t in around)
    same_target = sum(1 for t in around
                      if (t["to"] or "").lower() == (ctx.to_address or "").lower())
    adj = [idx.get(ctx.tx_index - 1), idx.get(ctx.tx_index + 1)]
    adjacent = any(t and (t["to"] or "").lower() == (ctx.to_address or "").lower()
                   for t in adj)
    return {"same_sender_around": bool(same_sender),
            "adjacent_same_target": bool(adjacent),
            "same_target_around": int(same_target)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", required=True)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--max-per-set", type=int, default=4000)
    args = ap.parse_args()

    loaded, blocks, dates = {}, set(), set()
    for name in args.datasets:
        cs = load_contexts(ROOT / f"data/splits/{name}.pkl.gz")[:args.max_per_set]
        loaded[name] = cs
        for c in cs:
            if c.block_number:
                blocks.add(c.block_number)
        print(f"  {name}: {len(cs)} 笔")
    # 日期由 timestamp 推；缺失则由 BigQuery blocks 表补
    bq = BigQuerySource(project=PROJECT)
    import datetime
    for cs in loaded.values():
        for c in cs:
            if c.timestamp:
                dates.add(datetime.datetime.utcfromtimestamp(c.timestamp).strftime("%Y-%m-%d"))
    if not dates:
        print("无 timestamp，用 blocks 表补日期...")
        mp = bq.dates_for_blocks(sorted(blocks))
        dates = set(mp.values())
    print(f"涉及区块 {len(blocks)} 个，日期 {len(dates)} 个")

    if args.dry_run:
        gb = bq.gb(bq.block_neighbors(sorted(blocks), sorted(dates), dry_run=True))
        print(f"干跑：扫描 ~{gb:.1f} GB")
        return

    nb = bq.block_neighbors(sorted(blocks), sorted(dates))
    print(f"取到 {len(nb)} 个区块的交易列表")
    for name, cs in loaded.items():
        n = 0
        for c in cs:
            txs = nb.get(c.block_number)
            if txs:
                bc = compute(c, txs)
                if bc:
                    c.latent["block_ctx"] = bc
                    n += 1
        save_contexts(cs, ROOT / f"data/splits/{name}_blk.pkl.gz")
        hit = Counter(c.latent.get("block_ctx", {}).get("same_sender_around")
                      for c in cs)
        print(f"  {name}: 补充 {n}/{len(cs)} -> {name}_blk.pkl.gz  同发送方夹击={hit.get(True,0)}")


if __name__ == "__main__":
    main()
