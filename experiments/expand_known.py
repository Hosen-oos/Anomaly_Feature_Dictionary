"""扩充 D_known：phishing + sandwich 各 ~400（分散日期取样保多样性），供可扩展性
实验与稳定的 per-type 指标。经济类样本本就 ~50 不扩。

    python -m experiments.expand_known --dry-run   # 先看 BigQuery 成本
    python -m experiments.expand_known             # 真跑，存 d_known_ext.pkl.gz
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mlusd.collect.bigquery import BigQuerySource            # noqa: E402
from mlusd.collect.sources import BQPrefetchSource            # noqa: E402
from mlusd.dataset.build import build_contexts, save_contexts  # noqa: E402
from mlusd.dataset.seeds import (                             # noqa: E402
    DEFAULT_SEED_DIR, STRONG_SEED, fill_missing_dates, load_seeds)

PROJECT = "project-b471d110-9146-4221-872"
TARGET_TYPES = ("phishing", "sandwich")


def select_diverse(seeds, target, per_date_cap=60):
    """密集日期优先取样：成本 = 去重日期数，故优先取样本密集的日期（少日期省成本），
    但仍跨多个日期保多样性。per_date_cap 上限避免全挤在一天。"""
    by_date = defaultdict(list)
    for s in seeds:
        if s.date:
            by_date[s.date].append(s)
    dates_by_density = sorted(by_date, key=lambda d: -len(by_date[d]))
    out = []
    for d in dates_by_density:
        out.extend(by_date[d][:per_date_cap])
        if len(out) >= target:
            break
    return out[:target]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--target", type=int, default=400)
    args = ap.parse_args()

    bq = BigQuerySource(project=PROJECT)
    all_seeds = load_seeds(DEFAULT_SEED_DIR / STRONG_SEED)
    selected = []
    for t in TARGET_TYPES:
        pool = [s for s in all_seeds if s.attack_type == t]
        sel = select_diverse(pool, args.target)
        sel = fill_missing_dates(sel, bq)
        sel = [s for s in sel if s.date]
        selected.extend(sel)
        print(f"{t}: 选 {len(sel)} 个, 覆盖 {len(set(s.date for s in sel))} 个日期")

    hashes = [s.tx_hash for s in selected]
    dates = [s.date for s in selected]
    ndate = len(set(dates))
    if args.dry_run:
        gb = bq.gb(bq.prefetch_by_dates(hashes, dates, dry_run=True))
        print(f"\n干跑：{len(hashes)} 笔 / {ndate} 个去重日期 → 扫描 ~{gb:.1f} GB (剩余~700GB)")
        return

    type_of = {s.tx_hash: s.attack_type for s in selected}
    txs, logs, traces = bq.prefetch_by_dates(hashes, dates)
    src = BQPrefetchSource(txs, logs)
    ctxs = build_contexts(list(txs.keys()), src, bq_traces=traces)
    for c in ctxs:
        c.latent["attack_type"] = type_of.get(c.tx_hash, "")
    save_contexts(ctxs, ROOT / "data/splits/d_known_ext.pkl.gz")
    from collections import Counter
    print(f"D_known_ext: {len(ctxs)} 笔 -> d_known_ext.pkl.gz")
    print("类型:", dict(Counter(c.latent.get('attack_type') for c in ctxs)))


if __name__ == "__main__":
    main()
