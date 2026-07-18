"""构建 D_open 真未知集：研究一里不属于六类的其他攻击（重入/访问控制/业务逻辑等），
字典中完全没有 → 终极开放集测试（能否检出从未见过的攻击类型）。

    python -m experiments.build_dopen --dry-run
    python -m experiments.build_dopen
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mlusd.collect.bigquery import BigQuerySource            # noqa: E402
from mlusd.dataset.build import build_dknown_from_seeds       # noqa: E402
from mlusd.dataset.seeds import (                             # noqa: E402
    DEFAULT_SEED_DIR, STRONG_SEED, fill_missing_dates, load_seeds)

PROJECT = "project-b471d110-9146-4221-872"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    seeds = load_seeds(DEFAULT_SEED_DIR / STRONG_SEED, keep_unknown=True)
    unk = [s for s in seeds if s.attack_type == "unknown"]
    print("真未知原始类型:", dict(Counter(s.raw_type for s in unk)))

    bq = BigQuerySource(project=PROJECT)
    unk = fill_missing_dates(unk, bq)
    unk = [s for s in unk if s.date]
    dates = sorted(set(s.date for s in unk))
    print(f"可用 {len(unk)} 笔, 覆盖 {len(dates)} 个日期")

    if args.dry_run:
        gb = bq.gb(bq.prefetch_by_dates([s.tx_hash for s in unk], dates, dry_run=True))
        print(f"干跑：扫描 ~{gb:.1f} GB")
        return

    ctxs = build_dknown_from_seeds(unk, bq, out_dir=ROOT / "data/splits", save_name="d_open")
    print(f"D_open: {len(ctxs)} 笔 -> d_open.pkl.gz")
    print("原始类型:", dict(Counter(c.latent.get('attack_type') for c in ctxs)))


if __name__ == "__main__":
    main()
