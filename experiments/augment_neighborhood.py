"""邻域增强单数据集（改进 B）。后台跑：python -m experiments.augment_neighborhood dcal|dknown"""
from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mlusd.collect.bigquery import BigQuerySource                # noqa: E402
from mlusd.collect.neighbor import augment_with_neighborhood     # noqa: E402
from mlusd.dataset.build import load_contexts, save_contexts     # noqa: E402
from mlusd.dataset.seeds import (                                # noqa: E402
    DEFAULT_SEED_DIR, STRONG_SEED, fill_missing_dates, load_seeds, select_per_type)

PROJECT = "project-b471d110-9146-4221-872"
DCAL_DAYS = ["2021-05-15", "2021-11-15", "2022-03-15",
             "2022-09-15", "2023-03-15", "2023-08-15"]


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "dknown"
    bq = BigQuerySource(project=PROJECT)
    t0 = time.time()
    if which == "dcal":
        ctxs = load_contexts(ROOT / "data/splits/d_cal.pkl.gz")
        n = augment_with_neighborhood(ctxs, bq, DCAL_DAYS)
        save_contexts(ctxs, ROOT / "data/splits/d_cal_nbr.pkl.gz")
    else:
        seeds = fill_missing_dates(select_per_type(
            load_seeds(DEFAULT_SEED_DIR / STRONG_SEED), 20), bq)
        dates = sorted({s.date for s in seeds if s.date})
        ctxs = load_contexts(ROOT / "data/splits/d_known.pkl.gz")
        n = augment_with_neighborhood(ctxs, bq, dates)
        save_contexts(ctxs, ROOT / "data/splits/d_known_nbr.pkl.gz")
    print(f"[{which}] 邻域增强 {n}/{len(ctxs)} 笔, 耗时 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
