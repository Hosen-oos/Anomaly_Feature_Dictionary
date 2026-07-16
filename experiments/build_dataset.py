"""数据集构建 CLI（设计架构 §7）。纯 BigQuery 路径，只需 gcloud ADC，无需 RPC key。

    cd D:\\科研\\开题\\mlusd
    # 正常校准集（单日期窗口示例；大规模分多窗口跑再合并）
    python -m experiments.build_dataset dcal --project <GCP_PROJECT_ID> \
        --ts-lo 2023-08-01 --ts-hi 2023-08-01 --frac 0.001 --n 20000
    # 已知异常标注集（按攻击 hash 清单，日期窗口需覆盖清单中攻击发生日）
    python -m experiments.build_dataset dknown --project <GCP_PROJECT_ID> \
        --ts-lo 2021-01-01 --ts-hi 2023-12-31 --attack-csv data/labels/attack_hashes.csv

提示：跑前可加 --dry-run 只估算 BigQuery 扫描字节（不计费、不真跑）。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["dcal", "dknown"])
    ap.add_argument("--project", required=True, help="GCP 项目 ID")
    ap.add_argument("--ts-lo", required=True, help="日期窗口起 (YYYY-MM-DD)")
    ap.add_argument("--ts-hi", required=True, help="日期窗口止 (YYYY-MM-DD)")
    ap.add_argument("--n", type=int, default=20_000)
    ap.add_argument("--frac", type=float, default=0.001)
    ap.add_argument("--attack-csv", default="data/labels/attack_hashes.csv")
    ap.add_argument("--out-dir", default="data/splits")
    ap.add_argument("--dry-run", action="store_true", help="只估算扫描字节，不真跑")
    args = ap.parse_args()

    from mlusd.collect.bigquery import BigQuerySource
    bq = BigQuerySource(project=args.project)

    if args.dry_run:
        dummy = ["0x" + "0" * 64]
        est = {
            "sample": bq.sample_normal_tx_hashes(args.ts_lo, args.ts_hi,
                                                 frac=args.frac, dry_run=True),
            "transactions": bq.transactions_for(dummy, args.ts_lo, args.ts_hi, dry_run=True),
            "logs": bq.logs_for(dummy, args.ts_lo, args.ts_hi, dry_run=True),
            "traces": bq.traces_for_transactions(dummy, args.ts_lo, args.ts_hi, dry_run=True),
        }
        total = sum(est.values())
        for k, v in est.items():
            print(f"  {k:14s}: {bq.gb(v):6.2f} GB")
        print(f"  {'合计':14s}: {bq.gb(total):6.2f} GB / 1024 GB 月免费额度")
        return

    from mlusd.dataset.build import build_dcal, build_dknown
    if args.mode == "dcal":
        ctxs = build_dcal(bq, args.ts_lo, args.ts_hi, n=args.n, frac=args.frac,
                          out_dir=args.out_dir)
        print(f"D_cal: {len(ctxs)} 笔 -> {args.out_dir}/d_cal.pkl.gz")
    else:
        ctxs = build_dknown(args.attack_csv, bq, args.ts_lo, args.ts_hi,
                            out_dir=args.out_dir)
        print(f"D_known: {len(ctxs)} 笔 -> {args.out_dir}/d_known.pkl.gz")


if __name__ == "__main__":
    main()
