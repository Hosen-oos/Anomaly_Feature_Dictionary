"""L4 广义化：为已有数据集补充**不泄漏**的合约结构性信号。

背景（设计定稿 §1 与实测）：L4 是广义链下信息，不等于恶意地址黑名单。实测公开黑名单
（MyEtherWallet darklist 715 条）对本文 DeFi 攻击覆盖为 0，而用研究一种子自带标签会
构成标签泄漏。故改用合约元信息——部署时长、字节码规模、是否标准代币——这些是通用
属性，不由"该地址是否作恶"推出，可安全用于离线评测。

数据源：BigQuery `crypto_ethereum.contracts`（一次查询，按地址集合过滤）。

    python -m experiments.augment_l4 --dataset d_known_nbr --dry-run
    python -m experiments.augment_l4 --dataset d_known_nbr
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
from mlusd.types import OffchainRecord                        # noqa: E402

PROJECT = "project-b471d110-9146-4221-872"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", required=True,
                    help="data/splits 下的名字（不含扩展名），可多个——合并为一次查询摊薄成本")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    loaded = {}
    all_addrs: set[str] = set()
    for name in args.datasets:
        cs = load_contexts(ROOT / f"data/splits/{name}.pkl.gz")
        loaded[name] = cs
        a = {c.to_address.lower() for c in cs if c.to_address}
        all_addrs |= a
        print(f"  {name}: {len(cs)} 笔，唯一交互地址 {len(a)}")
    addrs = sorted(all_addrs)
    print(f"合计唯一地址 {len(addrs)}（一次查询覆盖全部数据集）")

    bq = BigQuerySource(project=PROJECT)
    if args.dry_run:
        gb = bq.gb(bq.contract_meta(addrs, dry_run=True))
        print(f"干跑：contracts 表扫描 ~{gb:.1f} GB（与地址数无关，为一次全表列扫描）")
        return

    meta = bq.contract_meta(addrs)
    print(f"命中合约 {len(meta)} / {len(addrs)}（未命中者为 EOA 或极早期合约）")

    for name, ctxs in loaded.items():
        _augment(name, ctxs, meta)


def _augment(name, ctxs, meta):
    n_aug = 0
    for c in ctxs:
        m = meta.get((c.to_address or "").lower())
        if not m:
            continue
        age_days = None
        if m["created_ts"] and c.block_number:
            # 用交易所在区块的时间近似（无 timestamp 时退化为按区块号估算 12s/块）
            tx_ts = c.timestamp or (m["created_ts"] + max(0, c.block_number - m["created_block"]) * 12)
            age_days = max(0.0, (tx_ts - m["created_ts"]) / 86400.0)
        cm = {"bytecode_len": m["bytecode_len"],
              "is_token": bool(m["is_erc20"] or m["is_erc721"])}
        if age_days is not None:
            cm["age_days"] = age_days
        if c.offchain is None:
            c.offchain = OffchainRecord(contract_meta=cm)
        else:
            c.offchain.contract_meta = cm
        n_aug += 1

    out = ROOT / f"data/splits/{name}_l4.pkl.gz"
    save_contexts(ctxs, out)
    print(f"  {name}: 补充 L4 结构信号 {n_aug}/{len(ctxs)} 笔 -> {out.name}  "
          f"可用性组={dict(Counter(''.join(map(str, c.availability)) for c in ctxs))}")


if __name__ == "__main__":
    main()
