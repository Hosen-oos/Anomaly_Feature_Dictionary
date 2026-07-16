"""研究一攻击/正常种子的加载与映射（设计架构 §7，衔接研究内容一）。

研究一在 D:\\实验\\研究一\\data-collect\\defi_verified_augmentation 下产出了带标注的
真实交易种子（DeFiHackLabs/DefiLlama/LABPAAD 来源），字段含 tx_hash、attack_type、
event_date、block_number、trace_summary.main_tx_rpc.block_timestamp 等。本模块把它们
的 attack_type 映射到研究二的六类字典，并抽取用于 BigQuery 分区裁剪的精确日期。

注意：研究一的富化是**汇总**（事件计数、top 合约），不含逐条事件与调用树，故仍需
BigQuery 按 tx_hash 取原始 logs/traces（见 dataset/build.py）。研究一在此提供的价值是
**hash + 类型标签 + 日期**——最费人工的标注部分。
"""
from __future__ import annotations

import datetime
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional

# 研究一 attack_type → 研究二六类字典；其余类型标为 "unknown"（开放集真未知素材）
TYPE_MAP = {
    "flash_loan_attack": "flash_loan",
    "price_oracle_manipulation": "price_manipulation",
    "sandwich_mev": "sandwich",
    "rug_pull": "rug_pull",
    "phishing_approval": "phishing",
    "ponzi_scheme": "ponzi",
}

DEFAULT_SEED_DIR = Path(r"D:\实验\研究一\data-collect\defi_verified_augmentation")
BALANCED_SEED = "seed_events_real_v3_50_final_structured.jsonl"
STRONG_SEED = "seed_events_real_v4_4000_strong.jsonl"


@dataclass
class Seed:
    tx_hash: str
    attack_type: str          # 已映射到六类之一，或 "unknown"
    raw_type: str             # 研究一原始 attack_type
    date: Optional[str]       # YYYY-MM-DD（BigQuery 分区裁剪用），可能为 None
    block_number: Optional[int]
    event_name: str = ""
    source: str = ""


def _extract_date(rec: dict) -> Optional[str]:
    """精确日期优先级：main_tx_rpc.block_timestamp > 完整 event_date > None。"""
    ts = rec.get("trace_summary", {}).get("main_tx_rpc", {}).get("block_timestamp")
    if ts:
        return datetime.datetime.utcfromtimestamp(int(ts)).strftime("%Y-%m-%d")
    ed = rec.get("event_date") or ""
    if len(ed) >= 10:
        return ed[:10]
    return None


def load_seeds(path: str | Path, keep_unknown: bool = False) -> list[Seed]:
    seeds = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            raw = r.get("attack_type", "")
            mapped = TYPE_MAP.get(raw, "unknown")
            if mapped == "unknown" and not keep_unknown:
                continue
            th = r.get("tx_hash")
            if not th:
                continue
            seeds.append(Seed(
                tx_hash=th.lower(), attack_type=mapped, raw_type=raw,
                date=_extract_date(r), block_number=r.get("block_number"),
                event_name=r.get("event_name", ""), source=r.get("source", "")))
    return seeds


def select_per_type(seeds: list[Seed], n_per_type: int,
                    prefer_dated: bool = True) -> list[Seed]:
    """每类取 n_per_type 个，优先挑有精确日期的（BigQuery 分区裁剪更省）。"""
    from collections import defaultdict
    buckets: dict[str, list[Seed]] = defaultdict(list)
    for s in seeds:
        buckets[s.attack_type].append(s)
    out = []
    for t, lst in buckets.items():
        if prefer_dated:
            lst = sorted(lst, key=lambda s: (s.date is None, s.tx_hash))
        out.extend(lst[:n_per_type])
    return out


def fill_missing_dates(seeds: list[Seed], bq_source) -> list[Seed]:
    """对无日期但有 block_number 的种子，用 BigQuery blocks 表补精确日期。"""
    need = [s for s in seeds if s.date is None and s.block_number]
    if not need:
        return seeds
    mapping = bq_source.dates_for_blocks([s.block_number for s in need])
    for s in need:
        s.date = mapping.get(s.block_number)
    return seeds


def to_csv(seeds: list[Seed], path: str | Path) -> None:
    import csv
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["tx_hash", "attack_type", "date", "block_number",
                    "raw_type", "event_name", "source"])
        for s in seeds:
            w.writerow([s.tx_hash, s.attack_type, s.date or "", s.block_number or "",
                        s.raw_type, s.event_name, s.source])


def group_by_date(seeds: list[Seed]) -> dict[str, list[str]]:
    """按日期分组 tx_hash，供逐日期窗口的 BigQuery 取数（最小化扫描分区）。"""
    from collections import defaultdict
    g: dict[str, list[str]] = defaultdict(list)
    for s in seeds:
        if s.date:
            g[s.date].append(s.tx_hash)
    return dict(g)
