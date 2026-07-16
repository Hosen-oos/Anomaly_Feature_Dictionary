"""数据集构建流水线（设计架构 §7）。

用法（ADC 就绪后）：
    from mlusd.collect.sources import JsonRpcSource
    from mlusd.collect.bigquery import BigQuerySource
    from mlusd.dataset.build import build_dcal, build_dknown

数据落 data/splits/*.jsonl（TxContext 序列化）+ data/cache/（原始响应）。
攻击 hash 清单是 CSV：tx_hash,attack_type,event_name[,block_number]。
"""
from __future__ import annotations

import csv
import gzip
import json
import pickle
from dataclasses import asdict
from pathlib import Path
from typing import Iterable, Optional

from mlusd.collect.context import build_context
from mlusd.collect.graph import NeighborSource
from mlusd.collect.labels import LabelStore
from mlusd.collect.sources import ChainDataSource
from mlusd.types import TxContext


# ---------------------------------------------------------------- 攻击清单

def load_attack_hashes(csv_path: str | Path) -> list[dict]:
    """读取攻击交易清单 CSV，返回 [{tx_hash, attack_type, event_name, ...}]。

    来源建议：DeFiHackLabs、慢雾/PeckShield 事件库、Etherscan 标签，逐笔核对 hash。
    """
    rows = []
    with open(csv_path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            r["tx_hash"] = r["tx_hash"].strip().lower()
            rows.append(r)
    return rows


# ---------------------------------------------------------------- 批量采集

def build_contexts(tx_hashes: Iterable[str],
                   source: ChainDataSource,
                   neighbor_source: Optional[NeighborSource] = None,
                   labels: Optional[LabelStore] = None,
                   bq_traces: Optional[dict[str, list[dict]]] = None,
                   on_progress=None) -> list[TxContext]:
    """批量组装 TxContext。单笔失败置 None 跳过，不中断（设计架构 §4 M1）。"""
    out: list[TxContext] = []
    for i, h in enumerate(tx_hashes):
        rows = (bq_traces or {}).get(h.lower())
        ctx = build_context(h, source, neighbor_source=neighbor_source,
                            labels=labels, bq_trace_rows=rows)
        if ctx is not None:
            out.append(ctx)
        if on_progress is not None:
            on_progress(i + 1, ctx)
    return out


# ---------------------------------------------------------------- 序列化

def save_contexts(contexts: list[TxContext], path: str | Path) -> None:
    """落盘为 .pkl.gz（TxContext 含 networkx 图，用 pickle 保真）。"""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(p, "wb") as f:
        pickle.dump(contexts, f)


def load_contexts(path: str | Path) -> list[TxContext]:
    with gzip.open(path, "rb") as f:
        return pickle.load(f)


def save_manifest(path: str | Path, meta: dict) -> None:
    """落盘数据集清单（规模、可用性组分布、时间范围等），入库版本控制。"""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(meta, ensure_ascii=False, indent=2),
                          encoding="utf-8")


# ---------------------------------------------------------------- 端到端构建

def _prefetch_contexts(bq_source, hashes: list[str], ts_lo: str, ts_hi: str,
                       labels: Optional[LabelStore] = None) -> list[TxContext]:
    """纯 BigQuery 路径：一个日期窗口预取 tx/logs/traces，逐笔组装（不调 RPC）。"""
    from mlusd.collect.sources import BQPrefetchSource
    txs = bq_source.transactions_for(hashes, ts_lo, ts_hi)
    logs = bq_source.logs_for(hashes, ts_lo, ts_hi)
    traces = bq_source.traces_for_transactions(hashes, ts_lo, ts_hi)
    src = BQPrefetchSource(txs, logs)
    return build_contexts(list(txs.keys()), src, labels=labels, bq_traces=traces)


def build_dcal(bq_source, ts_lo: str, ts_hi: str,
               n: int = 50_000, frac: float = 0.001,
               labels: Optional[LabelStore] = None,
               out_dir: str | Path = "data/splits") -> list[TxContext]:
    """正常校准集 D_cal：BigQuery 采样交易 hash → 纯 BQ 组装（无需 RPC）。

    ts_lo/ts_hi 为日期窗口（分区裁剪，控制费用）。剔除命中标签库的地址（正常集
    不含已知恶意）。大规模采集应分多个日期窗口调用本函数再合并，摊薄单窗口字节。
    """
    hashes = bq_source.sample_normal_tx_hashes(ts_lo, ts_hi, frac=frac, limit=n)
    ctxs = _prefetch_contexts(bq_source, hashes, ts_lo, ts_hi, labels=labels)
    if labels is not None:   # 剔除任何命中恶意标签的"疑似非正常"样本
        ctxs = [c for c in ctxs if not (c.offchain and c.offchain.label_hits)]
    save_contexts(ctxs, Path(out_dir) / "d_cal.pkl.gz")
    save_manifest(Path(out_dir) / "d_cal.manifest.json", _manifest(ctxs))
    return ctxs


def build_dknown_from_seeds(seeds, bq_source,
                            labels: Optional[LabelStore] = None,
                            out_dir: str | Path = "data/splits",
                            save_name: str = "d_known") -> list[TxContext]:
    """从研究一种子构建 D_known：按日期分组，逐日期窗口纯 BQ 取数（最省分区扫描）。

    seeds: list[dataset.seeds.Seed]（已 fill_missing_dates）。同日期的多笔攻击合并到
    一次查询。attack_type 存入 latent（仅评测/权重更新用，不进 M2 fit）。
    """
    from mlusd.collect.sources import BQPrefetchSource
    type_of = {s.tx_hash: s.attack_type for s in seeds}
    dated = [s for s in seeds if s.date]
    hashes = [s.tx_hash for s in dated]
    dates = [s.date for s in dated]
    # 一次性取完所有攻击的 tx/logs/traces（3 次查询，DATE IN UNNEST 裁剪分区）
    txs, logs, traces = bq_source.prefetch_by_dates(hashes, dates)
    src = BQPrefetchSource(txs, logs)
    all_ctxs = build_contexts(list(txs.keys()), src, labels=labels, bq_traces=traces)
    for c in all_ctxs:
        c.latent["attack_type"] = type_of.get(c.tx_hash, "")
    by_date = {d: 1 for d in dates}
    n_missing = sum(1 for s in seeds if not s.date)
    save_contexts(all_ctxs, Path(out_dir) / f"{save_name}.pkl.gz")
    save_manifest(Path(out_dir) / f"{save_name}.manifest.json",
                  {**_manifest(all_ctxs), "seeds": len(seeds),
                   "no_date_skipped": n_missing, "dates": len(by_date)})
    return all_ctxs


def build_dknown(attack_csv: str | Path, bq_source,
                 ts_lo: str, ts_hi: str,
                 labels: Optional[LabelStore] = None,
                 out_dir: str | Path = "data/splits") -> list[TxContext]:
    """已知异常标注集 D_known：按攻击 hash 清单纯 BQ 采集，attack_type 存入 latent。

    ts_lo/ts_hi 应覆盖清单中所有攻击的发生日期（可取 min~max，或分批按日期窗口调用）。
    """
    rows = load_attack_hashes(attack_csv)
    hashes = [r["tx_hash"] for r in rows]
    type_of = {r["tx_hash"]: r.get("attack_type", "") for r in rows}
    ctxs = _prefetch_contexts(bq_source, hashes, ts_lo, ts_hi, labels=labels)
    for c in ctxs:               # 记录金标签（仅评测/权重更新用，不进 M2 fit）
        c.latent["attack_type"] = type_of.get(c.tx_hash, "")
    save_contexts(ctxs, Path(out_dir) / "d_known.pkl.gz")
    save_manifest(Path(out_dir) / "d_known.manifest.json", _manifest(ctxs))
    return ctxs


def _manifest(ctxs: list[TxContext], block_lo: int = 0, block_hi: int = 0) -> dict:
    from collections import Counter
    groups = Counter("".join(map(str, c.availability)) for c in ctxs)
    types = Counter(c.latent.get("attack_type", "") for c in ctxs)
    return {
        "n": len(ctxs),
        "availability_groups": dict(groups),
        "attack_types": {k: v for k, v in types.items() if k},
        "block_range": [block_lo, block_hi],
    }
