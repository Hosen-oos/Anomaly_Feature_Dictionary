"""M1 采集编排：build_context(tx_hash) -> TxContext（设计架构 §4 M1）。

纯装配：从 ChainDataSource 取原始数据 → decode.py 解码 → graph.py 建图 →
labels.py 查标签 → 组装 TxContext。任一层采集失败置 None 并由 availability
体现，不抛异常中断批处理。输出与 tests/synthetic.py 同构，直接喂 Detector。
"""
from __future__ import annotations

from typing import Optional

from mlusd.collect.decode import (
    build_trace_summary, build_trace_summary_from_bq, decode_logs,
)
from mlusd.collect.graph import NeighborSource, build_ego_graph, ego_graph_from_receipt
from mlusd.collect.labels import LabelStore
from mlusd.collect.sources import ChainDataSource
from mlusd.types import TxContext


def _hexint(x, default: int = 0) -> int:
    if x is None:
        return default
    if isinstance(x, int):
        return x
    try:
        return int(x, 16)
    except (ValueError, TypeError):
        return default


def _involved_addresses(ctx_from: str, ctx_to: str, decoded_logs) -> list[str]:
    addrs = {ctx_from, ctx_to}
    for d in decoded_logs:
        for v in d.args.values():
            if isinstance(v, str) and v.startswith("0x") and len(v) == 42:
                addrs.add(v)
        addrs.add(d.address)
    return [a for a in addrs if a]


def build_context(tx_hash: str,
                  source: ChainDataSource,
                  neighbor_source: Optional[NeighborSource] = None,
                  labels: Optional[LabelStore] = None,
                  bq_trace_rows: Optional[list[dict]] = None,
                  neighbor_hops: int = 2,
                  neighbor_window: int = 100_000) -> Optional[TxContext]:
    """组装一笔交易的完整上下文。tx 不存在返回 None。

    trace 优先用 RPC debug_traceTransaction；免费档取不到时，若提供 bq_trace_rows
    （BigQuery traces 行）则用 call 级 trace 兜底（设计架构 §4 M1 双通道）。
    """
    tx = source.get_transaction(tx_hash)
    if tx is None:
        return None
    receipt = source.get_receipt(tx_hash)

    from_addr = (tx.get("from") or "").lower()
    to_addr = (tx.get("to") or "").lower()
    block_number = _hexint(tx.get("blockNumber"))
    tx_index = _hexint(tx.get("transactionIndex"))
    value = _hexint(tx.get("value"))
    status = _hexint(receipt.get("status"), 1) == 1 if receipt else True

    # L2 语义：解码事件日志
    decoded_logs = decode_logs(receipt.get("logs", [])) if receipt else []
    event_logs = decoded_logs or None

    # L3 轨迹：优先 RPC callTracer，免费档取不到时用 BigQuery call 级 trace 兜底
    call_trace = source.get_call_trace(tx_hash)
    struct_logs = source.get_struct_logs(tx_hash)
    trace = build_trace_summary(call_trace, struct_logs)
    if trace is None and bq_trace_rows:
        trace = build_trace_summary_from_bq(bq_trace_rows)
    internal_calls = trace.calls if trace is not None else None

    # L1 图：优先用邻域源建 k 跳图；否则降级为交易内 Transfer 图（保证 m1=1）
    timestamp = 0
    if neighbor_source is not None:
        edges = neighbor_source.neighbor_transfers(
            from_addr, max(0, block_number - neighbor_window),
            block_number + neighbor_window, limit=1000)
        ego = build_ego_graph(from_addr, edges, hops=neighbor_hops)
    else:
        ego = ego_graph_from_receipt(from_addr, decoded_logs, timestamp)

    # L4 情报：查标签库
    offchain = None
    if labels is not None:
        involved = _involved_addresses(from_addr, to_addr, decoded_logs)
        offchain = labels.lookup(involved, contract=to_addr)

    return TxContext(
        tx_hash=tx_hash, from_address=from_addr, to_address=to_addr,
        block_number=block_number, tx_index=tx_index, timestamp=timestamp,
        value=value, status=status,
        ego_graph=ego, input_data=tx.get("input"), event_logs=event_logs,
        internal_calls=internal_calls, trace=trace, offchain=offchain)
