"""邻域交易图构建（设计架构 §4 M1）。

邻域来源（地址交易历史）需要 Etherscan API 或 BigQuery——RPC 无法高效列出
地址历史。此处定义 NeighborSource 抽象 + 由转账列表建 ego 图的纯函数；
真实 Etherscan/BigQuery 实现见 bigquery.py / etherscan.py（后续接入）。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Protocol

import networkx as nx


@dataclass
class TransferEdge:
    frm: str
    to: str
    value: float          # 以 wei 或代币最小单位计
    timestamp: int


class NeighborSource(Protocol):
    def neighbor_transfers(self, address: str, block_lo: int, block_hi: int,
                           limit: int) -> list[TransferEdge]: ...


def build_ego_graph(center: str, transfers: list[TransferEdge],
                    hops: int = 2, per_dir_cap: int = 100) -> nx.MultiDiGraph:
    """由一批转账边构造以 center 为中心的 ego 图。

    transfers 应已是 center 的 k 跳邻域转账（由 NeighborSource 提供）；此处只做
    截断与装配：每方向按金额降序保留前 per_dir_cap 条，避免热钱包爆炸。
    """
    center = center.lower()
    g = nx.MultiDiGraph()
    g.add_node(center)
    out_edges = sorted((t for t in transfers if t.frm.lower() == center),
                       key=lambda t: -t.value)[:per_dir_cap]
    in_edges = sorted((t for t in transfers if t.to.lower() == center),
                      key=lambda t: -t.value)[:per_dir_cap]
    rest = [t for t in transfers
            if t.frm.lower() != center and t.to.lower() != center]
    for t in out_edges + in_edges + rest:
        g.add_edge(t.frm.lower(), t.to.lower(), value=float(t.value),
                   timestamp=int(t.timestamp))
    return g


def ego_graph_from_receipt(center: str, decoded_transfers: list,
                           timestamp: int) -> nx.MultiDiGraph:
    """降级方案：无邻域源时，用本交易内的 Transfer 事件建一个最小 ego 图。

    保证 L1 提取器在无 Etherscan/BigQuery 时仍有可用输入（m1 恒为 1）。
    """
    edges = []
    for d in decoded_transfers:
        if d.event != "Transfer":
            continue
        frm = str(d.args.get("from", ""))
        to = str(d.args.get("to", ""))
        if frm and to:
            edges.append(TransferEdge(frm, to, float(d.args.get("value", 0) or 0),
                                      timestamp))
    return build_ego_graph(center, edges)
