"""邻域图增强（改进 B，设计架构 §4 M1 L1 层）。

现有 context 的 ego_graph 只含单笔交易内的转账，看不到跨交易的归集/扇入结构
（phishing drainer 聚集、ponzi 合约扇入）。本模块从 BigQuery token_transfers 取
关键地址的同期邻域，重建更丰富的 ego 图。复用已缓存的 tx/logs/traces，只补 transfers。

一致性：D_known 与 D_cal 用同一函数增强，L1 信号仍可比（校准有效）。
"""
from __future__ import annotations

from collections import defaultdict

import networkx as nx

from mlusd.types import TxContext


def key_addresses(ctx: TxContext, top_k: int = 6) -> list[str]:
    """一笔交易的关键地址：from/to + 事件转账里最常出现的对手方。"""
    addrs = {ctx.from_address, ctx.to_address}
    freq: dict[str, int] = defaultdict(int)
    for lg in (ctx.event_logs or []):
        for k in ("from", "to", "owner", "spender", "src", "dst"):
            v = lg.args.get(k)
            if isinstance(v, str) and v.startswith("0x") and len(v) == 42:
                freq[v] += 1
    for a, _ in sorted(freq.items(), key=lambda x: -x[1])[:top_k]:
        addrs.add(a)
    return [a for a in addrs if a and a.startswith("0x")]


def _build_graph(center: str, transfers: list[dict], per_addr_cap: int = 200) -> nx.MultiDiGraph:
    g = nx.MultiDiGraph()
    g.add_node(center)
    seen_out: dict[str, int] = defaultdict(int)
    seen_in: dict[str, int] = defaultdict(int)
    for t in sorted(transfers, key=lambda x: -x["value"]):
        f, to = t["from"], t["to"]
        if not f or not to:
            continue
        if seen_out[f] >= per_addr_cap and seen_in[to] >= per_addr_cap:
            continue
        g.add_edge(f, to, value=float(t["value"]), timestamp=int(t.get("ts", 0)))
        seen_out[f] += 1
        seen_in[to] += 1
    return g


def augment_with_neighborhood(contexts: list[TxContext], bq_source,
                              dates: list[str], per_addr_cap: int = 200) -> int:
    """就地把每个 context 的 ego_graph 替换为含真实邻域的图。

    一次查询取所有关键地址在 dates 内的转移，再按地址索引分配给各 context。
    返回成功增强的 context 数。dates 应覆盖这些交易的发生日期。
    """
    # 汇总全部关键地址
    ctx_keys = {c.tx_hash: set(key_addresses(c)) for c in contexts}
    all_addrs = sorted({a for s in ctx_keys.values() for a in s})
    if not all_addrs:
        return 0
    transfers = bq_source.transfers_touching(all_addrs, dates)
    # 按地址索引转移（from 或 to 命中即归入）
    by_addr: dict[str, list[dict]] = defaultdict(list)
    for t in transfers:
        if t["from"]:
            by_addr[t["from"]].append(t)
        if t["to"]:
            by_addr[t["to"]].append(t)
    # 每地址只保留金额最高的若干转移，界定每个 context 的建图开销（热钱包封顶）
    cap_per_addr = max(per_addr_cap * 3, 600)
    for a in list(by_addr):
        lst = by_addr[a]
        if len(lst) > cap_per_addr:
            by_addr[a] = sorted(lst, key=lambda x: -x["value"])[:cap_per_addr]
    n = 0
    for c in contexts:
        keys = ctx_keys[c.tx_hash]
        edges = []
        seen = set()
        for a in keys:
            for t in by_addr.get(a, []):
                eid = (t["from"], t["to"], t["value"], t.get("ts", 0))
                if eid not in seen:
                    seen.add(eid)
                    edges.append(t)
        if edges:
            c.ego_graph = _build_graph(c.from_address, edges, per_addr_cap)
            n += 1
    return n
