"""合成交易上下文生成器（设计架构 §9 的黄金测试交易 + 校准集替身）。

在接入归档节点前，用它生成结构可控的 TxContext：一批"正常"交易做校准集，
若干条按攻击机理构造的交易做端到端判定测试。所有金额单位任意、仅供打通链路。
"""
from __future__ import annotations

import random

import networkx as nx

from mlusd.types import (
    Call, DecodedLog, OffchainRecord, TraceSummary, TxContext,
)

ETHER = 10 ** 18
UNLIMITED = 2 ** 256 - 1


def _ego_graph(center: str, n_out: int, n_in: int, seed: int,
               ring: bool = False) -> nx.MultiDiGraph:
    rng = random.Random(seed)
    g = nx.MultiDiGraph()
    g.add_node(center)
    t0 = 1_700_000_000
    outs = [f"0xout{i:03d}" for i in range(n_out)]
    ins = [f"0xin{i:03d}" for i in range(n_in)]
    for i, o in enumerate(outs):
        g.add_edge(center, o, value=rng.uniform(0.1, 5) * ETHER,
                   timestamp=t0 + i * 60)
    for i, s in enumerate(ins):
        g.add_edge(s, center, value=rng.uniform(0.1, 5) * ETHER,
                   timestamp=t0 + i * 45)
    if ring and outs:
        g.add_edge(outs[0], center, value=rng.uniform(0.1, 5) * ETHER,
                   timestamp=t0 + 1000)
    return g


def normal_tx(seed: int) -> TxContext:
    """一笔普通交易：小邻域、简单转账或单次 swap、浅调用、无风险标签。"""
    rng = random.Random(seed)
    center = f"0xuser{seed:04d}"
    logs = [DecodedLog(address="0xtoken", event="Transfer",
                       args={"from": center, "to": f"0xrecv{seed}",
                             "value": rng.uniform(0.1, 3) * ETHER})]
    if rng.random() < 0.4:      # 部分正常交易含一次 swap
        logs.append(DecodedLog(address="0xdex", event="Swap",
                               args={"sender": center, "amount0": 1 * ETHER,
                                     "amount1": 1 * ETHER}))
    trace = TraceSummary(
        calls=[Call(frm=center, to="0xtoken", kind="CALL", depth=1)],
        max_depth=1, sstore_count=rng.randint(0, 3), reverted_subcalls=0)
    return TxContext(
        tx_hash=f"0xnormal{seed:06d}", from_address=center, to_address="0xtoken",
        block_number=18_000_000 + seed, tx_index=rng.randint(0, 200),
        timestamp=1_700_000_000 + seed, value=0, status=True,
        ego_graph=_ego_graph(center, rng.randint(1, 4), rng.randint(1, 4), seed),
        event_logs=logs, internal_calls=trace.calls, trace=trace,
        offchain=OffchainRecord(contract_verified=True, audited=rng.random() < 0.3))


def flash_loan_tx() -> TxContext:
    """闪电贷攻击：flashloan + 连环 swap + repay，深调用、状态写入多。"""
    center = "0xattacker"
    logs = [
        DecodedLog(address="0xaave", event="FlashLoan",
                   args={"sender": center, "amount": 3000 * ETHER}),
        DecodedLog(address="0xdex1", event="Swap",
                   args={"sender": center, "amount0": 3000 * ETHER, "amount1": 6000 * ETHER}),
        DecodedLog(address="0xdex2", event="Swap",
                   args={"sender": center, "amount0": 6000 * ETHER, "amount1": 3400 * ETHER}),
        DecodedLog(address="0xaave", event="Repay",
                   args={"from": center, "amount": 3000 * ETHER}),
        DecodedLog(address="0xweth", event="Transfer",
                   args={"from": "0xdex2", "to": center, "value": 400 * ETHER}),
    ]
    calls = [Call(frm=center, to="0xaave", kind="CALL", depth=1),
             Call(frm="0xaave", to="0xdex1", kind="CALL", depth=2),
             Call(frm="0xdex1", to="0xdex2", kind="CALL", depth=3),
             Call(frm="0xdex2", to="0xaave", kind="CALL", depth=2)]
    trace = TraceSummary(calls=calls, max_depth=3, sstore_count=25, reverted_subcalls=0)
    return TxContext(
        tx_hash="0xflashloan_attack", from_address=center, to_address="0xaave",
        block_number=18_500_000, tx_index=3, timestamp=1_700_100_000, value=0,
        status=True, ego_graph=_ego_graph(center, 6, 2, 99),
        event_logs=logs, internal_calls=calls, trace=trace,
        offchain=OffchainRecord(contract_verified=False))


def phishing_tx() -> TxContext:
    """钓鱼（ice phishing）：无限授权后资金即被 spender 转走，命中恶意标签。"""
    victim = "0xvictim"
    spender = "0xdrainer"
    logs = [
        DecodedLog(address="0xtoken", event="Approval",
                   args={"owner": victim, "spender": spender, "value": UNLIMITED}),
        DecodedLog(address="0xtoken", event="Transfer",
                   args={"from": victim, "to": spender, "value": 50000 * ETHER}),
    ]
    trace = TraceSummary(
        calls=[Call(frm=victim, to="0xtoken", kind="CALL", depth=1)],
        max_depth=1, sstore_count=2, reverted_subcalls=0)
    g = _ego_graph(spender, 1, 15, 7)   # drainer 大量扇入
    return TxContext(
        tx_hash="0xphishing_attack", from_address=victim, to_address="0xtoken",
        block_number=18_600_000, tx_index=50, timestamp=1_700_200_000, value=0,
        status=True, ego_graph=g, event_logs=logs, internal_calls=trace.calls,
        trace=trace,
        offchain=OffchainRecord(
            label_hits=[{"address": spender, "label": "Fake_Phishing",
                         "source": "Etherscan", "severity": 0.9}],
            contract_verified=False))


def fan_in_anomaly_tx() -> TxContext:
    """结构异常：以 from_address 为中心的极端扇入（30 入 1 出），供 L1 图模型测试。"""
    center = "0xcollector"
    g = nx.MultiDiGraph()
    g.add_node(center)
    for i in range(30):
        g.add_edge(f"0xsrc{i:03d}", center, value=(i + 1) * ETHER,
                   timestamp=1_700_000_000 + i * 10)
    g.add_edge(center, "0xsink", value=100 * ETHER, timestamp=1_700_000_500)
    return TxContext(
        tx_hash="0xfanin_anomaly", from_address=center, to_address="0xsink",
        block_number=18_700_000, tx_index=5, timestamp=1_700_000_600,
        value=0, status=True, ego_graph=g,
        event_logs=None, internal_calls=None, trace=None, offchain=None)


def pure_transfer_tx() -> TxContext:
    """一笔纯 ETH 转账（黄金测试：应判 NORMAL，多层不可用）。"""
    center = "0xalice"
    g = nx.MultiDiGraph()
    g.add_node(center)
    g.add_edge(center, "0xbob", value=1 * ETHER, timestamp=1_700_000_000)
    return TxContext(
        tx_hash="0xpure_transfer", from_address=center, to_address="0xbob",
        block_number=18_000_001, tx_index=10, timestamp=1_700_000_500,
        value=1 * ETHER, status=True, ego_graph=g,
        event_logs=None, internal_calls=None, trace=None, offchain=None)


def normal_pure_transfer(seed: int) -> TxContext:
    """正常纯转账（可用性组 "1000"）：校准集必须含此类，否则真实纯转账无同组基线。"""
    import random
    rng = random.Random(seed)
    center = f"0xp{seed:05d}"
    g = nx.MultiDiGraph()
    g.add_node(center)
    g.add_edge(center, f"0xq{seed}", value=rng.uniform(0.05, 4) * ETHER,
               timestamp=1_700_000_000 + seed)
    if rng.random() < 0.5:
        g.add_edge(f"0xr{seed}", center, value=rng.uniform(0.05, 4) * ETHER,
                   timestamp=1_700_000_000 + seed + 30)
    return TxContext(
        tx_hash=f"0xpure{seed:06d}", from_address=center, to_address=f"0xq{seed}",
        block_number=18_000_000 + seed, tx_index=rng.randint(0, 200),
        timestamp=1_700_000_000 + seed, value=rng.uniform(0.05, 4) * ETHER,
        status=True, ego_graph=g,
        event_logs=None, internal_calls=None, trace=None, offchain=None)


def calibration_set(n: int = 2000) -> list[TxContext]:
    """混合校准集：~70% 合约交互交易 + ~30% 纯转账，覆盖多个可用性组。"""
    n_pure = n // 3
    return ([normal_tx(i) for i in range(n - n_pure)]
            + [normal_pure_transfer(i) for i in range(n_pure)])
