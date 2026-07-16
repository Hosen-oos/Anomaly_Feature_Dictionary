"""M1 交易上下文采集器（设计架构 §4 M1）。

build_context(tx_hash, sources...) -> TxContext，与 tests/synthetic.py 的输出同构，
可直接喂给 M2-M6，无需改动检测流水线。数据来源 provider 无关：单笔交易数据走
RPC（JsonRpcSource），邻域/批量走 Etherscan 或 BigQuery，全程落盘缓存。
"""
from mlusd.collect.context import build_context
from mlusd.collect.sources import ChainDataSource, JsonRpcSource, MockSource

__all__ = ["build_context", "ChainDataSource", "JsonRpcSource", "MockSource"]
