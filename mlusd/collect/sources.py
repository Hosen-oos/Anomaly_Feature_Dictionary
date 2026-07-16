"""链上数据来源抽象（设计架构 §4 M1：provider 无关）。

ChainDataSource 只负责取"原始"数据（tx / receipt / callTrace / structLog）；
解码、建图、组装 TxContext 在 context.py。这样换 provider（Alchemy/QuickNode/
自建 Erigon）或换成缓存/mock，检测链路完全不受影响。
"""
from __future__ import annotations

from typing import Optional, Protocol, runtime_checkable

from mlusd.collect.cache import DiskCache


@runtime_checkable
class ChainDataSource(Protocol):
    def get_transaction(self, tx_hash: str) -> Optional[dict]: ...
    def get_receipt(self, tx_hash: str) -> Optional[dict]: ...
    def get_call_trace(self, tx_hash: str) -> Optional[dict]: ...
    def get_struct_logs(self, tx_hash: str) -> Optional[list[dict]]: ...


class JsonRpcSource:
    """标准 JSON-RPC 归档节点来源（Alchemy/QuickNode/自建）。

    trace 用 debug_traceTransaction；callTracer 取调用树，
    可选 structLog（含 SSTORE/opcode，体积大，默认关闭以省额度）。
    """

    def __init__(self, rpc_url: str, cache: Optional[DiskCache] = None,
                 fetch_call_trace: bool = True, fetch_struct_logs: bool = False,
                 timeout: float = 30.0):
        self.rpc_url = rpc_url
        self.cache = cache or DiskCache()
        # 免费档（如 Alchemy Free）不支持 debug_traceTransaction，置 False 直接跳过，
        # 改由 BigQuery 提供 call 级 trace（见 bigquery.py 与设计架构 §4 M1）。
        self.fetch_call_trace = fetch_call_trace
        self.fetch_struct_logs = fetch_struct_logs
        self.timeout = timeout

    def _rpc(self, method: str, params: list) -> Optional[dict]:
        import requests
        payload = {"jsonrpc": "2.0", "id": 1, "method": method, "params": params}
        try:
            resp = requests.post(self.rpc_url, json=payload, timeout=self.timeout)
            resp.raise_for_status()
            return resp.json().get("result")
        except Exception:
            return None   # 采集失败不中断批处理（设计架构 §4 M1）

    def get_transaction(self, tx_hash: str) -> Optional[dict]:
        return self.cache.cached("tx", tx_hash,
                                 lambda: self._rpc("eth_getTransactionByHash", [tx_hash]))

    def get_receipt(self, tx_hash: str) -> Optional[dict]:
        return self.cache.cached("receipt", tx_hash,
                                 lambda: self._rpc("eth_getTransactionReceipt", [tx_hash]))

    def get_call_trace(self, tx_hash: str) -> Optional[dict]:
        if not self.fetch_call_trace:
            return None
        return self.cache.cached(
            "calltrace", tx_hash,
            lambda: self._rpc("debug_traceTransaction",
                              [tx_hash, {"tracer": "callTracer"}]))

    def get_struct_logs(self, tx_hash: str) -> Optional[list[dict]]:
        if not self.fetch_struct_logs:
            return None
        res = self.cache.cached(
            "structlog", tx_hash,
            lambda: self._rpc("debug_traceTransaction",
                              [tx_hash, {"disableStack": True, "disableMemory": True}]))
        return res.get("structLogs") if isinstance(res, dict) else None


class BQPrefetchSource:
    """纯 BigQuery 数据源：把批量预取的 tx/logs/traces 当作 ChainDataSource 服务。

    D_cal/D_known 的批量构建走这条——一个日期窗口用几条 BQ 查询预取全部数据，
    再逐笔组装，完全不调 RPC（省额度、无 key 暴露、比逐笔 RPC 快几个数量级）。
    构造：BQPrefetchSource(bq.transactions_for(...), bq.logs_for(...))。
    trace 通过 build_context(bq_trace_rows=...) 单独传入，此处 get_call_trace 返 None。
    """

    def __init__(self, txs: dict[str, dict], logs: dict[str, list[dict]]):
        self._txs = txs
        self._logs = logs

    def get_transaction(self, tx_hash: str) -> Optional[dict]:
        return self._txs.get(tx_hash.lower()) or self._txs.get(tx_hash)

    def get_receipt(self, tx_hash: str) -> Optional[dict]:
        tx = self.get_transaction(tx_hash)
        if tx is None:
            return None
        h = tx["hash"]
        return {"status": tx.get("receipt_status", 1),
                "logs": self._logs.get(h, self._logs.get(h.lower(), []))}

    def get_call_trace(self, tx_hash: str) -> Optional[dict]:
        return None      # trace 走 build_context 的 bq_trace_rows 通道

    def get_struct_logs(self, tx_hash: str) -> Optional[list[dict]]:
        return None


class MockSource:
    """内存 mock：用固定 dict 喂数据，供离线测试/黄金交易回归（无需 key）。"""

    def __init__(self, txs: dict[str, dict], receipts: dict[str, dict],
                 traces: Optional[dict[str, dict]] = None,
                 struct_logs: Optional[dict[str, list[dict]]] = None):
        self._txs = txs
        self._receipts = receipts
        self._traces = traces or {}
        self._struct = struct_logs or {}

    def get_transaction(self, tx_hash: str) -> Optional[dict]:
        return self._txs.get(tx_hash)

    def get_receipt(self, tx_hash: str) -> Optional[dict]:
        return self._receipts.get(tx_hash)

    def get_call_trace(self, tx_hash: str) -> Optional[dict]:
        return self._traces.get(tx_hash)

    def get_struct_logs(self, tx_hash: str) -> Optional[list[dict]]:
        return self._struct.get(tx_hash)
