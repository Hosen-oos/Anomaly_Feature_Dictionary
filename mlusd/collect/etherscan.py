"""Etherscan 邻域来源（设计架构 §4 M1：地址交易历史 → ego 图）。

免费 API key 即可（5 req/s）。适合按需取单地址邻域；大规模校准集用 BigQuery。
需要 requests；失败静默返回空列表，不中断批处理。
"""
from __future__ import annotations

from typing import Optional

from mlusd.collect.cache import DiskCache
from mlusd.collect.graph import TransferEdge

_BASE = "https://api.etherscan.io/api"


class EtherscanNeighborSource:
    def __init__(self, api_key: str, cache: Optional[DiskCache] = None,
                 timeout: float = 20.0):
        self.api_key = api_key
        self.cache = cache or DiskCache()
        self.timeout = timeout

    def _get(self, params: dict) -> list[dict]:
        import requests
        params = {**params, "apikey": self.api_key}
        try:
            r = requests.get(_BASE, params=params, timeout=self.timeout)
            r.raise_for_status()
            data = r.json()
            return data.get("result", []) if data.get("status") == "1" else []
        except Exception:
            return []

    def neighbor_transfers(self, address: str, block_lo: int, block_hi: int,
                           limit: int) -> list[TransferEdge]:
        addr = address.lower()
        key = f"{addr}_{block_lo}_{block_hi}_{limit}"

        def _fetch():
            # 普通转账 + ERC20 转账各取一页
            edges = []
            for action in ("txlist", "tokentx"):
                rows = self._get({
                    "module": "account", "action": action, "address": addr,
                    "startblock": block_lo, "endblock": block_hi,
                    "page": 1, "offset": min(limit, 1000), "sort": "desc"})
                for t in rows:
                    edges.append({
                        "frm": (t.get("from") or "").lower(),
                        "to": (t.get("to") or "").lower(),
                        "value": float(t.get("value", 0) or 0),
                        "timestamp": int(t.get("timeStamp", 0) or 0)})
            return edges

        raw = self.cache.cached("neighbor", key, _fetch)
        return [TransferEdge(e["frm"], e["to"], e["value"], e["timestamp"])
                for e in raw if e["frm"] and e["to"]]
