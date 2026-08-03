"""BigQuery 批量来源（设计架构 §5 数据通道：正常校准集走 BigQuery 省 RPC 额度）。

bigquery-public-data.crypto_ethereum.traces 有 call 级内部交易（from/to/value/
input/call_type/trace_address），足够建调用树与邻域图；但**不含 SSTORE/opcode**，
细节 trace 仍需 debug_traceTransaction（对少量攻击交易调用）。

⚠️ 费用：这些表按 DATE(block_timestamp) 分区，且几 TB 级。**必须带日期范围**
（ts_lo/ts_hi）裁剪分区，否则一条查询就扫几百 GB（每月仅 1TB 免费）。所有方法
都要求日期界，并提供 estimate_bytes 干跑估算——真跑前先看要扫多少字节。

google-cloud-bigquery 为可选依赖，import 延迟到实际使用。
"""
from __future__ import annotations

from typing import Optional

from mlusd.collect.graph import TransferEdge

# 取一批区块内的 call 级 trace（构造调用树 n-gram 校准集）——带日期分区裁剪
TRACES_BY_BLOCKS = """
SELECT transaction_hash, from_address, to_address, value,
       call_type, trace_address, status
FROM `bigquery-public-data.crypto_ethereum.traces`
WHERE DATE(block_timestamp) BETWEEN @ts_lo AND @ts_hi
  AND block_number BETWEEN @lo AND @hi
  AND trace_type = 'call'
ORDER BY transaction_hash, trace_address
"""

# 取地址邻域的代币转移（用于 ego 图）——带日期分区裁剪
NEIGHBOR_TRANSFERS = """
SELECT from_address, to_address, value, block_timestamp
FROM `bigquery-public-data.crypto_ethereum.token_transfers`
WHERE DATE(block_timestamp) BETWEEN @ts_lo AND @ts_hi
  AND (from_address = @addr OR to_address = @addr)
ORDER BY value DESC
LIMIT @limit
"""

# 按交易哈希批量取 call 级 trace（免费档无 debug 时的 trace 主路径）——带日期分区裁剪
TRACES_FOR_TXS = """
SELECT transaction_hash, from_address, to_address, value,
       call_type, trace_address, status
FROM `bigquery-public-data.crypto_ethereum.traces`
WHERE DATE(block_timestamp) BETWEEN @ts_lo AND @ts_hi
  AND transaction_hash IN UNNEST(@hashes)
  AND trace_type = 'call'
ORDER BY transaction_hash, trace_address
"""

# 随机采样正常交易的哈希（构造 D_cal）——带日期分区裁剪
SAMPLE_NORMAL_TX = """
SELECT `hash` AS tx_hash, from_address, to_address, block_number
FROM `bigquery-public-data.crypto_ethereum.transactions`
WHERE DATE(block_timestamp) BETWEEN @ts_lo AND @ts_hi
  AND RAND() < @frac
LIMIT @limit
"""

# 按 hash 批量取交易本体（含 receipt_status，省去单独的 receipt 查询）——纯 BQ 路径
TXS_FOR_HASHES = """
SELECT `hash`, from_address, to_address, value, block_number,
       transaction_index, input, receipt_status
FROM `bigquery-public-data.crypto_ethereum.transactions`
WHERE DATE(block_timestamp) BETWEEN @ts_lo AND @ts_hi
  AND `hash` IN UNNEST(@hashes)
"""

# 按 hash 批量取事件日志（L2 语义层）——纯 BQ 路径
LOGS_FOR_HASHES = """
SELECT transaction_hash, address, topics, data, log_index
FROM `bigquery-public-data.crypto_ethereum.logs`
WHERE DATE(block_timestamp) BETWEEN @ts_lo AND @ts_hi
  AND transaction_hash IN UNNEST(@hashes)
ORDER BY transaction_hash, log_index
"""

# 邻域：取一组地址在给定日期集合内涉及的代币转移（建 k=1 跳邻域图）
# 按金额降序 + LIMIT 界定返回行数（热钱包会返回海量转移，须封顶）
TRANSFERS_TOUCHING = """
SELECT from_address, to_address, value, block_timestamp
FROM `bigquery-public-data.crypto_ethereum.token_transfers`
WHERE DATE(block_timestamp) IN UNNEST(@dates)
  AND (from_address IN UNNEST(@addrs) OR to_address IN UNNEST(@addrs))
ORDER BY value DESC
LIMIT @limit
"""

# 区块号 -> 日期（补种子缺失日期；blocks 表仅两列，一次全表扫较小）
DATES_FOR_BLOCKS = """
SELECT number, DATE(timestamp) AS d
FROM `bigquery-public-data.crypto_ethereum.blocks`
WHERE number IN UNNEST(@blocks)
"""

# 合约元信息（L4 广义化：**不泄漏**的结构性链下信号）
# 部署时间/标准符合性/字节码规模都是通用属性，不是从"该地址是否作恶"推出来的，
# 故不构成标签泄漏——区别于恶意地址黑名单（公开黑名单对 DeFi 攻击覆盖为 0）。
CONTRACT_META = """
SELECT address,
       UNIX_SECONDS(block_timestamp) AS created_ts,
       block_number AS created_block,
       is_erc20, is_erc721,
       LENGTH(bytecode) AS bytecode_len
FROM `bigquery-public-data.crypto_ethereum.contracts`
WHERE address IN UNNEST(@addrs)
"""

# 指定日期集合 + 指定 hash 集合的单次查询（D_known：3 次查询取完所有攻击，非逐日期）
# DATE IN UNNEST 裁剪到恰好那些分区，扫描字节与逐日期相同但查询次数从 3N 降到 3。
TXS_FOR_HASHES_DATES = """
SELECT `hash`, from_address, to_address, value, block_number,
       transaction_index, input, receipt_status
FROM `bigquery-public-data.crypto_ethereum.transactions`
WHERE DATE(block_timestamp) IN UNNEST(@dates)
  AND `hash` IN UNNEST(@hashes)
"""
LOGS_FOR_HASHES_DATES = """
SELECT transaction_hash, address, topics, data, log_index
FROM `bigquery-public-data.crypto_ethereum.logs`
WHERE DATE(block_timestamp) IN UNNEST(@dates)
  AND transaction_hash IN UNNEST(@hashes)
ORDER BY transaction_hash, log_index
"""
TRACES_FOR_HASHES_DATES = """
SELECT transaction_hash, from_address, to_address, value,
       call_type, trace_address, status
FROM `bigquery-public-data.crypto_ethereum.traces`
WHERE DATE(block_timestamp) IN UNNEST(@dates)
  AND transaction_hash IN UNNEST(@hashes)
  AND trace_type = 'call'
ORDER BY transaction_hash, trace_address
"""


class BigQuerySource:
    """BigQuery 邻域/批量来源。需要 google-cloud-bigquery 与 GCP 凭据（ADC）。"""

    def __init__(self, project: Optional[str] = None):
        from google.cloud import bigquery  # 延迟 import
        self._client = bigquery.Client(project=project)
        self._bq = bigquery

    def _params(self, **kw):
        types = {int: "INT64", float: "FLOAT64", str: "STRING"}
        return [self._bq.ScalarQueryParameter(k, types[type(v)], v)
                for k, v in kw.items()]

    def _cfg(self, params, dry_run: bool = False):
        return self._bq.QueryJobConfig(
            query_parameters=params, dry_run=dry_run, use_query_cache=not dry_run)

    def estimate_bytes(self, sql: str, params) -> int:
        """干跑：返回该查询将扫描的字节数（不真正执行、不计费）。"""
        job = self._client.query(sql, job_config=self._cfg(params, dry_run=True))
        return int(job.total_bytes_processed)

    @staticmethod
    def gb(n_bytes: int) -> float:
        return n_bytes / 1024 ** 3

    def neighbor_transfers(self, address: str, ts_lo: str, ts_hi: str,
                           limit: int = 1000, dry_run: bool = False):
        params = self._params(addr=address.lower(), ts_lo=ts_lo, ts_hi=ts_hi, limit=limit)
        if dry_run:
            return self.estimate_bytes(NEIGHBOR_TRANSFERS, params)
        rows = self._client.query(NEIGHBOR_TRANSFERS, job_config=self._cfg(params)).result()
        return [TransferEdge(
            frm=r["from_address"], to=r["to_address"],
            value=float(r["value"] or 0),
            timestamp=int(r["block_timestamp"].timestamp()) if r["block_timestamp"] else 0,
        ) for r in rows]

    def traces_for_transactions(self, tx_hashes: list[str], ts_lo: str, ts_hi: str,
                                dry_run: bool = False):
        """一次查询取多笔交易的 call 级 trace，按 tx_hash 分组。ts_* 为攻击发生的
        日期范围（必须，用于分区裁剪）。"""
        params = [
            self._bq.ScalarQueryParameter("ts_lo", "DATE", ts_lo),
            self._bq.ScalarQueryParameter("ts_hi", "DATE", ts_hi),
            self._bq.ArrayQueryParameter("hashes", "STRING",
                                         [h.lower() for h in tx_hashes]),
        ]
        if dry_run:
            return self.estimate_bytes(TRACES_FOR_TXS, params)
        rows = self._client.query(TRACES_FOR_TXS, job_config=self._cfg(params)).result()
        out: dict[str, list[dict]] = {}
        for r in rows:
            out.setdefault(r["transaction_hash"], []).append(dict(r))
        return out

    def _dates_hashes_params(self, dates: list[str], hashes: list[str]):
        return [
            self._bq.ArrayQueryParameter("dates", "DATE", sorted(set(dates))),
            self._bq.ArrayQueryParameter("hashes", "STRING",
                                         [h.lower() for h in hashes]),
        ]

    def prefetch_by_dates(self, hashes: list[str], dates: list[str],
                          dry_run: bool = False):
        """一组 (hash, 日期) 的 tx/logs/traces 一次性取完（3 次查询）。

        返回 (txs, logs, traces) 三个 dict，直接喂 BQPrefetchSource + build_context。
        dry_run=True 时返回三者扫描字节之和。
        """
        p = self._dates_hashes_params(dates, hashes)
        if dry_run:
            return (self.estimate_bytes(TXS_FOR_HASHES_DATES, p)
                    + self.estimate_bytes(LOGS_FOR_HASHES_DATES, p)
                    + self.estimate_bytes(TRACES_FOR_HASHES_DATES, p))
        txrows = self._client.query(TXS_FOR_HASHES_DATES, job_config=self._cfg(p)).result()
        txs = {}
        for r in txrows:
            txs[r["hash"]] = {
                "hash": r["hash"], "from": r["from_address"], "to": r["to_address"],
                "value": int(r["value"] or 0), "blockNumber": int(r["block_number"] or 0),
                "transactionIndex": int(r["transaction_index"] or 0),
                "input": r["input"] or "0x",
                "receipt_status": int(r["receipt_status"]) if r["receipt_status"] is not None else 1,
            }
        logs: dict[str, list[dict]] = {}
        for r in self._client.query(LOGS_FOR_HASHES_DATES, job_config=self._cfg(p)).result():
            logs.setdefault(r["transaction_hash"], []).append({
                "address": r["address"], "topics": list(r["topics"] or []),
                "data": r["data"] or "0x"})
        traces: dict[str, list[dict]] = {}
        for r in self._client.query(TRACES_FOR_HASHES_DATES, job_config=self._cfg(p)).result():
            traces.setdefault(r["transaction_hash"], []).append(dict(r))
        return txs, logs, traces

    def sample_normal_tx_hashes(self, ts_lo: str, ts_hi: str,
                                frac: float = 0.001, limit: int = 50_000,
                                dry_run: bool = False):
        params = self._params(ts_lo=ts_lo, ts_hi=ts_hi, frac=frac, limit=limit)
        if dry_run:
            return self.estimate_bytes(SAMPLE_NORMAL_TX, params)
        rows = self._client.query(SAMPLE_NORMAL_TX, job_config=self._cfg(params)).result()
        return [r["tx_hash"] for r in rows]

    def _hashes_param(self, ts_lo: str, ts_hi: str, hashes: list[str]):
        return [
            self._bq.ScalarQueryParameter("ts_lo", "DATE", ts_lo),
            self._bq.ScalarQueryParameter("ts_hi", "DATE", ts_hi),
            self._bq.ArrayQueryParameter("hashes", "STRING",
                                         [h.lower() for h in hashes]),
        ]

    def transactions_for(self, hashes: list[str], ts_lo: str, ts_hi: str,
                         dry_run: bool = False):
        """按 hash 取交易本体，返回 {hash: RPC 形态 tx dict（含 status）}。"""
        params = self._hashes_param(ts_lo, ts_hi, hashes)
        if dry_run:
            return self.estimate_bytes(TXS_FOR_HASHES, params)
        rows = self._client.query(TXS_FOR_HASHES, job_config=self._cfg(params)).result()
        out = {}
        for r in rows:
            out[r["hash"]] = {
                "hash": r["hash"], "from": r["from_address"], "to": r["to_address"],
                "value": int(r["value"] or 0), "blockNumber": int(r["block_number"] or 0),
                "transactionIndex": int(r["transaction_index"] or 0),
                "input": r["input"] or "0x",
                "receipt_status": int(r["receipt_status"]) if r["receipt_status"] is not None else 1,
            }
        return out

    def transfers_touching(self, addresses: list[str], dates: list[str],
                           limit: int = 2_000_000, dry_run: bool = False):
        """取一组地址在给定日期涉及的代币转移，返回 (from,to,value,ts) 行列表。"""
        params = [
            self._bq.ArrayQueryParameter("dates", "DATE", sorted(set(dates))),
            self._bq.ArrayQueryParameter("addrs", "STRING",
                                         [a.lower() for a in addresses]),
            self._bq.ScalarQueryParameter("limit", "INT64", limit),
        ]
        if dry_run:
            return self.estimate_bytes(TRANSFERS_TOUCHING, params)
        rows = self._client.query(TRANSFERS_TOUCHING, job_config=self._cfg(params)).result()
        return [{"from": r["from_address"], "to": r["to_address"],
                 "value": float(r["value"] or 0),
                 "ts": int(r["block_timestamp"].timestamp()) if r["block_timestamp"] else 0}
                for r in rows]

    def contract_meta(self, addresses: list[str], dry_run: bool = False):
        """合约元信息 {address: {created_ts, created_block, is_erc20, is_erc721,
        bytecode_len}}。供 L4 广义化使用——不泄漏的结构性信号。"""
        params = [self._bq.ArrayQueryParameter(
            "addrs", "STRING", sorted({a.lower() for a in addresses}))]
        if dry_run:
            return self.estimate_bytes(CONTRACT_META, params)
        rows = self._client.query(CONTRACT_META, job_config=self._cfg(params)).result()
        return {r["address"]: {"created_ts": int(r["created_ts"] or 0),
                               "created_block": int(r["created_block"] or 0),
                               "is_erc20": bool(r["is_erc20"]),
                               "is_erc721": bool(r["is_erc721"]),
                               "bytecode_len": int(r["bytecode_len"] or 0)}
                for r in rows}

    def dates_for_blocks(self, block_numbers: list[int], dry_run: bool = False):
        """区块号 -> 'YYYY-MM-DD'。用于补研究一种子缺失的精确日期。"""
        params = [self._bq.ArrayQueryParameter("blocks", "INT64",
                                               [int(b) for b in block_numbers])]
        if dry_run:
            return self.estimate_bytes(DATES_FOR_BLOCKS, params)
        rows = self._client.query(DATES_FOR_BLOCKS, job_config=self._cfg(params)).result()
        return {r["number"]: r["d"].strftime("%Y-%m-%d") for r in rows}

    def logs_for(self, hashes: list[str], ts_lo: str, ts_hi: str,
                 dry_run: bool = False):
        """按 hash 取事件日志，返回 {hash: [RPC 形态 log dict]}。"""
        params = self._hashes_param(ts_lo, ts_hi, hashes)
        if dry_run:
            return self.estimate_bytes(LOGS_FOR_HASHES, params)
        rows = self._client.query(LOGS_FOR_HASHES, job_config=self._cfg(params)).result()
        out: dict[str, list[dict]] = {}
        for r in rows:
            out.setdefault(r["transaction_hash"], []).append({
                "address": r["address"], "topics": list(r["topics"] or []),
                "data": r["data"] or "0x"})
        return out
