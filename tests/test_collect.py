"""M1 采集器测试：原始 RPC 响应 → TxContext，并验证与 M2-M6 的贯通。"""
from __future__ import annotations

from mlusd.collect.context import build_context
from mlusd.collect.decode import build_trace_summary, decode_logs
from mlusd.collect.labels import LabelStore
from mlusd.types import TxContext
from tests import raw_fixtures as rf


def test_decode_logs_core_events():
    p = rf.phishing_raw()
    logs = decode_logs(p["receipt"]["logs"])
    events = [l.event for l in logs]
    assert events == ["Approval", "Transfer"]
    approval = logs[0]
    assert approval.args["spender"].endswith("deadbeef")
    assert approval.args["value"] == (1 << 256) - 1     # 无限授权正确解码


def test_build_trace_summary_flatten():
    trace = {"type": "CALL", "from": "0xa", "to": "0xb", "value": "0x0",
             "calls": [{"type": "DELEGATECALL", "from": "0xb", "to": "0xc",
                        "calls": [{"type": "CALL", "from": "0xc", "to": "0xd"}]}]}
    ts = build_trace_summary(trace)
    assert ts is not None
    assert len(ts.calls) == 3
    assert ts.max_depth == 3
    assert ts.calls[1].kind == "DELEGATECALL" and ts.calls[1].depth == 2


def test_build_context_produces_valid_txcontext():
    ctx = build_context(rf.PHISHING_HASH, rf.mock_source())
    assert isinstance(ctx, TxContext)
    assert ctx.from_address.endswith("cafebabe")
    assert ctx.event_logs and len(ctx.event_logs) == 2
    m = ctx.availability
    assert m[0] == 1 and m[1] == 1        # L1 恒可用；L2 有日志

def test_build_context_missing_tx_returns_none():
    assert build_context("0xdoesnotexist", rf.mock_source()) is None


def test_bq_trace_fallback_fills_l3():
    """免费档无 debug 时，BigQuery call 级 trace 兜底填 L3（reentrancy/revert 可用）。"""
    # 模拟 BigQuery traces 行：顶层 call + 两层子调用，其中一层 revert
    bq_rows = [
        {"from_address": "0xa", "to_address": "0xb", "value": 0,
         "call_type": "call", "trace_address": [], "status": 1},
        {"from_address": "0xb", "to_address": "0xc", "value": 0,
         "call_type": "delegatecall", "trace_address": [0], "status": 1},
        {"from_address": "0xc", "to_address": "0xb", "value": 0,
         "call_type": "call", "trace_address": [0, 0], "status": 0},
    ]
    ctx = build_context(rf.NORMAL_HASH, rf.mock_source(source_no_trace=True),
                        bq_trace_rows=bq_rows)
    assert ctx.trace is not None
    assert len(ctx.trace.calls) == 3
    assert ctx.trace.max_depth == 3
    assert ctx.trace.reverted_subcalls == 1
    assert ctx.availability[2] == 1        # L3 因 BQ trace 兜底而可用


def test_bq_prefetch_source_builds_context():
    """纯 BigQuery 路径：BQPrefetchSource + build_context 不调 RPC 组装上下文。"""
    from mlusd.collect.sources import BQPrefetchSource
    # 模拟 BigQuery transactions_for / logs_for 的输出（Python int 值、RPC 形态日志）
    h = "0xbq00000000000000000000000000000000000000000000000000000000000001"
    txs = {h: {"hash": h, "from": rf.VICTIM, "to": rf.TOKEN, "value": 0,
               "blockNumber": 18500000, "transactionIndex": 3, "input": "0xa9059cbb",
               "receipt_status": 1}}
    logs = {h: [{"address": rf.TOKEN,
                 "topics": [rf.TRANSFER_SIG, "0x"+"0"*24+rf.VICTIM[2:],
                            "0x"+"0"*24+rf.ATTACKER[2:]],
                 "data": "0x" + f"{10**18:064x}"}]}
    src = BQPrefetchSource(txs, logs)
    ctx = build_context(h, src)
    assert ctx is not None
    assert ctx.event_logs and ctx.event_logs[0].event == "Transfer"
    assert ctx.status is True
    assert ctx.availability[1] == 1        # L2 有日志


def test_labels_lookup_and_availability():
    store = LabelStore()
    store.add_label(rf.ATTACKER, "Fake_Phishing", "Etherscan")
    ctx = build_context(rf.PHISHING_HASH, rf.mock_source(), labels=store)
    assert ctx.offchain is not None
    assert ctx.availability[3] == 1        # L4 因命中标签而可用
    assert ctx.offchain.label_hits[0]["severity"] >= 0.85


def test_collected_context_flows_through_detector():
    """M1 输出直接喂 Detector：真实数据路径端到端贯通（无需改 M2-M6）。"""
    from pathlib import Path

    from mlusd.match.dictionary import load_dictionaries
    from mlusd.pipeline import Detector
    from mlusd.signals.factory import default_extractors
    from tests import synthetic

    det = Detector(default_extractors(),
                   load_dictionaries(Path(__file__).resolve().parents[1]
                                     / "configs" / "dictionaries"),
                   alpha=0.01, min_group_size=50)
    det.fit(synthetic.calibration_set(800))

    store = LabelStore()
    store.add_label(rf.ATTACKER, "Fake_Phishing", "Etherscan")
    ctx = build_context(rf.PHISHING_HASH, rf.mock_source(), labels=store)
    report = det.detect(ctx)
    assert report.tx_hash == rf.PHISHING_HASH
    assert report.to_dict()["verdict"] in ("KNOWN", "UNKNOWN", "INSUFFICIENT", "NORMAL")
