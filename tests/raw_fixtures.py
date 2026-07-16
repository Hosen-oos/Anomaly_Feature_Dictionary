"""原始 RPC 形态的测试夹具（模拟 Alchemy/QuickNode 返回），喂给 MockSource。

用于验证 M1 build_context 能把真实结构的原始响应组装成 TxContext，且与
tests/synthetic.py 同构、可直接过 Detector。金额为 32 字节十六进制字。
"""
from __future__ import annotations

TRANSFER_SIG = "0xddf252ad1be2c89b69c2b068fc378daa952ba7f163c4a11628f55a4df523b3ef"
APPROVAL_SIG = "0x8c5be1e5ebec7d5bd14f71427d1e84f3dd0314c0f7b2291e5b200ac8c7c3b925"
SWAP_V2_SIG = "0xd78ad95fa46c994b6551d0da85fc275fe613ce37657fb8d5e3d130840159d822"

ATTACKER = "0x00000000000000000000000000000000deadbeef"
VICTIM = "0x00000000000000000000000000000000cafebabe"
TOKEN = "0x000000000000000000000000000000000000t0ke".replace("t0ke", "1234")
DEX = "0x00000000000000000000000000000000000000dex".replace("dex", "999")


def _topic_addr(addr: str) -> str:
    return "0x" + "0" * 24 + addr[2:]


def _word(n: int) -> str:
    return f"{n:064x}"


PHISHING_HASH = "0xphish0000000000000000000000000000000000000000000000000000000001"
NORMAL_HASH = "0xnorm00000000000000000000000000000000000000000000000000000000001"


def phishing_raw() -> dict:
    """ice-phishing：无限 approve + 大额 transferFrom 到 spender。"""
    unlimited = (1 << 256) - 1
    tx = {
        "hash": PHISHING_HASH, "from": VICTIM, "to": TOKEN,
        "blockNumber": "0x11a2b3c", "transactionIndex": "0x2a", "value": "0x0",
        "input": "0x095ea7b3",
    }
    receipt = {
        "status": "0x1",
        "logs": [
            {"address": TOKEN, "topics": [APPROVAL_SIG, _topic_addr(VICTIM),
                                          _topic_addr(ATTACKER)],
             "data": "0x" + _word(unlimited)},
            {"address": TOKEN, "topics": [TRANSFER_SIG, _topic_addr(VICTIM),
                                          _topic_addr(ATTACKER)],
             "data": "0x" + _word(50000 * 10 ** 18)},
        ],
    }
    trace = {"type": "CALL", "from": VICTIM, "to": TOKEN, "value": "0x0",
             "input": "0x095ea7b3", "calls": []}
    return {"tx": tx, "receipt": receipt, "trace": trace}


def normal_raw() -> dict:
    tx = {"hash": NORMAL_HASH, "from": VICTIM, "to": TOKEN,
          "blockNumber": "0x11a2b00", "transactionIndex": "0x5", "value": "0x0",
          "input": "0xa9059cbb"}
    receipt = {"status": "0x1", "logs": [
        {"address": TOKEN, "topics": [TRANSFER_SIG, _topic_addr(VICTIM),
                                      _topic_addr(ATTACKER)],
         "data": "0x" + _word(3 * 10 ** 17)}]}
    trace = {"type": "CALL", "from": VICTIM, "to": TOKEN, "value": "0x0",
             "input": "0xa9059cbb", "calls": []}
    return {"tx": tx, "receipt": receipt, "trace": trace}


def mock_source(source_no_trace: bool = False):
    """source_no_trace=True 模拟免费档（无 debug_traceTransaction）。"""
    from mlusd.collect.sources import MockSource
    p, n = phishing_raw(), normal_raw()
    traces = None if source_no_trace else {PHISHING_HASH: p["trace"],
                                           NORMAL_HASH: n["trace"]}
    return MockSource(
        txs={PHISHING_HASH: p["tx"], NORMAL_HASH: n["tx"]},
        receipts={PHISHING_HASH: p["receipt"], NORMAL_HASH: n["receipt"]},
        traces=traces,
    )
