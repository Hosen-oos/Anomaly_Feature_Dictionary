"""原始 RPC 响应 → mlusd 数据结构的解码（设计架构 §4 M1）。

不依赖 web3/eth_abi：核心事件（Transfer/Approval/Swap/Deposit/Withdrawal）的
topic0 是常量，indexed 参数在 topics、非 indexed 在 data，按 32 字节字解码即可。
未识别事件按原始 topic0 短名保留，供 v1 阶段接入完整 ABI 解码（Etherscan/4byte）。
"""
from __future__ import annotations

from typing import Any, Optional

from mlusd.types import Call, DecodedLog, TraceSummary

# 事件签名 keccak256(topic0) -> (标准事件名, [indexed参数名], [非indexed参数名])
# 全部 topic0 用 keccak256(签名) 校验过（2026-07 修正 Deposit/UniV3 Swap 手写错误）。
# indexed 参数从 topics[1:]，非indexed 从 data 按 32 字节字。协议事件（Borrow/Repay/
# FlashLoan/Mint/Burn/Liquidation）暂只识别"存在"，args 留空（经济规则多数只需 presence）。
EVENT_SIGNATURES: dict[str, tuple[str, list[str], list[str]]] = {
    "0xddf252ad1be2c89b69c2b068fc378daa952ba7f163c4a11628f55a4df523b3ef":
        ("Transfer", ["from", "to"], ["value"]),
    "0x8c5be1e5ebec7d5bd14f71427d1e84f3dd0314c0f7b2291e5b200ac8c7c3b925":
        ("Approval", ["owner", "spender"], ["value"]),
    # Uniswap V2 Swap(sender, a0In, a1In, a0Out, a1Out, to)
    "0xd78ad95fa46c994b6551d0da85fc275fe613ce37657fb8d5e3d130840159d822":
        ("Swap", ["sender", "to"],
         ["amount0In", "amount1In", "amount0Out", "amount1Out"]),
    # Uniswap V3 Swap（修正：之前 topic0 被截断为 63 位从不匹配）
    "0xc42079f94a6350d7e6235f29174924f928cc2ac818eb64fed8004e115fbcca67":
        ("Swap", ["sender", "to"],
         ["amount0", "amount1", "sqrtPriceX96", "liquidity", "tick"]),
    # WETH Deposit / Withdrawal（Deposit 修正手写错误）
    "0xe1fffcc4923d04b559f4d29a8bfc6cda04eb5b0d3c460751c2402c5c5cc9109c":
        ("Deposit", ["dst"], ["value"]),
    "0x7fcf532c15f0a6db0bd6d0e038bea71d30d808c7d98cb3bf7268a95bf5081b65":
        ("Withdrawal", ["src"], ["value"]),
    # Aave 借贷 / 闪电贷（V2 + V3），presence-only
    "0xc6a898309e823ee50bac64e45ca8adba6690e99e7841c45d754e2a38e9019d9b":
        ("Borrow", [], []),                      # Aave V2 Borrow
    "0x42904e9dde19c6f4cc4010f05689d8d7072b468b0b341794f1e27372bace1a2c":
        ("Borrow", [], []),                      # Aave V3 Borrow
    "0x4cdde6e09bb755c9a5589ebaec640bbfedff1362d4b255ebf8339782b9942faa":
        ("Repay", [], []),                       # Aave V2 Repay
    "0xa534c8dbe71f871f9f3530e97a74601fea17b426cae02e1c5aee42c96c784051":
        ("Repay", [], []),                       # Aave V3 Repay
    "0x631042c832b07452973831137f2d73e395028b44b250dedc5abb0ee766e168ac":
        ("FlashLoan", [], []),                   # Aave V2 FlashLoan
    "0xefefaba5e921573100900a3ad9cf29f222d995fb3b6045797eaea7521bd8d6f0":
        ("FlashLoan", [], []),                   # Aave V3 FlashLoan
    "0xe413a321e8681d831f4dbccbca790d2952b56f977908e45be37335533e005286":
        ("LiquidationCall", [], []),             # Aave LiquidationCall
    # 流动性增减（UniV2 Mint/Burn, UniV3 Mint/Burn）
    "0x4c209b5fc8ad50758f13e2e1088ba56a560dff690a1c6fef26394f4c03821c4f":
        ("Mint", [], []),                        # UniV2 Mint (add liquidity)
    "0x7a53080ba414158be7ec69b987b5fb7d07dee101fe85488f0853ae16239d0bde":
        ("Mint", [], []),                        # UniV3 Mint
    "0xdccd412f0b1252819cb1fd330b93224ca42612892bb3f4f789976e6d81936496":
        ("Burn", [], []),                        # UniV2 Burn (remove liquidity)
    "0x0c396cd989a39f4459b5fa1aed6a9a8dcdbc45908acfd67e028cd568da98982c":
        ("Burn", [], []),                        # UniV3 Burn
}

# topic0 前 8 位（不含 0x）-> 标准事件名。用于从已缓存 context 里的
# Unknown_<前8位> 事件免重取恢复语义（decode 修好前采的数据也能用）。
TOPIC0_PREFIX_EVENT: dict[str, str] = {
    k[2:10]: v[0] for k, v in EVENT_SIGNATURES.items()
}


def _word(data_hex: str, i: int) -> int:
    """取 data 的第 i 个 32 字节字为 uint256。"""
    h = data_hex[2:] if data_hex.startswith("0x") else data_hex
    chunk = h[i * 64:(i + 1) * 64]
    return int(chunk, 16) if chunk else 0


def _addr_from_topic(topic: str) -> str:
    """32 字节 topic 的低 20 字节为地址。"""
    return "0x" + topic[-40:].lower()


def decode_log(raw: dict) -> Optional[DecodedLog]:
    """解码一条 RPC receipt.log。未知事件返回 event='Unknown_<前8位>'。"""
    topics = raw.get("topics") or []
    if not topics:
        return None
    topic0 = topics[0].lower()
    address = (raw.get("address") or "").lower()
    data = raw.get("data") or "0x"
    sig = EVENT_SIGNATURES.get(topic0)
    if sig is None:
        return DecodedLog(address=address, event=f"Unknown_{topic0[2:10]}", args={})
    event, indexed_names, data_names = sig
    args: dict[str, Any] = {}
    for name, topic in zip(indexed_names, topics[1:]):
        args[name] = _addr_from_topic(topic)
    for i, name in enumerate(data_names):
        args[name] = _word(data, i)
    # 归一 Swap：把 in/out 折叠成便于 L2 使用的 value（取较大的转移量）
    if event == "Swap" and "amount0In" in args:
        args["value"] = max(args.get("amount0In", 0), args.get("amount1In", 0),
                            args.get("amount0Out", 0), args.get("amount1Out", 0))
    return DecodedLog(address=address, event=event, args=args)


def decode_logs(raw_logs: list[dict]) -> list[DecodedLog]:
    out = []
    for r in raw_logs or []:
        d = decode_log(r)
        if d is not None:
            out.append(d)
    return out


def _flatten_calltracer(node: dict, depth: int, out: list[Call]) -> None:
    """debug_traceTransaction(callTracer) 的嵌套树 → 展平的 Call 列表。"""
    out.append(Call(
        frm=(node.get("from") or "").lower(),
        to=(node.get("to") or "").lower(),
        kind=node.get("type", "CALL"),
        depth=depth,
        value=int(node.get("value", "0x0"), 16) if isinstance(node.get("value"), str) else int(node.get("value", 0) or 0),
        func=(node.get("input", "0x")[:10] if node.get("input") else None),
        reverted=bool(node.get("error")),
    ))
    for child in node.get("calls", []) or []:
        _flatten_calltracer(child, depth + 1, out)


def build_trace_summary_from_bq(rows: list[dict]) -> Optional[TraceSummary]:
    """由 BigQuery traces 行（call 级）构造 TraceSummary（免费档的 trace 主路径）。

    每行含 from_address/to_address/value/call_type/trace_address/status。
    depth = len(trace_address) + 1（顶层 trace_address 为空）。BigQuery traces
    **不含 SSTORE/opcode**，故 sstore_count=0——L3-j3 的重入/回滚规则仍可用，
    仅 sstore_spike 子规则失效（贡献 0，优雅降级）。行应按 trace_address 排序。
    """
    if not rows:
        return None

    def _depth(ta) -> int:
        if ta is None or ta == "":
            return 1
        if isinstance(ta, (list, tuple)):
            return len(ta) + 1
        return len(str(ta).split(",")) + 1 if str(ta) else 1

    calls = [Call(
        frm=(r.get("from_address") or "").lower(),
        to=(r.get("to_address") or "").lower(),
        kind=(r.get("call_type") or "call").upper(),
        depth=_depth(r.get("trace_address")),
        value=int(float(r.get("value") or 0)),
        reverted=(str(r.get("status", 1)) == "0"),
    ) for r in rows]
    max_depth = max((c.depth for c in calls), default=0)
    reverted = sum(1 for c in calls if c.reverted)
    return TraceSummary(calls=calls, max_depth=max_depth, sstore_count=0,
                        reverted_subcalls=reverted, opcodes=None, storage_writes={})


def build_trace_summary(call_trace: Optional[dict],
                        struct_logs: Optional[list[dict]] = None) -> Optional[TraceSummary]:
    """由 callTracer 结果（+ 可选 structLog）构造 TraceSummary。"""
    if call_trace is None:
        return None
    calls: list[Call] = []
    _flatten_calltracer(call_trace, 1, calls)
    max_depth = max((c.depth for c in calls), default=0)
    reverted = sum(1 for c in calls if c.reverted)
    # SSTORE 次数：优先用 structLog（opcode 级），否则留 0（callTracer 不含 opcode）
    sstore = 0
    opcodes = None
    storage_writes: dict[str, int] = {}
    if struct_logs:
        opcodes = [s.get("op", "") for s in struct_logs]
        sstore = sum(1 for op in opcodes if op == "SSTORE")
    return TraceSummary(
        calls=calls, max_depth=max_depth, sstore_count=sstore,
        reverted_subcalls=reverted, opcodes=opcodes, storage_writes=storage_writes)
