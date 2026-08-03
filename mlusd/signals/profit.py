"""利润归因（DeFort 式，针对 flash_loan vs price_manipulation 的真语义重叠）。

从已解码的 Transfer 事件重建 (地址 × 代币) 的资金流台账，导出**与代币小数位无关**的
比值型特征——不同 ERC20 的 decimals 不同，原始额度不可跨代币比较，故一律用比值。

核心判别直觉：
- **闪电贷提供方**：借出巨额又被归还 → gross 巨大但 net≈0（"高流转、零净额"签名）
- **被操控/抽干的池子**：net 单向大幅为负 → |net|/gross ≈ 1（纯流出）
两者形态正交，故 passthrough_max 与 drain_imbalance 可分离这两类攻击。

仅使用已有 event_logs，不需要重新采数。
"""
from __future__ import annotations

from collections import defaultdict

EPS = 1e-9

# 价值锚定代币（主网规范地址，小写）：用它们把利润折算成可比的"美元量级"，
# 无需外部价格源——WETH 按近似 ETH 价、稳定币按 1:1。这是 DeFort 式利润口径的
# 轻量实现：多数攻击的获利腿本就是 WETH 或稳定币。
ANCHORS: dict[str, tuple[int, float]] = {
    # 地址: (decimals, 单位美元价值)
    "0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2": (18, 2000.0),  # WETH
    "0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48": (6, 1.0),      # USDC
    "0xdac17f958d2ee523a2206206994597c13d831ec7": (6, 1.0),      # USDT
    "0x6b175474e89094c44da98b954eedeac495271d0f": (18, 1.0),     # DAI
    "0x2260fac5e5542a773aa44fbcfedf7c193bc2c599": (8, 60000.0),  # WBTC
    "0x4fabb145d64652a948d72533023f6e7a623c7c53": (18, 1.0),     # BUSD
    "0x0000000000085d4780b73119b644ae5ecd22b376": (18, 1.0),     # TUSD
}


def anchor_usd(token: str, raw_amount: float) -> float | None:
    """把锚定代币的原始额度折算为美元量级；非锚定代币返回 None。
    单位价值取粗略常数——目的是**量级可比**（log 后进 ECDF），非精确估值。"""
    info = ANCHORS.get((token or "").lower())
    if not info:
        return None
    dec, px = info
    return raw_amount / (10 ** dec) * px


def build_ledger(ctx) -> dict:
    """(地址, 代币) -> {'in': 流入, 'out': 流出}。代币用发出 Transfer 的合约地址标识。"""
    led: dict[tuple[str, str], dict[str, float]] = defaultdict(
        lambda: {"in": 0.0, "out": 0.0})
    for log in ctx.event_logs or []:
        if log.event != "Transfer":
            continue
        v = float(log.args.get("value", 0) or 0)
        if v <= 0:
            continue
        token = log.address
        frm = str(log.args.get("from", ""))
        to = str(log.args.get("to", ""))
        if frm:
            led[(frm, token)]["out"] += v
        if to:
            led[(to, token)]["in"] += v
    return led


def attribution_params(ctx) -> dict[str, float]:
    """利润归因特征（全部为 [0,1] 比值或计数，跨代币可比）。"""
    led = build_ledger(ctx)
    if not led:
        return {}
    r: dict[str, float] = {}

    # 每个 (地址,代币) 的净额与流转量
    recs = []
    for (addr, token), f in led.items():
        gross = f["in"] + f["out"]
        net = f["in"] - f["out"]
        if gross > 0:
            recs.append((addr, token, net, gross, f["in"], f["out"]))

    # 1) 闪电贷签名：存在地址在某代币上"大进大出、净额≈0"
    #    passthrough = min(in,out)/max(in,out) → 1 表示完美对冲式流转
    #    只在流转量占本交易该代币总量较大时才算，避免小额噪声
    tok_gross: dict[str, float] = defaultdict(float)
    for _, token, _, gross, _, _ in recs:
        tok_gross[token] += gross
    passthrough = 0.0
    for _, token, _, gross, i, o in recs:
        if tok_gross[token] <= 0 or gross / tok_gross[token] < 0.2:
            continue
        pt = min(i, o) / (max(i, o) + EPS)
        passthrough = max(passthrough, pt)
    r["passthrough_max"] = passthrough

    # 2) 抽干签名：最大净损失方的单向程度 |net|/gross（→1 表示纯流出，池子被抽干）
    losers = [(abs(net), gross) for _, _, net, gross, _, _ in recs if net < 0]
    if losers:
        loss, g = max(losers, key=lambda x: x[0])
        r["drain_imbalance"] = float(loss / (g + EPS))
    else:
        r["drain_imbalance"] = 0.0

    # 3) 利润集中度：最大赢家占全部净收益的比例（单一获利方 vs 分散）
    gains = [net for _, _, net, _, _, _ in recs if net > 0]
    if gains:
        r["profit_concentration"] = float(max(gains) / (sum(gains) + EPS))
    else:
        r["profit_concentration"] = 0.0

    # 4) 损失集中度：最大输家占全部净损失的比例（单一受害池 → 价格操控典型）
    losses = [abs(net) for _, _, net, _, _, _ in recs if net < 0]
    if losses:
        r["loss_concentration"] = float(max(losses) / (sum(losses) + EPS))
    else:
        r["loss_concentration"] = 0.0

    # 5) 发起方是否为最大赢家（自利型攻击 vs 通过合约获利）
    sender = ctx.from_address
    sender_gain = sum(net for a, _, net, _, _, _ in recs if a == sender and net > 0)
    r["sender_is_winner"] = 1.0 if (gains and sender_gain >= max(gains) - EPS
                                    and sender_gain > 0) else 0.0

    # 6) 参与代币种数（价格操控多为多代币兑换回路）——用 log 压缩后归一
    n_tok = len(tok_gross)
    r["n_tokens_norm"] = min(1.0, n_tok / 10.0)

    # 7) 回环：发起方在同一代币上既收又付（套利回路）
    roundtrip = 0.0
    for a, _, _, _, i, o in recs:
        if a == sender and i > 0 and o > 0:
            roundtrip = 1.0
            break
    r["sender_roundtrip"] = roundtrip

    # 8) 受害者身份：最大净损失方是否为 AMM 池（本交易内发出过 Swap/Sync 事件）
    #    价格操控 → 抽干的是池子；闪电贷攻击 → 受害者常是借贷/逻辑漏洞协议而非池子。
    #    这是资金流拓扑之外的语义判别（两类都用闪电贷，故 passthrough 无法分离）。
    pools = {l.address for l in (ctx.event_logs or [])
             if l.event in ("Swap", "Sync", "Mint", "Burn")}
    if pools and recs:
        loser = min(recs, key=lambda x: x[2])          # net 最小 = 最大净损失
        winner = max(recs, key=lambda x: x[2])
        r["victim_is_pool"] = 1.0 if loser[0] in pools else 0.0
        r["winner_is_pool"] = 1.0 if winner[0] in pools else 0.0
        # 净损失中来自池子地址的占比（价格操控应接近 1）
        tot_loss = sum(abs(n) for _, _, n, _, _, _ in recs if n < 0)
        pool_loss = sum(abs(n) for a, _, n, _, _, _ in recs if n < 0 and a in pools)
        r["pool_loss_share"] = float(pool_loss / (tot_loss + EPS)) if tot_loss > 0 else 0.0

    # 9) 锚定美元利润（DeFort 式利润口径的轻量实现）
    #    仅统计锚定代币（WETH/稳定币/WBTC）的净额——攻击的获利腿多为这些资产；
    #    非锚定代币无价格无法折算，跳过而非猜测。log 压缩后交由 ECDF 校准。
    import math
    best_gain = 0.0
    sender_gain = 0.0
    sender = ctx.from_address
    for addr, token, net, _, _, _ in recs:
        if net <= 0:
            continue
        usd = anchor_usd(token, net)
        if usd is None:
            continue
        best_gain = max(best_gain, usd)
        if addr == sender:
            sender_gain = max(sender_gain, usd)
    if best_gain > 0:
        r["usd_profit_mag"] = float(math.log1p(best_gain))
        r["usd_profit_sender"] = float(math.log1p(sender_gain))
        # 大额获利标记（≥10 万美元量级，攻击典型）
        r["large_usd_profit"] = 1.0 if best_gain >= 1e5 else 0.0
    return r
