"""强三明治判定（按文献共识条件实现）。

文献共识（Züst 2021 / Qin et al. / SandWatch / GasTrace 等）：一次三明治攻击需满足
  ① 攻击者两笔交易 TA1、TA2 在**同一区块**且顺序为 TA1 → 受害 → TA2；
  ② 两笔在**同一流动性池**上 swap；
  ③ 两笔 swap **方向相反**（TA1 买入、TA2 卖出）；
  ④ TA2 的输入量 ≈ TA1 的输出量（金额链接）。

此前实现只用了"同区块 + 同 to 地址"（最弱条件），提升有限（sandwich +0.017）。
本模块解析同区块的 Swap 事件，判定池子与方向，输出强判定信号。

Swap 方向定义：
  UniV2 Swap(sender, a0In, a1In, a0Out, a1Out, to) → a0In>0 记为 +1（token0 入池），
      a1In>0 记为 -1。
  UniV3 Swap(sender, recipient, amount0, amount1, ...) → amount0 为 int256，
      >0 表示 token0 流入池（等价 UniV2 的 a0In>0）。
"""
from __future__ import annotations

UNIV2 = "0xd78ad95fa46c994b6551d0da85fc275fe613ce37657fb8d5e3d130840159d822"
UNIV3 = "0xc42079f94a6350d7e6235f29174924f928cc2ac818eb64fed8004e115fbcca67"


def _word(data: str, i: int) -> int:
    h = data[2:] if data.startswith("0x") else data
    chunk = h[i * 64:(i + 1) * 64]
    return int(chunk, 16) if chunk else 0


def _to_int256(u: int) -> int:
    return u - (1 << 256) if u >= (1 << 255) else u


def parse_swap(log: dict):
    """→ (pool, direction, amount_in, amount_out) 或 None。

    direction ∈ {+1,-1} 表示 token0 入池 / token1 入池。同时返回**入池量与出池量**——
    文献的金额链接条件是"TA2 输入 ≈ TA1 输出"（两者是同一种代币的两条腿），
    只比入池量会永远对不上（此前 full_pattern 恒为 0 的原因）。
    """
    t0 = (log.get("topics") or [None])[0]
    data = log.get("data", "0x")
    pool = (log.get("pool") or "").lower()
    if t0 == UNIV2:
        a0in, a1in = _word(data, 0), _word(data, 1)
        a0out, a1out = _word(data, 2), _word(data, 3)
        if a0in > 0:
            return pool, +1, a0in, a1out      # 投入 token0，取出 token1
        if a1in > 0:
            return pool, -1, a1in, a0out      # 投入 token1，取出 token0
        return None
    if t0 == UNIV3:
        a0, a1 = _to_int256(_word(data, 0)), _to_int256(_word(data, 1))
        if a0 > 0:
            return pool, +1, a0, max(0, -a1)
        if a0 < 0:
            return pool, -1, max(0, a1), -a0
        return None
    return None


def sandwich_signals(ctx, block_logs: list[dict]) -> dict:
    """对目标交易，在同区块 Swap 事件中检验强三明治条件。

    因数据集只含攻击者的**某一笔**（前置或后置），故从"本交易所在的池子"出发，
    检验同区块是否存在方向相反的配对 swap，以及是否形成 TA1→受害→TA2 结构。
    """
    my_hash = ctx.tx_hash.lower()
    parsed = []
    for lg in block_logs:
        p = parse_swap(lg)
        if p:
            parsed.append((lg["tx_hash"].lower(), lg["log_index"], *p))
    if not parsed:
        return {}
    mine = [x for x in parsed if x[0] == my_hash]
    if not mine:
        return {"sw_in_swap_block": 1.0}

    others = [x for x in parsed if x[0] != my_hash]
    opposite_same_pool = 0     # 同池反向的他方 swap 数
    amount_linked = 0.0        # 金额链接（TA2 输入 ≈ TA1 输出，同一代币的两条腿）
    victim_between = 0         # 本交易与配对交易之间是否夹着第三方 swap（受害者）
    my_idx = min(li for _, li, *_ in mine)

    for _, _, pool, d, a_in, a_out in mine:
        for oh, oli, opool, od, o_in, o_out in others:
            if opool != pool or od == d:
                continue
            opposite_same_pool += 1
            # 两种配对次序都验：我方为 TA1（我的 out ≈ 对方 in）或我方为 TA2（对方 out ≈ 我的 in）
            for x, y in ((a_out, o_in), (o_out, a_in)):
                if x > 0 and y > 0:
                    amount_linked = max(amount_linked, min(x, y) / max(x, y))
            lo, hi = min(my_idx, oli), max(my_idx, oli)
            if any(lo < li < hi and h not in (my_hash, oh)
                   for h, li, *_ in parsed):
                victim_between = 1

    return {
        "sw_in_swap_block": 1.0,
        "sw_opposite_same_pool": float(min(1.0, opposite_same_pool / 2.0)),
        "sw_amount_linked": float(amount_linked),
        "sw_victim_between": float(victim_between),
        # 完整三明治：同池反向 + 中间夹着第三方 + 金额链接
        "sw_full_pattern": 1.0 if (opposite_same_pool and victim_between
                                   and amount_linked >= 0.5) else 0.0,
    }
