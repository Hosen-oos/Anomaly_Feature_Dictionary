"""L2 合约交互语义层：语义提升器 + 三个 v0 提取器（设计架构 §4 M2、§5.3）。

语义提升器把解码事件提升为 DeFi 动作序列——ActLifter（CCS 2023）"事件+资产
转移模式"路线的简化实现，v1 阶段按其论文规则补全 10 类动作的完整识别逻辑。

L2-j1 分布偏离：动作序列 bigram 稀有度（正常集统计，add-one 平滑）。
L2-j2 经济异常：借还闭环/闪电贷/连环 swap/流动性撤出/无本获利 规则分
        （利润口径参考 DeFort，价格操控信号参考 DeFiRanger/SMARTCAT）。
L2-j3 信息差异：授权-转账不一致（ice phishing 模式，参考 NDSS 2025 payload 钓鱼）。
"""
from __future__ import annotations

import math
from collections import Counter, defaultdict
from typing import Optional

from mlusd.signals.base import SignalExtractor
from mlusd.types import DeFiAction, TxContext

UNLIMITED_APPROVE = 2 ** 255   # 金额超过此值视为无限授权

# 事件名 -> DeFi 动作类型（ActLifter 式映射的 v0 子集）
_EVENT_ACTION = {
    "Transfer": "transfer",
    "Approval": "approve",
    "Swap": "swap",
    "Borrow": "borrow",
    "Repay": "repay",
    "FlashLoan": "flashloan",
    "Mint": "add_liquidity",
    "Burn": "remove_liquidity",
    "Deposit": "deposit",
    "Withdrawal": "withdraw",
    "Withdraw": "withdraw",
    "LiquidationCall": "liquidate",
}


def _event_name(log) -> str:
    """标准事件名。已缓存 context 里未识别事件存为 Unknown_<topic0前8位>，
    用 TOPIC0_PREFIX_EVENT 反查恢复（免重取，见 decode.py 说明）。"""
    if log.event.startswith("Unknown_"):
        from mlusd.collect.decode import TOPIC0_PREFIX_EVENT
        return TOPIC0_PREFIX_EVENT.get(log.event[8:], log.event)
    return log.event


def lift_actions(ctx: TxContext) -> list[DeFiAction]:
    """语义提升：解码事件 -> 按发生顺序的 DeFi 动作序列。"""
    if not ctx.event_logs:
        return []
    actions = []
    for log in ctx.event_logs:
        kind = _EVENT_ACTION.get(_event_name(log))
        if kind is None:
            continue
        a = log.args
        actions.append(DeFiAction(
            kind=kind,
            actor=str(a.get("from", a.get("owner", a.get("sender", "")))),
            protocol=log.address,
            token_in=str(a.get("token", log.address)),
            token_out=str(a.get("token_out", "")),
            amount_in=float(a.get("value", a.get("amount", a.get("amount0", 0))) or 0),
            amount_out=float(a.get("amount_out", a.get("amount1", 0)) or 0),
        ))
    return actions


class ActionSequenceRarity(SignalExtractor):
    """L2-j1：动作序列 bigram 稀有度，score = 平均 -log2 P(a_i | a_{i-1})。"""

    layer, angle, name = 2, 1, "l2j1_action_ngram"

    def __init__(self):
        self._bigram: Counter = Counter()
        self._unigram: Counter = Counter()
        self._vocab: int = 1

    @staticmethod
    def _seq(ctx: TxContext) -> list[str]:
        return ["<s>"] + [a.kind for a in lift_actions(ctx)] + ["</s>"]

    def fit(self, normal_contexts: list[TxContext]) -> None:
        for ctx in normal_contexts:
            seq = self._seq(ctx)
            if len(seq) <= 2:
                continue
            self._unigram.update(seq[:-1])
            self._bigram.update(zip(seq[:-1], seq[1:]))
        self._vocab = max(len(self._unigram), 1)

    def score(self, ctx: TxContext) -> Optional[float]:
        seq = self._seq(ctx)
        if len(seq) <= 2:       # 无可提升动作
            return None
        nll = 0.0
        for prev, cur in zip(seq[:-1], seq[1:]):
            p = (self._bigram[(prev, cur)] + 1) / (self._unigram[prev] + self._vocab)
            nll += -math.log2(p)
        return nll / (len(seq) - 1)

    def evidence(self, ctx: TxContext) -> str:
        kinds = [a.kind for a in lift_actions(ctx)]
        return "动作序列: " + " → ".join(kinds[:10]) + ("…" if len(kinds) > 10 else "")


class EconomicAnomalyScore(SignalExtractor):
    """L2-j2：协议操作经济异常规则分。"""

    layer, angle, name = 2, 2, "l2j2_economic_rules"

    def params(self, ctx: TxContext) -> dict[str, float]:
        actions = lift_actions(ctx)
        if not actions:
            return {}
        kinds = [a.kind for a in actions]
        n = len(actions)
        r: dict[str, float] = {}
        # 闪电贷（单交易内借入即归还）
        r["flashloan"] = 1.0 if "flashloan" in kinds else 0.0
        # 借-还闭环：同一交易同时出现 borrow 与 repay
        r["borrow_repay"] = 1.0 if ("borrow" in kinds and "repay" in kinds) else 0.0
        # 连环 swap：单交易 >=3 次兑换（价格操控/套利回路的典型形态）
        r["swap_chain"] = min(1.0, max(0, kinds.count("swap") - 1) / 3.0)
        # 流动性撤出占比
        r["liquidity_exit"] = min(1.0, kinds.count("remove_liquidity") / max(n * 0.3, 1))
        # 发起方净流入（DeFort 利润口径）+ 幅度参数（量值化，从已解码 Transfer 计算，无需重取）
        sender = ctx.from_address
        net = 0.0
        amts = []
        for log in ctx.event_logs or []:
            if log.event != "Transfer":
                continue
            v = float(log.args.get("value", 0) or 0)
            amts.append(v)
            if str(log.args.get("to", "")) == sender:
                net += v
            if str(log.args.get("from", "")) == sender:
                net -= v
        r["free_profit"] = 1.0 if (net > 0 and ctx.value == 0 and n >= 3) else 0.0
        # 量值化参数：利润幅度、单笔最大转移、总交易量（log，ECDF 处理量纲）。use_magnitude 供消融
        if getattr(self, "use_magnitude", True):
            r["net_profit_mag"] = math.log1p(net) if net > 0 else 0.0
            if amts:
                r["max_transfer_mag"] = math.log1p(max(amts))
                r["total_volume_mag"] = math.log1p(sum(amts))
        return r

    def score(self, ctx: TxContext) -> Optional[float]:
        r = self.params(ctx)
        if not r:
            return None
        w = {"flashloan": 0.30, "borrow_repay": 0.20, "swap_chain": 0.20,
             "liquidity_exit": 0.15, "free_profit": 0.15}
        return float(sum(w[k] * v for k, v in r.items() if k in w))

    def evidence(self, ctx: TxContext) -> str:
        r = self.params(ctx)
        zh = {"flashloan": "闪电贷", "borrow_repay": "借还闭环",
              "swap_chain": "连环swap", "liquidity_exit": "流动性撤出",
              "free_profit": "零成本净获利"}
        hits = [zh[k] for k, v in sorted(r.items(), key=lambda x: -x[1])
                if k in zh and v >= 0.5]
        return "经济模式: " + ("、".join(hits) if hits else "无显著模式")


class ApprovalMismatchScore(SignalExtractor):
    """L2-j3：授权-转账不一致（ice phishing 模式）。"""

    layer, angle, name = 2, 3, "l2j3_approval_mismatch"

    def params(self, ctx: TxContext) -> dict[str, float]:
        logs = ctx.event_logs or []
        approvals = [l for l in logs if l.event == "Approval"]
        transfers = [l for l in logs if l.event == "Transfer"]
        if not approvals and not transfers:
            return {}
        r: dict[str, float] = {"unlimited": 0.0, "approve_drain": 0.0, "spender_drain": 0.0}
        spenders = set()
        for ap in approvals:
            amt = float(ap.args.get("value", 0) or 0)
            spenders.add(str(ap.args.get("spender", "")))
            if amt >= UNLIMITED_APPROVE:
                r["unlimited"] = 1.0
        for tr in transfers:
            owner_out = str(tr.args.get("from", ""))
            # 授权后同交易内即发生大额转出（drain）
            if approvals and owner_out and owner_out == ctx.from_address:
                r["approve_drain"] = 1.0
            # 转出操作的受益人正是刚获授权的 spender
            if str(tr.args.get("to", "")) in spenders and spenders:
                r["spender_drain"] = 1.0
        return r

    def score(self, ctx: TxContext) -> Optional[float]:
        r = self.params(ctx)
        if not r:
            return None
        w = {"unlimited": 0.4, "approve_drain": 0.3, "spender_drain": 0.3}
        return float(sum(w[k] * v for k, v in r.items()))

    def evidence(self, ctx: TxContext) -> str:
        r = self.params(ctx)
        zh = {"unlimited": "无限额度授权", "approve_drain": "授权后立即转出",
              "spender_drain": "资金流向新授权地址"}
        hits = [zh[k] for k, v in r.items() if v >= 0.5]
        return "授权异常: " + ("、".join(hits) if hits else "无")
