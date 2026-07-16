"""L2-j1 v1：DeFi 动作序列的微型自回归 Transformer 困惑度（设计架构 §5 v1）。

v0 是动作序列 bigram 稀有度；v1 换成 tiny-LM 困惑度——同一个量（-logP 的指数），
故 v0→v1 是 apples-to-apples 消融。参考 BERT4ETH（C16）的"学习式行为序列表征"，
但作用在单交易的 DeFi 动作序列上。
"""
from __future__ import annotations

from mlusd.signals.l2_semantic import lift_actions
from mlusd.signals.nn import SequenceLMExtractor
from mlusd.types import TxContext


def _action_tokens(ctx: TxContext) -> list[str]:
    return [a.kind for a in lift_actions(ctx)]


class ActionSequenceTransformer(SequenceLMExtractor):
    layer, angle, name = 2, 1, "l2j1_action_transformer"

    def __init__(self, **kw):
        super().__init__(tokens_of=_action_tokens, **kw)

    def evidence(self, ctx: TxContext) -> str:
        kinds = _action_tokens(ctx)
        return "动作序列(LM): " + " → ".join(kinds[:10]) + ("…" if len(kinds) > 10 else "")
