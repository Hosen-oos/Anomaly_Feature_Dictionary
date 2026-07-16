"""L3-j1 v1：调用树 token 序列的微型语言模型困惑度（设计架构 §5 v1）。

即 BlockGPT（arXiv 2304.12749）的轻量复现：trace 树 tokenization + 从头训 LM +
按困惑度排序异常。v0 的 4-gram 稀有度是其无参数替代，二者对比构成一个实验点。
token 复用 l3_trace 的 kind@depth 编码。
"""
from __future__ import annotations

from mlusd.signals.l3_trace import _call_token
from mlusd.signals.nn import SequenceLMExtractor
from mlusd.types import TxContext


def _trace_tokens(ctx: TxContext) -> list[str]:
    if ctx.trace is None or not ctx.trace.calls:
        return []
    return [_call_token(c) for c in ctx.trace.calls]


class TraceTransformer(SequenceLMExtractor):
    layer, angle, name = 3, 1, "l3j1_trace_transformer"

    def __init__(self, **kw):
        super().__init__(tokens_of=_trace_tokens, **kw)

    def evidence(self, ctx: TxContext) -> str:
        t = ctx.trace
        if t is None:
            return ""
        return f"调用树(LM): {len(t.calls)}次调用, 最大深度{t.max_depth}"
