"""L3 EVM 执行轨迹层的两个 v0 提取器（设计架构 §4 M2、§5.3）。

L3-j1 分布偏离：调用树 4-gram 稀有度（正常 trace 统计）——BlockGPT
       （arXiv 2304.12749）trace 语言模型的无参数轻量替代，v1 换成 trace 树 LM。
L3-j3 信息差异：执行路径 vs 声明功能偏离，用"逻辑关系 + 规则查询"形式表达
       （TxSpector, USENIX Sec 2020）——重入 / revert 后继续获利 / SSTORE 突增。
"""
from __future__ import annotations

import math
from collections import Counter
from typing import Optional

from mlusd.signals.base import SignalExtractor
from mlusd.types import Call, TraceSummary, TxContext


def _call_token(c: Call) -> str:
    """调用节点 tokenization：类型@深度（BlockGPT trace token 的粗粒度版）。"""
    return f"{c.kind}@{min(c.depth, 8)}"


def _call_seq(trace: Optional[TraceSummary]) -> list[str]:
    if trace is None or not trace.calls:
        return []
    return ["<s>"] + [_call_token(c) for c in trace.calls] + ["</s>"]


class TraceNgramRarity(SignalExtractor):
    """L3-j1：调用序列 4-gram 稀有度，score = 平均 -log2 P(gram)。"""

    layer, angle, name = 3, 1, "l3j1_trace_ngram"

    def __init__(self, n: int = 4):
        self.n = n
        self._grams: Counter = Counter()
        self._total: int = 0
        self._vocab: int = 1

    def fit(self, normal_contexts: list[TxContext]) -> None:
        for ctx in normal_contexts:
            seq = _call_seq(ctx.trace)
            for g in self._iter_grams(seq):
                self._grams[g] += 1
        self._total = sum(self._grams.values())
        self._vocab = max(len(self._grams), 1)

    def _iter_grams(self, seq: list[str]):
        if len(seq) < self.n:
            if seq:
                yield tuple(seq)
            return
        for i in range(len(seq) - self.n + 1):
            yield tuple(seq[i:i + self.n])

    def score(self, ctx: TxContext) -> Optional[float]:
        seq = _call_seq(ctx.trace)
        if not seq:
            return None
        grams = list(self._iter_grams(seq))
        if not grams:
            return None
        nll = 0.0
        for g in grams:
            p = (self._grams[g] + 1) / (self._total + self._vocab)
            nll += -math.log2(p)
        return nll / len(grams)

    def evidence(self, ctx: TxContext) -> str:
        t = ctx.trace
        if t is None:
            return ""
        return (f"调用树: {len(t.calls)}次调用, 最大深度{t.max_depth}, "
                f"SSTORE {t.sstore_count}次")


class TracePropertyScore(SignalExtractor):
    """L3-j3：TxSpector 式逻辑规则——重入 / revert 获利 / 状态写入突增。"""

    layer, angle, name = 3, 3, "l3j3_trace_rules"

    def __init__(self):
        # SSTORE 次数的正常上界（fit 时以正常集 99 分位标定，未 fit 用默认）
        self._sstore_p99: float = 20.0

    def fit(self, normal_contexts: list[TxContext]) -> None:
        counts = [c.trace.sstore_count for c in normal_contexts
                  if c.trace is not None]
        if counts:
            import numpy as np
            self._sstore_p99 = float(np.percentile(counts, 99)) + 1.0

    def params(self, ctx: TxContext) -> dict[str, float]:
        t = ctx.trace
        if t is None or not t.calls:
            return {}
        r: dict[str, float] = {}
        # 重入：调用栈中同一 (from->to) 合约对在未返回时重复入栈
        active: list[tuple[str, str]] = []
        reentrant = False
        prev_depth = 0
        for c in t.calls:
            if c.depth <= prev_depth:
                active = active[:c.depth]
            pair = (c.to,)
            if c.to in [a[0] for a in active]:
                reentrant = True
            active.append((c.to,))
            prev_depth = c.depth
        r["reentrancy"] = 1.0 if reentrant else 0.0
        # revert 后继续获利：存在被 revert 的子调用但顶层交易成功
        r["revert_profit"] = 1.0 if (t.reverted_subcalls > 0 and ctx.status) else 0.0
        # 状态写入突增：SSTORE 次数远超正常上界
        r["sstore_spike"] = min(1.0, max(0.0, t.sstore_count - self._sstore_p99)
                                / max(self._sstore_p99, 1.0))
        return r

    def score(self, ctx: TxContext) -> Optional[float]:
        r = self.params(ctx)
        if not r:
            return None
        w = {"reentrancy": 0.45, "revert_profit": 0.30, "sstore_spike": 0.25}
        return float(sum(w[k] * v for k, v in r.items()))

    def evidence(self, ctx: TxContext) -> str:
        r = self.params(ctx)
        zh = {"reentrancy": "重入调用", "revert_profit": "回滚后获利",
              "sstore_spike": "状态写入突增"}
        hits = [zh[k] for k, v in r.items() if v >= 0.5]
        return "执行轨迹: " + ("、".join(hits) if hits else "无异常模式")
