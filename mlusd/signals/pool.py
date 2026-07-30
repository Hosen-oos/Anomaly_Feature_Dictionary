"""格内参数池 + 非稀释聚合（方案定稿 §2.1）。

一个信号格 (l,j) 内部是多个参数（来自多篇论文/多条规则）。每个参数先各自对正常集
做全局 ECDF 得到分位数，再取 **max 分位数**作为格值（"任一参数极端即格异常"，
避免加权平均把强信号稀释——实验四已证明均值稀释之害）。该格值随后由 M3 做组内二次
ECDF 校准恢复共形有效性（max 分位数有上偏，组内再校准修正）。

三层递归母题：格内(多参数→max→组内校准) 与 M5(多格→Fisher→组内校准) 同构。
"""
from __future__ import annotations

from collections import defaultdict
from typing import Optional, Protocol

import numpy as np

from mlusd.signals.base import SignalExtractor
from mlusd.types import TxContext


class ParamProvider(Protocol):
    """被 DictSignal 包裹的内层：给出该格的多参数原始值（越大越异常）。"""
    layer: int
    angle: int
    name: str
    def params(self, ctx: TxContext) -> dict[str, float]: ...
    def evidence(self, ctx: TxContext) -> str: ...
    def fit(self, normal_contexts: list[TxContext]) -> None: ...


class DictSignal(SignalExtractor):
    """把多参数内层聚合成单个格值（max 分位数），对外仍是标准 SignalExtractor。"""

    def __init__(self, inner: ParamProvider, agg: str = "max",
                 two_sided: bool = False):
        self.inner = inner
        self.layer = inner.layer
        self.angle = inner.angle
        self.name = inner.name
        self.agg = agg            # max（非稀释，默认）/ mean（消融，稀释对照）
        # two_sided：参数级双侧——"异常地小"也算异常（零值/粉尘转账类钓鱼）
        self.two_sided = two_sided
        self._ecdf: dict[str, np.ndarray] = {}

    def fit(self, normal_contexts: list[TxContext]) -> None:
        # 内层可能有自身状态需拟合（如 SSTORE 分位阈值）
        self.inner.fit(normal_contexts)
        buckets: dict[str, list[float]] = defaultdict(list)
        for c in normal_contexts:
            for k, v in (self.inner.params(c) or {}).items():
                if v is not None and np.isfinite(v):
                    buckets[k].append(float(v))
        self._ecdf = {k: np.sort(np.asarray(vs, dtype=float))
                      for k, vs in buckets.items()}

    def _param_quantiles(self, ctx: TxContext) -> list[tuple[str, float]]:
        d = self.inner.params(ctx)
        if not d:
            return []
        out = []
        for k, v in d.items():
            ref = self._ecdf.get(k)
            if ref is None or len(ref) == 0 or v is None or not np.isfinite(v):
                continue
            q = float(np.searchsorted(ref, v, side="left") / (len(ref) + 1))
            if self.two_sided:
                q = min(1.0 - 1.0 / (len(ref) + 1), 2.0 * abs(q - 0.5))
            out.append((k, q))
        return out

    def score(self, ctx: TxContext) -> Optional[float]:
        qs = self._param_quantiles(ctx)
        if not qs:
            return None
        vals = [q for _, q in qs]
        if self.agg == "mean":
            return float(sum(vals) / len(vals))    # 消融：均值（稀释）
        return max(vals)                           # 非稀释：任一参数极端即格异常

    def top_param(self, ctx: TxContext) -> Optional[str]:
        qs = self._param_quantiles(ctx)
        return max(qs, key=lambda x: x[1])[0] if qs else None

    def param_quantiles(self, ctx: TxContext) -> dict[str, float]:
        """公开接口：格内各参数的校准分位数 {参数名: q}。供 M4 per-param 字典匹配
        与否定证据（veto）使用——格值聚合会抹掉"哪个参数触发"这一判别信息。"""
        return dict(self._param_quantiles(ctx))

    def evidence(self, ctx: TxContext) -> str:
        return self.inner.evidence(ctx)
