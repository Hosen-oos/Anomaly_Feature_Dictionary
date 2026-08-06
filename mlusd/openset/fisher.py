"""M5 未知异常开放识别（报告 4.5.2.3）。

Fisher 聚合：U = -2 Σ log(1 - Q[l,j])，只在可用位置求和。
同一校准集导出的共形 p 值相互相关，U 的卡方分布不严格成立；因此不用
卡方阈值，而是对 U 再做一次组内经验分位数校准：
    Ū = ECDF_g(U)   （在同组正常交易的 U 分布上）
判定阈值直接取 Ū ≥ 1 - α，α 为目标误报率——这一步绕开了独立性假设，
是方案的方法论点之一（设计架构 §5.1）。
"""
from __future__ import annotations

import numpy as np

from mlusd.calibrate.groups import GroupResolver
from mlusd.types import VALID_POSITIONS


def fisher_u(Q: np.ndarray, mask: tuple[int, int, int, int],
             mode: str = "fisher") -> float:
    """跨格聚合原始统计量。mode: fisher(Σ) / max / mean —— 实验三消融用。"""
    terms = []
    for (l, j) in VALID_POSITIONS:
        if not mask[l - 1]:
            continue
        q = Q[l - 1, j - 1]
        if not np.isfinite(q):
            continue
        terms.append(-2.0 * np.log(max(1.0 - q, 1e-12)))
    if not terms:
        return 0.0
    if mode == "max":
        return float(max(terms))
    if mode == "mean":
        return float(sum(terms) / len(terms))
    return float(sum(terms))     # fisher（默认）


class OpenSetCalibrator:
    """存各校准组正常交易的 U 分布；查询时给出组内相对异常分数 Ū。"""

    def __init__(self, alpha: float = 0.01, mode: str = "fisher"):
        self.alpha = alpha
        self.mode = mode          # fisher / max / mean（消融）
        self._sorted_u: dict[str, np.ndarray] = {}

    def fit(self, Qs: list[np.ndarray],
            masks: list[tuple[int, int, int, int]],
            resolver: GroupResolver, param_vecs=None) -> None:
        """param_vecs 仅为与 LearnedOpenSetCalibrator 接口一致而接受，Fisher 路不使用。"""
        buckets: dict[str, list[float]] = {}
        for Q, m in zip(Qs, masks):
            g = resolver.resolve(m)
            buckets.setdefault(g, []).append(fisher_u(Q, m, self.mode))
        self._sorted_u = {g: np.sort(np.asarray(v)) for g, v in buckets.items()}

    def ubar(self, Q: np.ndarray, mask: tuple[int, int, int, int],
             group: str, param_vec=None) -> float:
        """Ū：同组正常交易中整体异常程度不超过当前交易的比例。"""
        ref = self._sorted_u.get(group)
        if ref is None or len(ref) == 0:
            return 0.0
        u = fisher_u(Q, mask, self.mode)
        n = len(ref)
        # side="left"：U 与正常样本并列时取低分位（保守，见 ecdf.py 顶部说明）
        return float(np.searchsorted(ref, u, side="left") / (n + 1))

    @property
    def threshold(self) -> float:
        """τ_u：Ū 超过该值判为未知异常（误报率约为 α）。"""
        return 1.0 - self.alpha
