"""M5 学习型开放识别（替换 Fisher 聚合，诊断验证 AUROC 0.639→0.767）。

母题不变——"聚合原始信号 → 对聚合量再校准以恢复有效性"，只是把弱的 Fisher 求和换成
学习型聚合器（IsolationForest，无需 GPU）：
    校准 Q 矩阵(8维,缺失位置=0) → IForest 异常分 → 组内 ECDF 再校准 → Ū
再校准保证各可用性组 FP ≈ α（与 Fisher-Ū 相同的 FP 控制），同时拿回检测力。
per-cell 贡献度解释仍由 -2log(1-Q) 给出（可解释性不受影响）。

接口与 OpenSetCalibrator 一致（fit/ubar/threshold），可在 Detector 中互换。
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np

from mlusd.calibrate.groups import GroupResolver
from mlusd.types import VALID_POSITIONS


class LearnedOpenSetCalibrator:
    def __init__(self, alpha: float = 0.01, n_estimators: int = 200, seed: int = 0):
        self.alpha = alpha
        self.n_estimators = n_estimators
        self.seed = seed
        self.mode = "learned"
        self._model = None
        self._sorted: dict[str, np.ndarray] = {}

    def _vec(self, Q: np.ndarray, mask) -> np.ndarray:
        """8 维校准分位向量；缺失/不可用位置填 0（=该信号不异常）。"""
        return np.asarray([
            Q[l - 1, j - 1] if (mask[l - 1] and np.isfinite(Q[l - 1, j - 1])) else 0.0
            for (l, j) in VALID_POSITIONS], dtype=float)

    def fit(self, Qs, masks, resolver: GroupResolver) -> None:
        from sklearn.ensemble import IsolationForest
        X = np.vstack([self._vec(Q, m) for Q, m in zip(Qs, masks)])
        self._model = IsolationForest(n_estimators=self.n_estimators,
                                      random_state=self.seed).fit(X)
        raw = -self._model.score_samples(X)      # 越大越异常
        buckets: dict[str, list[float]] = defaultdict(list)
        for r, m in zip(raw, masks):
            buckets[resolver.resolve(m)].append(float(r))
        self._sorted = {g: np.sort(np.asarray(v)) for g, v in buckets.items()}

    def raw_score(self, Q: np.ndarray, mask) -> float:
        """全局原始异常分（越大越异常），用于排序/AUROC。组内再校准会压掉全局排序，
        故检测质量应以此为准；组内 FP 控制走 ubar（组内分位阈值），二者决策一致。"""
        if self._model is None:
            return 0.0
        return float(-self._model.score_samples(self._vec(Q, mask).reshape(1, -1))[0])

    def ubar(self, Q: np.ndarray, mask, group: str) -> float:
        """组内分位数（决策变量）：ubar≥1−α ⟺ 原始分≥该组 (1−α) 分位阈值，
        实现各组 FP≈α 的控制（side="left" 保守）。"""
        ref = self._sorted.get(group)
        if ref is None or len(ref) == 0 or self._model is None:
            return 0.0
        r = self.raw_score(Q, mask)
        return float(np.searchsorted(ref, r, side="left") / (len(ref) + 1))

    @property
    def threshold(self) -> float:
        return 1.0 - self.alpha
