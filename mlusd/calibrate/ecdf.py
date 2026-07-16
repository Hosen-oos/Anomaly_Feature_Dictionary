"""M3 统一度量：组内经验分位数 + 尾部放大（报告 4.5.2.1）。

统计上这就是共形 p 值（Bates et al., Ann. Statist. 2023）：
    Q[l,j] = #{正常校准样本分数 < 当前分数} / (n + 1)
严格小于（searchsorted side="left"）而非小于等于：规则型信号的分数高度离散
（正常交易大量并列 0），若并列值计入分位数，一笔分数为 0 的正常交易会得到
Q≈1 而被系统性误报；取严格小于对应保守的共形 p 值 p=(#{cal>=s}+1)/(n+1)，
误报率仍受 α 控制。
尾部概率 p = 1 - Q 的下界为 1/(n+1)（校准集分辨率极限），因此尾部放大做
组内归一化：
    T[l,j] = -log(1 - Q) / log(n + 1)   ∈ [0, 1]
T 接近 1 表示该信号超出了组内全部正常样本。与报告公式的差异仅在归一化
分母：报告用固定 eps，实现中改用 log(n+1) 使 T 的量纲不随校准集规模变化。
"""
from __future__ import annotations

import numpy as np

from mlusd.calibrate.groups import GroupResolver
from mlusd.types import N_ANGLES, N_LAYERS, VALID_POSITIONS


class ECDFCalibrator:
    """按 (校准组, 层, 角度) 存正常分数的排序数组，查询时给出 Q/T。"""

    def __init__(self, min_group_size: int = 500):
        self.resolver = GroupResolver(min_group_size)
        # {(group, layer, angle): 升序 ndarray}
        self._sorted: dict[tuple[str, int, int], np.ndarray] = {}

    def fit(self, matrices: list[np.ndarray],
            masks: list[tuple[int, int, int, int]]) -> None:
        assert len(matrices) == len(masks)
        self.resolver.fit(masks)
        buckets: dict[tuple[str, int, int], list[float]] = {}
        for S, m in zip(matrices, masks):
            g = self.resolver.resolve(m)
            for (l, j) in VALID_POSITIONS:
                v = S[l - 1, j - 1]
                if np.isfinite(v):
                    buckets.setdefault((g, l, j), []).append(float(v))
        self._sorted = {
            key: np.sort(np.asarray(vals)) for key, vals in buckets.items()
        }

    def transform(self, S: np.ndarray,
                  mask: tuple[int, int, int, int]) -> tuple[np.ndarray, np.ndarray, str]:
        """返回 (Q, T, 使用的校准组)。缺失/无校准数据的位置为 NaN。"""
        g = self.resolver.resolve(mask)
        Q = np.full((N_LAYERS, N_ANGLES), np.nan)
        T = np.full((N_LAYERS, N_ANGLES), np.nan)
        for (l, j) in VALID_POSITIONS:
            if not mask[l - 1]:
                continue
            v = S[l - 1, j - 1]
            if not np.isfinite(v):
                continue
            ref = self._sorted.get((g, l, j))
            if ref is None or len(ref) == 0:
                continue
            n = len(ref)
            q = np.searchsorted(ref, v, side="left") / (n + 1)
            Q[l - 1, j - 1] = q
            T[l - 1, j - 1] = -np.log(1.0 - q) / np.log(n + 1)
        return Q, T, g
