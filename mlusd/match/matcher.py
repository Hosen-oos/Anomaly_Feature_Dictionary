"""M4 已知类型信号匹配（报告 4.5.2.2）。

覆盖率   c_k    = Σ_l r_k[l]·m[l] / Σ_l r_k[l]
基础分   base_k = Σ W_k[l,j]·T[l,j] / Σ W_k[l,j]      （只在可用且非 NaN 的位置求和）
最终分   final_k = base_k · c_k
贡献度   contrib_k[l,j] = W_k[l,j]·T[l,j] / Σ W·T
"""
from __future__ import annotations

import numpy as np

from mlusd.match.dictionary import AttackDictionary
from mlusd.types import Contribution, MatchResult, VALID_POSITIONS


def match_one(d: AttackDictionary, T: np.ndarray,
              mask: tuple[int, int, int, int]) -> MatchResult:
    req = d.layer_requirements
    coverage = float((req * np.asarray(mask)).sum() / req.sum())

    num = 0.0
    den = 0.0
    weighted: list[tuple[int, int, float]] = []
    for (l, j) in VALID_POSITIONS:
        w = d.weights[l - 1, j - 1]
        if w <= 0 or not mask[l - 1]:
            continue
        t = T[l - 1, j - 1]
        if not np.isfinite(t):
            continue
        num += w * t
        den += w
        weighted.append((l, j, w * t))

    base = num / den if den > 0 else 0.0
    total_wt = sum(v for _, _, v in weighted)
    contributions = [
        Contribution(layer=l, angle=j, value=(v / total_wt if total_wt > 0 else 0.0))
        for l, j, v in sorted(weighted, key=lambda x: -x[2])
    ]
    return MatchResult(
        attack_type=d.attack_type,
        base_score=float(base),
        coverage=coverage,
        final_score=float(base * coverage),
        match_threshold=d.match_threshold,
        coverage_threshold=d.coverage_threshold,
        contributions=contributions,
    )


def match_all(dicts: list[AttackDictionary], T: np.ndarray,
              mask: tuple[int, int, int, int]) -> list[MatchResult]:
    """按最终匹配分数降序返回全部类型的匹配结果。"""
    results = [match_one(d, T, mask) for d in dicts]
    return sorted(results, key=lambda r: -r.final_score)
