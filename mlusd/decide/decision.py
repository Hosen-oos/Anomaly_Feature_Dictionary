"""M6 决策与解释（报告 4.3.3 表 8、设计架构 §4 M6）。

判定优先级：
1. final_{k*} ≥ τ_{k*} 且 c_{k*} ≥ c_min      -> KNOWN(k*)
2. base_{k*}  ≥ τ_{k*} 且 c_{k*} <  c_min      -> INSUFFICIENT（证据缺失，不硬分类）
3. 以上不满足 且 Ū ≥ τ_u                       -> UNKNOWN
4. 其余                                         -> NORMAL
不训练独立解释模型：解释直接来自匹配贡献度 / Fisher 贡献度与提取器证据片段。
"""
from __future__ import annotations

import numpy as np

from mlusd.types import (
    Contribution, DetectionReport, LAYER_NAMES, MatchResult, VALID_POSITIONS,
    Verdict,
)


def _unknown_contributions(Q: np.ndarray,
                           mask: tuple[int, int, int, int]) -> list[Contribution]:
    """未知异常贡献度：ucontrib[l,j] = -2·log(1-Q) / U。"""
    parts = []
    for (l, j) in VALID_POSITIONS:
        if not mask[l - 1]:
            continue
        q = Q[l - 1, j - 1]
        if not np.isfinite(q):
            continue
        parts.append((l, j, -2.0 * np.log(max(1.0 - q, 1e-12))))
    total = sum(v for _, _, v in parts)
    if total <= 0:
        return []
    return [Contribution(layer=l, angle=j, value=v / total)
            for l, j, v in sorted(parts, key=lambda x: -x[2])]


def _missing_evidence(best: MatchResult,
                      req: np.ndarray,
                      mask: tuple[int, int, int, int]) -> list[str]:
    return [
        f"{LAYER_NAMES[l + 1]}（判定 {best.attack_type} 依赖度 {req[l]:.2f}）"
        for l in range(4)
        if req[l] > 0 and not mask[l]
    ]


def decide(tx_hash: str,
           matches: list[MatchResult],
           Q: np.ndarray,
           mask: tuple[int, int, int, int],
           group: str,
           ubar: float,
           tau_u: float,
           layer_requirements: dict[str, np.ndarray]) -> DetectionReport:
    best = matches[0]
    report = DetectionReport(
        tx_hash=tx_hash,
        verdict=Verdict.NORMAL,
        availability_group=group,
        unknown_score=ubar,
        all_matches=matches,
    )

    if best.final_score >= best.match_threshold and best.coverage >= best.coverage_threshold:
        report.verdict = Verdict.KNOWN
        report.known_type = best.attack_type
        report.match_score = best.final_score
        report.evidence_coverage = best.coverage
        report.contributions = best.contributions[:5]
        return report

    if best.base_score >= best.match_threshold and best.coverage < best.coverage_threshold:
        report.verdict = Verdict.INSUFFICIENT
        report.known_type = best.attack_type
        report.match_score = best.base_score
        report.evidence_coverage = best.coverage
        report.contributions = best.contributions[:5]
        report.missing_evidence = _missing_evidence(
            best, layer_requirements[best.attack_type], mask)
        return report

    if ubar >= tau_u:
        report.verdict = Verdict.UNKNOWN
        report.contributions = _unknown_contributions(Q, mask)[:5]
        return report

    return report
