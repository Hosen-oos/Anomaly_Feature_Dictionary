"""端到端检测流水线：把 M2-M6 组装为一个 Detector（设计架构 §2）。

用法：
    det = Detector(extractors, dictionaries, alpha=0.01)
    det.fit(normal_contexts)          # 只喂正常交易
    report = det.detect(ctx)          # -> DetectionReport
"""
from __future__ import annotations

import numpy as np

from mlusd.calibrate.ecdf import ECDFCalibrator
from mlusd.decide.decision import decide
from mlusd.match.dictionary import AttackDictionary
from mlusd.match.matcher import match_all
from mlusd.openset.fisher import OpenSetCalibrator
from mlusd.signals.base import SignalExtractor
from mlusd.types import DetectionReport, TxContext, empty_signal_matrix


class Detector:
    def __init__(self,
                 extractors: list[SignalExtractor],
                 dictionaries: list[AttackDictionary],
                 alpha: float = 0.01,
                 min_group_size: int = 500,
                 single_group: bool = False,
                 openset_mode: str = "fisher",
                 openset_aggregator: str = "learned",
                 two_sided: bool = False,
                 dual_tail: bool = False):
        seen = set()
        for e in extractors:
            key = (e.layer, e.angle)
            assert key not in seen, f"信号位置重复: L{e.layer}-j{e.angle}"
            seen.add(key)
        self.extractors = extractors
        self.dictionaries = dictionaries
        self._req = {d.attack_type: d.layer_requirements for d in dictionaries}
        self.calibrator = ECDFCalibrator(min_group_size=min_group_size,
                                         single_group=single_group,
                                         two_sided=two_sided)
        # 参数级也同步双侧（零值/粉尘类参数的"异常地小"在格内就该被捕获）
        for e in extractors:
            if hasattr(e, "two_sided"):
                e.two_sided = two_sided
        if openset_aggregator == "learned":
            from mlusd.openset.learned import LearnedOpenSetCalibrator
            self.openset = LearnedOpenSetCalibrator(alpha=alpha, dual_tail=dual_tail)
        else:
            self.openset = OpenSetCalibrator(alpha=alpha, mode=openset_mode)
        self._fitted = False

    # ------------------------------------------------------------- 内部

    def _raw_matrix(self, ctx: TxContext) -> tuple[np.ndarray, tuple]:
        mask = ctx.availability
        S = empty_signal_matrix()
        for e in self.extractors:
            if not mask[e.layer - 1]:
                continue
            v = e.score(ctx)
            if v is not None and np.isfinite(v):
                S[e.layer - 1, e.angle - 1] = float(v)
        return S, mask

    # ------------------------------------------------------------- API

    def fit(self, normal_contexts: list[TxContext]) -> "Detector":
        """在正常交易校准集上拟合提取器、ECDF 校准和开放集阈值。"""
        for e in self.extractors:
            e.fit(normal_contexts)
        mats, masks = [], []
        for ctx in normal_contexts:
            S, m = self._raw_matrix(ctx)
            mats.append(S)
            masks.append(m)
        self.calibrator.fit(mats, masks)
        Qs = [self.calibrator.transform(S, m)[0] for S, m in zip(mats, masks)]
        self.openset.fit(Qs, masks, self.calibrator.resolver)
        self._fitted = True
        return self

    def detect(self, ctx: TxContext) -> DetectionReport:
        assert self._fitted, "先调用 fit(normal_contexts)"
        S, mask = self._raw_matrix(ctx)
        Q, T, group = self.calibrator.transform(S, mask)
        matches = match_all(self.dictionaries, T, mask)
        ubar = self.openset.ubar(Q, mask, group)
        report = decide(
            tx_hash=ctx.tx_hash, matches=matches, Q=Q, mask=mask, group=group,
            ubar=ubar, tau_u=self.openset.threshold,
            layer_requirements=self._req,
        )
        # 用提取器的证据片段充实解释输出
        ev = {(e.layer, e.angle): e for e in self.extractors}
        for c in report.contributions:
            e = ev.get((c.layer, c.angle))
            if e is not None:
                c.evidence = e.evidence(ctx)
        return report

    def detect_batch(self, contexts: list[TxContext]) -> list[DetectionReport]:
        return [self.detect(c) for c in contexts]


class DualPathDetector:
    """两路校准并联（单侧 + 折叠双侧），取全局分位数的 max（实测净收益为正）。

    动机：异常方向随信号/攻击类型而变——经济类攻击"值大即异常"（单侧最优 0.99），
    ponzi/地址投毒类"值小即异常"（折叠最优 0.80/0.60），而无监督聚合器无法自行推断
    方向（双尾特征实测无效：正常 q 均匀 → 两尾在边缘分布上都不显著）。故两路并联，
    "任一路极端即异常"——与格内参数池的非稀释 max 聚合同一母题。

    误报控制：两个 (1-α/2) 拒绝域的并集 ≈ α，故各路用 α/2。
    检测分数用 detect_score()；类型匹配/解释走单侧路径的 detect()（方向性由 M4 对比式
    权重处理，不受此影响）。
    """

    def __init__(self, extractors_factory, dictionaries, alpha: float = 0.01,
                 min_group_size: int = 500, ref_n: int = 4000):
        self.alpha = alpha
        self.ref_n = ref_n
        self.single = Detector(extractors_factory(), dictionaries, alpha=alpha / 2,
                               min_group_size=min_group_size,
                               openset_aggregator="learned")
        self.folded = Detector(extractors_factory(), dictionaries, alpha=alpha / 2,
                              min_group_size=min_group_size,
                              openset_aggregator="learned", two_sided=True)
        self._ref: dict[str, np.ndarray] = {}

    def _raw(self, det, ctx) -> float:
        S, m = det._raw_matrix(ctx)
        Q, _, _ = det.calibrator.transform(S, m)
        return det.openset.raw_score(Q, m)

    def fit(self, normal_contexts):
        self.single.fit(normal_contexts)
        self.folded.fit(normal_contexts)
        ref = normal_contexts[:self.ref_n]
        self._ref = {"single": np.sort([self._raw(self.single, c) for c in ref]),
                     "folded": np.sort([self._raw(self.folded, c) for c in ref])}
        return self

    def detect_score(self, ctx) -> float:
        """并联异常分 ∈ [0,1]：两路全局分位数取 max。"""
        out = []
        for name, det in (("single", self.single), ("folded", self.folded)):
            ref = self._ref[name]
            v = self._raw(det, ctx)
            out.append(np.searchsorted(ref, v, side="left") / (len(ref) + 1))
        return float(max(out))

    def detect(self, ctx) -> DetectionReport:
        """完整报告走单侧路径（类型匹配与解释），并把并联分数写入 unknown_score。"""
        rep = self.single.detect(ctx)
        rep.unknown_score = self.detect_score(ctx)
        return rep
