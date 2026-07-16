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
                 min_group_size: int = 500):
        seen = set()
        for e in extractors:
            key = (e.layer, e.angle)
            assert key not in seen, f"信号位置重复: L{e.layer}-j{e.angle}"
            seen.add(key)
        self.extractors = extractors
        self.dictionaries = dictionaries
        self._req = {d.attack_type: d.layer_requirements for d in dictionaries}
        self.calibrator = ECDFCalibrator(min_group_size=min_group_size)
        self.openset = OpenSetCalibrator(alpha=alpha)
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
