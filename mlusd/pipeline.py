"""端到端检测流水线：把 M2-M6 组装为一个 Detector（设计架构 §2）。

用法：
    det = Detector(extractors, dictionaries, alpha=0.01)
    det.fit(normal_contexts)                          # 只喂正常交易（无监督部分）
    det.fit_types(attacks_by_type, normal_contexts)   # 可选：参数级字典 + 阈值标定
    report = det.detect(ctx)                          # -> DetectionReport

fit() 完成后 detect() 即可工作，此时 M4 走位级字典。fit_types() 另需带类型标签的攻击
样本，它构建参数级对比权重并把各类 τ_k 重标定到正常匹配分的 (1-alpha_fp) 分位数；此后
M4 改走参数级分 score_k。两条路径的覆盖率缩放、判定优先级与贡献度分解完全一致。
"""
from __future__ import annotations

import numpy as np

from mlusd.calibrate.ecdf import ECDFCalibrator
from mlusd.decide.decision import decide
from mlusd.match.dictionary import AttackDictionary
from mlusd.match.matcher import match_all, match_one
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
                 dual_tail: bool = False,
                 dual_repr: bool = True,
                 param_dict: bool = True,
                 param_dict_lam: float = 1.0,
                 param_dict_n0: float = 10.0,
                 alpha_fp: float = 0.005):
        # dual_repr：开放集检测采用**双路表示**（8 维格值 + 参数向量，取全局分位数 max）。
        # 实测整体 0.837→0.877、sandwich 0.779→0.929，phishing 基本持平。见 learned.py::fit。
        self.dual_repr = dual_repr
        # param_dict：M4 使用参数级对比字典（需 fit_types 提供带标签攻击样本）。
        # 位级 max 聚合会抹平机理相近类型间的差异，参数级对比权重恢复该判别力。
        self.param_dict = param_dict
        self.param_dict_lam = param_dict_lam
        self.param_dict_n0 = param_dict_n0
        self.alpha_fp = alpha_fp
        self._profiles: dict = {}
        self._param_names: list[str] = []
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

    def _param_quantiles(self, ctx: TxContext) -> dict[str, float]:
        """各 DictSignal 格内参数的校准分位数（掩码感知：层不可用则不产出）。"""
        from mlusd.signals.pool import DictSignal
        out: dict[str, float] = {}
        mask = ctx.availability
        for e in self.extractors:
            if not isinstance(e, DictSignal) or not mask[e.layer - 1]:
                continue
            for k, q in e.param_quantiles(ctx).items():
                out[f"L{e.layer}j{e.angle}.{k}"] = q
        return out

    def _param_vec(self, ctx: TxContext) -> np.ndarray:
        """定长参数向量（缺失填中性 0.5）+ 可用性掩码，供双路表示的参数路使用。"""
        qs = self._param_quantiles(ctx)
        v = np.full(len(self._param_names) + 4, 0.5)
        for i, k in enumerate(self._param_names):
            if k in qs:
                v[i] = qs[k]
        v[len(self._param_names):] = ctx.availability
        return v

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

        param_vecs = None
        if self.dual_repr and hasattr(self.openset, "_model_p"):
            pqs = [self._param_quantiles(c) for c in normal_contexts]
            self._param_names = sorted({k for d in pqs for k in d})
            if self._param_names:
                param_vecs = [self._param_vec(c) for c in normal_contexts]
        self.openset.fit(Qs, masks, self.calibrator.resolver, param_vecs=param_vecs)
        self._fitted = True
        return self

    def fit_types(self, attacks_by_type: dict[str, list[TxContext]],
                  normal_contexts: list[TxContext] | None = None,
                  alpha_fp: float | None = None) -> dict[str, float]:
        """构建参数级对比字典，并逐类型把 τ_k 标定到正常匹配分的 (1-alpha_fp) 分位数。

        阈值必须重标定：score_k 与位级 base_k 量纲不同，YAML 中的先验 τ_k 不可直接沿用。
        标定按逐类误报率而非 F1 最优——类型判定是跨类型 argmax，多个低阈值叠加会把大量
        正常交易误判为某个已知类型（见 weight_update.tune_thresholds）。

        返回 {攻击类型: 标定后的 τ_k}；param_dict=False 时为空操作。
        """
        assert self._fitted, "先调用 fit(normal_contexts)"
        if not self.param_dict:
            return {}
        from mlusd.match.contrastive import build_profiles
        known = {d.attack_type for d in self.dictionaries}
        profs = build_profiles(self, attacks_by_type, n0=self.param_dict_n0)
        self._profiles = {t: p for t, p in profs.items() if t in known}
        if not normal_contexts:
            return {d.attack_type: d.match_threshold for d in self.dictionaries}

        rows = []
        for c in normal_contexts:
            S, m = self._raw_matrix(c)
            _, T, _ = self.calibrator.transform(S, m)
            rows.append((T, m, self._type_scores(c)))
        a = self.alpha_fp if alpha_fp is None else alpha_fp
        tuned = {}
        for d in self.dictionaries:
            neg = [match_one(d, T, m, None if sc is None else sc.get(d.attack_type)).final_score
                   for T, m, sc in rows]
            if neg:
                d.match_threshold = max(float(np.quantile(neg, 1.0 - a)), 0.05)
            tuned[d.attack_type] = d.match_threshold
        return tuned

    def _type_scores(self, ctx: TxContext) -> dict[str, float] | None:
        """参数级对比分 {类型: score_k}；未构建档案时返回 None（M4 退回位级）。"""
        if not self._profiles:
            return None
        from mlusd.match.contrastive import contrastive_score
        qs = self._param_quantiles(ctx)
        return {t: contrastive_score(p, qs, self.param_dict_lam)[0]
                for t, p in self._profiles.items()}

    def detect(self, ctx: TxContext) -> DetectionReport:
        assert self._fitted, "先调用 fit(normal_contexts)"
        S, mask = self._raw_matrix(ctx)
        Q, T, group = self.calibrator.transform(S, mask)
        matches = match_all(self.dictionaries, T, mask, self._type_scores(ctx))
        pv = self._param_vec(ctx) if self._param_names else None
        ubar = (self.openset.ubar(Q, mask, group, pv)
                if self._param_names else self.openset.ubar(Q, mask, group))
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

    def fit_types(self, attacks_by_type, normal_contexts=None, alpha_fp=None):
        """类型匹配只走单侧路径（见类文档），故参数级字典也只在该路径上构建。"""
        return self.single.fit_types(attacks_by_type, normal_contexts, alpha_fp)

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
