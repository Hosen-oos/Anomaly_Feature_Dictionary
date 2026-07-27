"""M4 增强：per-param 对比式字典 + 否定证据（veto）。

动机（真实数据验证）：经济三类（flash_loan/price_manipulation/rug_pull）在 8 个聚合
格值上"同向为正"，格层面分不开；但在**参数/动作层面**判别力强——例如 rug_pull 的
借贷/流动性动作出现率为 0%，而 flash/price 为 20–36%。故 veto 必须下沉到参数层。

三点设计（与主框架一致）：
1. **对比式权重**：权重学"类型 k vs 其他攻击类型"（非 vs 正常）——所有攻击相对正常
   的偏离方向相似，只有类间对比才产生判别力；**不 clip 负值**，负值即否定证据。
2. **掩码感知**：否定证据只在该层可用（m[l]=1）且参数确实算出时计入。层不可用 ≠
   该行为没发生（守住异构可用性主张，见设计定稿 §1）。
3. **少样本收缩**：每类仅 ~20 样本，权重按 n/(n+n0) 向 0 收缩，抑制过拟合。

匹配分：score_k = pos_k − λ·neg_k，其中 pos/neg 分别为正/负权重下的加权平均分位数。
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np

from mlusd.signals.pool import DictSignal


@dataclass
class ParamProfile:
    """一个攻击类型的 per-param 对比式权重档案。"""
    attack_type: str
    weights: dict[str, float] = field(default_factory=dict)   # 参数名 -> 对比权重(可负)
    layers: dict[str, int] = field(default_factory=dict)      # 参数名 -> 所属层(掩码用)

    def positive(self) -> dict[str, float]:
        return {k: v for k, v in self.weights.items() if v > 0}

    def negative(self) -> dict[str, float]:
        return {k: -v for k, v in self.weights.items() if v < 0}


def collect_param_quantiles(det, ctx) -> tuple[dict[str, float], dict[str, int]]:
    """从所有 DictSignal 提取器收集 {参数名: 分位数} 与 {参数名: 层}。
    参数名加 'L{l}j{j}.' 前缀避免跨格重名。"""
    qs, layers = {}, {}
    mask = ctx.availability
    for e in det.extractors:
        if not isinstance(e, DictSignal):
            continue
        if not mask[e.layer - 1]:          # 掩码感知：层不可用则不产出任何证据
            continue
        for k, q in e.param_quantiles(ctx).items():
            name = f"L{e.layer}j{e.angle}.{k}"
            qs[name] = q
            layers[name] = e.layer
    return qs, layers


def build_profiles(det, attacks_by_type: dict, n0: float = 10.0,
                   contrast: str = "vs_others") -> dict[str, ParamProfile]:
    """构建 per-param 权重档案。

    contrast="vs_others"（默认，对比式）：w[p] = mean_q[k][p] − mean_q[others][p]
        ——学类间差异，这才产生判别力与否定证据。
    contrast="vs_normal"（消融）：w[p] = mean_q[k][p] − 0.5
        ——参数分位数本就以正常集为基准校准，故正常均值≈0.5；等价于"攻击 vs 正常"。
        用于分离"per-param 粒度"与"对比式目标"各自的贡献。
    """
    # 每类每参数的均值分位数
    per_type: dict[str, dict[str, list[float]]] = {}
    layer_of: dict[str, int] = {}
    for t, ctxs in attacks_by_type.items():
        acc: dict[str, list[float]] = defaultdict(list)
        for c in ctxs:
            qs, lys = collect_param_quantiles(det, c)
            for k, q in qs.items():
                acc[k].append(q)
            layer_of.update(lys)
        per_type[t] = acc

    means = {t: {k: float(np.mean(v)) for k, v in acc.items() if v}
             for t, acc in per_type.items()}
    counts = {t: {k: len(v) for k, v in acc.items()} for t, acc in per_type.items()}

    profiles = {}
    for t in means:
        prof = ParamProfile(attack_type=t)
        for k, mine in means[t].items():
            if contrast == "vs_normal":
                raw = mine - 0.5              # 正常基准（分位数以正常集校准）
            else:
                others = [means[o][k] for o in means if o != t and k in means[o]]
                if not others:
                    continue
                raw = mine - float(np.mean(others))   # 对比：正=支持, 负=否定证据
            n = counts[t].get(k, 0)
            prof.weights[k] = raw * (n / (n + n0))       # 少样本收缩
            prof.layers[k] = layer_of.get(k, 1)
        profiles[t] = prof
    return profiles


def contrastive_score(prof: ParamProfile, qs: dict[str, float],
                      lam: float = 1.0) -> tuple[float, float, list[str]]:
    """返回 (综合分, 否定分, 触发的否定证据参数名)。qs 已是掩码过滤后的参数分位数。"""
    pos_w, neg_w = prof.positive(), prof.negative()

    def wavg(weights):
        num = den = 0.0
        for k, w in weights.items():
            if k in qs:                    # 掩码感知：不可用层的参数不在 qs 中
                num += w * qs[k]
                den += w
        return (num / den) if den > 0 else 0.0

    pos = wavg(pos_w)
    neg = wavg(neg_w)
    fired = [k for k in neg_w if k in qs and qs[k] >= 0.9]   # 高分位的否定证据
    return pos - lam * neg, neg, fired
