"""L1 交易图拓扑层的两个 v0 提取器（设计架构 §4 M2）。

L1-j1 分布偏离：邻域图统计特征 + Isolation Forest（特征定义参考 Elliptic++/TTAGN）。
L1-j2 经济异常：资金流模式规则分（FlowScope 的多跳中转 + AntiBenford 的首位数偏离）。
"""
from __future__ import annotations

import math
from collections import Counter
from typing import Optional

import numpy as np

from mlusd.signals.base import SignalExtractor
from mlusd.types import TxContext

# Benford 定律的首位数概率 P(d) = log10(1 + 1/d)
_BENFORD = np.array([math.log10(1 + 1 / d) for d in range(1, 10)])


def _graph_features(ctx: TxContext) -> Optional[np.ndarray]:
    """从 ego 图提取中心地址（交易发起方）的统计特征向量。"""
    g = ctx.ego_graph
    center = ctx.from_address
    if g is None or center not in g:
        return None
    out_edges = list(g.out_edges(center, data=True))
    in_edges = list(g.in_edges(center, data=True))
    out_vals = [e[2].get("value", 0.0) for e in out_edges]
    in_vals = [e[2].get("value", 0.0) for e in in_edges]
    times = sorted(e[2]["timestamp"] for e in g.edges(center, data=True)
                   if "timestamp" in e[2])
    gaps = np.diff(times) if len(times) > 1 else np.array([0.0])

    def _entropy(vals: np.ndarray) -> float:
        if len(vals) < 2 or vals.sum() <= 0:
            return 0.0
        p = vals / vals.sum()
        p = p[p > 0]
        return float(-(p * np.log(p)).sum())

    def _stats(vals: list[float]) -> tuple[float, float, float]:
        if not vals:
            return 0.0, 0.0, 0.0
        a = np.asarray(vals, dtype=float)
        return float(np.log1p(a.sum())), float(np.log1p(a.max())), _entropy(a)

    out_sum, out_max, out_ent = _stats(out_vals)
    in_sum, in_max, in_ent = _stats(in_vals)
    # 图级特征：任意节点的最大入/出度（捕捉 drainer 归集 / ponzi 扇入，
    # 该结构常不在中心节点 from_address 上，改进 B 关键）
    import networkx as nx
    dg = nx.DiGraph(g)
    max_in = max((dg.in_degree(n) for n in dg.nodes()), default=0)
    max_out = max((dg.out_degree(n) for n in dg.nodes()), default=0)
    return np.array([
        len(out_edges), len(in_edges),
        len({e[1] for e in out_edges}), len({e[0] for e in in_edges}),
        out_sum, out_max, out_ent,
        in_sum, in_max, in_ent,
        _entropy(np.asarray(gaps, dtype=float) + 1.0),   # 时间间隔熵
        g.number_of_nodes(), g.number_of_edges(),
        max_in, max_out,                                  # 图级扇入/扇出峰值
    ], dtype=float)


class GraphDistributionScore(SignalExtractor):
    """L1-j1：邻域图统计特征的 Isolation Forest 异常分。"""

    layer, angle, name = 1, 1, "l1j1_graph_iforest"

    def __init__(self, n_estimators: int = 100, random_state: int = 0):
        self.n_estimators = n_estimators
        self.random_state = random_state
        self._model = None
        self._mu: Optional[np.ndarray] = None
        self._sigma: Optional[np.ndarray] = None

    def fit(self, normal_contexts: list[TxContext]) -> None:
        feats = [f for f in (_graph_features(c) for c in normal_contexts)
                 if f is not None]
        if not feats:
            return
        X = np.vstack(feats)
        self._mu = X.mean(axis=0)
        self._sigma = X.std(axis=0) + 1e-9
        try:
            from sklearn.ensemble import IsolationForest
            self._model = IsolationForest(
                n_estimators=self.n_estimators,
                random_state=self.random_state).fit(X)
        except ImportError:
            self._model = None   # 退化为鲁棒 z 分数（下方 score 的兜底分支）

    def score(self, ctx: TxContext) -> Optional[float]:
        f = _graph_features(ctx)
        if f is None or self._mu is None:
            return None
        if self._model is not None:
            return float(-self._model.decision_function(f.reshape(1, -1))[0])
        z = np.abs((f - self._mu) / self._sigma)
        return float(np.sort(z)[-3:].mean())    # top-3 维度的平均偏离

    def evidence(self, ctx: TxContext) -> str:
        f = _graph_features(ctx)
        if f is None:
            return ""
        return (f"邻域图统计: 出边{int(f[0])}/入边{int(f[1])}, "
                f"对手方{int(f[2])}出/{int(f[3])}入, 节点{int(f[11])}")


class FundFlowScore(SignalExtractor):
    """L1-j2：资金流模式规则分，各子模式 ∈ [0,1] 加权求和。

    子模式：扇入聚集（多来源汇入单一地址）、环形转账（资金回流发起方）、
    多跳中转（FlowScope 式高容量路径）、金额首位数偏离 Benford。
    """

    layer, angle, name = 1, 2, "l1j2_fundflow_rules"

    def params(self, ctx: TxContext) -> dict[str, float]:
        g = ctx.ego_graph
        center = ctx.from_address
        if g is None or center not in g:
            return {}
        import networkx as nx
        dg = nx.DiGraph(g)   # 折叠多重边，便于路径分析
        pat: dict[str, float] = {}
        # 扇入聚集：图中任意节点从多少个不同来源收款（drainer/ponzi 合约的归集，
        # 通常不在中心节点上 → 取全图最大入度，改进 B）
        max_in = max((dg.in_degree(n) for n in dg.nodes()), default=0)
        pat["fan_in"] = min(1.0, max_in / 20.0)
        # 扇出分发：任意节点向多少个地址付款（ponzi 分红 / rug 分散）
        max_out = max((dg.out_degree(n) for n in dg.nodes()), default=0)
        pat["fan_out"] = min(1.0, max_out / 20.0)
        # 环形转账：从中心出发的后继能回到中心
        cyc = 0.0
        for succ in list(dg.successors(center))[:20]:
            if nx.has_path(dg, succ, center):
                cyc = 1.0
                break
        pat["cycle"] = cyc
        # 多跳中转：图中"纯中转"节点（入=出=1）占比
        relay = sum(1 for n in dg.nodes()
                    if dg.in_degree(n) == 1 and dg.out_degree(n) == 1)
        pat["relay"] = min(1.0, relay / max(dg.number_of_nodes(), 1))
        # Benford 首位数偏离（边金额）
        vals = [d.get("value", 0.0) for _, _, d in g.edges(data=True) if d.get("value", 0) > 0]
        if len(vals) >= 20:
            digits = Counter(int(str(f"{v:.6e}")[0]) for v in vals)
            obs = np.array([digits.get(d, 0) for d in range(1, 10)], dtype=float)
            obs = obs / obs.sum()
            pat["benford"] = float(min(1.0, np.abs(obs - _BENFORD).sum()))
        # 量值化参数：资金流总量、单边最大金额（log；ECDF 处理量纲）。use_magnitude 供消融
        if getattr(self, "use_magnitude", True) and vals:
            pat["total_flow_mag"] = float(np.log1p(sum(vals)))
            pat["max_edge_mag"] = float(np.log1p(max(vals)))

        # 跨交易上下文：三明治的"前后夹击"本质在**同区块的相邻交易**里，单笔交易看不见。
        # 由 experiments/augment_block_context.py 预填 latent["block_ctx"]；缺失则不产出，
        # 由掩码/参数池自然承接（异构可用性）。
        bc = ctx.latent.get("block_ctx")
        if bc:
            pat["same_sender_around"] = 1.0 if bc.get("same_sender_around") else 0.0
            pat["adjacent_same_target"] = 1.0 if bc.get("adjacent_same_target") else 0.0
            pat["same_target_around"] = float(min(1.0, bc.get("same_target_around", 0) / 2.0))
        # 强三明治判定（文献共识条件：同池 + 反向 + 金额链接 + 中间夹受害者）
        sw = ctx.latent.get("sandwich_ctx")
        if sw:
            pat.update({k: float(v) for k, v in sw.items()})
        return pat

    def score(self, ctx: TxContext) -> Optional[float]:
        # 独立调用时的兜底（加权和）；流水线中由 DictSignal 以 max 分位数聚合
        pat = self.params(ctx)
        if not pat:
            return None
        w = {"fan_in": 0.3, "fan_out": 0.2, "cycle": 0.2, "relay": 0.15, "benford": 0.15}
        return float(sum(w[k] * v for k, v in pat.items() if k in w))

    def evidence(self, ctx: TxContext) -> str:
        pat = self.params(ctx)
        zh = {"fan_in": "扇入聚集", "fan_out": "扇出分发", "cycle": "环形转账回流",
              "relay": "多跳中转节点", "benford": "金额分布偏离Benford"}
        hits = [zh[k] for k, v in sorted(pat.items(), key=lambda x: -x[1])
                if k in zh and v > 0.3]
        return "资金流模式: " + ("、".join(hits) if hits else "无显著模式")
