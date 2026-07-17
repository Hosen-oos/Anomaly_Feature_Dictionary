"""TxContext → 扁平特征向量（基线方法用）。

对比要点：扁平特征法**被迫对缺失层填零**（异构可用性下丢信息），这正是本框架用
可用性掩码 + 分组校准所避免的。基线在这些特征上跑通用检测器，与本框架同数据同任务对比。
"""
from __future__ import annotations

from collections import Counter

import numpy as np

from mlusd.signals.l2_semantic import lift_actions

_ACTION_KINDS = ["transfer", "approve", "swap", "borrow", "repay", "flashloan",
                 "add_liquidity", "remove_liquidity", "deposit", "withdraw", "liquidate"]

FEATURE_NAMES = (
    ["log_value"]
    + [f"act_{k}" for k in _ACTION_KINDS]
    + ["n_logs", "n_calls", "max_depth", "sstore", "reverted"]
    + ["g_nodes", "g_edges", "c_out", "c_in", "g_max_in", "g_max_out"]
    + ["m1", "m2", "m3", "m4", "n_labels"]
)


def flat_features(ctx) -> np.ndarray:
    import networkx as nx
    f = [np.log1p(float(ctx.value))]
    acts = Counter(a.kind for a in lift_actions(ctx))
    f += [float(acts.get(k, 0)) for k in _ACTION_KINDS]
    f.append(float(len(ctx.event_logs or [])))
    t = ctx.trace
    f += [float(len(t.calls)) if t else 0.0,
          float(t.max_depth) if t else 0.0,
          float(t.sstore_count) if t else 0.0,
          float(t.reverted_subcalls) if t else 0.0]
    g = ctx.ego_graph
    if g is not None and g.number_of_nodes() > 0:
        dg = nx.DiGraph(g)
        c = ctx.from_address
        f += [float(g.number_of_nodes()), float(g.number_of_edges()),
              float(dg.out_degree(c)) if c in dg else 0.0,
              float(dg.in_degree(c)) if c in dg else 0.0,
              float(max((dg.in_degree(n) for n in dg.nodes()), default=0)),
              float(max((dg.out_degree(n) for n in dg.nodes()), default=0))]
    else:
        f += [0.0] * 6
    f += [float(b) for b in ctx.availability]
    off = ctx.offchain
    f.append(float(len(off.label_hits)) if off else 0.0)
    return np.asarray(f, dtype=float)


def feature_matrix(contexts) -> np.ndarray:
    return np.vstack([flat_features(c) for c in contexts]) if contexts else np.empty((0, len(FEATURE_NAMES)))
