"""L1-j1 v1：ego 图自编码器重构误差（设计架构 §5 v1，AnomalyDAE 风格）。

v0 是"图统计特征 + IsolationForest"；v1 换成图自编码器：GCN 编码 + 结构解码
（内积重构邻接，GAE）+ 属性解码（重构节点特征，AnomalyDAE 的双重构），异常分 =
中心节点的重构误差。纯 torch 实现，不依赖 PyG/PyGOD（选型池见 I72/I74）；
真实数据阶段可替换为 PyGOD 的 AnomalyDAE/GGAD 官方实现做对照。
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from mlusd.signals.base import SignalExtractor
from mlusd.types import TxContext

_FEAT_DIM = 5


def _graph_tensors(ctx: TxContext):
    """ego 图 → (节点特征 X[n,F], 邻接 A[n,n], 中心节点下标)。"""
    g = ctx.ego_graph
    center = ctx.from_address
    if g is None or g.number_of_nodes() == 0 or center not in g:
        return None
    import networkx as nx
    dg = nx.DiGraph(g)
    nodes = list(dg.nodes())
    idx = {u: i for i, u in enumerate(nodes)}
    n = len(nodes)
    X = np.zeros((n, _FEAT_DIM), dtype=np.float32)
    for u in nodes:
        i = idx[u]
        out_v = float(sum(d.get("value", 0.0) for _, _, d in g.out_edges(u, data=True)))
        in_v = float(sum(d.get("value", 0.0) for _, _, d in g.in_edges(u, data=True)))
        X[i] = [dg.out_degree(u), dg.in_degree(u),
                np.log1p(out_v), np.log1p(in_v), 1.0 if u == center else 0.0]
    A = np.zeros((n, n), dtype=np.float32)
    for u, v in dg.edges():
        A[idx[u], idx[v]] = 1.0
    return X, A, idx[center]


def _norm_adj(np_A):
    n = np_A.shape[0]
    A = np_A + np.eye(n, dtype=np.float32)
    d = A.sum(axis=1)
    dinv = np.zeros_like(d)
    nz = d > 0
    dinv[nz] = np.power(d[nz], -0.5)
    return (dinv[:, None] * A) * dinv[None, :]


class GraphAutoencoder(SignalExtractor):
    layer, angle, name = 1, 1, "l1j1_graph_ae"

    def __init__(self, hidden: int = 16, emb: int = 8, epochs: int = 3,
                 lr: float = 1e-2, max_graphs: int = 800, seed: int = 0,
                 alpha: float = 0.7):
        self.hidden, self.emb = hidden, emb
        self.epochs, self.lr, self.max_graphs = epochs, lr, max_graphs
        self.seed, self.alpha = seed, alpha
        self._model = None
        self._mu = None
        self._sigma = None

    def _standardize(self, X):
        return (X - self._mu) / self._sigma

    def fit(self, normal_contexts: list[TxContext]) -> None:
        import torch
        import torch.nn as nn
        torch.manual_seed(self.seed)
        graphs = [t for t in (_graph_tensors(c) for c in normal_contexts) if t]
        if not graphs:
            return
        allX = np.vstack([X for X, _, _ in graphs])
        self._mu = allX.mean(axis=0)
        self._sigma = allX.std(axis=0) + 1e-6
        rng = np.random.default_rng(self.seed)
        if len(graphs) > self.max_graphs:
            graphs = [graphs[i] for i in rng.choice(len(graphs), self.max_graphs, replace=False)]

        class GAE(nn.Module):
            def __init__(s, fdim, h, e):
                super().__init__()
                s.w1 = nn.Linear(fdim, h)
                s.w2 = nn.Linear(h, e)
                s.dec = nn.Linear(e, fdim)   # 属性解码

            def encode(s, Xn, An):
                h = torch.relu(An @ s.w1(Xn))
                return An @ s.w2(h)

            def forward(s, Xn, An):
                z = s.encode(Xn, An)
                A_hat = torch.sigmoid(z @ z.t())      # 结构解码（GAE 内积）
                X_hat = s.dec(z)                       # 属性解码
                return A_hat, X_hat

        self._model = GAE(_FEAT_DIM, self.hidden, self.emb)
        opt = torch.optim.Adam(self._model.parameters(), lr=self.lr)
        self._model.train()
        for _ in range(self.epochs):
            for X, A, _ in graphs:
                Xn = torch.tensor(self._standardize(X))
                An = torch.tensor(_norm_adj(A))
                At = torch.tensor(A)
                A_hat, X_hat = self._model(Xn, An)
                loss = (self.alpha * nn.functional.mse_loss(A_hat, At)
                        + (1 - self.alpha) * nn.functional.mse_loss(X_hat, Xn))
                opt.zero_grad(); loss.backward(); opt.step()
        self._model.eval()

    def score(self, ctx: TxContext) -> Optional[float]:
        if self._model is None:
            return None
        t = _graph_tensors(ctx)
        if t is None:
            return None
        import torch
        X, A, c = t
        with torch.no_grad():
            Xn = torch.tensor(self._standardize(X))
            An = torch.tensor(_norm_adj(A))
            A_hat, X_hat = self._model(Xn, An)
            # 中心节点重构误差 = 结构行误差 + 属性误差
            struct_err = float(((A_hat[c] - torch.tensor(A[c])) ** 2).mean())
            attr_err = float(((X_hat[c] - Xn[c]) ** 2).mean())
        return self.alpha * struct_err + (1 - self.alpha) * attr_err

    def evidence(self, ctx: TxContext) -> str:
        return "图自编码器重构误差 (v1)"
