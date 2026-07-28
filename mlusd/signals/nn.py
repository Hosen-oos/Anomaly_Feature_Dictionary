"""v1 学习提取器的共享组件：微型自回归 Transformer 语言模型（设计架构 §5 v1）。

统一范式：只在正常校准集上训练，异常分 = 序列困惑度（perplexity）。L2-j1（DeFi
动作序列）与 L3-j1（调用树 token 序列）共用本模块，仅 tokenizer 不同——对应
BERT4ETH（C16，行为序列表征）与 BlockGPT（trace 树语言模型）两条参考线的轻量实现。

torch 为可选依赖：本模块被 factory.v1_extractors() 延迟 import，v0 用户无需安装。
"""
from __future__ import annotations

from typing import Callable, Optional

import numpy as np

from mlusd.signals.base import SignalExtractor
from mlusd.types import TxContext

PAD, BOS, EOS, UNK = "<pad>", "<s>", "</s>", "<unk>"


def _lazy_torch():
    import torch  # noqa: F401
    import torch.nn as nn
    return torch, nn


class _Vocab:
    def __init__(self, tokens: list[str]):
        base = [PAD, BOS, EOS, UNK]
        uniq = sorted({t for t in tokens if t not in base})
        self.itos = base + uniq
        self.stoi = {t: i for i, t in enumerate(self.itos)}

    def __len__(self) -> int:
        return len(self.itos)

    def encode(self, seq: list[str]) -> list[int]:
        unk = self.stoi[UNK]
        return ([self.stoi[BOS]]
                + [self.stoi.get(t, unk) for t in seq]
                + [self.stoi[EOS]])


def _build_model(nn, vocab_size: int, d_model: int, n_head: int,
                 n_layer: int, block_size: int):
    class TinyLM(nn.Module):
        def __init__(self):
            super().__init__()
            self.tok = nn.Embedding(vocab_size, d_model)
            self.pos = nn.Embedding(block_size, d_model)
            layer = nn.TransformerEncoderLayer(
                d_model=d_model, nhead=n_head, dim_feedforward=d_model * 4,
                batch_first=True, dropout=0.0)
            self.enc = nn.TransformerEncoder(layer, num_layers=n_layer)
            self.ln = nn.LayerNorm(d_model)
            self.head = nn.Linear(d_model, vocab_size)
            self.block_size = block_size

        def forward(self, idx):
            import torch
            b, t = idx.shape
            pos = torch.arange(t, device=idx.device).unsqueeze(0)
            x = self.tok(idx) + self.pos(pos)
            mask = torch.triu(torch.ones(t, t, device=idx.device), diagonal=1).bool()
            x = self.enc(x, mask=mask)
            return self.head(self.ln(x))
    return TinyLM()


class SequenceLMExtractor(SignalExtractor):
    """在 token 序列上训练微型 LM，score = 困惑度。子类只需给出 tokens_of。"""

    layer, angle, name = 0, 0, "seq_lm"

    def __init__(self, tokens_of: Callable[[TxContext], list[str]],
                 d_model: int = 64, n_head: int = 2, n_layer: int = 2,
                 block_size: int = 48, epochs: int = 8, lr: float = 3e-3,
                 batch_size: int = 128, seed: int = 0, device: str = "auto"):
        self.tokens_of = tokens_of
        self.d_model, self.n_head, self.n_layer = d_model, n_head, n_layer
        self.block_size, self.epochs, self.lr = block_size, epochs, lr
        self.batch_size, self.seed = batch_size, seed
        self.device = device            # "auto"|"cuda"|"cpu"
        self._model = None
        self._vocab: Optional[_Vocab] = None

    def _dev(self):
        import torch
        if self.device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return self.device

    # ------------------------------------------------------------- 训练

    def fit(self, normal_contexts: list[TxContext]) -> None:
        torch, nn = _lazy_torch()
        torch.manual_seed(self.seed)
        seqs = [self.tokens_of(c) for c in normal_contexts]
        seqs = [s for s in seqs if s]
        if not seqs:
            return
        flat = [t for s in seqs for t in s]
        self._vocab = _Vocab(flat)
        encoded = [self._vocab.encode(s)[:self.block_size] for s in seqs]

        dev = self._dev()
        self._model = _build_model(nn, len(self._vocab), self.d_model,
                                   self.n_head, self.n_layer, self.block_size).to(dev)
        opt = torch.optim.Adam(self._model.parameters(), lr=self.lr)
        pad = self._vocab.stoi[PAD]
        self._model.train()
        rng = np.random.default_rng(self.seed)
        for _ in range(self.epochs):
            order = rng.permutation(len(encoded))
            for i in range(0, len(order), self.batch_size):
                batch = [encoded[j] for j in order[i:i + self.batch_size]]
                x, y = self._to_xy(torch, batch, pad)
                x, y = x.to(dev), y.to(dev)
                logits = self._model(x)
                loss = nn.functional.cross_entropy(
                    logits.reshape(-1, logits.size(-1)), y.reshape(-1),
                    ignore_index=pad)
                opt.zero_grad(); loss.backward(); opt.step()
        self._model.eval()

    def _to_xy(self, torch, batch, pad):
        maxlen = max(len(s) for s in batch)
        x = torch.full((len(batch), maxlen - 1), pad, dtype=torch.long)
        y = torch.full((len(batch), maxlen - 1), pad, dtype=torch.long)
        for r, s in enumerate(batch):
            t = torch.tensor(s, dtype=torch.long)
            x[r, :len(s) - 1] = t[:-1]
            y[r, :len(s) - 1] = t[1:]
        return x, y

    # ------------------------------------------------------------- 打分

    def score(self, ctx: TxContext) -> Optional[float]:
        if self._model is None or self._vocab is None:
            return None
        seq = self.tokens_of(ctx)
        if not seq:
            return None
        torch, nn = _lazy_torch()
        ids = self._vocab.encode(seq)[:self.block_size]
        if len(ids) < 2:
            return None
        dev = self._dev()
        with torch.no_grad():
            x = torch.tensor(ids[:-1], dtype=torch.long).unsqueeze(0).to(dev)
            y = torch.tensor(ids[1:], dtype=torch.long).unsqueeze(0).to(dev)
            logits = self._model(x)
            ce = nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        return float(torch.exp(ce))     # 困惑度，越大越异常
