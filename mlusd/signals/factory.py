"""组装 8 个 v0 信号提取器（设计架构 §3.2 的有效信号位置表）。"""
from __future__ import annotations

from mlusd.signals.base import SignalExtractor
from mlusd.signals.l1_graph import FundFlowScore, GraphDistributionScore
from mlusd.signals.l2_semantic import (
    ActionSequenceRarity, ApprovalMismatchScore, EconomicAnomalyScore,
)
from mlusd.signals.l3_trace import TraceNgramRarity, TracePropertyScore
from mlusd.signals.l4_offchain import OffchainConsistencyScore
from mlusd.signals.pool import DictSignal


def default_extractors(magnitude: bool = True) -> list[SignalExtractor]:
    """8 个有效信号位置的 v0 提取器，覆盖 (l,j) ∈ VALID_POSITIONS。

    多参数规则格（L1-j2/L2-j2/L2-j3/L3-j3）用 DictSignal 做非稀释 max 分位数聚合
    （方案定稿 §2.1）；单参数学习格（L1-j1/L2-j1/L3-j1）与 L4-j3 直接用其提取器。
    magnitude=False 关闭量值化参数（供消融）。
    """
    ff, econ = FundFlowScore(), EconomicAnomalyScore()
    ff.use_magnitude = magnitude
    econ.use_magnitude = magnitude
    return [
        GraphDistributionScore(),          # L1-j1（单参数：IForest）
        DictSignal(ff),                    # L1-j2（参数池）
        ActionSequenceRarity(),            # L2-j1（单参数：n-gram）
        DictSignal(econ),                  # L2-j2（参数池）
        DictSignal(ApprovalMismatchScore()),  # L2-j3（参数池）
        TraceNgramRarity(),                # L3-j1（单参数：n-gram）
        DictSignal(TracePropertyScore()),  # L3-j3（参数池）
        DictSignal(OffchainConsistencyScore()),  # L4-j3（参数池：标签 + 合约结构性信号）
    ]


def ablation_extractors(cell_agg: str = "max") -> list[SignalExtractor]:
    """消融用：cell_agg='mean' 让参数池格用均值聚合（稀释对照），其余同 default。"""
    return [
        GraphDistributionScore(),
        DictSignal(FundFlowScore(), agg=cell_agg),
        ActionSequenceRarity(),
        DictSignal(EconomicAnomalyScore(), agg=cell_agg),
        DictSignal(ApprovalMismatchScore(), agg=cell_agg),
        TraceNgramRarity(),
        DictSignal(TracePropertyScore(), agg=cell_agg),
        DictSignal(OffchainConsistencyScore(), agg=cell_agg),
    ]


def v1_extractors(**nn_kwargs) -> list[SignalExtractor]:
    """v1：把 L1-j1/L2-j1/L3-j1 换成学习模型，其余保持 v0 规则版。

    需要 torch（延迟 import，v0 用户无需安装）。三个学习位置统一为
    "正常数据似然模型，异常分=重构误差/困惑度"。nn_kwargs 透传给三个学习提取器
    （如 epochs、d_model），便于实验三的 v0-vs-v1 消融与超参扫描。
    """
    from mlusd.signals.l1_graph_v1 import GraphAutoencoder
    from mlusd.signals.l2_semantic_v1 import ActionSequenceTransformer
    from mlusd.signals.l3_trace_v1 import TraceTransformer
    return [
        GraphAutoencoder(),                       # L1-j1 (v1)
        FundFlowScore(),                          # L1-j2
        ActionSequenceTransformer(**nn_kwargs),   # L2-j1 (v1)
        EconomicAnomalyScore(),                   # L2-j2
        ApprovalMismatchScore(),                  # L2-j3
        TraceTransformer(**nn_kwargs),            # L3-j1 (v1)
        TracePropertyScore(),                     # L3-j3
        OffchainConsistencyScore(),               # L4-j3
    ]
