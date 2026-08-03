"""核心数据结构：所有模块间的契约（设计架构 §3）。

信号矩阵约定：S 为 4x3 numpy 数组（层 x 观测角度，0-indexed 存储、1-indexed 论述），
无效/缺失位置为 NaN。层级可用性掩码 m 为长度 4 的 0/1 元组。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

import numpy as np

N_LAYERS = 4
N_ANGLES = 3

# 8 个有效信号位置 (layer, angle)，1-indexed。见设计架构 §3.2 的表格。
VALID_POSITIONS: tuple[tuple[int, int], ...] = (
    (1, 1), (1, 2),
    (2, 1), (2, 2), (2, 3),
    (3, 1), (3, 3),
    (4, 3),
)

LAYER_NAMES = {1: "交易图拓扑层", 2: "合约交互语义层", 3: "EVM执行轨迹层", 4: "链下情报层"}
ANGLE_NAMES = {1: "分布偏离", 2: "经济异常", 3: "信息差异"}


# ---------------------------------------------------------------- 链上原始结构

@dataclass
class DecodedLog:
    """解码后的事件日志。event 用标准名（Transfer/Swap/Approval/Borrow/...）。"""
    address: str                  # 发出事件的合约
    event: str
    args: dict[str, Any] = field(default_factory=dict)


@dataclass
class Call:
    """内部调用树中的一条调用记录（按执行顺序展平）。"""
    frm: str
    to: str
    kind: str                     # CALL / DELEGATECALL / STATICCALL / CREATE
    depth: int
    value: int = 0
    func: Optional[str] = None    # 函数选择器或已解析函数名
    reverted: bool = False


@dataclass
class TraceSummary:
    """EVM 执行轨迹摘要（来自 debug_traceTransaction）。"""
    calls: list[Call] = field(default_factory=list)
    max_depth: int = 0
    sstore_count: int = 0
    reverted_subcalls: int = 0
    opcodes: Optional[list[str]] = None            # 可选的粗粒度 opcode 序列
    storage_writes: dict[str, int] = field(default_factory=dict)  # 合约 -> 写入次数


@dataclass
class OffchainRecord:
    """链下情报：标签命中、项目背景与合约结构性元信息。

    contract_meta 为**不泄漏**的通用属性（部署时长/字节码规模/标准符合性），
    区别于恶意地址标签——后者在离线评测中要么覆盖为 0，要么构成标签泄漏。
    """
    label_hits: list[dict] = field(default_factory=list)  # {address,label,source,severity(0-1)}
    contract_verified: Optional[bool] = None
    audited: Optional[bool] = None
    # {age_days: float, bytecode_len: int, is_token: bool}
    contract_meta: dict = field(default_factory=dict)


@dataclass
class DeFiAction:
    """语义提升器输出的高层 DeFi 动作（ActLifter/DeFiRanger 风格）。"""
    kind: str                     # transfer/swap/borrow/repay/flashloan/approve/
    #                               add_liquidity/remove_liquidity/deposit/withdraw/liquidate
    actor: str = ""
    protocol: str = ""            # 相关协议合约
    token_in: str = ""
    token_out: str = ""
    amount_in: float = 0.0
    amount_out: float = 0.0


# ---------------------------------------------------------------- 交易上下文

@dataclass
class TxContext:
    """M1 输出：一笔待检测交易及其全部可用上下文。"""
    tx_hash: str
    from_address: str = ""
    to_address: str = ""
    block_number: int = 0
    tx_index: int = 0             # 区块内位置（三明治结构判断用）
    timestamp: int = 0
    value: int = 0
    status: bool = True           # 顶层交易是否成功

    ego_graph: Any = None         # networkx.MultiDiGraph | None（k 跳邻域）
    input_data: Optional[str] = None
    event_logs: Optional[list[DecodedLog]] = None
    internal_calls: Optional[list[Call]] = None
    trace: Optional[TraceSummary] = None
    offchain: Optional[OffchainRecord] = None

    # 测试/合成数据可直接放置隐变量，正式流程不使用
    latent: dict = field(default_factory=dict)

    @property
    def availability(self) -> tuple[int, int, int, int]:
        """层级可用性向量 m（设计架构 §3.1 的规则）。"""
        m1 = 1  # 任何交易都有交易图邻域
        m2 = 1 if (self.event_logs or self.internal_calls) else 0
        m3 = 1 if (self.trace is not None and (self.trace.max_depth > 1 or self.trace.sstore_count > 0)) else 0
        m4 = 1 if (self.offchain is not None and (
            self.offchain.label_hits
            or self.offchain.contract_verified is not None
            or self.offchain.audited is not None
            or self.offchain.contract_meta)) else 0   # 含合约结构性元信息（L4 广义化）
        return (m1, m2, m3, m4)


# ---------------------------------------------------------------- 检测输出

class Verdict(str, Enum):
    KNOWN = "KNOWN"
    UNKNOWN = "UNKNOWN"
    INSUFFICIENT = "INSUFFICIENT"
    NORMAL = "NORMAL"


@dataclass
class Contribution:
    layer: int
    angle: int
    value: float
    evidence: str = ""

    def to_dict(self) -> dict:
        return {
            "layer": f"L{self.layer}",
            "layer_name": LAYER_NAMES[self.layer],
            "angle": ANGLE_NAMES[self.angle],
            "value": round(float(self.value), 4),
            "evidence": self.evidence,
        }


@dataclass
class MatchResult:
    attack_type: str
    base_score: float
    coverage: float
    final_score: float
    match_threshold: float
    coverage_threshold: float
    contributions: list[Contribution] = field(default_factory=list)


@dataclass
class DetectionReport:
    tx_hash: str
    verdict: Verdict
    availability_group: str
    known_type: Optional[str] = None
    match_score: float = 0.0
    evidence_coverage: float = 0.0
    unknown_score: float = 0.0          # Ū，组内相对整体异常程度
    contributions: list[Contribution] = field(default_factory=list)
    missing_evidence: list[str] = field(default_factory=list)
    all_matches: list[MatchResult] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "tx_hash": self.tx_hash,
            "verdict": self.verdict.value,
            "known_type": self.known_type,
            "match_score": round(float(self.match_score), 4),
            "evidence_coverage": round(float(self.evidence_coverage), 4),
            "unknown_score": round(float(self.unknown_score), 4),
            "availability_group": self.availability_group,
            "contributions": [c.to_dict() for c in self.contributions],
            "missing_evidence": self.missing_evidence,
        }


def empty_signal_matrix() -> np.ndarray:
    """全 NaN 的 4x3 信号矩阵。"""
    return np.full((N_LAYERS, N_ANGLES), np.nan)
