"""M2 信号提取器统一接口（设计架构 §4 M2）。

纪律：fit 只允许使用正常交易校准集；异常标签只进入字典（M4）和评测。
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

from mlusd.types import TxContext


class SignalExtractor(ABC):
    """单个信号位置 (layer, angle) 的提取器。score 越大越异常。"""

    layer: int = 0
    angle: int = 0
    name: str = "base"

    def fit(self, normal_contexts: list[TxContext]) -> None:  # noqa: B027
        """在正常交易校准集上拟合。默认无参数，规则型提取器不需要重写。"""

    @abstractmethod
    def score(self, ctx: TxContext) -> Optional[float]:
        """原始异常分数；该层数据缺失时返回 None。"""

    def evidence(self, ctx: TxContext) -> str:
        """人类可读的证据片段，供解释输出。"""
        return ""
