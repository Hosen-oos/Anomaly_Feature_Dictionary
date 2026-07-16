"""M2 信号提取器。default_extractors() 返回 8 个 v0 提取器实例。"""
from mlusd.signals.base import SignalExtractor
from mlusd.signals.factory import default_extractors

__all__ = ["SignalExtractor", "default_extractors"]
