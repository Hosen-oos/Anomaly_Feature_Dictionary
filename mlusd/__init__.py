"""mlusd — Multi-Layer Unified Signal Detection（研究内容二）。

端到端用法见 mlusd.pipeline.Detector 与 mlusd.signals.factory.default_extractors。
"""
from mlusd.types import (
    DetectionReport, TxContext, Verdict,
)

__all__ = ["DetectionReport", "TxContext", "Verdict"]
__version__ = "0.1.0"
