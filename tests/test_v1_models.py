"""v1 学习提取器测试（torch 缺失时自动跳过）。

验证三点：(1) 能在正常合成数据上训练并打分；(2) 攻击样本的异常分高于正常样本
（模型学到了正常分布）；(3) v1 提取器能整体接入 Detector 端到端出结果。
"""
from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("torch")   # 无 torch 则跳过整个模块

from mlusd.match.dictionary import load_dictionaries        # noqa: E402
from mlusd.pipeline import Detector                          # noqa: E402
from mlusd.signals.factory import v1_extractors              # noqa: E402
from mlusd.signals.l1_graph_v1 import GraphAutoencoder       # noqa: E402
from mlusd.signals.l2_semantic_v1 import ActionSequenceTransformer  # noqa: E402
from tests import synthetic                                  # noqa: E402

DICT_DIR = Path(__file__).resolve().parents[1] / "configs" / "dictionaries"


def test_action_transformer_trains_and_scores():
    ext = ActionSequenceTransformer(epochs=5)
    ext.fit(synthetic.calibration_set(400))
    s = ext.score(synthetic.flash_loan_tx())
    assert s is not None and s > 0


def test_graph_ae_higher_on_anomaly():
    ext = GraphAutoencoder(epochs=3, max_graphs=300)
    ext.fit(synthetic.calibration_set(400))
    normal = [ext.score(synthetic.normal_tx(50000 + i)) for i in range(30)]
    normal = [x for x in normal if x is not None]
    anom = ext.score(synthetic.fan_in_anomaly_tx())   # 极端扇入，结构异常
    assert anom is not None and normal
    # 结构异常图的中心节点重构误差应高于正常样本中位数
    normal.sort()
    assert anom > normal[len(normal) // 2]


def test_v1_extractors_flow_through_detector():
    det = Detector(v1_extractors(epochs=4),
                   load_dictionaries(DICT_DIR), alpha=0.01, min_group_size=50)
    det.fit(synthetic.calibration_set(500))
    r = det.detect(synthetic.flash_loan_tx())
    assert r.verdict.value in ("KNOWN", "UNKNOWN", "INSUFFICIENT", "NORMAL")
    assert r.to_dict()["tx_hash"] == "0xflashloan_attack"
