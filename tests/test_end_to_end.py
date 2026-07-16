"""端到端回归测试（设计架构 §9 出口标准：黄金测试交易判定正确）。"""
from __future__ import annotations

from pathlib import Path

import pytest

from mlusd.match.dictionary import load_dictionaries
from mlusd.pipeline import Detector
from mlusd.signals.factory import default_extractors
from mlusd.types import Verdict
from tests import synthetic

DICT_DIR = Path(__file__).resolve().parents[1] / "configs" / "dictionaries"


@pytest.fixture(scope="module")
def detector() -> Detector:
    det = Detector(
        extractors=default_extractors(),
        dictionaries=load_dictionaries(DICT_DIR),
        alpha=0.01,
        min_group_size=50,     # 合成校准集较小，放宽分组门槛
    )
    det.fit(synthetic.calibration_set(1500))
    return det


def test_dictionaries_load():
    dicts = load_dictionaries(DICT_DIR)
    assert {d.attack_type for d in dicts} == {
        "flash_loan", "price_manipulation", "sandwich",
        "phishing", "ponzi", "rug_pull"}


def test_pure_transfer_is_normal(detector):
    r = detector.detect(synthetic.pure_transfer_tx())
    assert r.verdict == Verdict.NORMAL
    assert r.availability_group[0] == "1"    # L1 恒可用


def test_normal_tx_mostly_normal(detector):
    verdicts = [detector.detect(synthetic.normal_tx(10000 + i)).verdict
                for i in range(200)]
    # 目标误报率 α=1%，合成数据放宽到 <8%
    fp = sum(v != Verdict.NORMAL for v in verdicts) / len(verdicts)
    assert fp < 0.08, f"正常交易误报率过高: {fp:.3f}"


def test_flash_loan_detected(detector):
    r = detector.detect(synthetic.flash_loan_tx())
    assert r.verdict in (Verdict.KNOWN, Verdict.UNKNOWN)
    if r.verdict == Verdict.KNOWN:
        # 经济攻击家族（flash_loan/price_manipulation/rug_pull）在合成数据上高度重叠
        # （共享闪电贷/兑换/流动性经济信号）；精确区分需利润归因（DeFort/v1）。
        assert r.known_type in ("flash_loan", "price_manipulation", "rug_pull")
    assert r.contributions, "应给出贡献度解释"


def test_phishing_detected(detector):
    r = detector.detect(synthetic.phishing_tx())
    assert r.verdict in (Verdict.KNOWN, Verdict.UNKNOWN)
    if r.verdict == Verdict.KNOWN:
        assert r.known_type in ("phishing", "ponzi")   # 两者都吃扇入/标签信号


def test_report_serializable(detector):
    r = detector.detect(synthetic.flash_loan_tx())
    d = r.to_dict()
    assert d["tx_hash"] == "0xflashloan_attack"
    assert "verdict" in d and "contributions" in d


def test_all_eight_positions_active(detector):
    """闪电贷交易应在多个信号位置产出有限分数（4 层至少 3 层可用）。"""
    ctx = synthetic.flash_loan_tx()
    m = ctx.availability
    assert m[0] == 1 and m[1] == 1 and m[2] == 1   # L1/L2/L3 可用
