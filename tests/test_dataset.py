"""数据集构建流水线测试（用 MockSource，无需 key）。"""
from __future__ import annotations

import csv

from mlusd.dataset.build import (
    build_contexts, load_attack_hashes, load_contexts, save_contexts, _manifest,
)
from tests import raw_fixtures as rf


def test_build_contexts_batch_skips_missing():
    hashes = [rf.PHISHING_HASH, "0xmissing", rf.NORMAL_HASH]
    ctxs = build_contexts(hashes, rf.mock_source())
    assert len(ctxs) == 2                      # 缺失的被跳过，不中断
    assert {c.tx_hash for c in ctxs} == {rf.PHISHING_HASH, rf.NORMAL_HASH}


def test_save_load_roundtrip(tmp_path):
    ctxs = build_contexts([rf.PHISHING_HASH], rf.mock_source())
    p = tmp_path / "d.pkl.gz"
    save_contexts(ctxs, p)
    loaded = load_contexts(p)
    assert len(loaded) == 1
    assert loaded[0].tx_hash == rf.PHISHING_HASH
    assert loaded[0].event_logs is not None     # networkx 图/日志 pickle 保真


def test_load_attack_hashes(tmp_path):
    p = tmp_path / "attacks.csv"
    with open(p, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["tx_hash", "attack_type", "event_name"])
        w.writerow(["0xABC", "flash_loan", "FlashLoan"])
    rows = load_attack_hashes(p)
    assert rows[0]["tx_hash"] == "0xabc"        # 归一为小写
    assert rows[0]["attack_type"] == "flash_loan"


def test_manifest_reports_availability_groups():
    ctxs = build_contexts([rf.PHISHING_HASH, rf.NORMAL_HASH], rf.mock_source())
    m = _manifest(ctxs)
    assert m["n"] == 2
    assert sum(m["availability_groups"].values()) == 2
