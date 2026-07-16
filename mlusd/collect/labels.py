"""链下标签库（设计架构 §4 M1：统一成 (address,label,source,date) 表）。

来源：Etherscan 标签、CryptoScamDB、慢雾/PeckShield 事件库、研究内容一种子样本。
v0 只做结构化"有/无风险记录"匹配，不做 NLP 舆情（§1 明确列为扩展项）。
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Optional

from mlusd.types import OffchainRecord

# 标签 -> 严重度（0-1），用于 L4-j3 评分
_SEVERITY = {
    "phishing": 0.9, "fake_phishing": 0.9, "scam": 0.85, "exploit": 0.9,
    "hack": 0.9, "heist": 0.9, "rugpull": 0.85, "ponzi": 0.8,
    "mixer": 0.5, "sanctioned": 0.7, "high_risk": 0.6,
}


def _severity(label: str) -> float:
    key = label.lower().replace(" ", "_")
    for k, v in _SEVERITY.items():
        if k in key:
            return v
    return 0.4   # 未知风险标签的保守严重度


class LabelStore:
    """地址 -> 标签命中列表；合约 -> 是否已验证/已审计。"""

    def __init__(self):
        self._labels: dict[str, list[dict]] = {}
        self._verified: dict[str, bool] = {}
        self._audited: dict[str, bool] = {}

    def add_label(self, address: str, label: str, source: str,
                  severity: Optional[float] = None) -> None:
        addr = address.lower()
        self._labels.setdefault(addr, []).append({
            "address": addr, "label": label, "source": source,
            "severity": severity if severity is not None else _severity(label),
        })

    def set_contract_meta(self, address: str, verified: Optional[bool] = None,
                          audited: Optional[bool] = None) -> None:
        addr = address.lower()
        if verified is not None:
            self._verified[addr] = verified
        if audited is not None:
            self._audited[addr] = audited

    def lookup(self, addresses: list[str],
               contract: Optional[str] = None) -> Optional[OffchainRecord]:
        """汇总涉及地址的标签命中与合约元信息。全部无记录时返回 None（m4=0）。"""
        hits: list[dict] = []
        for a in addresses:
            hits.extend(self._labels.get(a.lower(), []))
        verified = audited = None
        if contract is not None:
            c = contract.lower()
            verified = self._verified.get(c)
            audited = self._audited.get(c)
        if not hits and verified is None and audited is None:
            return None
        return OffchainRecord(label_hits=hits, contract_verified=verified,
                              audited=audited)

    # ---------------------------------------------------------- 加载器

    @classmethod
    def from_csv(cls, path: str | Path) -> "LabelStore":
        """CSV 列: address,label,source[,severity]。"""
        store = cls()
        with open(path, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                sev = row.get("severity")
                store.add_label(row["address"], row["label"], row.get("source", ""),
                                float(sev) if sev else None)
        return store

    @classmethod
    def from_json(cls, path: str | Path) -> "LabelStore":
        """JSON: {"labels":[{address,label,source,severity}],
                  "contracts":[{address,verified,audited}]}。"""
        store = cls()
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        for r in data.get("labels", []):
            store.add_label(r["address"], r["label"], r.get("source", ""),
                            r.get("severity"))
        for c in data.get("contracts", []):
            store.set_contract_meta(c["address"], c.get("verified"), c.get("audited"))
        return store
