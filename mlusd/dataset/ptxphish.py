"""PTXPHISH 数据集加载（NDSS 2025《Dissecting Payload-based Transaction Phishing on Ethereum》）。

xlsx 布局（已逐列核实）：
  行2/行3 = 大类分组（Exploiting legitimate contracts / Deploying phishing contracts；
            Ice phishing scam / NFT order scam / address poisoning scam / payable function scam）
  行4     = 具体手法列名，其后紧跟 'Source' 列（Twitter 证据链接）
  行5起   = 交易哈希
  **末两列 Benign KOL / Benign Developer 是良性对照，不是攻击**（曾误解，已核实修正）

规模：钓鱼攻击 4,998 笔（11 种手法子类型）+ 良性对照 13,557 笔。
- 攻击样本用于扩容 D_known 的 phishing（研究一种子仅 2625，且无手法级标注）；
- 良性样本是**硬负样本**（形似钓鱼的合法交易：KOL/开发者），用于更锐利的校准。
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path

DEFAULT_PATH = Path(r"D:\实验\研究一\data-collect\ptxphish\PTXPHISH.xlsx")

# 行4 列名 -> (归一化子类型, 是否为攻击)
COLUMN_MAP: dict[str, tuple[str, bool]] = {
    "approve":            ("ice_phishing_approve", True),
    "permit":             ("ice_phishing_permit", True),
    "setapproveforall":   ("ice_phishing_setapprovalforall", True),
    "bulk transfer":      ("nft_order_bulk_transfer", True),
    "proxy upgrade":      ("phishing_proxy_upgrade", True),
    "free buy order":     ("nft_order_free_buy", True),
    "zero value transfer": ("address_poisoning_zero_value", True),
    "fake token transfer": ("address_poisoning_fake_token", True),
    "dust value transfer": ("address_poisoning_dust_value", True),
    "airdrop function":   ("payable_airdrop_function", True),
    "wallet function":    ("payable_wallet_function", True),
    "benign kol":         ("benign_kol", False),
    "benign developer":   ("benign_developer", False),
}


@dataclass
class PtxSeed:
    tx_hash: str
    subtype: str
    is_attack: bool


def load_ptxphish(path: str | Path = DEFAULT_PATH) -> list[PtxSeed]:
    """解析 xlsx → [PtxSeed]。未在 COLUMN_MAP 中的列（含 Source/空列）跳过。"""
    import openpyxl
    wb = openpyxl.load_workbook(path)
    ws = wb.worksheets[0]
    rows = list(ws.iter_rows(values_only=True))
    if len(rows) < 5:
        return []

    col_info: dict[int, tuple[str, bool]] = {}
    for i, h in enumerate(rows[3]):
        if not h:
            continue
        info = COLUMN_MAP.get(str(h).strip().lower())
        if info:
            col_info[i] = info

    seen: set[str] = set()
    out: list[PtxSeed] = []
    for r in rows[4:]:
        for i, (sub, is_atk) in col_info.items():
            if i >= len(r) or r[i] is None:
                continue
            s = str(r[i]).strip().lower()
            if s.startswith("0x") and len(s) == 66 and s not in seen:
                seen.add(s)
                out.append(PtxSeed(tx_hash=s, subtype=sub, is_attack=is_atk))
    return out


def attacks(seeds: list[PtxSeed]) -> list[PtxSeed]:
    return [s for s in seeds if s.is_attack]


def benign(seeds: list[PtxSeed]) -> list[PtxSeed]:
    return [s for s in seeds if not s.is_attack]


def subtype_counts(seeds: list[PtxSeed]) -> dict[str, int]:
    return dict(Counter(s.subtype for s in seeds).most_common())
