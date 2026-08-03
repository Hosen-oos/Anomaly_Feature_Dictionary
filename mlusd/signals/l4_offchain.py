"""L4 链下情报层的 v0 提取器（设计架构 §4 M2）。

L4-j3 信息差异：链下声明 vs 链上行为一致性。v0 为结构化标签命中评分
       （链下 metadata + 链上行为融合的做法参考 TokenScout/RPHunter），
       不做 NLP 舆情（设计架构 §1 明确列为扩展项）。
"""
from __future__ import annotations

from typing import Optional

from mlusd.signals.base import SignalExtractor
from mlusd.types import TxContext


class OffchainConsistencyScore(SignalExtractor):
    """L4-j3：链下情报。含标签命中与**不泄漏的合约结构性信号**。

    L4 广义化（设计定稿 §1）：L4 是广义链下信息，不等于恶意地址黑名单。实测公开黑名单
    对 DeFi 攻击覆盖为 0，且用种子自带标签会构成泄漏；故引入合约元信息——部署时间、
    标准符合性、字节码规模——这些是通用属性，不由"是否作恶"推出，无泄漏风险。
    参数化后由 DictSignal 做非稀释聚合（新鲜度/微型合约等各自成参数）。
    """

    layer, angle, name = 4, 3, "l4j3_offchain"

    def params(self, ctx: TxContext) -> dict[str, float]:
        off = ctx.offchain
        if off is None:
            return {}
        r: dict[str, float] = {}
        # --- 标签类（可选；离线评测通常缺失，部署期可用）---
        sev = [float(h.get("severity", 0.5)) for h in off.label_hits]
        r["malicious_label"] = max(sev) if sev else 0.0
        if off.contract_verified is not None:
            r["unverified"] = 1.0 if off.contract_verified is False else 0.0
        if off.audited is not None:
            r["unaudited"] = 0.0 if off.audited else 1.0

        # --- 合约结构性信号（不泄漏）---
        meta = getattr(off, "contract_meta", None) or {}
        if meta:
            age_d = meta.get("age_days")
            if age_d is not None:
                # 新部署合约风险更高：≤1 天→1.0，≥365 天→0
                r["contract_freshness"] = float(max(0.0, min(1.0, 1.0 - age_d / 365.0)))
            blen = meta.get("bytecode_len")
            if blen:
                # 极小字节码常见于一次性攻击/代理壳合约
                r["tiny_bytecode"] = 1.0 if blen < 500 else 0.0
                r["bytecode_len_norm"] = float(min(1.0, blen / 24000.0))
            if meta.get("is_token") is not None:
                # 非标准代币合约（既非 ERC20 也非 ERC721）交互
                r["nonstandard_contract"] = 0.0 if meta["is_token"] else 1.0
        return r

    def _components(self, ctx: TxContext) -> Optional[dict[str, float]]:
        p = self.params(ctx)
        return p or None

    def score(self, ctx: TxContext) -> Optional[float]:
        p = self.params(ctx)
        if not p:
            return None
        w = {"malicious_label": 0.5, "unverified": 0.2, "unaudited": 0.1,
             "contract_freshness": 0.3, "tiny_bytecode": 0.2,
             "nonstandard_contract": 0.1}
        raw = sum(w[k] * v for k, v in p.items() if k in w)
        return float(max(0.0, min(1.0, raw)))

    def evidence(self, ctx: TxContext) -> str:
        off = ctx.offchain
        if off is None:
            return ""
        parts = []
        if off.label_hits:
            labels = ", ".join(f"{h.get('label')}({h.get('source')})"
                               for h in off.label_hits[:3])
            parts.append(f"标签命中: {labels}")
        if off.contract_verified is False:
            parts.append("合约未验证")
        if off.audited is True:
            parts.append("有审计记录")
        return "链下情报: " + ("; ".join(parts) if parts else "无风险记录")
