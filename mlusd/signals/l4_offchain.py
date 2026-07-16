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
    """L4-j3：标签命中加权评分。恶意标签(+)、未验证合约(+)、有审计(-)。"""

    layer, angle, name = 4, 3, "l4j3_offchain_labels"

    def _components(self, ctx: TxContext) -> Optional[dict[str, float]]:
        off = ctx.offchain
        if off is None:
            return None
        comp: dict[str, float] = {}
        # 恶意标签命中：取命中中的最高 severity
        sev = [float(h.get("severity", 0.5)) for h in off.label_hits]
        comp["malicious"] = max(sev) if sev else 0.0
        # 合约未验证：不透明代码是风险加分
        comp["unverified"] = 1.0 if off.contract_verified is False else 0.0
        # 有审计记录：风险减分
        comp["audited"] = -1.0 if off.audited is True else 0.0
        return comp

    def score(self, ctx: TxContext) -> Optional[float]:
        comp = self._components(ctx)
        if comp is None:
            return None
        w = {"malicious": 0.7, "unverified": 0.3, "audited": 0.2}
        raw = sum(w[k] * v for k, v in comp.items())
        return float(max(0.0, min(1.0, raw)))   # 截断到 [0,1]

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
