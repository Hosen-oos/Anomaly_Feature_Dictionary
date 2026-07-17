"""M3 可用性分组（设计架构 §4 M3、报告 4.5.2.1）。

同组交易才互相比较：纯转账（"1000"）与复杂 DeFi 交易（"1110"）各用各的正常基线。
样本不足的组向"父组"合并：去掉编号最大的可用层（如 "1110" -> "1100"）。
"""
from __future__ import annotations


def group_key(mask: tuple[int, int, int, int]) -> str:
    return "".join(str(int(b)) for b in mask)


def parent_group(g: str) -> str | None:
    """去掉编号最大的可用层。"1000" 已是根，返回 None。"""
    for i in range(len(g) - 1, 0, -1):
        if g[i] == "1":
            return g[:i] + "0" + g[i + 1:]
    return None


class GroupResolver:
    """记录校准集里各组样本量，把小组解析到样本充足的父组。"""

    def __init__(self, min_group_size: int = 500, single_group: bool = False):
        self.min_group_size = min_group_size
        self.single_group = single_group    # 消融：True 时全体共用一个 ECDF
        self.counts: dict[str, int] = {}
        self._fitted_groups: set[str] = set()

    def fit(self, masks: list[tuple[int, int, int, int]]) -> None:
        if self.single_group:
            self.counts = {"ALL": len(masks)}
            self._fitted_groups = {"ALL"}
            return
        self.counts = {}
        for m in masks:
            g = group_key(m)
            self.counts[g] = self.counts.get(g, 0) + 1
        self._fitted_groups = {
            g for g, n in self.counts.items() if n >= self.min_group_size
        }
        if not self._fitted_groups:
            # 极端情况：全部合并到样本最多的组
            biggest = max(self.counts, key=self.counts.get)
            self._fitted_groups = {biggest}

    def resolve(self, mask: tuple[int, int, int, int]) -> str:
        """返回该交易实际使用的校准组（沿父链向下走到有校准数据的组）。"""
        if self.single_group:
            return "ALL"
        g = group_key(mask)
        cur: str | None = g
        while cur is not None:
            if cur in self._fitted_groups:
                return cur
            cur = parent_group(cur)
        # 父链走完仍无（如 "1000" 组样本不足）：退回任一已拟合组
        return next(iter(sorted(self._fitted_groups)))

    @property
    def fitted_groups(self) -> set[str]:
        return set(self._fitted_groups)
