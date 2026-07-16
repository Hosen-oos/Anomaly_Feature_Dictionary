"""落盘缓存（设计架构 §4 M1：所有原始响应落盘，保证可复现、断点续跑）。

每个 (namespace, key) 存为 data/cache/{namespace}/{key}.json.gz。
"""
from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Any, Callable, Optional


class DiskCache:
    def __init__(self, root: str | Path = "data/cache"):
        self.root = Path(root)

    def _path(self, namespace: str, key: str) -> Path:
        safe = key.replace("/", "_")
        return self.root / namespace / f"{safe}.json.gz"

    def get(self, namespace: str, key: str) -> Optional[Any]:
        p = self._path(namespace, key)
        if not p.exists():
            return None
        with gzip.open(p, "rt", encoding="utf-8") as f:
            return json.load(f)

    def put(self, namespace: str, key: str, value: Any) -> None:
        p = self._path(namespace, key)
        p.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(p, "wt", encoding="utf-8") as f:
            json.dump(value, f, ensure_ascii=False)

    def cached(self, namespace: str, key: str,
               producer: Callable[[], Any]) -> Any:
        """缓存未命中时调用 producer 并落盘；命中直接返回。None 结果也缓存（负缓存）。"""
        hit = self.get(namespace, key)
        if hit is not None:
            return hit.get("v") if isinstance(hit, dict) and "v" in hit else hit
        value = producer()
        self.put(namespace, key, {"v": value})
        return value
