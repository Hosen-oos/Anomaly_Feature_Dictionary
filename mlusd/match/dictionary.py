"""M4 信号字典：已知攻击类型的 YAML 配置（设计架构 §3.3、报告 4.5.2.2）。

新增攻击类型 = 新增一个 YAML 文件，不需要重训模型。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml

from mlusd.types import N_ANGLES, N_LAYERS


@dataclass
class AttackDictionary:
    attack_type: str
    name_zh: str
    weights: np.ndarray                 # W_k, 4x3，无效位置为 0
    layer_requirements: np.ndarray      # r_k, 长度 4
    match_threshold: float              # τ_k
    coverage_threshold: float           # c_min
    hard_evidence: list[str] = field(default_factory=list)
    references: list[str] = field(default_factory=list)

    def validate(self) -> None:
        assert self.weights.shape == (N_LAYERS, N_ANGLES), self.attack_type
        assert self.layer_requirements.shape == (N_LAYERS,), self.attack_type
        assert self.weights.sum() > 0, f"{self.attack_type}: 权重全零"
        assert self.layer_requirements.sum() > 0, f"{self.attack_type}: 层级需求全零"
        assert 0 < self.match_threshold < 1 and 0 < self.coverage_threshold <= 1


def load_dictionary(path: str | Path) -> AttackDictionary:
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    d = AttackDictionary(
        attack_type=raw["type"],
        name_zh=raw.get("name_zh", raw["type"]),
        weights=np.asarray(raw["weights"], dtype=float),
        layer_requirements=np.asarray(raw["layer_requirements"], dtype=float),
        match_threshold=float(raw["match_threshold"]),
        coverage_threshold=float(raw["coverage_threshold"]),
        hard_evidence=raw.get("hard_evidence", []) or [],
        references=raw.get("references", []) or [],
    )
    d.validate()
    return d


def load_dictionaries(dir_path: str | Path) -> list[AttackDictionary]:
    dicts = [load_dictionary(p) for p in sorted(Path(dir_path).glob("*.yaml"))]
    assert dicts, f"字典目录为空: {dir_path}"
    types = [d.attack_type for d in dicts]
    assert len(types) == len(set(types)), f"攻击类型重复: {types}"
    return dicts
