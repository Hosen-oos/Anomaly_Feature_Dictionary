"""P1 数据集构建（设计架构 §7）。

三个数据集：正常校准集 D_cal、已知异常标注集 D_known、未知模拟集 D_open。
build.py 提供 BigQuery 采样 + 攻击 hash 清单加载 + 批量 build_context 落缓存。
"""
from mlusd.dataset.build import (
    build_contexts, load_attack_hashes, save_contexts,
)

__all__ = ["build_contexts", "load_attack_hashes", "save_contexts"]
