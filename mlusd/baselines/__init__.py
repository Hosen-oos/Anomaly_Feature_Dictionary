"""基线方法（A 刊对比必需）。features: TxContext → 扁平特征向量。"""
from mlusd.baselines.features import feature_matrix, flat_features

__all__ = ["flat_features", "feature_matrix"]
