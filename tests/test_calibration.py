"""M3/M5 校准核心的单元测试（共形 p 值有效性、分组、Fisher 聚合）。"""
from __future__ import annotations

import numpy as np

from mlusd.calibrate.ecdf import ECDFCalibrator
from mlusd.calibrate.groups import GroupResolver, parent_group
from mlusd.openset.fisher import OpenSetCalibrator, fisher_u
from mlusd.types import empty_signal_matrix


def _matrix(l, j, v):
    S = empty_signal_matrix()
    S[l - 1, j - 1] = v
    return S


def test_parent_group():
    assert parent_group("1110") == "1100"
    assert parent_group("1100") == "1000"
    assert parent_group("1000") is None


def test_group_resolver_merges_small_groups():
    masks = [(1, 1, 1, 0)] * 600 + [(1, 1, 1, 1)] * 10
    r = GroupResolver(min_group_size=500)
    r.fit(masks)
    # "1111" 样本不足 -> 合并到父组 "1110"
    assert r.resolve((1, 1, 1, 1)) == "1110"
    assert r.resolve((1, 1, 1, 0)) == "1110"


def test_ecdf_quantile_monotone():
    mask = (1, 1, 1, 0)
    cal = [_matrix(2, 1, float(x)) for x in range(1000)]
    c = ECDFCalibrator(min_group_size=10)
    c.fit(cal, [mask] * len(cal))
    q_low, _, _ = c.transform(_matrix(2, 1, 10.0), mask)
    q_hi, _, _ = c.transform(_matrix(2, 1, 990.0), mask)
    assert q_low[1, 0] < q_hi[1, 0]
    assert 0.0 <= q_low[1, 0] <= 1.0


def test_conformal_pvalue_validity():
    """核心性质：正常分数的尾部概率 p=1-Q 近似均匀 -> P(Q>=1-α)≈α。"""
    mask = (1, 1, 1, 0)
    rng = np.random.default_rng(0)
    cal_vals = rng.normal(size=5000)
    cal = [_matrix(2, 1, float(v)) for v in cal_vals]
    c = ECDFCalibrator(min_group_size=10)
    c.fit(cal, [mask] * len(cal))
    test_vals = rng.normal(size=5000)
    exceed = 0
    for v in test_vals:
        q, _, _ = c.transform(_matrix(2, 1, float(v)), mask)
        if q[1, 0] >= 0.95:
            exceed += 1
    rate = exceed / len(test_vals)
    assert 0.03 < rate < 0.07, f"5% 名义误报率实测 {rate:.3f}，超出容差"


def test_fisher_u_increases_with_anomaly():
    from mlusd.types import empty_signal_matrix
    Q1 = empty_signal_matrix(); Q1[0, 0] = 0.5; Q1[1, 0] = 0.5
    Q2 = empty_signal_matrix(); Q2[0, 0] = 0.99; Q2[1, 0] = 0.99
    mask = (1, 1, 0, 0)
    assert fisher_u(Q2, mask) > fisher_u(Q1, mask)


def test_openset_ubar_flags_tail():
    mask = (1, 1, 0, 0)
    rng = np.random.default_rng(1)
    Qs, masks = [], []
    for _ in range(3000):
        Q = empty_signal_matrix()
        Q[0, 0] = rng.uniform(0, 1)
        Q[1, 0] = rng.uniform(0, 1)
        Qs.append(Q); masks.append(mask)
    resolver = GroupResolver(min_group_size=10); resolver.fit(masks)
    os = OpenSetCalibrator(alpha=0.01); os.fit(Qs, masks, resolver)
    extreme = empty_signal_matrix(); extreme[0, 0] = 0.999; extreme[1, 0] = 0.999
    assert os.ubar(extreme, mask, "1100") >= os.threshold
