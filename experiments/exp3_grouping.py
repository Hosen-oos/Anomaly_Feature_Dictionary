"""实验三补充：分组校准的正确度量 —— 各可用性组的误报率是否都受控在 α。

整体 AUROC 测不出分组的价值；分组的目的是"不同信息画像的交易，误报率都受控"，
即避免复杂交易被系统性误报。对比有/无分组下各组 FP。
"""
from __future__ import annotations

import random
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mlusd.dataset.build import load_contexts               # noqa: E402
from mlusd.match.dictionary import load_dictionaries         # noqa: E402
from mlusd.pipeline import Detector                          # noqa: E402
from mlusd.signals.factory import default_extractors         # noqa: E402


def group_fp(fit_norm, test_norm, single):
    det = Detector(default_extractors(), load_dictionaries(ROOT / "configs/dictionaries"),
                   alpha=0.01, min_group_size=150, single_group=single).fit(fit_norm)
    tau = det.openset.threshold
    by = defaultdict(lambda: [0, 0])
    for c in test_norm:
        S, m = det._raw_matrix(c)
        Q, _, g = det.calibrator.transform(S, m)
        ub = det.openset.ubar(Q, m, g)
        key = "".join(map(str, m))
        by[key][1] += 1
        if ub >= tau:
            by[key][0] += 1
    return by


def main():
    dcal = load_contexts(ROOT / "data/splits/d_cal_nbr.pkl.gz")
    random.Random(0).shuffle(dcal)
    fit_norm = dcal[:6000]
    test_norm = dcal[6000:8500]
    on = group_fp(fit_norm, test_norm, False)
    off = group_fp(fit_norm, test_norm, True)
    print("分组校准价值 = 各可用性组误报率都受控在 α=1%（目标）")
    print(f"{'可用性组':<10}{'样本':>6}{'有分组':>9}{'无分组':>9}")
    for k in sorted(set(on) | set(off)):
        a = on.get(k, [0, 1]); b = off.get(k, [0, 1])
        print(f"{k:<10}{a[1]:>6}{a[0]/max(a[1],1)*100:>8.1f}%{b[0]/max(b[1],1)*100:>8.1f}%")


if __name__ == "__main__":
    main()
