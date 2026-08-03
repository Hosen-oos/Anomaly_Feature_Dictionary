"""效率实验（设计 §8 实验五）：端到端耗时分解，定位瓶颈。

分解为：信号提取(M2) / 校准(M3) / 字典匹配(M4) / 开放识别(M5) / 决策解释(M6)。
数据采集(M1)单独报告——它是网络 I/O，与检测算法不同量级，正是要说明的"瓶颈在采集
而非检测"。同时报吞吐量（笔/秒）用于与实时性需求对照。
"""
from __future__ import annotations

import random
import statistics
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mlusd.dataset.build import load_contexts               # noqa: E402
from mlusd.decide.decision import decide                     # noqa: E402
from mlusd.match.dictionary import load_dictionaries         # noqa: E402
from mlusd.match.matcher import match_all                    # noqa: E402
from mlusd.pipeline import Detector                          # noqa: E402
from mlusd.signals.factory import default_extractors         # noqa: E402


def main():
    n_eval = 300
    dcal = load_contexts(ROOT / "data/splits/d_cal_nbr.pkl.gz")
    dknown = load_contexts(ROOT / "data/splits/d_known_nbr.pkl.gz")
    random.Random(0).shuffle(dcal)
    fit_norm = dcal[:8000]

    t0 = time.perf_counter()
    det = Detector(default_extractors(), load_dictionaries(ROOT / "configs/dictionaries"),
                   alpha=0.01, min_group_size=150,
                   openset_aggregator="learned").fit(fit_norm)
    t_fit = time.perf_counter() - t0
    print(f"一次性拟合（{len(fit_norm)} 正常样本）: {t_fit:.1f}s  "
          f"[{t_fit/len(fit_norm)*1000:.2f} ms/笔]")

    sample = (dknown + dcal[8000:8000 + n_eval])[:n_eval]
    stages = {"M2 信号提取": [], "M3 校准": [], "M4 字典匹配": [],
              "M5 开放识别": [], "M6 决策解释": []}
    for c in sample:
        t = time.perf_counter(); S, m = det._raw_matrix(c)
        stages["M2 信号提取"].append(time.perf_counter() - t)

        t = time.perf_counter(); Q, T, g = det.calibrator.transform(S, m)
        stages["M3 校准"].append(time.perf_counter() - t)

        t = time.perf_counter(); matches = match_all(det.dictionaries, T, m)
        stages["M4 字典匹配"].append(time.perf_counter() - t)

        t = time.perf_counter(); ub = det.openset.ubar(Q, m, g)
        stages["M5 开放识别"].append(time.perf_counter() - t)

        t = time.perf_counter()
        decide(tx_hash=c.tx_hash, matches=matches, Q=Q, mask=m, group=g,
               ubar=ub, tau_u=det.openset.threshold, layer_requirements=det._req)
        stages["M6 决策解释"].append(time.perf_counter() - t)

    print(f"\n=== 单笔检测耗时分解（n={len(sample)}）===")
    print(f"{'阶段':<18}{'均值(ms)':>12}{'中位(ms)':>12}{'占比':>8}")
    means = {k: statistics.mean(v) * 1000 for k, v in stages.items()}
    total = sum(means.values())
    for k in stages:
        med = statistics.median(stages[k]) * 1000
        print(f"{k:<18}{means[k]:>12.3f}{med:>12.3f}{means[k]/total*100:>7.1f}%")
    print(f"{'合计':<18}{total:>12.3f}{'':>12}{'100.0%':>8}")
    print(f"\n检测吞吐量: {1000/total:.0f} 笔/秒（单线程，不含数据采集）")

    print("\n=== 数据采集（M1）实测参照 ===")
    print("  BigQuery 批量: 一个日期窗口 ~4.2 GB 扫描，数千笔/次查询")
    print("  RPC 逐笔(20并发): ~0.43 s/笔（18555 笔耗时 7992 s）")
    print("  → 采集比检测慢 3-4 个数量级，瓶颈在数据获取而非检测算法")


if __name__ == "__main__":
    main()
