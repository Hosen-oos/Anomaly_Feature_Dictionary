"""强三明治信号对**类型归因**的贡献（而非检测）。

动机：opposite_same_pool 在 sandwich 上命中 82.5%、正常仅 1.7%（48 倍富集），
是近乎完美的类型指纹；但开放集检测分是**类型无关**的，单类型强信号在其中被稀释
（检测仅 +0.014）。此类证据的用武之地是 M4 类型归因——per-param 对比式字典正是
用类型特异证据的机制。本实验验证这一点。
"""
from __future__ import annotations

import random
import sys
import warnings
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mlusd.dataset.build import load_contexts                     # noqa: E402
from mlusd.match.contrastive import (                              # noqa: E402
    build_profiles, collect_param_quantiles, contrastive_score)
from mlusd.match.dictionary import load_dictionaries               # noqa: E402
from mlusd.pipeline import Detector                                # noqa: E402
from mlusd.signals.factory import default_extractors               # noqa: E402


def run(dknown, dcal, strip_sw, lam=0.5):
    if strip_sw:
        for c in dknown + dcal:
            c.latent.pop("sandwich_ctx", None)
    random.Random(0).shuffle(dcal)
    det = Detector(default_extractors(), load_dictionaries(ROOT / "configs/dictionaries"),
                   alpha=0.01, min_group_size=150,
                   openset_aggregator="learned").fit(dcal[:6000])
    by = defaultdict(list)
    for c in dknown:
        by[c.latent.get("attack_type")].append(c)
    train, test = {}, {}
    for t, cs in by.items():
        cs = cs[:]; random.Random(1).shuffle(cs)
        k = max(1, len(cs) // 2)
        train[t], test[t] = cs[:k], cs[k:]
    profiles = build_profiles(det, train)
    correct, per, tot = 0, Counter(), Counter()
    for t, cs in test.items():
        for c in cs:
            qs, _ = collect_param_quantiles(det, c)
            sc = {k: contrastive_score(p, qs, lam)[0] for k, p in profiles.items()}
            pred = max(sc, key=sc.get)
            tot[t] += 1
            if pred == t:
                correct += 1; per[t] += 1
    return correct, sum(tot.values()), per, tot


def main():
    dknown = load_contexts(ROOT / "data/splits/d_known_nbr_blk_l4_sw.pkl.gz")
    dcal = load_contexts(ROOT / "data/splits/d_cal_nbr_blk_l4_sw.pkl.gz")
    print("=== 强三明治信号对类型归因的贡献 ===")
    res = {}
    for tag, strip in [("含强三明治信号", False), ("剥离(消融)", True)]:
        c, n, per, tot = run(list(dknown), list(dcal), strip)
        res[tag] = (c, n, per, tot)
        print(f"\n[{tag}] 类型归因准确率 {c}/{n} = {c/n*100:.0f}%")
        for t in sorted(tot):
            print(f"    {t:<22}{per[t]}/{tot[t]}")
    a = res["含强三明治信号"][0] / res["含强三明治信号"][1]
    b = res["剥离(消融)"][0] / res["剥离(消融)"][1]
    print(f"\n整体提升: {a*100:.0f}% vs {b*100:.0f}%  ({(a-b)*100:+.0f} 个百分点)")
    sa = res["含强三明治信号"][2]["sandwich"]
    sb = res["剥离(消融)"][2]["sandwich"]
    print(f"sandwich 归因: {sa} vs {sb} (共 {res['含强三明治信号'][3]['sandwich']} 笔)")


if __name__ == "__main__":
    main()
