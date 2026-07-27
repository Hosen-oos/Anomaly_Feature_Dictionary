"""λ 敏感性扫描 + veto 消融（λ=0 即"仅对比式权重、无否定证据"）。
一次 fit + 一次建档，评估多个 λ，省算力。
"""
from __future__ import annotations

import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mlusd.dataset.build import load_contexts                    # noqa: E402
from mlusd.match.contrastive import (                             # noqa: E402
    build_profiles, collect_param_quantiles, contrastive_score)
from mlusd.match.dictionary import load_dictionaries              # noqa: E402
from mlusd.match.matcher import match_one                         # noqa: E402
from mlusd.pipeline import Detector                               # noqa: E402
from mlusd.signals.factory import default_extractors              # noqa: E402

ECON = {"flash_loan", "price_manipulation", "rug_pull"}


def main():
    dcal = load_contexts(ROOT / "data/splits/d_cal_nbr.pkl.gz")
    dknown = load_contexts(ROOT / "data/splits/d_known_nbr.pkl.gz")
    random.Random(0).shuffle(dcal)
    det = Detector(default_extractors(), load_dictionaries(ROOT / "configs/dictionaries"),
                   alpha=0.01, min_group_size=150).fit(dcal[:6000])

    by = defaultdict(list)
    for c in dknown:
        by[c.latent.get("attack_type")].append(c)
    train, test = {}, {}
    for t, cs in by.items():
        cs = cs[:]; random.Random(1).shuffle(cs)
        k = max(1, len(cs) // 2)
        train[t], test[t] = cs[:k], cs[k:]
    profiles = build_profiles(det, train)
    dicts = load_dictionaries(ROOT / "configs/dictionaries")
    types = sorted(by)

    # 预取测试样本的参数分位数（避免重复计算）
    cache = [(t, collect_param_quantiles(det, c)[0]) for t, cs in test.items() for c in cs]

    # 基线
    base_correct = 0
    for t, cs in test.items():
        for c in cs:
            S, m = det._raw_matrix(c)
            _, T, _ = det.calibrator.transform(S, m)
            sc = {d.attack_type: match_one(d, T, m).final_score for d in dicts}
            if max(sc, key=sc.get) == t:
                base_correct += 1
    n = len(cache)

    def evaluate(profs, lam):
        correct, per = 0, Counter()
        for t, qs in cache:
            sc = {k: contrastive_score(p, qs, lam)[0] for k, p in profs.items()}
            if max(sc, key=sc.get) == t:
                correct += 1; per[t] += 1
        return correct, per

    print(f"{'配置':<34}{'准确率':>8}   每类正确 (fl/ph/po/pr/ru/sa)")
    print(f"{'A 基线(手写先验+聚合格值)':<34}{base_correct/n*100:>7.0f}%")

    # 消融：per-param 但用 vs_normal 权重（隔离"对比式目标"的贡献）
    prof_vn = build_profiles(det, train, contrast="vs_normal")
    c, per = evaluate(prof_vn, 0.0)
    print(f"{'B per-param + vs正常权重(无对比)':<34}{c/n*100:>7.0f}%   "
          + "/".join(str(per[t]) for t in types))

    print(f"\n{'λ':>5}{'准确率':>9}   每类正确 (fl/ph/po/pr/ru/sa)")
    for lam in [0.0, 0.5, 1.0, 2.0, 3.0]:
        correct, per = evaluate(profiles, lam)
        tag = "  ← C per-param+对比, 无veto" if lam == 0 else ""
        cnt = "/".join(str(per[t]) for t in types)
        print(f"{lam:>5.1f}{correct/n*100:>8.0f}%   {cnt}{tag}")


if __name__ == "__main__":
    main()
