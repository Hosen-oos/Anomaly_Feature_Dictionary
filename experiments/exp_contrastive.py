"""验证 per-param 对比式字典 + 否定证据(veto) 能否缓解经济类混淆。

对照实验（同数据同划分）：
  A. 基线：现有聚合格值匹配（match_one）
  B. 改进：per-param 对比式 + veto（contrastive_score）
指标：类型归因准确率（argmax 是否为真类）+ 混淆矩阵（重点看经济三类互吸）。
每类 50/50 train/test，权重只在 train 上学。
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
    lam = float(sys.argv[1]) if len(sys.argv) > 1 else 1.0
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

    print(f"=== 类型归因对比（λ={lam}，每类 train/test 各半）===")
    for tag in ["A 基线(聚合格值)", "B 改进(per-param对比+veto)"]:
        conf = defaultdict(Counter); correct = tot = 0
        econ_cross = 0
        for true_t, ctxs in test.items():
            for c in ctxs:
                if tag.startswith("A"):
                    S, m = det._raw_matrix(c)
                    _, T, _ = det.calibrator.transform(S, m)
                    scores = {d.attack_type: match_one(d, T, m).final_score for d in dicts}
                else:
                    qs, _ = collect_param_quantiles(det, c)
                    scores = {t: contrastive_score(p, qs, lam)[0] for t, p in profiles.items()}
                pred = max(scores, key=scores.get)
                conf[true_t][pred] += 1
                tot += 1
                if pred == true_t:
                    correct += 1
                elif true_t in ECON and pred in ECON:
                    econ_cross += 1
        print(f"\n[{tag}] 类型归因准确率 {correct}/{tot} = {correct/tot*100:.0f}%"
              f" | 经济三类互相误判 {econ_cross}")
        print(f"  {'真实\\预测':<20}" + "".join(f"{t[:8]:>10}" for t in types))
        for t in types:
            print(f"  {t:<20}" + "".join(f"{conf[t].get(p,0):>10}" for p in types))

    # 展示 veto 实例
    print("\n=== 否定证据触发实例（前5）===")
    shown = 0
    for true_t, ctxs in test.items():
        for c in ctxs:
            qs, _ = collect_param_quantiles(det, c)
            for t, p in profiles.items():
                if t == true_t:
                    continue
                _, _, fired = contrastive_score(p, qs, lam)
                if fired and shown < 5:
                    print(f"  真实={true_t}: 否定 {t} ← {fired[:3]}")
                    shown += 1
                    break
            if shown >= 5:
                break
        if shown >= 5:
            break


if __name__ == "__main__":
    main()
