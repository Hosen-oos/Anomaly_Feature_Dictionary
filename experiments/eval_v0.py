"""实验一（v0）真实数据评测：fit 正常校准集 → 留出正常误报率 + 攻击检测率。

    cd D:\\科研\\开题\\mlusd
    python -m experiments.eval_v0

依赖 data/splits/d_cal.pkl.gz 与 d_known.pkl.gz（由 BigQuery 采集，见 seeds/build）。
"""
from __future__ import annotations

import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mlusd.dataset.build import load_contexts          # noqa: E402
from mlusd.match.dictionary import load_dictionaries    # noqa: E402
from mlusd.pipeline import Detector                      # noqa: E402
from mlusd.signals.factory import default_extractors     # noqa: E402


def main(fit_n: int = 0) -> None:
    dcal = load_contexts(ROOT / "data/splits/d_cal.pkl.gz")
    dknown = load_contexts(ROOT / "data/splits/d_known.pkl.gz")
    random.seed(0); random.shuffle(dcal)
    split = int(len(dcal) * 0.8)
    fit_set = dcal[:split] if fit_n == 0 else dcal[:fit_n]
    held = dcal[split:]

    det = Detector(default_extractors(), load_dictionaries(ROOT / "configs/dictionaries"),
                   alpha=0.01, min_group_size=200).fit(fit_set)

    hv = Counter(det.detect(c).verdict.value for c in held)
    fp = sum(v for k, v in hv.items() if k != "NORMAL") / len(held)
    print(f"留出正常 {len(held)} 笔  误报率 {fp*100:.2f}% (目标α=1%)  {dict(hv)}")

    by_type = defaultdict(Counter); correct = defaultdict(int); tot = defaultdict(int)
    for c in dknown:
        r = det.detect(c); t = c.latent.get("attack_type"); tot[t] += 1
        by_type[t][r.verdict.value] += 1
        if r.verdict.value == "KNOWN" and r.known_type == t:
            correct[t] += 1
    flagged = sum(sum(v for k, v in by_type[t].items() if k != "NORMAL") for t in tot)
    print(f"攻击检测率 {flagged}/{sum(tot.values())} = {flagged/sum(tot.values())*100:.1f}%")
    for t in sorted(tot):
        dr = sum(v for k, v in by_type[t].items() if k != "NORMAL")
        print(f"  {t:<20} 检出{dr}/{tot[t]} KNOWN正确{correct[t]}  {dict(by_type[t])}")


if __name__ == "__main__":
    main()
