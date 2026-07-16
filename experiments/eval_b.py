"""改进 B 评测：邻域增强前后对比（可用性组 + 检测率 + 误报）。

    python -m experiments.eval_b nbr    # 邻域增强数据
    python -m experiments.eval_b base   # 原始（单交易内转账）数据
"""
from __future__ import annotations

import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mlusd.dataset.build import load_contexts               # noqa: E402
from mlusd.match.dictionary import load_dictionaries         # noqa: E402
from mlusd.pipeline import Detector                          # noqa: E402
from mlusd.signals.factory import default_extractors         # noqa: E402


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "nbr"
    suffix = "_nbr" if mode == "nbr" else ""
    dcal = load_contexts(ROOT / f"data/splits/d_cal{suffix}.pkl.gz")
    dknown = load_contexts(ROOT / f"data/splits/d_known{suffix}.pkl.gz")
    random.seed(0); random.shuffle(dcal)
    fit = dcal[:8000]
    held = dcal[8000:10000]

    det = Detector(default_extractors(), load_dictionaries(ROOT / "configs/dictionaries"),
                   alpha=0.01, min_group_size=150).fit(fit)

    hv = Counter(det.detect(c).verdict.value for c in held)
    fp = 1 - hv.get("NORMAL", 0) / len(held)
    print(f"=== 模式 {mode} ===")
    print(f"留出正常 {len(held)} 笔 误报率 {fp*100:.2f}%  {dict(hv)}")

    by = defaultdict(Counter); flagged = 0
    for c in dknown:
        r = det.detect(c); t = c.latent.get("attack_type"); by[t][r.verdict.value] += 1
        if r.verdict.value != "NORMAL":
            flagged += 1
    print(f"攻击检测率 {flagged}/{len(dknown)} = {flagged/len(dknown)*100:.1f}%")
    for t in sorted(by):
        d = sum(v for k, v in by[t].items() if k != "NORMAL")
        print(f"  {t:<20} 检出{d}/{sum(by[t].values())}  {dict(by[t])}")


if __name__ == "__main__":
    main()
