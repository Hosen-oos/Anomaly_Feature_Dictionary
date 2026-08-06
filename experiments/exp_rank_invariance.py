"""验证：ECDF 校准不改变任何单信号的 AUROC（秩不变性），故"绕开校准看 v1 真实优势"
这一实验在数学上必然为空。

论证：ECDF 为单调非降变换 q = rank(v)/(n+1)；AUROC 只依赖排序；故对任一信号位，
AUROC(原始 S) ≡ AUROC(校准 Q)。唯一损失来自离散化（校准集 n 个参考点 → 分辨率 1/n），
在 n=8000 时可忽略。

推论（对论文的意义）：
- 校准层**不可能**抹掉学习表征的判别力——若 v1 的格子更有判别力，AUROC 会如实反映；
  故 "v1 无增益" 的归因是干净的，不存在"被校准抵消"这一替代解释。
- 校准真正影响的是**多信号聚合**（把不同量纲的信号变得可比），而非单信号判别。
- 秩变换丢弃的是**幅度**：这解释了 rug_pull（极端量级签名被压缩）与 v1（容量改变尺度
  而非排序）两个现象——同一性质的两种表现。
"""
from __future__ import annotations

import random
import sys
import warnings
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mlusd.dataset.build import load_contexts               # noqa: E402
from mlusd.match.dictionary import load_dictionaries         # noqa: E402
from mlusd.pipeline import Detector                          # noqa: E402
from mlusd.signals.factory import default_extractors         # noqa: E402
from mlusd.types import LAYER_NAMES, VALID_POSITIONS         # noqa: E402


def auroc(sa, sn):
    m = np.isfinite(sa); n = np.isfinite(sn)
    if m.sum() < 2 or n.sum() < 2:
        return float("nan")
    a, b = sa[m], sn[n]
    return roc_auc_score(np.r_[np.ones(len(a)), np.zeros(len(b))], np.r_[a, b])


def main():
    dcal = load_contexts(ROOT / "data/splits/d_cal_nbr_blk_l4_sw.pkl.gz")
    dknown = load_contexts(ROOT / "data/splits/d_known_nbr_blk_l4_sw.pkl.gz")
    random.Random(0).shuffle(dcal)
    fit_norm, test_norm = dcal[:8000], dcal[8000:10000]

    det = Detector(default_extractors(), load_dictionaries(ROOT / "configs/dictionaries"),
                   alpha=0.01, min_group_size=150,
                   openset_aggregator="learned").fit(fit_norm)

    def mats(ctxs):
        S_all, Q_all = [], []
        for c in ctxs:
            S, m = det._raw_matrix(c)
            Q, _, _ = det.calibrator.transform(S, m)
            S_all.append(S); Q_all.append(Q)
        return np.asarray(S_all), np.asarray(Q_all)

    Sa, Qa = mats(dknown)
    Sn, Qn = mats(test_norm)

    print("=== 单信号位 AUROC：校准前(S) vs 校准后(Q) ===")
    print(f"{'信号位':<12}{'层':<16}{'AUROC(原始S)':>14}{'AUROC(校准Q)':>14}{'差':>9}")
    diffs = []
    for (l, j) in VALID_POSITIONS:
        a_s = auroc(Sa[:, l - 1, j - 1], Sn[:, l - 1, j - 1])
        a_q = auroc(Qa[:, l - 1, j - 1], Qn[:, l - 1, j - 1])
        if np.isfinite(a_s) and np.isfinite(a_q):
            diffs.append(abs(a_s - a_q))
        print(f"{'L'+str(l)+'-j'+str(j):<12}{LAYER_NAMES[l][:14]:<16}"
              f"{a_s:>14.4f}{a_q:>14.4f}{a_q - a_s:>+9.4f}")
    if diffs:
        print(f"\n最大绝对差 = {max(diffs):.5f}（理论应为 0，非零部分来自 ECDF 离散化 1/{len(fit_norm)}）")

    print("\n结论：ECDF 为单调变换、AUROC 为秩指标，二者恒等 →")
    print("  · '绕开校准检验 v1 真实优势' 的实验必然为空，无需跑；")
    print("  · 'v1 无增益' 不存在'被校准抵消'的替代解释，归因是干净的；")
    print("  · 校准影响的是多信号**聚合**（量纲可比），而非单信号判别力；")
    print("  · 秩变换丢弃的是**幅度**——同时解释 rug_pull 的极端量级签名被压缩。")


if __name__ == "__main__":
    main()
