"""有监督协议评测：与主流论文（有监督 + 闭集 + 平衡采样）同协议的可比数字。

我们的主结果是**无监督开放集**（只在正常上 fit），与那些论文不可直比。本实验补一个
同协议评测，回答"若也用有监督闭集协议，本框架的信号能拿到多少"。

三种特征表示 × 同一分类器（RF / HGB），5 折分层交叉验证，报 AUROC / F1 / P / R：
  a) 扁平手工特征（25 维）           ← 常见做法
  b) **本框架 8 维校准信号**          ← 我们的贡献（分组 ECDF + 尾部放大）
  c) a+b 拼接                        ← 互补性
任务：① phishing 二分类（PTXPHISH 攻击 vs 硬负样本，最难设定）
     ② 六类攻击 vs 正常二分类（研究一 D_known）
纪律：校准器只在**训练折的正常样本**上 fit，避免标签泄漏。
"""
from __future__ import annotations

import random
import sys
import warnings
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mlusd.baselines.features import feature_matrix          # noqa: E402
from mlusd.dataset.build import load_contexts                # noqa: E402
from mlusd.match.dictionary import load_dictionaries          # noqa: E402
from mlusd.pipeline import Detector                           # noqa: E402
from mlusd.signals.factory import default_extractors          # noqa: E402
from mlusd.types import VALID_POSITIONS                       # noqa: E402


def calibrated_signals(det, ctxs) -> np.ndarray:
    """本框架的 8 维校准信号（尾部放大后的 T）。"""
    out = []
    for c in ctxs:
        S, m = det._raw_matrix(c)
        _, T, _ = det.calibrator.transform(S, m)
        out.append([T[l - 1, j - 1] if np.isfinite(T[l - 1, j - 1]) else 0.0
                    for (l, j) in VALID_POSITIONS])
    return np.asarray(out)


def run_task(name, pos_ctxs, neg_ctxs, fit_norm, n_max=3000, seed=0):
    rng = random.Random(seed)
    pos = pos_ctxs[:] ; neg = neg_ctxs[:]
    rng.shuffle(pos); rng.shuffle(neg)
    n = min(len(pos), len(neg), n_max)          # 平衡采样（与主流论文协议一致）
    pos, neg = pos[:n], neg[:n]
    ctxs = pos + neg
    y = np.r_[np.ones(len(pos)), np.zeros(len(neg))]

    det = Detector(default_extractors(), load_dictionaries(ROOT / "configs/dictionaries"),
                   alpha=0.01, min_group_size=150,
                   openset_aggregator="learned").fit(fit_norm)
    X_flat = feature_matrix(ctxs)
    X_sig = calibrated_signals(det, ctxs)
    reps = {"a) 扁平手工特征(25维)": X_flat,
            "b) 本框架校准信号(8维)": X_sig,
            "c) a+b 拼接(33维)": np.c_[X_flat, X_sig]}

    print(f"\n=== {name}（平衡采样 {len(pos)} vs {len(neg)}，5折分层CV）===")
    print(f"{'表示':<26}{'模型':<8}{'AUROC':>8}{'F1':>8}{'Prec':>8}{'Rec':>8}")
    for rep_name, X in reps.items():
        for mdl_name, mk in [("RF", lambda: RandomForestClassifier(
                                  n_estimators=300, random_state=0, n_jobs=-1)),
                             ("HGB", lambda: HistGradientBoostingClassifier(random_state=0))]:
            aucs, f1s, ps, rs = [], [], [], []
            for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(X, y):
                clf = mk().fit(X[tr], y[tr])
                pr = clf.predict_proba(X[te])[:, 1]
                yp = (pr >= 0.5).astype(int)
                aucs.append(roc_auc_score(y[te], pr))
                f1s.append(f1_score(y[te], yp))
                ps.append(precision_score(y[te], yp, zero_division=0))
                rs.append(recall_score(y[te], yp))
            print(f"{rep_name:<26}{mdl_name:<8}{np.mean(aucs):>8.3f}{np.mean(f1s):>8.3f}"
                  f"{np.mean(ps):>8.3f}{np.mean(rs):>8.3f}")


def main():
    dcal = load_contexts(ROOT / "data/splits/d_cal.pkl.gz")
    random.Random(0).shuffle(dcal)
    fit_norm = dcal[:8000]

    # 任务①：PTXPHISH 钓鱼 vs 硬负样本（最难，形似钓鱼的合法交易）
    ptx_a = load_contexts(ROOT / "data/splits/d_phish_ptx.pkl.gz")
    ptx_b = load_contexts(ROOT / "data/splits/d_benign_ptx.pkl.gz")
    run_task("任务① PTXPHISH 钓鱼 vs 硬负样本", ptx_a, ptx_b, fit_norm)

    # 任务①b：仅授权类子集（我们声称有能力的部分）
    keep = {"ice_phishing_approve", "ice_phishing_permit",
            "ice_phishing_setapprovalforall", "nft_order_free_buy",
            "nft_order_bulk_transfer"}
    sub = [c for c in ptx_a if c.latent.get("subtype") in keep]
    run_task("任务①b 授权/NFT订单类钓鱼 vs 硬负样本", sub, ptx_b, fit_norm)

    # 任务②：六类攻击 vs 正常
    dknown = load_contexts(ROOT / "data/splits/d_known.pkl.gz")
    run_task("任务② 六类攻击 vs 正常交易", dknown, dcal[10000:14000], fit_norm)


if __name__ == "__main__":
    main()
