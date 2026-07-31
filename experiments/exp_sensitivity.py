"""虚假相关敏感性分析：0.984 的高分是"识别钓鱼"还是"识别 KOL/开发者人群"？

PTXPHISH 的硬负样本是 KOL 与开发者交易——一个特定人群，可能在钓鱼性之外还系统性
不同（活跃度、合约类型、时间段）。若分类器学的是"是否 KOL"而非"是否钓鱼"，
换负样本后性能应显著下降。三组对照：

  N1 硬负样本(KOL/开发者)       ← 原设定
  N2 随机正常交易(D_cal)         ← 换人群
  N3 硬负 + 随机正常 混合         ← 混合人群

另做**特征消融**：若性能主要来自与钓鱼机制无关的特征（如可用性掩码、图规模），
说明存在捷径；若来自 L2 授权/语义信号，则是真机制。
"""
from __future__ import annotations

import random
import sys
import warnings
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mlusd.baselines.features import FEATURE_NAMES, feature_matrix   # noqa: E402
from mlusd.dataset.build import load_contexts                        # noqa: E402


def cv(X, y, seed=0):
    aucs, f1s = [], []
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(X, y):
        clf = RandomForestClassifier(n_estimators=300, random_state=0, n_jobs=-1).fit(X[tr], y[tr])
        pr = clf.predict_proba(X[te])[:, 1]
        aucs.append(roc_auc_score(y[te], pr))
        f1s.append(f1_score(y[te], (pr >= 0.5).astype(int)))
    return float(np.mean(aucs)), float(np.mean(f1s))


def main():
    rng = random.Random(0)
    ptx_a = load_contexts(ROOT / "data/splits/d_phish_ptx.pkl.gz")
    ptx_b = load_contexts(ROOT / "data/splits/d_benign_ptx.pkl.gz")
    dcal = load_contexts(ROOT / "data/splits/d_cal.pkl.gz")
    rng.shuffle(dcal)

    keep = {"ice_phishing_approve", "ice_phishing_permit",
            "ice_phishing_setapprovalforall", "nft_order_free_buy",
            "nft_order_bulk_transfer"}
    pos = [c for c in ptx_a if c.latent.get("subtype") in keep]
    rng.shuffle(pos); rng.shuffle(ptx_b)
    n = min(len(pos), 2000)
    pos = pos[:n]

    negs = {
        "N1 硬负样本(KOL/开发者)": ptx_b[:n],
        "N2 随机正常交易(D_cal)": dcal[:n],
        "N3 混合(各半)": ptx_b[:n // 2] + dcal[:n - n // 2],
    }

    print("=== 负样本人群敏感性（授权/NFT订单类钓鱼，RF，5折CV）===")
    print(f"{'负样本':<26}{'AUROC':>9}{'F1':>8}")
    Xp = feature_matrix(pos)
    for name, neg in negs.items():
        X = np.vstack([Xp, feature_matrix(neg)])
        y = np.r_[np.ones(len(pos)), np.zeros(len(neg))]
        a, f = cv(X, y)
        print(f"{name:<26}{a:>9.3f}{f:>8.3f}")

    # 特征消融：去掉可能构成"捷径"的特征组
    print("\n=== 特征消融（负样本=N1 硬负样本）===")
    neg = ptx_b[:n]
    X_all = np.vstack([Xp, feature_matrix(neg)])
    y = np.r_[np.ones(len(pos)), np.zeros(len(neg))]
    idx = {f: i for i, f in enumerate(FEATURE_NAMES)}
    groups = {
        "全部特征": [],
        "−可用性掩码(m1..m4)": ["m1", "m2", "m3", "m4"],
        "−图规模(g_*/c_*)": [f for f in FEATURE_NAMES if f.startswith(("g_", "c_"))],
        "−L2动作计数(act_*)": [f for f in FEATURE_NAMES if f.startswith("act_")],
        "仅L2动作计数": [f for f in FEATURE_NAMES if not f.startswith("act_")],
    }
    for name, drop in groups.items():
        cols = [i for f, i in idx.items() if f not in set(drop)]
        a, f = cv(X_all[:, cols], y)
        print(f"{name:<26}{a:>9.3f}{f:>8.3f}   (用 {len(cols)} 维)")

    # 特征重要性 top-8
    clf = RandomForestClassifier(n_estimators=300, random_state=0, n_jobs=-1).fit(X_all, y)
    top = np.argsort(clf.feature_importances_)[::-1][:8]
    print("\n=== 特征重要性 top-8 ===")
    for i in top:
        print(f"  {FEATURE_NAMES[i]:<20} {clf.feature_importances_[i]:.3f}")


if __name__ == "__main__":
    main()
