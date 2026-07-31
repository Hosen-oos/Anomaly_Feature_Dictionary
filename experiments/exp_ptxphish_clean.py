"""管线对齐后的 PTXPHISH 干净评测（消除采集偽影）。

偽影：此前 PTXPHISH(RPC, 0% 有trace) vs D_cal(BigQuery, 100% 有trace)，单凭
"是否有trace"即可完美分离（AUROC=1.000），使"vs正常"列被污染（且方向为**低估**：
缺失的 L3 信号被填 0 → 看起来更正常）。现改用 d_cal_rpc（同管线 RPC 采集）。

同时报有监督协议下的可比数字。
"""
from __future__ import annotations

import random
import sys
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mlusd.baselines.features import feature_matrix          # noqa: E402
from mlusd.dataset.build import load_contexts                # noqa: E402
from mlusd.match.dictionary import load_dictionaries          # noqa: E402
from mlusd.pipeline import Detector                           # noqa: E402
from mlusd.signals.factory import default_extractors          # noqa: E402

APPROVAL = {"ice_phishing_approve", "ice_phishing_permit",
            "ice_phishing_setapprovalforall", "nft_order_free_buy",
            "nft_order_bulk_transfer"}


def scores(det, ctxs):
    out = []
    for c in ctxs:
        S, m = det._raw_matrix(c)
        Q, _, g = det.calibrator.transform(S, m)
        out.append(det.openset.raw_score(Q, m))
    return np.asarray(out)


def auroc(sa, sn):
    return roc_auc_score(np.r_[np.ones(len(sa)), np.zeros(len(sn))], np.r_[sa, sn])


def main():
    rpc_norm = load_contexts(ROOT / "data/splits/d_cal_rpc.pkl.gz")
    random.Random(0).shuffle(rpc_norm)
    fit_norm, test_norm = rpc_norm[:4000], rpc_norm[4000:]
    ptx_a = load_contexts(ROOT / "data/splits/d_phish_ptx.pkl.gz")
    ptx_b = load_contexts(ROOT / "data/splits/d_benign_ptx.pkl.gz")
    print(f"管线对齐: 正常(RPC) fit {len(fit_norm)} / test {len(test_norm)} | "
          f"钓鱼 {len(ptx_a)} | 硬负 {len(ptx_b)}")

    det = Detector(default_extractors(), load_dictionaries(ROOT / "configs/dictionaries"),
                   alpha=0.01, min_group_size=150,
                   openset_aggregator="learned").fit(fit_norm)
    sn, sb = scores(det, test_norm), scores(det, ptx_b)
    by = defaultdict(list)
    for c in ptx_a:
        by[c.latent.get("subtype", "?")].append(c)

    print("\n=== 无监督检测 AUROC（管线对齐后）===")
    print(f"{'子类型':<34}{'n':>6}{'vs正常(干净)':>14}{'vs硬负样本':>12}")
    sa_all = scores(det, ptx_a)
    print(f"{'全部钓鱼':<34}{len(ptx_a):>6}{auroc(sa_all, sn):>14.3f}{auroc(sa_all, sb):>12.3f}")
    sub = [c for c in ptx_a if c.latent.get("subtype") in APPROVAL]
    ss = scores(det, sub)
    print(f"{'授权/NFT订单类':<34}{len(sub):>6}{auroc(ss, sn):>14.3f}{auroc(ss, sb):>12.3f}")
    for t, cs in sorted(by.items(), key=lambda x: -len(x[1])):
        if len(cs) >= 15:
            s = scores(det, cs)
            print(f"  {t:<32}{len(cs):>6}{auroc(s, sn):>14.3f}{auroc(s, sb):>12.3f}")

    # 有监督协议（管线对齐）
    print("\n=== 有监督协议（RF, 5折CV, 平衡采样，管线对齐）===")
    rng = random.Random(0)
    for name, pos, neg in [("授权类钓鱼 vs 正常(RPC)", sub, test_norm),
                           ("授权类钓鱼 vs 硬负样本", sub, ptx_b),
                           ("全部钓鱼 vs 正常(RPC)", ptx_a, test_norm)]:
        p, q = pos[:], neg[:]
        rng.shuffle(p); rng.shuffle(q)
        n = min(len(p), len(q), 1900)
        X = np.vstack([feature_matrix(p[:n]), feature_matrix(q[:n])])
        y = np.r_[np.ones(n), np.zeros(n)]
        aucs, f1s = [], []
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(X, y):
            clf = RandomForestClassifier(n_estimators=300, random_state=0, n_jobs=-1).fit(X[tr], y[tr])
            pr = clf.predict_proba(X[te])[:, 1]
            aucs.append(roc_auc_score(y[te], pr))
            f1s.append(f1_score(y[te], (pr >= 0.5).astype(int)))
        print(f"  {name:<28} n={n:<5} AUROC={np.mean(aucs):.3f}  F1={np.mean(f1s):.3f}")


if __name__ == "__main__":
    main()
