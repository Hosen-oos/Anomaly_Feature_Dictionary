"""v1 学习模型真实数据训练与评测（GPU 机器上跑；v0-vs-v1 消融）。

v1 把三个格子换成学习模型（统一范式：只在正常数据训练，异常分=重构误差/困惑度）：
  L1-j1 GraphAutoencoder（图自编码器）· L2-j1 ActionSequenceTransformer（动作序列困惑度）
  · L3-j1 TraceTransformer（调用树 LM 困惑度，BlockGPT 缩小版）
其余 5 格保持 v0 规则版。对比 v0/v1 的开放集检测 AUROC（整体 + 分类型 + D_open 真未知）。

    python -m experiments.exp_v1_train                 # 默认参数
    python -m experiments.exp_v1_train --epochs 20 --fit-n 12000 --device cuda
    python -m experiments.exp_v1_train --skip-v0       # 只跑 v1
"""
from __future__ import annotations

import argparse
import random
import sys
import time
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mlusd.dataset.build import load_contexts                    # noqa: E402
from mlusd.match.dictionary import load_dictionaries              # noqa: E402
from mlusd.pipeline import Detector                               # noqa: E402
from mlusd.signals.factory import default_extractors              # noqa: E402


def build_v1(epochs: int, device: str):
    """v1 提取器组：三个学习格 + 五个 v0 规则格。"""
    from mlusd.signals.l1_graph import FundFlowScore
    from mlusd.signals.l1_graph_v1 import GraphAutoencoder
    from mlusd.signals.l2_semantic import ApprovalMismatchScore, EconomicAnomalyScore
    from mlusd.signals.l2_semantic_v1 import ActionSequenceTransformer
    from mlusd.signals.l3_trace import TracePropertyScore
    from mlusd.signals.l3_trace_v1 import TraceTransformer
    from mlusd.signals.l4_offchain import OffchainConsistencyScore
    from mlusd.signals.pool import DictSignal
    return [
        GraphAutoencoder(epochs=max(3, epochs // 3), device=device),   # L1-j1 (v1)
        DictSignal(FundFlowScore()),                                   # L1-j2
        ActionSequenceTransformer(epochs=epochs, device=device),       # L2-j1 (v1)
        DictSignal(EconomicAnomalyScore()),                            # L2-j2
        DictSignal(ApprovalMismatchScore()),                           # L2-j3
        TraceTransformer(epochs=epochs, device=device),                # L3-j1 (v1)
        DictSignal(TracePropertyScore()),                              # L3-j3
        OffchainConsistencyScore(),                                    # L4-j3
    ]


def scores(det, ctxs):
    out = []
    for c in ctxs:
        S, m = det._raw_matrix(c)
        Q, _, g = det.calibrator.transform(S, m)
        out.append(det.openset.raw_score(Q, m)
                   if hasattr(det.openset, "raw_score") else det.openset.ubar(Q, m, g))
    return np.asarray(out)


def auroc(sa, sn):
    return roc_auc_score(np.r_[np.ones(len(sa)), np.zeros(len(sn))], np.r_[sa, sn])


def evaluate(det, dknown, dopen, test_norm, by, tag):
    sn = scores(det, test_norm)
    row = {"整体": auroc(scores(det, dknown), sn)}
    for t in sorted(by):
        row[t] = auroc(scores(det, by[t]), sn)
    if dopen:
        row["D_open真未知"] = auroc(scores(det, dopen), sn)
    print(f"\n[{tag}]")
    for k, v in row.items():
        print(f"  {k:<20} AUROC = {v:.3f}")
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=15, help="序列模型训练轮数")
    ap.add_argument("--fit-n", type=int, default=8000, help="用于 fit 的正常样本数")
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    ap.add_argument("--skip-v0", action="store_true")
    args = ap.parse_args()

    try:
        import torch
        print(f"torch {torch.__version__} | CUDA 可用: {torch.cuda.is_available()}"
              + (f" | {torch.cuda.get_device_name(0)}" if torch.cuda.is_available() else ""))
    except ImportError:
        sys.exit("需要 torch：pip install -e '.[learn]'")

    dcal = load_contexts(ROOT / "data/splits/d_cal_nbr.pkl.gz")
    dknown = load_contexts(ROOT / "data/splits/d_known_nbr.pkl.gz")
    p = ROOT / "data/splits/d_open.pkl.gz"
    dopen = load_contexts(p) if p.exists() else None
    random.Random(0).shuffle(dcal)
    fit_norm = dcal[:args.fit_n]
    test_norm = dcal[args.fit_n:args.fit_n + 2000]
    by = defaultdict(list)
    for c in dknown:
        by[c.latent.get("attack_type")].append(c)
    print(f"fit {len(fit_norm)} 正常 | test {len(test_norm)} 正常 | "
          f"{len(dknown)} 攻击 | {len(dopen) if dopen else 0} 真未知")

    dicts = load_dictionaries(ROOT / "configs/dictionaries")
    results = {}
    if not args.skip_v0:
        t0 = time.time()
        det0 = Detector(default_extractors(), dicts, alpha=0.01, min_group_size=150,
                        openset_aggregator="learned").fit(fit_norm)
        print(f"\nv0 拟合耗时 {time.time()-t0:.0f}s")
        results["v0"] = evaluate(det0, dknown, dopen, test_norm, by, "v0 规则版")

    t0 = time.time()
    det1 = Detector(build_v1(args.epochs, args.device), dicts, alpha=0.01,
                    min_group_size=150, openset_aggregator="learned").fit(fit_norm)
    print(f"\nv1 训练+拟合耗时 {time.time()-t0:.0f}s")
    results["v1"] = evaluate(det1, dknown, dopen, test_norm, by, "v1 学习版")

    if "v0" in results:
        print("\n=== v0 vs v1 对比（Δ>0 表示 v1 更优）===")
        for k in results["v0"]:
            d = results["v1"][k] - results["v0"][k]
            print(f"  {k:<20} v0={results['v0'][k]:.3f}  v1={results['v1'][k]:.3f}  Δ={d:+.3f}")


if __name__ == "__main__":
    main()
