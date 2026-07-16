"""实验四：字典权重数据驱动更新 + 逐类型阈值标定（设计架构 §4 M4、§8 实验四）。

协议（小样本诚实划分）：
  D_known 每类 50% train / 50% test；D_cal 70% fit / 15% val / 15% test。
  提取器+ECDF 只 fit 正常；权重更新用 attack-train vs normal-fit；阈值用 attack-train(正)
  + normal-val(负) 调 F1；最终在 attack-test + normal-test 上报 before(先验0.55) vs after。

    cd D:\\科研\\开题\\mlusd
    python -m experiments.exp4_tuning
"""
from __future__ import annotations

import random
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mlusd.dataset.build import load_contexts                    # noqa: E402
from mlusd.match.dictionary import load_dictionaries              # noqa: E402
from mlusd.match.weight_update import (                           # noqa: E402
    signal_tensors, tune_thresholds, update_dictionary_weights)
from mlusd.pipeline import Detector                               # noqa: E402
from mlusd.signals.factory import default_extractors              # noqa: E402
from mlusd.types import Verdict                                   # noqa: E402


def _split_by_type(dknown, frac_train=0.5, seed=0):
    by = defaultdict(list)
    for c in dknown:
        by[c.latent.get("attack_type")].append(c)
    train, test = [], []
    rng = random.Random(seed)
    for t, lst in by.items():
        lst = lst[:]; rng.shuffle(lst)
        k = max(1, int(len(lst) * frac_train))
        train += lst[:k]; test += lst[k:]
    return train, test


def _eval_known(det, atk_test, norm_test):
    """每类 KNOWN 分类 precision/recall/F1 + 正常误报为 KNOWN。"""
    by = defaultdict(list)
    for c in atk_test:
        by[c.latent.get("attack_type")].append(c)
    types = sorted(by)
    # 正常被误判为 KNOWN（任意类型）
    norm_known = sum(det.detect(c).verdict == Verdict.KNOWN for c in norm_test)
    rows = {}
    for t in types:
        detected = [det.detect(c) for c in by[t]]
        tp = sum(r.verdict == Verdict.KNOWN and r.known_type == t for r in detected)
        wrong = sum(r.verdict == Verdict.KNOWN and r.known_type != t for r in detected)
        rows[t] = (tp, len(by[t]), wrong)
    return rows, norm_known, len(norm_test)


def main():
    dcal = load_contexts(ROOT / "data/splits/d_cal.pkl.gz")
    dknown = load_contexts(ROOT / "data/splits/d_known.pkl.gz")
    random.Random(0).shuffle(dcal)
    n = len(dcal)
    fit_norm = dcal[:int(n * 0.7)][:9000]     # 上限 9000 控制时长
    val_norm = dcal[int(n * 0.7):int(n * 0.85)][:2000]
    test_norm = dcal[int(n * 0.85):][:2000]
    atk_train, atk_test = _split_by_type(dknown, 0.5)
    print(f"D_cal fit {len(fit_norm)} / val {len(val_norm)} / test {len(test_norm)}")
    print(f"D_known train {len(atk_train)} / test {len(atk_test)}")

    det = Detector(default_extractors(), load_dictionaries(ROOT / "configs/dictionaries"),
                   alpha=0.01, min_group_size=200).fit(fit_norm)

    def report(tag):
        rows, nk, nn = _eval_known(det, atk_test, test_norm)
        print(f"\n[{tag}] KNOWN 分类 (attack-test)  正常误判KNOWN {nk}/{nn}")
        macro = []
        for t, (tp, tot, wrong) in rows.items():
            rec = tp / tot if tot else 0
            print(f"  {t:<20} 正确KNOWN {tp}/{tot} 召回{rec:.2f} 误分其他类{wrong}  τ={det._dict(t).match_threshold:.3f}")
            macro.append(rec)
        print(f"  宏平均 KNOWN 召回: {sum(macro)/len(macro):.3f}")

    # 给 Detector 加一个按类型取字典的便捷方法
    det._dict = lambda t: next(d for d in det.dictionaries if d.attack_type == t)

    report("BEFORE 先验权重+τ0.55")

    # --- 权重更新 + 阈值标定 ---
    atk_tensors = defaultdict(list)
    for c in atk_train:
        S, m = det._raw_matrix(c); _, T, _ = det.calibrator.transform(S, m)
        atk_tensors[c.latent.get("attack_type")].append((T, m))
    norm_fit_tensors = signal_tensors(det, fit_norm[:3000])
    norm_val_tensors = signal_tensors(det, val_norm)

    update_dictionary_weights(det.dictionaries, atk_tensors, norm_fit_tensors, prior_ratio=0.5)
    tuned = tune_thresholds(det.dictionaries, atk_tensors, norm_val_tensors, alpha_fp=0.005)
    print("\n标定阈值(控正常误报≤0.5%/类):", {k: round(v, 3) for k, v in tuned.items()})

    report("AFTER 数据驱动权重+F1标定阈值")


if __name__ == "__main__":
    main()
