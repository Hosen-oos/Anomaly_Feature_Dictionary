"""LOTO（Leave-One-Type-Out）开放集协议：逐类型移出字典，检验其攻击是否被正确
判为 UNKNOWN（而非误塞进其他已知类型）。检测分类型无关，故重点看"false-KNOWN"率。
"""
from __future__ import annotations

import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mlusd.dataset.build import load_contexts                    # noqa: E402
from mlusd.match.dictionary import load_dictionaries              # noqa: E402
from mlusd.match.weight_update import (                           # noqa: E402
    signal_tensors, tune_thresholds, update_dictionary_weights)
from mlusd.pipeline import Detector                               # noqa: E402
from mlusd.signals.factory import default_extractors              # noqa: E402
from mlusd.types import Verdict                                   # noqa: E402

DICT = ROOT / "configs/dictionaries"


def main():
    dcal = load_contexts(ROOT / "data/splits/d_cal_nbr.pkl.gz")
    dknown = load_contexts(ROOT / "data/splits/d_known_nbr.pkl.gz")
    random.Random(0).shuffle(dcal)
    fit_norm, val_norm, test_norm = dcal[:8000], dcal[8000:9000], dcal[9000:11000]
    by = defaultdict(list)
    for c in dknown:
        by[c.latent.get("attack_type")].append(c)

    det = Detector(default_extractors(), load_dictionaries(DICT),
                   alpha=0.01, min_group_size=150, openset_aggregator="learned").fit(fit_norm)

    def raw(ctxs):
        out = []
        for c in ctxs:
            S, m = det._raw_matrix(c)
            Q, _, g = det.calibrator.transform(S, m)
            out.append(det.openset.raw_score(Q, m))
        return np.asarray(out)
    sn = raw(test_norm)
    norm_val_T = signal_tensors(det, val_norm)

    print("=== LOTO：逐类型移出字典，其攻击应判 UNKNOWN 而非误塞已知类 ===")
    print(f"{'留出类型':<20}{'检测AUROC':>10}{'UNKNOWN':>9}{'误判已知':>9}{'漏检NORMAL':>11}")
    for k in sorted(by):
        # 移出 k 的字典与训练样本
        dicts = [d for d in load_dictionaries(DICT) if d.attack_type != k]
        det.dictionaries = dicts
        det._req = {d.attack_type: d.layer_requirements for d in dicts}
        atk_T = {d.attack_type: signal_tensors(det, by[d.attack_type]) for d in dicts}
        update_dictionary_weights(dicts, atk_T, signal_tensors(det, fit_norm[:2000]), 0.5)
        tune_thresholds(dicts, atk_T, norm_val_T, alpha_fp=0.005)
        verd = Counter(det.detect(c).verdict for c in by[k])
        auroc = roc_auc_score(np.r_[np.ones(len(by[k])), np.zeros(len(sn))],
                              np.r_[raw(by[k]), sn])
        print(f"{k:<20}{auroc:>10.3f}{verd.get(Verdict.UNKNOWN,0):>9}"
              f"{verd.get(Verdict.KNOWN,0):>9}{verd.get(Verdict.NORMAL,0):>11}")


if __name__ == "__main__":
    main()
