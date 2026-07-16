"""端到端演示：拟合合成校准集，对 4 笔黄金测试交易出检测报告。

    cd D:\\科研\\开题\\mlusd
    python -m experiments.demo
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mlusd.match.dictionary import load_dictionaries      # noqa: E402
from mlusd.pipeline import Detector                        # noqa: E402
from mlusd.signals.factory import default_extractors       # noqa: E402
from tests import synthetic                                # noqa: E402

DICT_DIR = ROOT / "configs" / "dictionaries"


def main() -> None:
    print("拟合校准集（3000 笔正常交易）...")
    det = Detector(
        extractors=default_extractors(),
        dictionaries=load_dictionaries(DICT_DIR),
        alpha=0.01, min_group_size=50,
    ).fit(synthetic.calibration_set(3000))
    print(f"校准组: {sorted(det.calibrator.resolver.fitted_groups)}\n")

    cases = {
        "纯 ETH 转账（期望 NORMAL）": synthetic.pure_transfer_tx(),
        "普通合约交互（期望 NORMAL）": synthetic.normal_tx(999999),
        "闪电贷攻击（期望 KNOWN/UNKNOWN）": synthetic.flash_loan_tx(),
        "钓鱼 ice-phishing（期望 KNOWN/UNKNOWN）": synthetic.phishing_tx(),
    }
    for title, ctx in cases.items():
        r = det.detect(ctx)
        print("=" * 70)
        print(title)
        print(json.dumps(r.to_dict(), ensure_ascii=False, indent=2))
    print("=" * 70)


if __name__ == "__main__":
    main()
