"""深度基线：MLP AutoEncoder（重构误差异常检测），补全基线谱系。
与本框架、经典基线同数据同任务同指标(AUROC 攻击 vs 正常)。torch CPU 即可。
"""
from __future__ import annotations

import random
import sys
import warnings
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mlusd.baselines.features import feature_matrix              # noqa: E402
from mlusd.dataset.build import load_contexts                    # noqa: E402


def train_ae(Xn, dim, epochs=60, seed=0):
    import torch
    import torch.nn as nn
    torch.manual_seed(seed)
    enc = nn.Sequential(nn.Linear(dim, 16), nn.ReLU(), nn.Linear(16, 8))
    dec = nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, dim))
    model = nn.Sequential(enc, dec)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    X = torch.tensor(Xn, dtype=torch.float32)
    model.train()
    for _ in range(epochs):
        opt.zero_grad()
        loss = nn.functional.mse_loss(model(X), X)
        loss.backward(); opt.step()
    model.eval()
    return model


def recon_err(model, X):
    import torch
    with torch.no_grad():
        Xt = torch.tensor(X, dtype=torch.float32)
        return ((model(Xt) - Xt) ** 2).mean(dim=1).numpy()


def main():
    dcal = load_contexts(ROOT / "data/splits/d_cal_nbr.pkl.gz")
    dknown = load_contexts(ROOT / "data/splits/d_known_nbr.pkl.gz")
    random.Random(0).shuffle(dcal)
    fit_norm, test_norm = dcal[:8000], dcal[8000:10000]
    Xn = feature_matrix(fit_norm)
    sc = StandardScaler().fit(Xn)
    Zn = sc.transform(Xn)
    Ztn = sc.transform(feature_matrix(test_norm))
    Za = sc.transform(feature_matrix(dknown))

    model = train_ae(Zn, Zn.shape[1])
    sa, sn = recon_err(model, Za), recon_err(model, Ztn)
    y = np.r_[np.ones(len(sa)), np.zeros(len(sn))]
    print(f"AutoEncoder(扁平, 重构误差)  AUROC = {roc_auc_score(y, np.r_[sa, sn]):.3f}")


if __name__ == "__main__":
    main()
