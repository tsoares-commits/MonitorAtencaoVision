"""
train_mlp.py
------------
Treina um MLP para prever (altura_total, largura) a partir de:
    - 6 parametros do processo: incremento_z, n_camadas, altura_ideal,
      vel_arame, potencia, vel_robo
    - 4-5 features de area do video segmentado: area_mean, area_std,
      area_amplitude, frac_overshoot, A0

Arquitetura (small-data friendly):
    Input(~11) -> Linear(32) -> LeakyReLU + Dropout(0.3)
                -> Linear(16) -> LeakyReLU + Dropout(0.3)
                -> Linear(2)   # altura, largura

Avaliacao: K-fold cross-validation (default k=5). Dataset pequeno
demais para um split unico train/val/test ser confiavel.

Pre-requisito:
    python build_mlp_dataset.py   # gera mlp_dataset/features.csv

Uso:
    python train_mlp.py
    python train_mlp.py --features mlp_dataset\\features.csv --k 5 --epochs 800
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


HERE = Path(__file__).resolve().parent
DEFAULT_FEATURES = HERE / "mlp_dataset" / "features.csv"
DEFAULT_OUT      = HERE / "ckpt_mlp"

# Mesmas listas do build_mlp_dataset.py
PROC_COLS = ["incremento_z", "n_camadas", "altura_ideal",
             "vel_arame", "potencia", "vel_robo"]
AREA_COLS = ["area_mean", "area_std", "area_amplitude", "frac_overshoot", "A0"]
TARGET_COLS = ["altura_total", "largura"]


# -----------------------------------------------------------------------------
# Arquitetura
# -----------------------------------------------------------------------------
class WeldMLP(nn.Module):
    """MLP simples para regressao multi-task (altura, largura).

    Pequeno por design: dataset tem ~17 amostras, modelo grande sofre
    overfitting imediato.
    """

    def __init__(self, in_dim, hidden=(32, 16), drop=0.3):
        super().__init__()
        layers = []
        prev = in_dim
        for h in hidden:
            layers += [
                nn.Linear(prev, h),
                nn.LeakyReLU(0.01),
                nn.Dropout(drop),
            ]
            prev = h
        layers.append(nn.Linear(prev, len(TARGET_COLS)))   # 2 saidas
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# -----------------------------------------------------------------------------
# Helpers de scaling (z-score)
# -----------------------------------------------------------------------------
class ZScaler:
    """Standardiza por coluna: (x - mean) / std. Mantem mean/std p/ inverter."""

    def fit(self, X):
        self.mean = X.mean(axis=0)
        self.std = X.std(axis=0)
        self.std = np.where(self.std < 1e-9, 1.0, self.std)   # evita /0
        return self

    def transform(self, X):
        return (X - self.mean) / self.std

    def inverse_transform(self, X):
        return X * self.std + self.mean

    def to_dict(self):
        return {"mean": self.mean.tolist(), "std": self.std.tolist()}

    @classmethod
    def from_dict(cls, d):
        z = cls()
        z.mean = np.asarray(d["mean"])
        z.std = np.asarray(d["std"])
        return z


# -----------------------------------------------------------------------------
# Treino de uma dobra
# -----------------------------------------------------------------------------
def train_fold(X_tr, y_tr, X_va, y_va, in_dim,
               epochs=500, lr=1e-3, wd=1e-3, batch_size=4, patience=80):
    """Treina o MLP em uma dobra; retorna (model_state, mae_val_real_units)."""
    model = WeldMLP(in_dim=in_dim)
    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    loss_fn = nn.SmoothL1Loss()

    Xt = torch.from_numpy(X_tr).float()
    yt = torch.from_numpy(y_tr).float()
    Xv = torch.from_numpy(X_va).float()
    yv = torch.from_numpy(y_va).float()

    ds = TensorDataset(Xt, yt)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True)

    best_val = float("inf")
    best_state = {k: v.clone() for k, v in model.state_dict().items()}
    no_imp = 0

    for ep in range(epochs):
        model.train()
        for xb, yb in dl:
            optim.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            optim.step()

        model.eval()
        with torch.no_grad():
            vp = model(Xv)
            vloss = loss_fn(vp, yv).item()

        if vloss < best_val - 1e-6:
            best_val = vloss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            no_imp = 0
        else:
            no_imp += 1
            if no_imp >= patience:
                break

    return best_state, best_val


# -----------------------------------------------------------------------------
# Main: K-fold CV + treino final em todo o dataset
# -----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default=str(DEFAULT_FEATURES))
    ap.add_argument("--out_dir", default=str(DEFAULT_OUT))
    ap.add_argument("--k", type=int, default=5, help="K-fold CV (default 5)")
    ap.add_argument("--epochs", type=int, default=500)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-3, help="weight decay (L2 reg)")
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--drop", type=float, default=0.3, help="dropout entre camadas")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    df = pd.read_csv(args.features)
    print(f"[data] {len(df)} amostras lidas de {args.features}")

    feat_cols = PROC_COLS + [c for c in AREA_COLS if c in df.columns]
    missing_feat = [c for c in PROC_COLS if c not in df.columns] + \
                   [c for c in TARGET_COLS if c not in df.columns]
    if missing_feat:
        raise ValueError(f"Colunas faltando em features.csv: {missing_feat}")

    X = df[feat_cols].to_numpy(dtype=np.float64)
    y = df[TARGET_COLS].to_numpy(dtype=np.float64)
    print(f"[data] X.shape={X.shape} | y.shape={y.shape}")
    print(f"[data] features de entrada: {feat_cols}")

    n = len(df)
    if n < args.k:
        args.k = n
        print(f"[warn] k>n, ajustando k={n} (LOOCV)")

    # K-fold sem sklearn (simples)
    indices = np.arange(n)
    rng = np.random.RandomState(args.seed)
    rng.shuffle(indices)
    folds = np.array_split(indices, args.k)

    fold_mae = []
    fold_results = []
    for k_idx in range(args.k):
        val_idx = folds[k_idx]
        train_idx = np.concatenate([folds[i] for i in range(args.k) if i != k_idx])

        Xtr, Xva = X[train_idx], X[val_idx]
        ytr, yva = y[train_idx], y[val_idx]

        # scaling AJUSTADO so no train (evita data leakage)
        sx = ZScaler().fit(Xtr)
        sy = ZScaler().fit(ytr)
        Xtr_s = sx.transform(Xtr).astype(np.float32)
        Xva_s = sx.transform(Xva).astype(np.float32)
        ytr_s = sy.transform(ytr).astype(np.float32)
        yva_s = sy.transform(yva).astype(np.float32)

        state, val_loss = train_fold(
            Xtr_s, ytr_s, Xva_s, yva_s, in_dim=X.shape[1],
            epochs=args.epochs, lr=args.lr, wd=args.wd,
            batch_size=args.batch_size, patience=80,
        )

        # avalia em unidade real (mm)
        model = WeldMLP(in_dim=X.shape[1])
        model.load_state_dict(state)
        model.eval()
        with torch.no_grad():
            pred_s = model(torch.from_numpy(Xva_s).float()).cpu().numpy()
        pred = sy.inverse_transform(pred_s)
        mae_per_target = np.abs(pred - yva).mean(axis=0)
        fold_mae.append(mae_per_target)
        fold_results.append({
            "fold": k_idx + 1,
            "n_val": int(len(val_idx)),
            "val_smoothl1_z": float(val_loss),
            "mae_altura_mm": float(mae_per_target[0]),
            "mae_largura_mm": float(mae_per_target[1]),
        })
        print(f"[fold {k_idx+1}/{args.k}] n_val={len(val_idx)} | "
              f"MAE altura={mae_per_target[0]:.3f}mm | "
              f"MAE largura={mae_per_target[1]:.3f}mm")

    fold_mae = np.array(fold_mae)   # [k, 2]
    print(f"\n[CV summary] {args.k}-fold")
    print(f"  MAE altura : {fold_mae[:, 0].mean():.3f} ± {fold_mae[:, 0].std():.3f} mm")
    print(f"  MAE largura: {fold_mae[:, 1].mean():.3f} ± {fold_mae[:, 1].std():.3f} mm")

    # ===== treino final em todo o dataset =====
    print("\n[final] treinando modelo final em TODAS as amostras...")
    sx_full = ZScaler().fit(X)
    sy_full = ZScaler().fit(y)
    X_s = sx_full.transform(X).astype(np.float32)
    y_s = sy_full.transform(y).astype(np.float32)

    # split interno 80/20 so para early stopping
    n_val = max(1, int(0.2 * n))
    idx = np.arange(n); rng.shuffle(idx)
    tr_i, va_i = idx[n_val:], idx[:n_val]
    state, _ = train_fold(
        X_s[tr_i], y_s[tr_i], X_s[va_i], y_s[va_i],
        in_dim=X.shape[1], epochs=args.epochs, lr=args.lr, wd=args.wd,
        batch_size=args.batch_size, patience=80,
    )

    # ===== salva tudo =====
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": state,
        "in_dim": X.shape[1],
        "hidden": (32, 16),
        "drop": args.drop,
        "feat_cols": feat_cols,
        "target_cols": TARGET_COLS,
        "scaler_x": sx_full.to_dict(),
        "scaler_y": sy_full.to_dict(),
    }, out_dir / "mlp_best.pt")

    with open(out_dir / "cv_results.json", "w", encoding="utf-8") as f:
        json.dump({
            "n_samples": n,
            "k": args.k,
            "feat_cols": feat_cols,
            "target_cols": TARGET_COLS,
            "fold_results": fold_results,
            "mae_altura_mean_mm": float(fold_mae[:, 0].mean()),
            "mae_altura_std_mm": float(fold_mae[:, 0].std()),
            "mae_largura_mean_mm": float(fold_mae[:, 1].mean()),
            "mae_largura_std_mm": float(fold_mae[:, 1].std()),
        }, f, indent=2)

    print(f"\n[OK] modelo final salvo em: {out_dir / 'mlp_best.pt'}")
    print(f"     resultados CV: {out_dir / 'cv_results.json'}")


if __name__ == "__main__":
    main()
