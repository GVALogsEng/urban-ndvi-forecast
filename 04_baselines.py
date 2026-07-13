# %% Imports — pull the shared data contract from src/data.py
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.data import load_cube, make_windows, temporal_split

ndvi, observed = load_cube()
X, Y, M, times = make_windows(ndvi, observed, k=12)
train, test = temporal_split(X, Y, M, times, test_years=3)

Xtr, Ytr, Mtr, Ttr = train
Xte, Yte, Mte, Tte = test
print("train:", Xtr.shape, "| test:", Xte.shape)

# %% The masked scoring function — score only real-observation pixels
def masked_rmse(pred, truth, mask):
    """Root mean squared error over observed pixels only."""
    err = (pred - truth)[mask]
    return float(np.sqrt(np.mean(err ** 2)))

def skill(rmse_model, rmse_reference):
    """% improvement over a reference. 0 = no better; >0 = better; <0 = worse."""
    return 100.0 * (1.0 - rmse_model / rmse_reference)

# %% Baseline 1 — Persistence: next frame = last input frame
pred_persist = Xte[:, -1]
rmse_persist = masked_rmse(pred_persist, Yte, Mte)
print(f"Persistence  RMSE: {rmse_persist:.4f}")

# %% Baseline 2 — Seasonal climatology: predict the typical value for that month
tr_month = Ttr.astype("datetime64[M]").astype(int) % 12
te_month = Tte.astype("datetime64[M]").astype(int) % 12

assert len(tr_month) == len(Ytr)
assert len(te_month) == len(Yte)

clim = np.zeros((12, Ytr.shape[1], Ytr.shape[2]), dtype="float32")
for mth in range(12):
    sel = tr_month == mth
    if sel.any():
        clim[mth] = Ytr[sel].mean(axis=0)

pred_clim = np.stack([clim[m] for m in te_month])
rmse_clim = masked_rmse(pred_clim, Yte, Mte)
print(f"Climatology  RMSE: {rmse_clim:.4f}")

# %% Scoreboard — the bar the model must beat
print("\n--- Baseline scoreboard (test set, observed pixels only) ---")
print(f"Persistence  RMSE: {rmse_persist:.4f}")
print(f"Climatology  RMSE: {rmse_clim:.4f}")
print(f"Climatology vs Persistence skill: {skill(rmse_clim, rmse_persist):+.1f}%")

better = "Climatology" if rmse_clim < rmse_persist else "Persistence"
best = min(rmse_clim, rmse_persist)
print(f"\nBest baseline: {better}  (RMSE {best:.4f}) -- this is the bar to beat.")