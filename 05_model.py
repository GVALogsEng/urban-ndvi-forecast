# %% 1. Imports, device, data, tensors
import os
import sys
import copy
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
from src.data import load_cube, make_windows, temporal_split

torch.manual_seed(0)
np.random.seed(0)

# Apple Silicon GPU (MPS) if available, else CPU. If MPS ever misbehaves,
# set: device = torch.device("cpu")
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print("Device:", device)

# Same windows + split as the baselines (the shared contract).
ndvi, observed = load_cube()
X, Y, M, times = make_windows(ndvi, observed, k=12)
train, test = temporal_split(X, Y, M, times, test_years=3)
Xtr, Ytr, Mtr, Ttr = train
Xte, Yte, Mte, Tte = test

# Carve a small VALIDATION set from the END of training (last 12 months).
# Used only to decide WHEN to stop training. The test set stays untouched.
n_val = 12
Xtr2, Ytr2, Mtr2 = Xtr[:-n_val], Ytr[:-n_val], Mtr[:-n_val]
Xval, Yval, Mval = Xtr[-n_val:], Ytr[-n_val:], Mtr[-n_val:]
print(f"train: {len(Xtr2)} | val: {len(Xval)} | test: {len(Xte)}")

# To torch tensors. Inputs need a channel axis: (N, k, H, W) -> (N, k, 1, H, W).
def to_x(a):
    return torch.from_numpy(a).unsqueeze(2).float().to(device)

Xtr2_t = to_x(Xtr2)
Ytr2_t = torch.from_numpy(Ytr2).float().to(device)
Mtr2_t = torch.from_numpy(Mtr2).to(device)
Xval_t = to_x(Xval)
Xte_t  = to_x(Xte)


# %% 2. Define the ConvLSTM
# CNN (spatial pattern in each frame) fused with an LSTM (how the pattern
# changes across the 12 input months). Reads the year one month at a time,
# updating a memory (h = hidden state, c = cell memory), then maps the final
# memory to next month's NDVI.

class ConvLSTMCell(nn.Module):
    def __init__(self, in_ch, hid_ch, kernel=3):
        super().__init__()
        pad = kernel // 2
        # one conv computes all 4 LSTM gates at once, from [input, hidden]
        self.conv = nn.Conv2d(in_ch + hid_ch, 4 * hid_ch, kernel, padding=pad)
        self.hid_ch = hid_ch

    def forward(self, x, h, c):
        gates = self.conv(torch.cat([x, h], dim=1))      # (B, 4*hid, H, W)
        i, f, o, g = torch.chunk(gates, 4, dim=1)
        i, f, o = torch.sigmoid(i), torch.sigmoid(f), torch.sigmoid(o)
        g = torch.tanh(g)
        c = f * c + i * g                                 # update cell memory
        h = o * torch.tanh(c)                             # update hidden state
        return h, c


class ConvLSTM(nn.Module):
    def __init__(self, hid_ch=16, kernel=3):
        super().__init__()
        self.hid_ch = hid_ch
        self.cell = ConvLSTMCell(in_ch=1, hid_ch=hid_ch, kernel=kernel)
        self.head = nn.Conv2d(hid_ch, 1, kernel_size=1)   # memory -> 1 NDVI frame

    def forward(self, x):                                 # x: (B, T, 1, H, W)
        B, T, C, H, W = x.shape
        h = torch.zeros(B, self.hid_ch, H, W, device=x.device)
        c = torch.zeros(B, self.hid_ch, H, W, device=x.device)
        for t in range(T):                                # read months in order
            h, c = self.cell(x[:, t], h, c)
        return self.head(h).squeeze(1)                    # (B, H, W)


# %% 3. Scoring + baselines (recomputed here so the comparison is self-contained)
def masked_rmse(pred, truth, mask):
    """RMSE over observed pixels only (numpy)."""
    err = (pred - truth)[mask]
    return float(np.sqrt(np.mean(err ** 2)))

def skill(rmse_model, rmse_ref):
    return 100.0 * (1.0 - rmse_model / rmse_ref)

# Persistence and climatology on the SAME test set (matches 03_baselines.py).
rmse_persist = masked_rmse(Xte[:, -1], Yte, Mte)

tr_month = Ttr.astype("datetime64[M]").astype(int) % 12
te_month = Tte.astype("datetime64[M]").astype(int) % 12
clim = np.zeros((12, Ytr.shape[1], Ytr.shape[2]), dtype="float32")
for mth in range(12):
    sel = tr_month == mth
    if sel.any():
        clim[mth] = Ytr[sel].mean(axis=0)
pred_clim = np.stack([clim[m] for m in te_month])
rmse_clim = masked_rmse(pred_clim, Yte, Mte)
print(f"Baselines -> persistence {rmse_persist:.4f} | climatology {rmse_clim:.4f}")


# %% 4. The masked training loss
# Penalize ONLY real-observation pixels (filled guesses are multiplied out),
# so the model never wastes effort matching our interpolation.
def masked_mse_loss(pred, target, mask):
    m = mask.float()
    se = ((pred - target) ** 2) * m
    return se.sum() / m.sum().clamp(min=1.0)


# %% 5. Train to convergence with early stopping (patience)
model = ConvLSTM(hid_ch=16).to(device)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)

def eval_rmse(Xtensor, Ynp, Mnp):
    model.eval()
    with torch.no_grad():
        pred = model(Xtensor).cpu().numpy()
    return masked_rmse(pred, Ynp, Mnp)

max_epochs = 300      # safety cap, NOT the real stopping point
patience   = 30       # stop if val hasn't improved for this many epochs
min_delta  = 1e-4     # improvements smaller than this don't count (noise guard)
batch = 16
n = len(Xtr2_t)

best_val = np.inf
best_state = None
best_epoch = -1
since_improve = 0
history = []

for epoch in range(max_epochs):
    model.train()
    perm = torch.randperm(n).to(device)
    tot, nb = 0.0, 0
    for i in range(0, n, batch):
        idx = perm[i:i + batch]
        opt.zero_grad()
        pred = model(Xtr2_t[idx])
        loss = masked_mse_loss(pred, Ytr2_t[idx], Mtr2_t[idx])
        loss.backward()
        opt.step()
        tot += loss.item(); nb += 1

    val_rmse = eval_rmse(Xval_t, Yval, Mval)
    history.append((tot / nb, val_rmse))

    if val_rmse < best_val - min_delta:            # a real improvement
        best_val = val_rmse
        best_state = copy.deepcopy(model.state_dict())
        best_epoch = epoch + 1
        since_improve = 0
    else:
        since_improve += 1

    if (epoch + 1) % 10 == 0:
        print(f"epoch {epoch+1:3d} | train MSE {tot/nb:.5f} | "
              f"val RMSE {val_rmse:.4f} | best {best_val:.4f} @ ep{best_epoch} | "
              f"wait {since_improve}/{patience}")

    if since_improve >= patience:                   # converged: stop
        print(f"\nEarly stop at epoch {epoch+1}: no val improvement for {patience} epochs.")
        break

model.load_state_dict(best_state)                   # restore best-validation model
print(f"Best validation RMSE: {best_val:.4f} (epoch {best_epoch}, ran {len(history)} epochs)")


# %% 6. Final test evaluation + scoreboard + save the model
rmse_model = eval_rmse(Xte_t, Yte, Mte)

print("\n=== FINAL SCOREBOARD (test set, observed pixels only) ===")
print(f"Persistence   RMSE: {rmse_persist:.4f}")
print(f"Climatology   RMSE: {rmse_clim:.4f}   <- the bar")
print(f"ConvLSTM      RMSE: {rmse_model:.4f}")
print(f"\nConvLSTM skill vs climatology: {skill(rmse_model, rmse_clim):+.1f}%")
print("=> beats climatology" if rmse_model < rmse_clim
      else "=> does NOT beat climatology (a legitimate finding)")

os.makedirs(os.path.join(_ROOT, "checkpoints"), exist_ok=True)
torch.save(model.state_dict(), os.path.join(_ROOT, "checkpoints", "convlstm.pt"))
print("\nSaved -> checkpoints/convlstm.pt")


# %% 7. (Optional) plot the learning curves
tr_hist, val_hist = zip(*history)
fig, ax1 = plt.subplots(figsize=(9, 4))
ax1.plot(tr_hist, color="tab:blue", label="train MSE")
ax1.set_xlabel("epoch"); ax1.set_ylabel("train masked-MSE", color="tab:blue")
ax2 = ax1.twinx()
ax2.plot(val_hist, color="tab:red", label="val RMSE")
ax2.axhline(rmse_clim, color="gray", ls="--", label="climatology bar")
ax2.set_ylabel("val masked-RMSE", color="tab:red")
plt.title("Learning curves (watch for val RMSE rising = overfitting)")
fig.tight_layout(); plt.show()