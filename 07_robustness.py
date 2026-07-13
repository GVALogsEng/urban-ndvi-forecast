# %% 1. Setup: data, tensors, model class, and training helpers
import os
import sys
import copy
import numpy as np
import xarray as xr
import torch
import torch.nn as nn

_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
from src.data import load_cube, make_windows, temporal_split

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print("Device:", device)

ndvi, observed = load_cube()
X, Y, M, times = make_windows(ndvi, observed, k=12)
train, test = temporal_split(X, Y, M, times, test_years=3)
Xtr, Ytr, Mtr, Ttr = train
Xte, Yte, Mte, Tte = test

# Validation split (last 12 months of train), same as 04.
n_val = 12
Xtr2, Ytr2, Mtr2 = Xtr[:-n_val], Ytr[:-n_val], Mtr[:-n_val]
Xval, Yval, Mval = Xtr[-n_val:], Ytr[-n_val:], Mtr[-n_val:]

def to_x(a):
    return torch.from_numpy(a).unsqueeze(2).float().to(device)

Xtr2_t = to_x(Xtr2); Ytr2_t = torch.from_numpy(Ytr2).float().to(device)
Mtr2_t = torch.from_numpy(Mtr2).to(device)
Xval_t = to_x(Xval)
Xte_t  = to_x(Xte)
n_tr = len(Xtr2_t)


class ConvLSTMCell(nn.Module):
    def __init__(self, in_ch, hid_ch, kernel=3):
        super().__init__()
        self.conv = nn.Conv2d(in_ch + hid_ch, 4 * hid_ch, kernel, padding=kernel // 2)
        self.hid_ch = hid_ch
    def forward(self, x, h, c):
        i, f, o, g = torch.chunk(self.conv(torch.cat([x, h], dim=1)), 4, dim=1)
        i, f, o = torch.sigmoid(i), torch.sigmoid(f), torch.sigmoid(o)
        c = f * c + i * torch.tanh(g)
        return o * torch.tanh(c), c

class ConvLSTM(nn.Module):
    def __init__(self, hid_ch=16, kernel=3):
        super().__init__()
        self.hid_ch = hid_ch
        self.cell = ConvLSTMCell(1, hid_ch, kernel)
        self.head = nn.Conv2d(hid_ch, 1, 1)
    def forward(self, x):
        B, T, C, H, W = x.shape
        h = torch.zeros(B, self.hid_ch, H, W, device=x.device)
        c = torch.zeros(B, self.hid_ch, H, W, device=x.device)
        for t in range(T):
            h, c = self.cell(x[:, t], h, c)
        return self.head(h).squeeze(1)

def masked_mse_loss(pred, target, mask):
    m = mask.float()
    return (((pred - target) ** 2) * m).sum() / m.sum().clamp(min=1.0)

def masked_rmse(pred, truth, mask):
    err = (pred - truth)[mask]
    return float(np.sqrt(np.mean(err ** 2)))

def skill(r_model, r_ref):
    return 100.0 * (1.0 - r_model / r_ref)

def pooled_rmse(pred, truth, mask, pix2d):
    sel = mask & pix2d[None, :, :]
    err = (pred - truth)[sel]
    return float(np.sqrt(np.mean(err ** 2))) if err.size else np.nan

def eval_rmse(model, Xtensor, Ynp, Mnp):
    model.eval()
    with torch.no_grad():
        pred = model(Xtensor).cpu().numpy()
    return masked_rmse(pred, Ynp, Mnp)


# %% 2. Climatology + greenness tiers (seed-independent, computed once)
tr_month = Ttr.astype("datetime64[M]").astype(int) % 12
te_month = Tte.astype("datetime64[M]").astype(int) % 12
clim = np.zeros((12, Ytr.shape[1], Ytr.shape[2]), dtype="float32")
for mth in range(12):
    sel = tr_month == mth
    if sel.any():
        clim[mth] = Ytr[sel].mean(axis=0)
pred_clim = np.stack([clim[m] for m in te_month])
rmse_clim = masked_rmse(pred_clim, Yte, Mte)

ndvi_vals, obs_vals = ndvi.values, observed.values
sum_obs = np.where(obs_vals, ndvi_vals, 0.0).sum(axis=0)
cnt_obs = obs_vals.sum(axis=0)
mean_ndvi_pix = np.where(cnt_obs > 0, sum_obs / np.maximum(cnt_obs, 1), np.nan)
q33, q67 = np.percentile(mean_ndvi_pix[~np.isnan(mean_ndvi_pix)], [33, 67])
tiers = {
    "built": mean_ndvi_pix <= q33,
    "mid":   (mean_ndvi_pix > q33) & (mean_ndvi_pix <= q67),
    "green": mean_ndvi_pix > q67,
}
print(f"climatology RMSE: {rmse_clim:.4f} | tier thresholds: {q33:.3f}, {q67:.3f}")


# %% 3. Train across seeds and collect skill (this is the slow cell: ~5 trainings)
def train_one_seed(seed):
    torch.manual_seed(seed); np.random.seed(seed)
    model = ConvLSTM(hid_ch=16).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    best_val, best_state, since = np.inf, None, 0
    for epoch in range(300):
        model.train()
        perm = torch.randperm(n_tr).to(device)
        for i in range(0, n_tr, 16):
            idx = perm[i:i + 16]
            opt.zero_grad()
            loss = masked_mse_loss(model(Xtr2_t[idx]), Ytr2_t[idx], Mtr2_t[idx])
            loss.backward(); opt.step()
        vr = eval_rmse(model, Xval_t, Yval, Mval)
        if vr < best_val - 1e-4:
            best_val, best_state, since = vr, copy.deepcopy(model.state_dict()), 0
        else:
            since += 1
        if since >= 30:
            break
    model.load_state_dict(best_state)
    return model, epoch + 1

seeds = [0, 1, 2, 3, 4]
overall_skills, tier_rows, preds = [], [], []

for s in seeds:
    model, ran = train_one_seed(s)
    with torch.no_grad():
        p = model(Xte_t).cpu().numpy()
    preds.append(p)
    ov = skill(masked_rmse(p, Yte, Mte), rmse_clim)
    ts = {name: skill(pooled_rmse(p, Yte, Mte, pm), pooled_rmse(pred_clim, Yte, Mte, pm))
          for name, pm in tiers.items()}
    overall_skills.append(ov)
    tier_rows.append(ts)
    print(f"seed {s}: test RMSE {masked_rmse(p, Yte, Mte):.4f} | overall skill {ov:+5.1f}% | "
          f"built {ts['built']:+.1f} mid {ts['mid']:+.1f} green {ts['green']:+.1f} | epochs {ran}")


# %% 4. Aggregate across seeds -> the citable headline numbers
overall_skills = np.array(overall_skills)
print("\n=== SEED ROBUSTNESS (n = %d seeds) ===" % len(seeds))
print(f"Overall skill vs climatology: {overall_skills.mean():+.1f}% "
      f"(std {overall_skills.std(ddof=1):.1f}, range {overall_skills.min():+.1f} to {overall_skills.max():+.1f})")

print("\nPer-tier skill (mean +/- std across seeds):")
for name in ["built", "mid", "green"]:
    vals = np.array([r[name] for r in tier_rows])
    print(f"  {name:5s}: {vals.mean():+5.1f}%  +/- {vals.std(ddof=1):.1f}")
mono = all(r["built"] < r["mid"] < r["green"] for r in tier_rows)
print(f"  built < mid < green holds in ALL seeds: {mono}")


# %% 5. Bootstrap the built-vs-green gradient over pixels (significance of the gap)
# Uses the seed-ensemble mean prediction for a stable point estimate; resamples
# pixels (with replacement) to get a 95% CI on (green skill - built skill).
pred_ens = np.mean(preds, axis=0)                      # (T, H, W)
se_m = np.where(Mte, (pred_ens - Yte) ** 2, 0.0).sum(axis=0).ravel()
se_c = np.where(Mte, (pred_clim - Yte) ** 2, 0.0).sum(axis=0).ravel()
cnt  = Mte.sum(axis=0).ravel()

def tier_skill(idx):
    c = cnt[idx].sum()
    if c == 0:
        return np.nan
    return 100.0 * (1.0 - np.sqrt(se_m[idx].sum() / c) / np.sqrt(se_c[idx].sum() / c))

built_ix = np.where(tiers["built"].ravel() & (cnt > 0))[0]
green_ix = np.where(tiers["green"].ravel() & (cnt > 0))[0]

pt_built, pt_green = tier_skill(built_ix), tier_skill(green_ix)
rng = np.random.default_rng(0)
gaps = np.empty(2000)
for b in range(2000):
    gi = rng.choice(green_ix, green_ix.size, replace=True)
    bi = rng.choice(built_ix, built_ix.size, replace=True)
    gaps[b] = tier_skill(gi) - tier_skill(bi)

lo, hi = np.percentile(gaps, [2.5, 97.5])
print("\n=== TIER GRADIENT (seed-ensemble, pixel bootstrap) ===")
print(f"built skill {pt_built:+.1f}% | green skill {pt_green:+.1f}%")
print(f"green - built gap: {pt_green - pt_built:+.1f}%  (95% CI: {lo:+.1f} to {hi:+.1f})")
print("Gap CI excludes 0 -> gradient is significant:" , (lo > 0) or (hi < 0))