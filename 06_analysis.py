# %% 1. Load data + the trained model, and generate all test-set predictions
import os
import sys
import numpy as np
import xarray as xr
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
from src.data import load_cube, make_windows, temporal_split

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# --- Same data + split as everywhere else (the shared contract) ---
ndvi, observed = load_cube()
X, Y, M, times = make_windows(ndvi, observed, k=12)
train, test = temporal_split(X, Y, M, times, test_years=3)
Xtr, Ytr, Mtr, Ttr = train
Xte, Yte, Mte, Tte = test

# --- Rebuild the model class (same definition as 04_model.py) and load weights ---
# (A cleaner repo would import this from src/models.py; duplicated here to avoid
#  importing 04_model.py, which would re-run training.)
class ConvLSTMCell(nn.Module):
    def __init__(self, in_ch, hid_ch, kernel=3):
        super().__init__()
        pad = kernel // 2
        self.conv = nn.Conv2d(in_ch + hid_ch, 4 * hid_ch, kernel, padding=pad)
        self.hid_ch = hid_ch
    def forward(self, x, h, c):
        gates = self.conv(torch.cat([x, h], dim=1))
        i, f, o, g = torch.chunk(gates, 4, dim=1)
        i, f, o = torch.sigmoid(i), torch.sigmoid(f), torch.sigmoid(o)
        g = torch.tanh(g)
        c = f * c + i * g
        h = o * torch.tanh(c)
        return h, c

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

model = ConvLSTM(hid_ch=16).to(device)   # hid_ch MUST match the saved checkpoint
model.load_state_dict(torch.load(os.path.join(_ROOT, "checkpoints", "convlstm.pt"),
                                 map_location=device))
model.eval()

# --- Predictions on the test set: model, climatology, persistence ---
Xte_t = torch.from_numpy(Xte).unsqueeze(2).float().to(device)
with torch.no_grad():
    pred_model = model(Xte_t).cpu().numpy()          # (Nte, H, W)

pred_persist = Xte[:, -1]                             # last input frame

tr_month = Ttr.astype("datetime64[M]").astype(int) % 12
te_month = Tte.astype("datetime64[M]").astype(int) % 12
clim = np.zeros((12, Ytr.shape[1], Ytr.shape[2]), dtype="float32")
for mth in range(12):
    sel = tr_month == mth
    if sel.any():
        clim[mth] = Ytr[sel].mean(axis=0)
pred_clim = np.stack([clim[m] for m in te_month])    # (Nte, H, W)

print("Predictions ready:", pred_model.shape)


# %% 2. Scoring helpers + per-pixel & pooled skill
def masked_rmse(pred, truth, mask):
    err = (pred - truth)[mask]
    return float(np.sqrt(np.mean(err ** 2)))

def skill(r_model, r_ref):
    return 100.0 * (1.0 - r_model / r_ref)

def per_pixel_rmse(pred, truth, mask):
    """RMSE at each pixel over its OBSERVED test months. (H,W), NaN where none."""
    se = (pred - truth) ** 2
    cnt = mask.sum(axis=0)
    ssum = np.where(mask, se, 0.0).sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        rmse = np.sqrt(ssum / cnt)
    rmse[cnt == 0] = np.nan
    return rmse

def pooled_rmse(pred, truth, mask, pix_mask2d):
    """RMSE over observed timesteps AND pixels within pix_mask2d (H,W bool)."""
    sel = mask & pix_mask2d[None, :, :]
    err = (pred - truth)[sel]
    return float(np.sqrt(np.mean(err ** 2))) if err.size else np.nan

# Overall test numbers (sanity: should match 04_model.py).
print("Overall RMSE -> persistence:", round(masked_rmse(pred_persist, Yte, Mte), 4),
      "| climatology:", round(masked_rmse(pred_clim, Yte, Mte), 4),
      "| model:", round(masked_rmse(pred_model, Yte, Mte), 4))

# Helper to put a 2-D array back on the map with correct lat/lon axes.
def to_da(arr2d, name):
    return xr.DataArray(arr2d, dims=["latitude", "longitude"],
                        coords={"latitude": ndvi.latitude,
                                "longitude": ndvi.longitude}, name=name)


# %% 3. WHERE does the model beat climatology? (per-pixel skill map)
rmse_m_pix = per_pixel_rmse(pred_model, Yte, Mte)
rmse_c_pix = per_pixel_rmse(pred_clim,  Yte, Mte)
with np.errstate(invalid="ignore", divide="ignore"):
    skill_pix = 100.0 * (1.0 - rmse_m_pix / rmse_c_pix)
skill_pix[rmse_c_pix < 1e-6] = np.nan     # guard tiny denominators

lim = np.nanpercentile(np.abs(skill_pix), 95)   # symmetric color range
plt.figure(figsize=(6.5, 5))
to_da(skill_pix, "skill %").plot(cmap="RdBu_r", vmin=-lim, vmax=lim,
                                 cbar_kwargs={"label": "ConvLSTM skill vs climatology (%)"})
plt.title("Per-pixel skill: red = model beats climatology, blue = worse")
plt.tight_layout(); plt.show()

print("Per-pixel skill: median", round(np.nanmedian(skill_pix), 1), "%",
      "| % of pixels where model wins:",
      round(100 * np.nanmean(skill_pix > 0), 1), "%")


# %% 4. Greenness tiers (land-cover proxy) — does the edge track vegetation?
# Classify pixels by their long-term OBSERVED-mean NDVI into thirds.
ndvi_vals = ndvi.values
obs_vals = observed.values
sum_obs = np.where(obs_vals, ndvi_vals, 0.0).sum(axis=0)
cnt_obs = obs_vals.sum(axis=0)
mean_ndvi_pix = np.where(cnt_obs > 0, sum_obs / np.maximum(cnt_obs, 1), np.nan)

vals = mean_ndvi_pix[~np.isnan(mean_ndvi_pix)]
q33, q67 = np.percentile(vals, [33, 67])
tiers = {
    "low  (built-up)": mean_ndvi_pix <= q33,
    "mid            ": (mean_ndvi_pix > q33) & (mean_ndvi_pix <= q67),
    "high (green)   ": mean_ndvi_pix > q67,
}

print(f"\nGreenness thresholds (mean NDVI): q33={q33:.3f}, q67={q67:.3f}")
print("tier               | clim RMSE | model RMSE | skill")
print("-" * 56)
for name, pm in tiers.items():
    r_c = pooled_rmse(pred_clim,  Yte, Mte, pm)
    r_m = pooled_rmse(pred_model, Yte, Mte, pm)
    print(f"{name}    |   {r_c:.4f}  |   {r_m:.4f}   | {skill(r_m, r_c):+5.1f}%")

# Show the classification map so you can see the tiers spatially.
tier_map = np.full(mean_ndvi_pix.shape, np.nan)
tier_map[tiers["low  (built-up)"]] = 0
tier_map[tiers["mid            "]] = 1
tier_map[tiers["high (green)   "]] = 2
plt.figure(figsize=(6, 5))
to_da(tier_map, "tier").plot(cmap="YlGn", cbar_kwargs={"label": "0=built  1=mid  2=green"})
plt.title("Greenness tiers (long-term mean NDVI)")
plt.tight_layout(); plt.show()


# %% 5. Confound check — does skill track GREENNESS or just OBSERVED-NESS?
# The green areas were also the best-observed areas, so we must separate them.
obs_frac_pix = Mte.mean(axis=0)          # test-period observed fraction per pixel

def flat_clean(*arrs):
    a = [x.ravel() for x in arrs]
    good = np.ones_like(a[0], dtype=bool)
    for x in a:
        good &= ~np.isnan(x)
    return [x[good] for x in a]

sk, mn, ob = flat_clean(skill_pix, mean_ndvi_pix, obs_frac_pix)

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
axes[0].scatter(mn, sk, s=6, alpha=0.3)
axes[0].axhline(0, color="k", lw=0.8)
axes[0].set_xlabel("pixel mean NDVI (greenness)"); axes[0].set_ylabel("skill vs climatology (%)")
axes[0].set_title(f"skill vs greenness  (r = {np.corrcoef(mn, sk)[0,1]:.2f})")
axes[1].scatter(ob, sk, s=6, alpha=0.3, color="tab:orange")
axes[1].axhline(0, color="k", lw=0.8)
axes[1].set_xlabel("pixel observed fraction"); axes[1].set_ylabel("skill vs climatology (%)")
axes[1].set_title(f"skill vs observed-ness  (r = {np.corrcoef(ob, sk)[0,1]:.2f})")
plt.tight_layout(); plt.show()

print("\nConfound diagnostics:")
print("  corr(greenness, skill)     :", round(np.corrcoef(mn, sk)[0, 1], 3))
print("  corr(observedness, skill)  :", round(np.corrcoef(ob, sk)[0, 1], 3))
print("  corr(greenness, observed)  :", round(np.corrcoef(mn, ob)[0, 1], 3),
      " <- if high, greenness & coverage are entangled")


# %% 6. Prediction pictures — actual vs model vs climatology, a couple of months
te_mnum = Tte.astype("datetime64[M]").astype(int) % 12 + 1     # 1..12

def pick_month(m):
    cand = np.where(te_mnum == m)[0]
    if len(cand) == 0:
        return None
    return cand[int(np.argmax([Mte[c].sum() for c in cand]))]   # best-observed one

picks = [i for i in (pick_month(8), pick_month(2)) if i is not None]  # Aug, Feb

for idx in picks:
    date = str(Tte[idx])[:7]
    actual = np.where(Mte[idx], Yte[idx], np.nan)   # show only real pixels
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, arr, ttl in zip(
        axes,
        [actual, pred_model[idx], pred_clim[idx]],
        [f"Actual  {date}", "ConvLSTM prediction", "Climatology prediction"],
    ):
        im = to_da(arr, "NDVI").plot(ax=ax, cmap="RdYlGn", vmin=0, vmax=0.5, add_colorbar=False)
        ax.set_title(ttl)
    fig.colorbar(im, ax=axes, label="NDVI", shrink=0.8)
    plt.show()

print("\nDone. Figures: per-pixel skill map, tier map, confound scatters, month pictures.")