# %% Imports and load the raw 16-day cube into memory
import warnings
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

# Small cube (~21 MB) -> pull it fully into memory so everything below is fast
# and free of dask quirks.
ds = xr.open_zarr("data/shanghai_ndvi.zarr")
ndvi = ds["NDVI"].load()

raw_missing = float(ndvi.isnull().mean())
print("RAW 16-day cube")
print("  shape:", ndvi.shape)
print("  overall missing fraction:", round(raw_missing, 3))   # the 0.655 from before

# %% 1. Composite to monthly means -- the 'fill holes for free' step
# Average the (usually two) 16-day observations within each calendar month,
# skipping NaNs. A month is only NaN where BOTH halves were fully clouded.
# 'MS' = month-start. 572 sixteen-day steps -> ~299 monthly steps.
with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=RuntimeWarning)  # silence all-NaN-slice noise
    monthly = ndvi.resample(time="1MS").mean()

mon_missing = float(monthly.isnull().mean())
print("MONTHLY cube")
print("  shape:", monthly.shape)
print("  overall missing fraction:", round(mon_missing, 3))
print(f"  -> missingness went {round(raw_missing,3)}  ->  {round(mon_missing,3)}")

# %% 2. Missingness over time, AFTER monthly compositing
# Compare this to the violent 0-to-1 spikes in the 16-day version. We want to
# see far fewer fully-blank (==1.0) timesteps.
missing_time = monthly.isnull().mean(dim=["latitude", "longitude"])

plt.figure(figsize=(11, 4))
missing_time.plot()
plt.title("Monthly cube: fraction of missing pixels over time")
plt.ylabel("Missing fraction"); plt.ylim(0, 1); plt.grid(alpha=0.3); plt.show()

# %% 3. Per-pixel missingness map, AFTER compositing
# Which pixels are STILL chronically empty? Yellow = bad. This is the map that
# decides whether (and how much) we crop.
missing_map = monthly.isnull().mean(dim="time")

plt.figure(figsize=(6, 5))
missing_map.plot(cmap="viridis")
plt.title("Monthly: per-pixel missing fraction (0 = always present)")
plt.show()

# %% 4. The cropping-decision data: how many pixels are 'good enough'?
# For several missingness thresholds, count how many of the 67x67 = 4489 pixels
# fall under it. This tells us the COST of cropping, non-arbitrarily.
pix_missing = monthly.isnull().mean(dim="time").values
total_pix = pix_missing.size

print("Per-pixel missingness after monthly compositing:")
for thr in [0.10, 0.20, 0.30, 0.50]:
    n = int(np.nansum(pix_missing <= thr))
    print(f"  pixels with <= {int(thr*100):>2}% missing:  {n:>4} / {total_pix}  ({100*n/total_pix:.0f}%)")

plt.figure(figsize=(8, 4))
plt.hist(pix_missing.ravel(), bins=40, color="slateblue", edgecolor="white")
plt.title("Distribution of per-pixel missing fraction (monthly)")
plt.xlabel("Fraction of months missing"); plt.ylabel("Number of pixels")
plt.grid(alpha=0.3); plt.show()

# %% 4b. How much does cropping the TIME window (not pixels) buy us?
# For several start years, restrict to that year onward and recompute:
#   - overall missing fraction
#   - how many pixels clear the 20% / 30% bars
# This decides the window the way Cell 4 decided (and rejected) pixel-cropping.
for start in [2008, 2010, 2012, 2014]:
    sub = monthly.sel(time=slice(f"{start}-01-01", None))
    n_steps = sub.sizes["time"]
    overall = float(sub.isnull().mean())
    pm = sub.isnull().mean(dim="time").values
    n20 = int(np.nansum(pm <= 0.20))
    n30 = int(np.nansum(pm <= 0.30))
    print(f"start {start}:  {n_steps:>3} months | overall missing {overall:.2f} | "
          f"<=20% px: {n20:>4} ({100*n20/pm.size:.0f}%) | "
          f"<=30% px: {n30:>4} ({100*n30/pm.size:.0f}%)")

# %% 5. Signal-preservation check: did compositing keep the season + trend?
# Overlay the 16-day region-mean (noisy) against the monthly region-mean (smooth).
# We want the monthly line to track the sawtooth's shape AND keep the upward
# trend -- i.e. compositing cleaned the noise without destroying the signal.
raw_mean   = ndvi.mean(dim=["latitude", "longitude"])
month_mean = monthly.mean(dim=["latitude", "longitude"])

plt.figure(figsize=(11, 4))
raw_mean.plot(alpha=0.35, label="16-day (raw)")
month_mean.plot(linewidth=1.6, label="monthly (composited)")
plt.title("Region-mean NDVI: 16-day vs monthly")
plt.ylabel("NDVI"); plt.legend(); plt.grid(alpha=0.3); plt.show()

# %% 6. Save the monthly cube (STILL CONTAINS GAPS -- filling is the next decision)
monthly.to_dataset(name="NDVI").to_zarr("data/shanghai_ndvi_monthly.zarr", mode="w")
print("Saved -> data/shanghai_ndvi_monthly.zarr  (gaps not yet filled)")

# %% 7. Restrict to the well-observed window and record what's REAL vs filled
# 2012+ : 156 monthly steps, ~62% observed. Enough real data to learn from and
# score on, and the monthly frames are mostly-real (so filling is light bridging).
monthly = xr.open_zarr("data/shanghai_ndvi_monthly.zarr")["NDVI"].load()
cube = monthly.sel(time=slice("2012-01-01", None))

# The observed mask -- THE key to honesty. True = real measurement, False = gap.
# Later this removes filled values from both the training loss and the metric,
# so filled cells are pure scaffolding and can never inflate the result.
observed = cube.notnull()
print("Window:", str(cube.time.min().values)[:7], "->", str(cube.time.max().values)[:7])
print("Months:", cube.sizes["time"], "| observed fraction:", round(float(observed.mean()), 3))

# %% 8. Fill gaps for INPUT CONTINUITY ONLY (masked out of all learning/scoring).
# Per-pixel along time: interpolate between real obs, fill the ends of each
# series, then a global fallback so no NaN can ever reach the model.
filled = (cube
          .interpolate_na(dim="time", method="linear")   # bridge interior gaps
          .bfill(dim="time").ffill(dim="time"))           # leading/trailing gaps

spatial_mean = filled.mean(dim=["latitude", "longitude"])
filled = filled.fillna(spatial_mean)                       # any all-empty pixel
print("Remaining NaNs after fill:", int(filled.isnull().sum()))   # must print 0

# %% 9. Save the modeling-ready cube AND its observed-mask companion
out = xr.Dataset({"NDVI": filled, "observed": observed})
out.to_zarr("data/shanghai_ndvi_modelready.zarr", mode="w")
print("Saved -> data/shanghai_ndvi_modelready.zarr  (filled NDVI + observed mask)")