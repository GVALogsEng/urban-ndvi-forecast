# %% Imports and load the saved cube (no Earth Engine needed)
import os
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

print("Working directory:", os.getcwd())   # should be your project root

# Loads from disk in a second — no re-pulling from GEE ever again.
ds = xr.open_zarr("data/shanghai_ndvi.zarr")
ndvi = ds["NDVI"]
print(ndvi)

# %% 1. Orientation: shape, time span, value range, overall missingness
vals = ndvi.values
print("Shape (time, lat, lon):", ndvi.shape)
print("Time span:", str(ndvi.time.min().values)[:10], "->", str(ndvi.time.max().values)[:10])
print("NDVI  min / mean / max:",
      round(float(np.nanmin(vals)), 3),
      round(float(np.nanmean(vals)), 3),
      round(float(np.nanmax(vals)), 3))
print("Overall missing fraction:", round(float(np.isnan(vals).mean()), 3))

# %% 2. The whole-region signal: spatial-mean NDVI over time
# Average across all pixels at each timestep -> one number per 16-day step.
# The clearest single view of what you're forecasting: you should see a
# repeating yearly sawtooth (the seasonal cycle).
region_mean = ndvi.mean(dim=["latitude", "longitude"])   # skips NaN by default

plt.figure(figsize=(11, 4))
region_mean.plot()
plt.title("Shanghai region-mean NDVI over time (2000-2024)")
plt.ylabel("NDVI"); plt.grid(alpha=0.3); plt.show()

# %% 3. The seasonal cycle (a.k.a. climatology)
# Group every timestep by calendar month, average across all 25 years.
# This curve IS the 'seasonal climatology' baseline we build in Step 6 --
# seeing it now previews the bar your model will have to beat.
seasonal = region_mean.groupby("time.month").mean()

plt.figure(figsize=(8, 4))
seasonal.plot(marker="o")
plt.title("Average seasonal cycle of NDVI (mean by month)")
plt.xlabel("Month"); plt.ylabel("Mean NDVI")
plt.xticks(range(1, 13)); plt.grid(alpha=0.3); plt.show()

# %% 4. A few individual pixel time series
# Single pixels are noisier than the regional mean. Plotting a few interior
# spots shows real trajectories and how locations differ. Edit the index pairs.
pixels = [(20, 20), (33, 33), (50, 50)]   # interior locations on the 67x67 grid

plt.figure(figsize=(11, 4))
for (i, j) in pixels:
    s = ndvi.isel(latitude=i, longitude=j)
    s.plot(label=f"({float(s.latitude):.3f}, {float(s.longitude):.3f})", alpha=0.8)
plt.title("NDVI time series at individual pixels")
plt.ylabel("NDVI"); plt.legend(fontsize=8); plt.grid(alpha=0.3); plt.show()

# %% 5. Missingness over time: are the gaps clustered?
# Fraction of pixels NaN at each timestep. Spikes in certain seasons (cloudy
# summers) or years tell us where the quality mask bites -- those gaps have
# to be filled before modeling.
missing_time = ndvi.isnull().mean(dim=["latitude", "longitude"])

plt.figure(figsize=(11, 4))
missing_time.plot()
plt.title("Fraction of missing (masked) pixels over time")
plt.ylabel("Missing fraction"); plt.grid(alpha=0.3); plt.show()

# %% 6. Missingness per pixel: a map of where data is chronically absent
# Fraction of timesteps each pixel is NaN. Highlights the grid-edge artifacts
# from the reshape and any persistently cloudy/water spots.
missing_map = ndvi.isnull().mean(dim="time")

plt.figure(figsize=(6, 5))
missing_map.plot(cmap="viridis")
plt.title("Per-pixel missing fraction (0 = always present, 1 = always missing)")
plt.show()

# %% 7. Value distribution: a final scale-factor sanity check
# Vegetation should pile up in a sensible positive range; nothing in the thousands.
flat = vals[~np.isnan(vals)]

plt.figure(figsize=(8, 4))
plt.hist(flat, bins=60, color="seagreen", edgecolor="white")
plt.title("Distribution of all NDVI values")
plt.xlabel("NDVI"); plt.ylabel("Pixel-timesteps"); plt.grid(alpha=0.3); plt.show()

# %% 8. Autocorrelation at lag 1: why 'persistence' is a real baseline
# Each step's region-mean NDVI vs the previous step's. A tight diagonal means
# 'next step looks like this step' -- persistence is hard to beat, which is
# exactly why it's a baseline.
m = region_mean.values
x, y = m[:-1], m[1:]
mask = ~np.isnan(x) & ~np.isnan(y)
r = np.corrcoef(x[mask], y[mask])[0, 1]

plt.figure(figsize=(5, 5))
plt.scatter(x[mask], y[mask], s=10, alpha=0.5)
plt.plot([np.nanmin(m), np.nanmax(m)], [np.nanmin(m), np.nanmax(m)], "r--", lw=1)
plt.title(f"Lag-1 autocorrelation (r = {r:.3f})")
plt.xlabel("NDVI at step t"); plt.ylabel("NDVI at step t+1")
plt.grid(alpha=0.3); plt.show()