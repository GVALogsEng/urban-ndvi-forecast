# %% Imports and Earth Engine init
import ee
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt

ee.Initialize(project="urbanndvi-501020")

# %% 1. Study area (central Shanghai) and time range
# Small ~16 km core box (~65 x 65 pixels at 250 m). Starting small is deliberate:
# it keeps each pull under Earth Engine's request cap and makes iteration fast.
# Widen later by editing these four numbers: [west, south, east, north].
shanghai = ee.Geometry.Rectangle([121.40, 31.15, 121.55, 31.30])

START = "2000-01-01"
END   = "2025-01-01"

# %% 2. Load MODIS NDVI, mask by quality, apply scale factor, reproject to lat/lon
def clean_ndvi(img):
    good = img.select("SummaryQA").lte(1)
    ndvi = (img.select("NDVI")
               .multiply(0.0001)
               .updateMask(good)
               .reproject(crs="EPSG:4326", scale=250))   # <-- THE FIX: leave sinusoidal behind
    return ndvi.copyProperties(img, ["system:time_start"])

collection = (ee.ImageCollection("MODIS/061/MOD13Q1")
              .filterDate(START, END)
              .filterBounds(shanghai)
              .map(clean_ndvi))

print("Number of images:", collection.size().getInfo())  # expect ~572

# %% 3. Pull to a flat table with getRegion, one year at a time
# THE FIX: crs="EPSG:4326". MODIS is in a sinusoidal projection; without an explicit
# output CRS, getRegion can't intersect it with our lat/lon box (the error you saw).
# Yearly chunks keep each request under the per-call size limit.
frames = []
for yr in range(2000, 2025):
    rows = (collection
            .filterDate(f"{yr}-01-01", f"{yr+1}-01-01")
            .getRegion(geometry=shanghai, scale=250, crs="EPSG:4326")
            .getInfo())
    if len(rows) > 1:                       # row 0 is the column header
        frames.append(pd.DataFrame(rows[1:], columns=rows[0]))
    print(f"  {yr}: {max(len(rows) - 1, 0)} rows")

df = pd.concat(frames, ignore_index=True)
print("Total rows:", len(df))

# %% 3b. Reshape the flat table into a (time, latitude, longitude) cube
# 'time' comes back as epoch milliseconds; NDVI may have nulls (masked pixels) -> NaN.
df["time"] = pd.to_datetime(df["time"], unit="ms")
df["NDVI"] = pd.to_numeric(df["NDVI"], errors="coerce")

ndvi = (df.set_index(["time", "latitude", "longitude"])["NDVI"]
          .to_xarray()
          .sortby("time"))

print(ndvi)
print("Shape (time, lat, lon):", ndvi.shape)

# %% 4. Inspect
print("Time span:", str(ndvi.time.min().values)[:10], "->", str(ndvi.time.max().values)[:10])
print("N timesteps:", ndvi.sizes["time"])
print("Grid:", ndvi.sizes["latitude"], "lat x", ndvi.sizes["longitude"], "lon")

# %% 5. Visual sanity check — mean NDVI over all time
ndvi.mean(dim="time").plot()
plt.title("Shanghai mean NDVI (2000-2024)")
plt.show()

# %% 6. Save the cube so we never re-pull from GEE during development
ndvi.to_dataset(name="NDVI").to_zarr("data/shanghai_ndvi.zarr", mode="w")
print("Saved -> data/shanghai_ndvi.zarr")
