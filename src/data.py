"""
src/data.py
-----------
Shared data contract for the Shanghai NDVI forecasting project.

Both the baselines and the ConvLSTM import these functions so they train and
are evaluated on IDENTICAL windows and the IDENTICAL temporal split -- which is
what makes the eventual comparison fair and leak-free.

A "sample" is (X, y, m):
    X : the past `k` frames           -> (k, H, W)
    y : the next frame (the target)   -> (H, W)
    m : observed-mask for y           -> (H, W) bool, True where y is real
The mask travels with every target so filled values can be excluded from both
the loss and the metric. The split also carries its timestamps, so train/test
month labels can never drift out of sync with the arrays.
"""

import os
import numpy as np
import xarray as xr

# Project root = one level up from this file's folder (src/ -> project root).
# Makes data paths work no matter what the working directory is.
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)


def load_cube(path=None):
    """Load the model-ready dataset: filled NDVI + its observed mask."""
    if path is None:
        path = os.path.join(_ROOT, "data", "shanghai_ndvi_modelready.zarr")
    ds = xr.open_zarr(path)
    ndvi = ds["NDVI"].load()          # (time, lat, lon), no NaNs (filled)
    observed = ds["observed"].load()  # (time, lat, lon), bool: True = real obs
    return ndvi, observed


def make_windows(ndvi, observed, k=12):
    """
    Slide a length-k window over time to build supervised examples.
    For each t in [k, T): inputs = frames [t-k, t), target = frame t.

    Returns
    -------
    X : (N, k, H, W) float32  -- input sequences
    Y : (N, H, W)    float32  -- target frames
    M : (N, H, W)    bool     -- observed mask for each target frame
    times : (N,) datetime64   -- timestamp of each target
    """
    arr = ndvi.values.astype("float32")
    obs = observed.values.astype(bool)
    times = ndvi["time"].values
    T = arr.shape[0]

    X, Y, M, ts = [], [], [], []
    for t in range(k, T):
        X.append(arr[t - k:t])   # (k, H, W)
        Y.append(arr[t])         # (H, W)
        M.append(obs[t])         # (H, W) -- mask of the TARGET frame
        ts.append(times[t])
    return np.stack(X), np.stack(Y), np.stack(M), np.array(ts)


def temporal_split(X, Y, M, times, test_years=3):
    """
    Split STRICTLY by time -- never randomly. The last `test_years` years of
    targets become the test set; everything earlier is training.

    Returns the timestamps WITH each split so downstream code never recomputes
    the boundary (which is how train/test can silently drift apart).

    Each returned tuple is (X, Y, M, times).
    """
    times = np.asarray(times)
    yrs = times.astype("datetime64[Y]")
    cutoff = yrs.max() - np.timedelta64(test_years, "Y")
    is_test = yrs > cutoff        # most-recent block held out

    train = (X[~is_test], Y[~is_test], M[~is_test], times[~is_test])
    test  = (X[ is_test], Y[ is_test], M[ is_test], times[ is_test])
    return train, test


if __name__ == "__main__":
    # Run this file directly (F5) for a self-test of the shapes.
    ndvi, observed = load_cube()
    X, Y, M, times = make_windows(ndvi, observed, k=12)
    train, test = temporal_split(X, Y, M, times, test_years=3)
    # train and test are each (X, Y, M, times)
    print(f"train: {train[0].shape[0]} samples | test: {test[0].shape[0]} samples")
    print(f"window k = {train[0].shape[1]} | grid = {train[0].shape[2]}x{train[0].shape[3]}")
    print(f"observed frac -- train: {train[2].mean():.3f} | test: {test[2].mean():.3f}")