# Urban Vegetation (NDVI) Forecasting — Shanghai

One-month-ahead forecasting of urban greenness (NDVI) for central Shanghai using a
**ConvLSTM**, benchmarked honestly against seasonal baselines and evaluated only on
real (non-gap-filled) satellite observations.

> **Headline result:** the ConvLSTM improves one-month-ahead NDVI forecasts over
> seasonal climatology by **+18% ± 1%** (mean ± std across 5 random seeds). The
> improvement is spatially widespread (better at ~78% of pixels) and scales with
> vegetation cover — from **+13.5%** in built-up areas to **+21.2%** in the greenest
> tier (gradient significant, 95% CI on the gap [+6.7, +9.8]).

---

## Overview

Cities are driven by vegetation dynamics that matter for urban heat, air quality, and
liveability. This project asks a focused, testable question:

**Can a spatio-temporal deep-learning model forecast next-month urban greenness better
than simple baselines — and where does it succeed?**

The point is *not* to show that deep learning "works," but to measure exactly how much
it adds over transparent baselines, on real data, with honest handling of missingness.
The whole pipeline runs on free tools (Google Earth Engine + a laptop).

## Results

Evaluated on a held-out test period (last 3 years), scored **only on observed pixels**
(gap-filled values are excluded from all training and evaluation):

| Predictor | Test RMSE (NDVI) | Skill vs. climatology |
|---|---|---|
| Persistence (`next = last month`) | 0.0894 | — |
| Seasonal climatology (`next = typical for that month`) | 0.0884 | baseline |
| **ConvLSTM** (5-seed mean) | **0.0725** | **+18% ± 1%** |

**What the model learned (validated, not just asserted):**

- The edge is **widespread**, not a few lucky pixels — the model beats climatology at
  ~78% of pixels.
- The edge **scales with vegetation**: +13.5% (built-up) → +18.7% (mid) → +21.2%
  (green). This monotonic ordering holds in **all 5 seeds**, and the built-vs-green gap
  is statistically significant (bootstrap 95% CI [+6.7, +9.8]).
- At the **individual-pixel** level the greenness relationship is weak (r ≈ 0.24),
  reflecting the high variance of per-pixel skill estimated from limited cloud-free
  observations — the effect is real in aggregate, noisy per pixel.

Intuition: climatology captures the seasonal cycle and persistence captures month-to-
month inertia; these two near-tie as baselines. The ConvLSTM beats both because it
combines *where* a pixel is with *how it has recently been trending* — structure that
neither baseline can express.

## Method

- **Data:** MODIS `MOD13Q1` NDVI (250 m, 16-day), 2000–2024, from Google Earth Engine,
  over a ~16 km box of central Shanghai (121.40–121.55°E, 31.15–31.30°N; 67×67 grid).
- **Cleaning:** composited to monthly; restricted to 2012–2024 (156 months) where cloud
  coverage supports honest gap-filling; remaining gaps filled by per-pixel temporal
  interpolation **and recorded in an `observed` mask** so filled values are excluded
  from every loss and metric.
- **Baselines:** persistence and seasonal climatology.
- **Model:** a small ConvLSTM (16 hidden channels) reading 12 input months to predict
  the next month, trained with a **masked MSE loss** (observed pixels only), Adam
  (lr 1e-3), and **early stopping (patience 30)** on a validation split.
- **Split:** strictly temporal (last 3 years held out) — no random shuffling, no leakage.
- **Robustness:** 5-seed retraining (mean ± std) + pixel bootstrap on the vegetation
  gradient.

## Repository structure

```
urban-ndvi-forecast/
├── README.md
├── LICENSE
├── environment.yml            # conda environment to reproduce
├── .gitignore
├── src/
│   └── data.py                # shared data contract: windowing + temporal split
├── 01_extract.py              # pull MODIS NDVI from Earth Engine -> data/shanghai_ndvi.zarr
├── 02_explore.py              # seasonality, autocorrelation, missingness diagnostics
├── 03_clean.py                # monthly composite + window + gap-fill -> *_modelready.zarr
├── 04_baselines.py            # persistence + climatology (masked scoring)
├── 05_model.py                # ConvLSTM training -> checkpoints/convlstm.pt
├── 06_analysis.py             # skill map, greenness-tier breakdown, prediction figures
├── 07_robustness.py           # multi-seed + bootstrap (the citable numbers)
├── data/                      # (git-ignored) data cubes; regenerated or from the DOI below
├── checkpoints/               # (git-ignored) trained weights
└── figures/                   # saved figures for the writeup
```

> Scripts are numbered in execution order. `data/` and `checkpoints/` are git-ignored
> (regenerable / large); the model-ready dataset is published separately (see **Data**).

## Reproduce

```bash
# 1. Environment
conda env create -f environment.yml
conda activate urbanveg

# 2. Authenticate Earth Engine (needs a Google account + a Cloud project)
#    Run once; follow the browser prompt.
python -c "import ee; ee.Authenticate()"

# 3. Run the pipeline in order (edit the project id in 01_extract.py first)
python 01_extract.py        # -> data/shanghai_ndvi.zarr
python 03_clean.py          # -> data/shanghai_ndvi_modelready.zarr
python 04_baselines.py      # baseline scoreboard
python 05_model.py          # trains + saves checkpoints/convlstm.pt
python 06_analysis.py       # figures + tier breakdown
python 07_robustness.py     # 5-seed + bootstrap (a few minutes)
```

Steps 02 and 06 produce diagnostic figures and are optional for reproducing the
headline number. Scripts also run cell-by-cell in Spyder (`# %%` cells).

Alternatively, skip step 01 and download the model-ready cube directly from the DOI
below (unzip into `data/`), then start from `04_baselines.py`.

## Data

- **Source:** MODIS `MOD13Q1` via Google Earth Engine (NASA LP DAAC), free for research.
- **Published model-ready dataset:** the cleaned monthly cube + observed mask
  (`shanghai_ndvi_modelready.zarr`) is archived on Zenodo with a citable DOI:
  **[10.5281/zenodo.21345765](https://doi.org/10.5281/zenodo.21345765)** (CC-BY-4.0).

## Limitations

Stated plainly, because they matter and they motivate the next step:

- **Coarse resolution.** 250 m MODIS pixels blend streets, buildings, and trees; this
  cannot resolve intra-urban canopy. (The natural upgrade is 30 m Landsat.)
- **Cloud gaps.** Central Shanghai is cloudy; ~38% of monthly pixels are gap-filled.
  These are excluded from training and scoring, but the underlying record is sparse.
- **Single city, single split.** One study area and one temporal test period; results
  are not claimed to transport to other cities or climates.

## Citation

If you use this code or dataset, please cite:

```
Arone, G. (2026). Urban Vegetation (NDVI) Forecasting — Shanghai [Software].
GitHub: https://github.com/GVALogsEng/urban-ndvi-forecast

Arone, G. (2026). Model-ready monthly NDVI data cube for central Shanghai
(2012–2024) [Data set]. Zenodo. https://doi.org/10.5281/zenodo.21345765
```

## License

Code released under the MIT License (see `LICENSE`). The published dataset is released
under CC-BY-4.0 via the DOI above.

## Acknowledgments

Built with Google Earth Engine, PyTorch, and xarray. MODIS data courtesy of NASA LP DAAC.