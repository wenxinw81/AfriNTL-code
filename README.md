# AfriNTL: manuscript-aligned reference implementation

A 500 m annual nighttime light dataset optimized for Africa, 2000–2025.

This repository implements the core equations and evaluation protocols described in the final C13 manuscript and appendix supplied on 6 November 2026. It is a **new reference implementation**, not the recovered source code or checkpoints used to generate published experiments. It does not contain pretrained weights, proprietary LandScan data, manuscript files, or stored performance values. Training this implementation does not automatically reproduce the paper's tables.

## Installation

Python 3.10+ recommended:

```sh
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m unittest discover -p 'test_*.py' -v
```

Run commands from the repository root. CPU works for tests; real training requires suitable GPU memory. The 7×7 contextual branch materializes neighborhoods; use small patches/tiles. Width 32 and encoder depth are engineering defaults because the supplied documents do not fully specify the original layer-by-layer architecture.

## Files

| File | Purpose |
|---|---|
| `models.py` | Stage I spatial encoder/decoder and temporal attention; Stage II joint population support, enhancement, local contextual suppression, geographic fusion |
| `temporal.py` | Seven-position source construction, YD/SM/PM and held-out-reference exclusion |
| `train.py` | Independent Stage I/II training, masked L1, AdamW, cosine schedule, validation checkpointing |
| `predict.py` | Inference on prepared NPZ tensors; retains pre-cutoff predictions |
| `metrics.py` | Radiometric, detection, TNLI, annual growth and urban metrics |
| `raster_eval.py` | Windowed GeoTIFF evaluation on common product/reference masks |
| `preprocess.py` | Explicit threshold and quantile rules |
| `c13.py` | Fixed-domain annual preprocessing effects, including sequential upper-clipping baseline |
| `gdp.py` | Country-level common-year level and consecutive first-difference regressions |
| `PROTOCOL.md` | Inputs, exclusions, units, scope and omitted preprocessing |
| `test_reference.py` | Numerical-domain, leakage, gap handling and model gradient checks |

## Training and prepared data

Stage I production reference targets: 2000, 2003, 2004, 2006; held-out 2010. Five-fold tests independently retrain on the other four reference years. The target-year input is always Li, never target RNTL. Eligible neighboring RNTL inputs, offsets and temporal padding are constructed by `temporal.build_window`. SM means **source mask**, not saturation mask.

Stage II training target: 2013 VNL; 2012 is reserved for evaluation. Precompute Stage I estimates with frozen parameters. Population input is density (persons/km²), not raw persons per native cell. Geography consists of DEM relief and annual NDVI. All input tensors must be cleaned/aligned and spatial validity propagated before training.

Prepare train/validation NPZ files, each with a leading sample dimension:

* Stage I: `x [N,T,1,H,W]`, `yd/sm/pm [N,T]`, `y/mask [N,1,H,W]`, `block_id [N]`.
* Stage II: `stage1 [N,1,H,W]`, `population/pop_valid [N,1,2H,2W]`, `geography [N,2,2H,2W]`, `y/mask [N,1,2H,2W]`, `block_id [N]`.

Split 1°×1° geographic blocks 80/20 within each of the five subregions using seed 42. Keep allocations fixed across variants, exclude holdout blocks before splitting, and prevent neighboring input patch halos from leaking supervision across partitions. `train.py` rejects overlapping block identifiers but cannot infer geographic overlap from identifiers alone.

For Stage II, prepare balanced training batches/records (25% each background, weak, moderate, bright) using **fixed default-2013 stratum labels**: `<0.1`, `0.1..Q25+`, `>Q25+..Q75+`, `>Q75+`. The generic loader shuffles the supplied records; preparation must enforce the balance. Validation should retain its declared sampling population. This release does not claim an automated geospatial patch sampler.

Population descriptor statistics JSON: `{"mean": [intensity_mean, heterogeneity_mean], "std": [intensity_sd, heterogeneity_sd]}`. Fit these on training pixels only, after computing log1p 7×7 mean density and log1p CV. They are saved with each checkpoint.

```sh
python train.py --stage 1 --train data/stage1_train.npz --validation data/stage1_val.npz --output outputs/stage1.pt
python train.py --stage 2 --train data/stage2_train.npz --validation data/stage2_val.npz --population-stats data/population_stats.json --output outputs/stage2.pt
python predict.py --checkpoint outputs/stage2.pt --input data/stage2_inference.npz --output outputs/precutoff_prediction.npz
```

Train each ablation independently. Do not interpret module removal at inference as the trained ablation. Pre-cutoff predictions are essential for C13; zeros in released rasters cannot reveal the removed radiance.

## Raster evaluation: Tables 3–5 / Appendix D

All GeoTIFFs must already share CRS, affine, dimensions and pixel centers. Multiple `--product` arguments enforce a common valid footprint. Observed zeros are retained. Radiometric r² is squared Pearson correlation after ln(1+x); RMSE/Bias/MAE are on original radiance values. No prediction-positive filter is applied. Detection and TNLI use the full domain; radiometric metrics use reference ≥0.1. Negative handling must be declared upstream; it is not silently inferred.

```sh
python raster_eval.py --reference data/VNL/2012.tif --product AfriNTL=data/AfriNTL/2012.tif --product Chen=data/Chen/2012.tif --output outputs/common_comparison.json
python raster_eval.py --reference data/BM2012_QA0_mean.tif --coverage data/BM2012_QA0_valid_fraction.tif --min-coverage 0.8 --product AfriNTL=data/AfriNTL/2012.tif --output outputs/BM.json
```

For BM, source radiance must first be QA-screened and average-resampled independently of QA indicator fractions. Read actual Collection 2 HDF5 scale/offset/fill attributes; never multiply by a previous collection's 0.1 factor. Primary NearNadir QA=0 ≥80% coverage; QA0≥50% and QA0/2≥80% are sensitivities. BM itself defines the reference-light subset. This release accepts already aligned BM inputs; the full HDF5 mosaic pipeline is outside its compact scope.

Country/subregion summaries: pool pixels for Africa, independently calculate each country, then unweighted country means within subregions. ADM2: sum reference/product over identical valid pixels, retain units with ≥1 reference-light pixel and retain zero product totals; compute within-country log1p-total r². These aggregations require separately provided official polygons and are documented rather than bundled with data.

## C13

Edit `c13.example.json` to list every actual unprocessed annual input and an independently prepared 1/0 study-boundary raster. Never point it at already cut-off products and claim historical preprocessing effects. The script intersects all annual validity masks with that boundary and retains valid zeros/subthreshold positives. Missing years are not interpolated; growth is only computed for consecutive calendar years.

```sh
python c13.py --config c13.example.json
```

The final appendix defines the upper-tail row **relative to the lower-only sequence** (`upper_baseline=lower_only`); combined and lower rows use the unprocessed baseline. Set `upper_baseline=raw` only for an explicitly different independent-operation experiment. The quantile population is explicit: final C11 uses positive values after default 0.1 thresholding; some earlier preprocessing code used `gt1`. Do not silently substitute either rule. Quantile domain is configurable (`native` or `fixed`); record the actual preprocessing domain. SD is sample SD, ddof=1; annual growth differences are absolute percentage-point changes. No C13 result values are hard-coded.

## GDP

CSV columns: `ISO3,year,Subregion,gdp` and one TNLI column per product. GDP must use the declared constant-dollar series. Common years are selected jointly across products per country; first differences require consecutive calendar years. Outputs include slopes and sample sizes. Regional/Africa values are unweighted means of country-specific regression R², not pooled country–year fits.

```sh
python gdp.py --panel data/country_annual.csv --products AfriNTL Chen SVNL DeepNTL LRCC --output outputs/gdp
```

## Scope and reproducibility

See `PROTOCOL.md`. The supplied manuscript describes model/evaluation logic but is not sufficient to reconstruct historical checkpoints, all source processing, data splits or every figure. This repository makes implementation choices explicit. Full archive replication additionally requires original data licenses, aligned annual layers, fitted normalization statistics, reference exclusions, block/patch manifests and trained checkpoints. No fitted-to-reference raster adjustment or hypothetical result-generation scripts are included.

Redistribution license has not been selected by the author. No open-source license is implied; obtain author permission for redistribution until one is added.
