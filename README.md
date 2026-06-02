# BrightNTL Code

BrightNTL is an Africa-optimized two-stage nighttime light reconstruction framework for producing annual 500 m nighttime light data. The implementation follows the manuscript methods: Stage I calibrates and desaturates DMSP-OLS time series with a Dynamic Spatio-Temporal Feature Extraction Network (DSTFE), and Stage II reconstructs high-resolution 500 m nighttime lights with population-adaptive sparse convolution and an enhancement-denoising dual-branch network.

## Method Overview

Stage I uses a seven-year temporal window from `Y-3` to `Y+3`, year-difference vectors, saturation-mask vectors, a UNet encoder-decoder, and multi-head temporal attention to generate temporally consistent 1 km calibrated nighttime light fields.

Stage II integrates Stage I output, LandScan population density, DEM-derived terrain relief, NDVI, and MODIS/SDC500 background features. It adapts the receptive field between sparse rural settlements and dense urban agglomerations, enhances weak valid light signals, suppresses geographically induced noise, and outputs 500 m reconstructed radiance.

## Repository Layout

```text
brightntl/
  models/       PyTorch Stage I and Stage II model definitions
  data/         GeoTIFF IO, temporal windows, and dataset classes
  training/     L1-loss training and evaluation loops
  inference/    Annual raster prediction helpers
  metrics/      Radiometric, temporal, super-resolution, and urban metrics
configs/        Default YAML configuration
scripts/        Training, prediction, and validation entry points
examples/       Synthetic forward-pass example
tests/          Synthetic unit tests
```

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Configuration

Edit `configs/brightntl_default.yaml` to match the local project layout. The default configuration follows the manuscript data groups:

- harmonized DMSP-OLS annual GeoTIFFs
- RNTL calibration reference GeoTIFFs
- VIIRS VNL V2 annual GeoTIFFs
- LandScan population density GeoTIFFs
- DEM relief, NDVI, and MODIS/SDC500 auxiliary GeoTIFFs

Annual rasters are expected to follow the naming pattern:

```text
<root>/<year>.tif
```

For example:

```text
data/dmsp_harmonized/2004.tif
data/rntl/2004.tif
data/viirs_vnl_v2/2013.tif
data/landscan/2013.tif
```

## Training

Train Stage I DSTFE:

```bash
python scripts/train_stage1.py \
  --config configs/brightntl_default.yaml \
  --output outputs/checkpoints/stage1_dstfe.pt
```

Train Stage II reconstruction:

```bash
python scripts/train_stage2.py \
  --config configs/brightntl_default.yaml \
  --output outputs/checkpoints/stage2_reconstruction.pt
```

## Prediction

Predict a Stage I annual intermediate raster:

```bash
python scripts/predict_stage1.py \
  --config configs/brightntl_default.yaml \
  --checkpoint outputs/checkpoints/stage1_dstfe.pt \
  --year 2010 \
  --output outputs/stage1_1km/2010.tif
```

Predict an annual 500 m BrightNTL raster:

```bash
python scripts/predict_year.py \
  --config configs/brightntl_default.yaml \
  --checkpoint outputs/checkpoints/stage2_reconstruction.pt \
  --year 2010 \
  --output outputs/brightntl_500m/2010.tif
```

## Validation

Compute basic radiometric accuracy against a reference raster:

```bash
python scripts/validate_product.py \
  --prediction outputs/brightntl_500m/2012.tif \
  --reference data/viirs_vnl_v2/2012.tif
```

The `brightntl.metrics` module also includes PSNR, SSIM, temporal mutation, overall accuracy, Kappa, IoU, relative area accuracy, and expansion-rate error functions used by the manuscript validation workflow.

## Synthetic Check

Run a small forward pass without geospatial files:

```bash
python examples/synthetic_forward.py
```

Run tests:

```bash
pytest
```
