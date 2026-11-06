# Verification (6 November 2026)

Six protocol tests passed with Python3.9.6, NumPy2.0.2, pandas2.3.3, rasterio1.4.3. One neural-model test passed in a separate Python3 runtime with installed PyTorch. Models were checked for finite/nonnegative outputs, padding invariance, forward/backward gradients and empty population neighborhoods. C13 was tested with generated tiny GeoTIFFs for sequential baselines and missing-year exclusion. Reference leakage exclusion, retained zero predictions, confusion counts and growth gaps were tested.

These are software checks on generated test fixtures, not measured paper experiments. No original model checkpoints, full-scale training rerun, production tile-inference benchmark or reproduction of published performance tables is claimed. The separate environments were used because the bundled runtime's GDAL import stalled after importing PyTorch; an integrated pinned research environment remains recommended for production use.
