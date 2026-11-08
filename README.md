# AfriNTL network

PyTorch reference implementation of the two-stage AfriNTL network for annual nighttime-light reconstruction over Africa.

## Installation

```bash
pip install -r requirements.txt
```

## Network architecture

**Stage I — temporal refinement.** Annual single-channel nighttime-light inputs share a U-Net-style encoder. Year offsets and source indicators are embedded with the annual features. Multihead attention uses the target-year feature as its query and excludes padded temporal positions. The output is a nonnegative refinement on the input grid.

**Stage II — spatial reconstruction.** Stage I output is interpolated to the population grid. Population context blends 3 × 3 and 5 × 5 spatial supports through a learned sigmoid weight. The enhancement branch combines residual, spatial-attention and channel-attention features. The denoising branch aggregates local features using feature similarity and learned intensity response. DEM and NDVI features guide the final fusion of these branches, producing a nonnegative radiance map.

## Files

- `models.py`: Stage I, Stage II and their network modules.
- `requirements.txt`: PyTorch dependency.

## Tensor interface

`B` denotes batch size; `T` is an odd number of chronological temporal positions, with the target year at the center. `H, W` and `H2, W2` are the input and output grid sizes.

### Stage I

| Argument | Shape | Meaning |
| --- | --- | --- |
| `x` | `[B,T,1,H,W]` | Aligned annual nighttime-light inputs |
| `yd` | `[B,T]` | Signed year offsets from the target year |
| `sm` | `[B,T]` | Source indicator: 0 for harmonized Li, 1 for RNTL |
| `pm` | `[B,T]` | Boolean temporal availability mask |
| Output | `[B,1,H,W]` | Refined target-year nighttime light |

The center must be available and use Li (`sm = 0`). Fill padded positions with finite values and mark them false in `pm`. Handle spatial NoData before constructing network inputs.

### Stage II

| Argument | Shape | Meaning |
| --- | --- | --- |
| `stage1` | `[B,1,H,W]` | Stage I output |
| `population` | `[B,1,H2,W2]` | Nonnegative population density (persons/km²) |
| `geography` | `[B,2,H2,W2]` | DEM and NDVI features, in that order |
| `pop_valid` | `[B,1,H2,W2]` | Boolean population-validity mask |
| Output | `[B,1,H2,W2]` | Reconstructed radiance |

Population context uses valid-pixel 7 × 7 means and coefficients of variation. Both descriptors are transformed with `log1p` and standardized using `population_stats`, a dictionary containing two-element `mean` and `std` sequences in that descriptor order. Reuse the statistics fitted on training inputs with the corresponding model weights. Neighborhood means below 1 use a coefficient of variation of zero.

Continuous inputs should be floating-point tensors on the network device. Align all Stage II layers and retain the feature scaling used to fit the model.

## Usage

```python
from models import StageI, StageII

stage_i = StageI(width=32, heads=4)
stage_ii = StageII(width=32, population_stats=population_stats)

# Prepared tensors follow the conventions above.
refined = stage_i(x, yd, sm, pm)
radiance = stage_ii(refined, population, geography, pop_valid)
```

Load model weights with the standard PyTorch `load_state_dict` interface. Pretrained weights are not included.
