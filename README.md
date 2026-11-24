# EarthINR

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue.svg)](https://www.python.org/downloads/)

**EarthINR** is a pure-PyTorch library for geographic Implicit Neural Representations (INRs), coordinate encoders, embedding raster products, and **CoordBench** downstream evaluation.

Every encoder in EarthINR implements a single `torch.nn.Module` contract (`LocationEncoder`), loads stripped `safetensors` checkpoints with zero third-party wrapper dependencies (`rshf`, `geoclip`, `transformers`, or runtime `git clone`s are not required), and evaluates cleanly across random, regional block-CV, buffered-exclusion, and urban/rural splits.

## Installation

```bash
pip install earthinr
```

Optional extras:

```bash
# AlphaEarth & TESSERA Zarr embedding products (zarr, s3fs, pyproj, requests, numcodecs)
pip install 'earthinr[products]'

# FAISS-accelerated k-NN probing (falls back to scikit-learn when omitted)
pip install 'earthinr[knn]'

# ONNX Runtime inference backend for earthinr-zoo models
pip install "earthinr[onnx]"
```

## Universal Input/Output Contract

- **Spatial input**: Every `LocationEncoder` accepts `(lon, lat)` coordinates in **WGS84 degrees** (`EPSG:4326` / GeoJSON order) with dynamic batch shape `(B, 2)` in `forward(coords, year=None, month=None)` and 1D arrays `(lon, lat)` in `encode(lon, lat, year=None, month=None)`. Models whose upstream weights expect `(lat, lon)` (MIND, GeoCLIP, TaxaBind, SLED, TTE, GT-Loc) handle the swap internally.
- **Temporal input**: Optional `year` (and `month` for Climplicit) are supported directly on `forward` / `encode` with explicit defaults documented below.
- **Output**: `float32` embeddings of shape `(B, embed_dim)`.

## Model & Input/Output Contract Table

All 26 location-encoder-only checkpoints are hosted in [`taylor-geospatial/earthinr-zoo`](https://huggingface.co/taylor-geospatial/earthinr-zoo) as both stripped `<model_id>/model.safetensors` weights and self-contained `<model_id>/model.onnx` graphs (opset 18; opset 20 for `osmgraphclip-*`, dynamic batch `coords: (batch, 2)` — plus `year: (batch,)` for `terranova-st` — verified with `onnxruntime`), reproducible via [`scripts/export_zoo.py`](scripts/export_zoo.py) and [`scripts/export_onnx.py`](scripts/export_onnx.py). MIND remains hosted at [`taylor-geospatial/MIND`](https://huggingface.co/taylor-geospatial/MIND) (`mind.safetensors` and `mind.onnx`). For upcoming and requested encoders not yet integrated, see [docs/CANDIDATE_MODELS.md](docs/CANDIDATE_MODELS.md).

| Preset | Family | Spatial Input | Temporal Input (Default) | Output `embed_dim` | Weight Source | Stripped Size | License |
|---|---|---|---|---:|---|---:|---|
| `sincos` | Analytic | `(lon, lat)` deg, `(B, 2)` | Time-invariant | 4 | None | — | MIT |
| `xyz` | Analytic | `(lon, lat)` deg, `(B, 2)` | Time-invariant | 3 | None | — | MIT |
| `nerf` | Analytic | `(lon, lat)` deg, `(B, 2)` | Time-invariant | 96 | None | — | MIT |
| `spherical-harmonics` | Analytic | `(lon, lat)` deg, `(B, 2)` | Time-invariant | 16 | None | — | MIT |
| `sh40` | Analytic | `(lon, lat)` deg, `(B, 2)` | Time-invariant | 1681 | None | — | MIT |
| `rff3072` | Analytic | `(lon, lat)` deg, `(B, 2)` | Time-invariant | 3072 | None | — | MIT |
| `mind` | Neural INR | `(lon, lat)` deg, `(B, 2)` | Optional `year` (`2021`) | 64 | `taylor-geospatial/MIND` | 216.5 MiB | MIT |
| `mind-3072` | Neural INR | `(lon, lat)` deg, `(B, 2)` | Optional `year` (`2021`) | 3072 | `taylor-geospatial/MIND` | 216.5 MiB | MIT |
| `mind-head` | Neural INR | `(lon, lat)` deg, `(B, 2)` | Optional `year` (`2021`) | 64 | `taylor-geospatial/MIND` | 216.5 MiB | MIT |
| `satclip` | Neural INR | `(lon, lat)` deg, `(B, 2)` | Time-invariant | 256 | `taylor-geospatial/earthinr-zoo` | 4.85 MB | MIT |
| `satclip-vit16-l10` | Neural INR | `(lon, lat)` deg, `(B, 2)` | Time-invariant | 256 | `taylor-geospatial/earthinr-zoo` | 1.78 MB | MIT |
| `satclip-resnet18-l40` | Neural INR | `(lon, lat)` deg, `(B, 2)` | Time-invariant | 256 | `taylor-geospatial/earthinr-zoo` | 4.85 MB | MIT |
| `satclip-resnet18-l10` | Neural INR | `(lon, lat)` deg, `(B, 2)` | Time-invariant | 256 | `taylor-geospatial/earthinr-zoo` | 1.78 MB | MIT |
| `satclip-resnet50-l40` | Neural INR | `(lon, lat)` deg, `(B, 2)` | Time-invariant | 256 | `taylor-geospatial/earthinr-zoo` | 4.85 MB | MIT |
| `satclip-resnet50-l10` | Neural INR | `(lon, lat)` deg, `(B, 2)` | Time-invariant | 256 | `taylor-geospatial/earthinr-zoo` | 1.78 MB | MIT |
| `geoclip` | Neural INR | `(lon, lat)` deg, `(B, 2)` | Time-invariant | 512 | `taylor-geospatial/earthinr-zoo` | 37.8 MB | MIT |
| `sinr` | Neural INR | `(lon, lat)` deg, `(B, 2)` | Time-invariant | 256 | `taylor-geospatial/earthinr-zoo` | 2.11 MB | MIT |
| `climplicit` | Neural INR | `(lon, lat)` deg, `(B, 2)` | Optional `month` 1–12 (`[3,6,9,12]` concat) | 1024 (256 w/ `month`) | `taylor-geospatial/earthinr-zoo` | 16.3 MB | CC-BY-4.0 |
| `gair` | Neural INR | `(lon, lat)` deg, `(B, 2)` | Time-invariant | 768 | `taylor-geospatial/earthinr-zoo` | 40.9 MB | Apache-2.0 |
| `taxabind` | Neural INR | `(lon, lat)` deg, `(B, 2)` | Time-invariant | 512 | `taylor-geospatial/earthinr-zoo` | 37.8 MB | Apache-2.0 |
| `sled` | Neural INR | `(lon, lat)` deg, `(B, 2)` | Time-invariant | 768 | `taylor-geospatial/earthinr-zoo` | 56.7 MB | Apache-2.0 |
| `sled-s2` | Neural INR | `(lon, lat)` deg, `(B, 2)` | Time-invariant | 768 | `taylor-geospatial/earthinr-zoo` | 56.7 MB | Apache-2.0 |
| `tte` | Neural INR | `(lon, lat)` deg, `(B, 2)` | Time-invariant | 512 | `taylor-geospatial/earthinr-zoo` | 13.7 MB | MIT |
| `terranova` | Neural INR | `(lon, lat)` deg, `(B, 2)` | Time-invariant | 256 | `taylor-geospatial/earthinr-zoo` | 200.8 MB | CC-BY-4.0 |
| `terranova-st` | Neural INR | `(lon, lat)` deg, `(B, 2)` | Optional `year` 1900–2035 (`2020`) | 256 | `taylor-geospatial/earthinr-zoo` | 614.9 MB | CC-BY-4.0 |
| `csp-inat` | Neural INR | `(lon, lat)` deg, `(B, 2)` | Time-invariant | 256 | `taylor-geospatial/earthinr-zoo` | 0.79 MB | Unlicensed (pending author confirmation) |
| `csp-fmow` | Neural INR | `(lon, lat)` deg, `(B, 2)` | Time-invariant | 256 | `taylor-geospatial/earthinr-zoo` | 0.79 MB | Unlicensed (pending author confirmation) |
| `osmgraphclip` | Neural INR | `(lon, lat)` deg, `(B, 2)` | Time-invariant | 256 | `taylor-geospatial/earthinr-zoo` | 4.86 MB | Apache-2.0 |
| `osmgraphclip-ms-l10` | Neural INR | `(lon, lat)` deg, `(B, 2)` | Time-invariant | 256 | `taylor-geospatial/earthinr-zoo` | 1.78 MB | Apache-2.0 |
| `osmgraphclip-a-l40` | Neural INR | `(lon, lat)` deg, `(B, 2)` | Time-invariant | 256 | `taylor-geospatial/earthinr-zoo` | 4.86 MB | Apache-2.0 |
| `osmgraphclip-a-l10` | Neural INR | `(lon, lat)` deg, `(B, 2)` | Time-invariant | 256 | `taylor-geospatial/earthinr-zoo` | 1.78 MB | Apache-2.0 |
| `le-sinr` | Neural INR | `(lon, lat)` deg, `(B, 2)` | Time-invariant | 256 | `taylor-geospatial/earthinr-zoo` | 2.37 MB | MIT |
| `gtloc` | Neural INR | `(lon, lat)` deg, `(B, 2)` | Time-invariant | 512 | `taylor-geospatial/earthinr-zoo` | 37.8 MB | CC-BY-NC-4.0 |
| `hydroloc` | Neural INR | `(lon, lat)` deg, `(B, 2)` | Time-invariant | 512 (MRL 64/128/256/512) | `taylor-geospatial/earthinr-zoo` | 4.21 MB | Apache-2.0 |
| `geoclosp` | Neural INR | `(lon, lat)` deg, `(B, 2)` | Time-invariant | 384 | `taylor-geospatial/earthinr-zoo` | 2.18 MB | CreativeML-OpenRAIL-M |
| `aef` | Embedding Product | `(lon, lat)` deg, `(B, 2)` | Optional `year` 2017–2025 (`2021`) | 64 | Zarr S3 (`tge-labs/aef-mosaic`) | Streams on demand | CC-BY-4.0 |
| `tessera` | Embedding Product | `(lon, lat)` deg, `(B, 2)` | Optional `year` 2017–2025 (`2021`) | 128 | Zarr S3 (`tessera/v1.1-dclimate`) | Streams on demand | CC-BY-4.0 |

> **License & attribution note**: Code in this repository is MIT-licensed. Individual pretrained checkpoints carry their respective upstream licenses (including CC-BY-4.0 attribution for Climplicit, TerraNova, AlphaEarth, and TESSERA; **CC-BY-NC-4.0** non-commercial for GT-Loc; **CreativeML-OpenRAIL-M** for GeoCLOSP-RN; MIT for LE-SINR; and Apache-2.0 for HydroLoc, trained by Isaac Corley on the [`isaaccorley/hydro`](https://huggingface.co/datasets/isaaccorley/hydro) dataset with no associated paper). `csp-inat` and `csp-fmow` weights originate from the upstream CSP release (Mai et al., ICML 2023) without an explicit license file and are marked unlicensed pending author confirmation.

## Python API Quickstart

### Direct PyTorch `nn.Module` & Batched NumPy Inference

```python
import numpy as np
import torch

from earthinr.models import SatCLIPLocationEncoder, get_model, list_models

# Inspect all 37 registered presets (no heavy imports required)
print(list_models())

# Load via registry preset or class constructor
encoder = get_model('mind', device='cpu')
# Or: encoder = SatCLIPLocationEncoder(weights='VIT16_L40', device='cpu')

# 1. Direct nn.Module forward pass on (lon, lat) WGS84 degrees
coords = torch.tensor([[-122.4194, 37.7749], [8.5417, 47.3769]])
emb = encoder(coords)  # shape: (2, 64), dtype: float32

# 2. Chunked NumPy inference with optional year / month
lon = np.array([-122.4194, 8.5417])
lat = np.array([37.7749, 47.3769])
year = np.array([2021, 2023])
features = encoder.encode(lon, lat, year=year)  # shape: (2, 64), dtype: float32

# 3. Optional ONNX Runtime inference backend (pip install "earthinr[onnx]")
onnx_enc = get_model('sinr', backend='onnx')
onnx_features = onnx_enc.encode(lon, lat)  # shape: (2, 256), dtype: float32
```

### Downstream Probing & Spatial Baselines

```python
from earthinr.probes import (
    chunked_probe_score,
    kriging_probe_score,
    linear_probe_score,
    spatial_fold_ids,
)

# Nested-CV Ridge probe with exact unpenalized intercept
r2, fold_r2s = linear_probe_score(features, targets, task_type='regression', folds=5)

# Generalized Tikhonov / Chunked Penalty probe for MRL-ordered embeddings
r2_cp, _ = chunked_probe_score(features, targets, task_type='regression', folds=5)

# Spatial block cross-validation & encoder-free Ordinary/Indicator Kriging baseline
folds = spatial_fold_ids(lat, lon, folds=5, cell_deg=10.0, seed=0)
krig_r2, _ = kriging_probe_score(lat, lon, targets, fold_assign=folds, cell_deg=10.0)
```

### Programmatic CoordBench Evaluation

```python
from earthinr.coordbench import (
    CoordConfig,
    CoordEvaluationConfig,
    CoordModelConfig,
    CoordOutputConfig,
    run_coordbench,
)

cfg = CoordConfig(
    model=CoordModelConfig(name='satclip'),
    datasets=('worldclim', 'air_temp'),
    evaluation=CoordEvaluationConfig(
        methods=('linear',), split='both', cell_deg=(2.0, 10.0, 20.0)
    ),
    output=CoordOutputConfig(file='results/satclip.csv', resume=True),
)
results = run_coordbench(cfg)
```

## CLI Quickstart

`earthinr models` and `earthinr coord --dry-run` start instantaneously without importing `torch`, `numpy`, or `pandas`.

```bash
# List all registered model presets or inspect a single preset YAML
earthinr models
earthinr models mind

# Validate configuration and inspect effective YAML without running
earthinr coord --model mind --datasets official --split both --cell-deg 2 10 20 --dry-run

# Run random + multi-cell regional sweep with atomic CSV resume
earthinr coord \
  --model satclip \
  --datasets worldclim satclip-air-temp \
  --methods linear \
  --split both \
  --cell-deg 2 10 20 40 \
  --output results/coordbench.csv \
  --resume

# Run urban/rural stratified or holdout evaluation
earthinr coord \
  --model mind \
  --datasets worldclim-bio1 satclip-country \
  --methods linear chunked \
  --split urban_rural_holdout \
  --cities-path /path/to/cities15000.txt \
  --output results/urban_rural.csv

# Run encoder-free spatial interpolation baselines (IDW or Ordinary/Indicator Kriging)
earthinr coord --model kriging --methods kriging --datasets air_temp --split spatial --cell-deg 10

# Summarize results CSV into per-family and macro rankings
earthinr leaderboard results/coordbench.csv --method linear --split spatial
```

## CoordBench Leaderboard

Published benchmark results are tracked in [`results/coordbench_results.csv`](results/coordbench_results.csv) and can be explored interactively via [`docs/index.html`](docs/index.html) (serve locally with `python -m http.server` or open via GitHub Pages).

Implicit spatial resolution measurements (half-sill distances and suggested sampling grid spacings across all 35 encoders) are documented in [`docs/IMPLICIT_RESOLUTION.md`](docs/IMPLICIT_RESOLUTION.md) and stored in [`results/implicit_resolution.csv`](results/implicit_resolution.csv).

## License

EarthINR is released under the [MIT License](LICENSE). Copyright (c) Taylor Geospatial.
