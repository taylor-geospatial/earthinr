# NeuralFTW

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://github.com/taylor-geospatial/neuralftw/blob/main/LICENSE)
[![Python 3.13+](https://img.shields.io/badge/python-3.13%2B-blue.svg)](https://www.python.org/downloads/)
[![Leaderboard](https://img.shields.io/badge/CoordBench-Leaderboard-0ea5e9.svg)](https://research.taylorgeospatial.org/neuralftw/)
[![WebGPU Demo](https://img.shields.io/badge/WebGPU-Live%20Demo-22c55e.svg)](https://research.taylorgeospatial.org/neuralftw/demo.html)

**NeuralFTW** is a pure-PyTorch library for geographic Implicit Neural Representations (INRs), coordinate encoders, embedding raster products, and **CoordBench** downstream evaluation.

Every encoder in NeuralFTW implements a single `torch.nn.Module` contract (`LocationEncoder`), loads stripped `safetensors` checkpoints with zero third-party wrapper dependencies (`rshf`, `geoclip`, `transformers`, or runtime `git clone`s are not required), and evaluates cleanly across random, regional block-CV, buffered-exclusion, and urban/rural splits.

- **CoordBench Leaderboard**: https://research.taylorgeospatial.org/neuralftw/
- **Live Browser WebGPU Demo**: https://research.taylorgeospatial.org/neuralftw/demo.html
- **Model Zoo (Safetensors + ONNX FP32/FP16/INT8/INT4)**: https://huggingface.co/taylor-geospatial/neuralftw-zoo

## Installation

```bash
pip install neuralftw
```

Optional extra (only needed for streaming cloud Zarr embedding products):

```bash
# AlphaEarth & TESSERA Zarr embedding products (zarr, s3fs, pyproj, requests, numcodecs)
pip install 'neuralftw[products]'
```

`onnxruntime`, `matplotlib`, and `torchid>=0.6.0` (pure PyTorch, no FAISS) are included by default.

## Universal Input/Output Contract

- **Spatial input**: Every `LocationEncoder` accepts `(lon, lat)` coordinates in **WGS84 degrees** (`EPSG:4326` / GeoJSON order) with dynamic batch shape `(B, 2)` in `forward(coords, year=None, month=None)` and 1D arrays `(lon, lat)` in `encode(lon, lat, year=None, month=None)`. Models whose upstream weights expect `(lat, lon)` (MIND, GeoCLIP, TaxaBind, SLED, TTE, GT-Loc) handle the swap internally.
- **Temporal input**: Optional `year` (and `month` for Climplicit) are supported directly on `forward` / `encode` with explicit defaults documented below.
- **Output**: `float32` embeddings of shape `(B, embed_dim)`.

## Model & Input/Output Contract Table

All 26 location-encoder-only checkpoints are hosted in [`taylor-geospatial/neuralftw-zoo`](https://huggingface.co/taylor-geospatial/neuralftw-zoo) as stripped `<model_id>/model.safetensors` weights and self-contained dynamic-batch ONNX graphs (`model.onnx`, plus quantized `model_fp16.onnx`, `model_int8.onnx`, and `model_int4_bs32.onnx` across the 24 spatial models and quantized `mind`), reproducible via [`scripts/export_zoo.py`](https://github.com/taylor-geospatial/neuralftw/blob/main/scripts/export_zoo.py) and [`scripts/export_onnx.py`](https://github.com/taylor-geospatial/neuralftw/blob/main/scripts/export_onnx.py). MIND FP32 weights remain hosted at [`taylor-geospatial/MIND`](https://huggingface.co/taylor-geospatial/MIND). For upcoming and requested encoders not yet integrated, see [`docs/CANDIDATE_MODELS.md`](https://github.com/taylor-geospatial/neuralftw/blob/main/docs/CANDIDATE_MODELS.md).

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
| `satclip` | Neural INR | `(lon, lat)` deg, `(B, 2)` | Time-invariant | 256 | `taylor-geospatial/neuralftw-zoo` | 4.85 MB | MIT |
| `satclip-vit16-l10` | Neural INR | `(lon, lat)` deg, `(B, 2)` | Time-invariant | 256 | `taylor-geospatial/neuralftw-zoo` | 1.78 MB | MIT |
| `satclip-resnet18-l40` | Neural INR | `(lon, lat)` deg, `(B, 2)` | Time-invariant | 256 | `taylor-geospatial/neuralftw-zoo` | 4.85 MB | MIT |
| `satclip-resnet18-l10` | Neural INR | `(lon, lat)` deg, `(B, 2)` | Time-invariant | 256 | `taylor-geospatial/neuralftw-zoo` | 1.78 MB | MIT |
| `satclip-resnet50-l40` | Neural INR | `(lon, lat)` deg, `(B, 2)` | Time-invariant | 256 | `taylor-geospatial/neuralftw-zoo` | 4.85 MB | MIT |
| `satclip-resnet50-l10` | Neural INR | `(lon, lat)` deg, `(B, 2)` | Time-invariant | 256 | `taylor-geospatial/neuralftw-zoo` | 1.78 MB | MIT |
| `geoclip` | Neural INR | `(lon, lat)` deg, `(B, 2)` | Time-invariant | 512 | `taylor-geospatial/neuralftw-zoo` | 37.8 MB | MIT |
| `sinr` | Neural INR | `(lon, lat)` deg, `(B, 2)` | Time-invariant | 256 | `taylor-geospatial/neuralftw-zoo` | 2.11 MB | MIT |
| `climplicit` | Neural INR | `(lon, lat)` deg, `(B, 2)` | Optional `month` 1–12 (`[3,6,9,12]` concat) | 1024 (256 w/ `month`) | `taylor-geospatial/neuralftw-zoo` | 16.3 MB | CC-BY-4.0 |
| `gair` | Neural INR | `(lon, lat)` deg, `(B, 2)` | Time-invariant | 768 | `taylor-geospatial/neuralftw-zoo` | 40.9 MB | Apache-2.0 |
| `taxabind` | Neural INR | `(lon, lat)` deg, `(B, 2)` | Time-invariant | 512 | `taylor-geospatial/neuralftw-zoo` | 37.8 MB | Apache-2.0 |
| `sled` | Neural INR | `(lon, lat)` deg, `(B, 2)` | Time-invariant | 768 | `taylor-geospatial/neuralftw-zoo` | 56.7 MB | Apache-2.0 |
| `sled-s2` | Neural INR | `(lon, lat)` deg, `(B, 2)` | Time-invariant | 768 | `taylor-geospatial/neuralftw-zoo` | 56.7 MB | Apache-2.0 |
| `tte` | Neural INR | `(lon, lat)` deg, `(B, 2)` | Time-invariant | 512 | `taylor-geospatial/neuralftw-zoo` | 13.7 MB | MIT |
| `terranova` | Neural INR | `(lon, lat)` deg, `(B, 2)` | Time-invariant | 256 | `taylor-geospatial/neuralftw-zoo` | 200.8 MB | CC-BY-4.0 |
| `terranova-st` | Neural INR | `(lon, lat)` deg, `(B, 2)` | Optional `year` 1900–2035 (`2020`) | 256 | `taylor-geospatial/neuralftw-zoo` | 614.9 MB | CC-BY-4.0 |
| `csp-inat` | Neural INR | `(lon, lat)` deg, `(B, 2)` | Time-invariant | 256 | `taylor-geospatial/neuralftw-zoo` | 0.79 MB | Unlicensed (pending author confirmation) |
| `csp-fmow` | Neural INR | `(lon, lat)` deg, `(B, 2)` | Time-invariant | 256 | `taylor-geospatial/neuralftw-zoo` | 0.79 MB | Unlicensed (pending author confirmation) |
| `osmgraphclip` | Neural INR | `(lon, lat)` deg, `(B, 2)` | Time-invariant | 256 | `taylor-geospatial/neuralftw-zoo` | 4.86 MB | Apache-2.0 |
| `osmgraphclip-ms-l10` | Neural INR | `(lon, lat)` deg, `(B, 2)` | Time-invariant | 256 | `taylor-geospatial/neuralftw-zoo` | 1.78 MB | Apache-2.0 |
| `osmgraphclip-a-l40` | Neural INR | `(lon, lat)` deg, `(B, 2)` | Time-invariant | 256 | `taylor-geospatial/neuralftw-zoo` | 4.86 MB | Apache-2.0 |
| `osmgraphclip-a-l10` | Neural INR | `(lon, lat)` deg, `(B, 2)` | Time-invariant | 256 | `taylor-geospatial/neuralftw-zoo` | 1.78 MB | Apache-2.0 |
| `le-sinr` | Neural INR | `(lon, lat)` deg, `(B, 2)` | Time-invariant | 256 | `taylor-geospatial/neuralftw-zoo` | 2.37 MB | MIT |
| `gtloc` | Neural INR | `(lon, lat)` deg, `(B, 2)` | Time-invariant | 512 | `taylor-geospatial/neuralftw-zoo` | 37.8 MB | CC-BY-NC-4.0 |
| `hydroloc` | Neural INR | `(lon, lat)` deg, `(B, 2)` | Time-invariant | 512 (MRL 64/128/256/512) | `taylor-geospatial/neuralftw-zoo` | 4.21 MB | Apache-2.0 |
| `geoclosp` | Neural INR | `(lon, lat)` deg, `(B, 2)` | Time-invariant | 384 | `taylor-geospatial/neuralftw-zoo` | 2.18 MB | CreativeML-OpenRAIL-M |
| `aef` | Embedding Product | `(lon, lat)` deg, `(B, 2)` | Optional `year` 2017–2025 (`2021`) | 64 | Zarr S3 (`tge-labs/aef-mosaic`) | Streams on demand | CC-BY-4.0 |
| `tessera` | Embedding Product | `(lon, lat)` deg, `(B, 2)` | Optional `year` 2017–2025 (`2021`) | 128 | Zarr S3 (`tessera/v1.1-dclimate`) | Streams on demand | CC-BY-4.0 |

> **License & attribution note**: Code in this repository is MIT-licensed. Bundled urban/rural city coordinates (`cities15000.parquet`) are derived from [GeoNames](https://www.geonames.org/) under CC-BY-4.0. Individual pretrained checkpoints carry their respective upstream licenses (including CC-BY-4.0 attribution for Climplicit, TerraNova, AlphaEarth, and TESSERA; **CC-BY-NC-4.0** non-commercial for GT-Loc; **CreativeML-OpenRAIL-M** for GeoCLOSP-RN; MIT for LE-SINR; and Apache-2.0 for HydroLoc, trained by Isaac Corley on [`isaaccorley/hydro`](https://huggingface.co/datasets/isaaccorley/hydro) with no associated paper). `csp-inat` and `csp-fmow` weights originate from the upstream CSP release (Mai et al., ICML 2023) without an explicit license file and are marked unlicensed pending author confirmation.

## Python API Quickstart

### Direct PyTorch `nn.Module` & Batched NumPy Inference

```python
import numpy as np
import torch

import neuralftw

# Inspect all 37 registered presets (no heavy imports required)
print(neuralftw.list_models())

# Load via registry preset (or neuralftw.models.SatCLIPLocationEncoder(...))
encoder = neuralftw.get_model('mind', device='cpu')

# 1. Direct nn.Module forward pass on (lon, lat) WGS84 degrees
coords = torch.tensor([[-122.4194, 37.7749], [8.5417, 47.3769]])
emb = encoder(coords)  # shape: (2, 64), dtype: float32

# 2. Chunked NumPy inference with optional year / month
lon = np.array([-122.4194, 8.5417])
lat = np.array([37.7749, 47.3769])
year = np.array([2021, 2023])
features = encoder.encode(lon, lat, year=year)  # shape: (2, 64), dtype: float32

# 3. ONNX Runtime backend with quantized weights (FP32, FP16, INT8, INT4)
onnx_enc = neuralftw.get_model('geoclip', backend='onnx', precision='int8')
onnx_features = onnx_enc.encode(lon, lat)  # shape: (2, 512), dtype: float32
```

### Grid Inference & Plotting

Generate a regular `(lon, lat)` grid, run batched inference, and project embeddings to RGB:

```python
import neuralftw

# Global 1-degree grid (or pass bounds=(lon_min, lat_min, lon_max, lat_max) for a region)
lon, lat, emb = neuralftw.encode_grid('sinr', step_deg=1.0, batch_size=8192)

# Pure-NumPy PCA RGB projection (works without matplotlib)
rgb = neuralftw.embeddings_to_rgb(emb, components=(0, 1, 2))

# Or plot directly with matplotlib
fig, ax = neuralftw.plot_embedding_map(lon, lat, emb, title='SINR (PC 1–3)')
```

### Adding a Custom Model

Subclass `neuralftw.LocationEncoder` and implement `forward(coords, year=None, month=None)` accepting `(B, 2)` `(lon, lat)` degrees. You can register it in Python or pass it directly to the CLI:

```python
import torch
from torch import Tensor

import neuralftw


@neuralftw.register_model('my-encoder', description='Custom coordinate MLP.')
class MyEncoder(neuralftw.LocationEncoder):
    name = 'my-encoder'
    embed_dim = 64

    def __init__(self, hidden_dim: int = 128, device: str = 'cpu') -> None:
        super().__init__(device=device)
        self.net = torch.nn.Sequential(
            torch.nn.Linear(2, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_dim, self.embed_dim),
        ).to(self.device)

    def forward(
        self, coords: Tensor, year: Tensor | None = None, month: Tensor | None = None
    ) -> Tensor:
        return self.net(coords.to(torch.float32))


enc = neuralftw.get_model('my-encoder', hidden_dim=256)
lon, lat, emb = neuralftw.encode_grid(enc, step_deg=2.0)
```

Or evaluate any importable `LocationEncoder` subclass directly from the CLI without writing a YAML config:

```bash
neuralftw coord \
  --model my-encoder \
  --target my_pkg.models.MyEncoder \
  --model-kwarg hidden_dim=256 \
  --datasets worldclim air_temp \
  --split both
```

### Downstream Probing & Spatial Baselines

```python
from neuralftw.probes import (
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
from neuralftw.coordbench import (
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

`neuralftw models`, `neuralftw datasets`, and `neuralftw coord --dry-run` start instantaneously without importing `torch`, `numpy`, or `pandas`.

```bash
# List registered model presets or inspect a single preset YAML
neuralftw models
neuralftw models mind

# List CoordBench dataset families and benchmark names
neuralftw datasets
neuralftw datasets --official

# Validate configuration and inspect effective YAML without running
neuralftw coord --model mind --datasets official --split both --cell-deg 2 10 20 --dry-run

# Run random + multi-cell regional sweep with atomic CSV resume
neuralftw coord \
  --model satclip \
  --datasets worldclim satclip-air-temp \
  --methods linear \
  --split both \
  --cell-deg 2 10 20 40 \
  --output results/coordbench.csv \
  --resume

# Run urban/rural stratified or holdout evaluation (uses bundled GeoNames cities15000)
neuralftw coord \
  --model mind \
  --datasets worldclim-bio1 satclip-country \
  --methods linear chunked \
  --split urban_rural_holdout \
  --output results/urban_rural.csv

# Run encoder-free spatial interpolation baselines (IDW or Ordinary/Indicator Kriging)
neuralftw coord --model kriging --methods kriging --datasets air_temp --split spatial --cell-deg 10

# Summarize results CSV into per-family and macro rankings
neuralftw leaderboard results/coordbench_results.csv --method linear --split spatial
```

## Benchmarks & Analysis

- **CoordBench Leaderboard**: [interactive leaderboard](https://research.taylorgeospatial.org/neuralftw/) backed by [`results/coordbench_results.csv`](https://github.com/taylor-geospatial/neuralftw/blob/main/results/coordbench_results.csv).
- **Implicit Spatial Resolution & Intrinsic Dimension**: half-sill distances, suggested sampling grids, and `torchid` intrinsic dimensionality across 35 encoders in [`docs/IMPLICIT_RESOLUTION.md`](https://github.com/taylor-geospatial/neuralftw/blob/main/docs/IMPLICIT_RESOLUTION.md) ([`implicit_resolution.csv`](https://github.com/taylor-geospatial/neuralftw/blob/main/results/implicit_resolution.csv), [`intrinsic_dim.csv`](https://github.com/taylor-geospatial/neuralftw/blob/main/results/intrinsic_dim.csv)).
- **ONNX Quantization**: FP32, mixed-FP16, weight-only INT8/INT4, and browser WebGPU benchmarks in [`docs/QUANTIZATION.md`](https://github.com/taylor-geospatial/neuralftw/blob/main/docs/QUANTIZATION.md).

## License

NeuralFTW is released under the [MIT License](https://github.com/taylor-geospatial/neuralftw/blob/main/LICENSE). Copyright (c) Taylor Geospatial.
