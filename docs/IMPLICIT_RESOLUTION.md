# Implicit Spatial Resolution of Geographic Location Encoders

Continuous coordinate encoders and geographic Implicit Neural Representations (INRs) map any `(lon, lat)` coordinate to an embedding vector, so they can be queried on arbitrarily fine grids. Querying an encoder far below the distance over which its representations vary creates nearly redundant neighboring features and increases compute and storage cost without adding spatial detail, whereas querying far above that scale aliases local structure that the encoder can resolve. Measuring an encoder's intrinsic spatial scale therefore informs grid spacing for precomputed embedding rasters, spatial cross-validation block sizes, and model selection for local versus continental tasks.

## Half-Sill Characteristic Distance

EarthINR quantifies implicit spatial resolution through the isotropic empirical semivariogram of an encoder $f(x) \in \mathbb{R}^D$:

$$\gamma(h) = \frac{1}{2}\,\mathbb{E}\left[\|f(x_i) - f(x_j)\|_2^2\right]$$

where $h$ is the great-circle separation between coordinates $x_i$ and $x_j$ on a sphere of radius $R = 6{,}371\text{ km}$. In [`src/earthinr/analysis/variogram.py`](../src/earthinr/analysis/variogram.py), the default `geodesic` estimator displaces each sampled anchor point along a uniform random bearing by an exact great-circle lag $h$. Solving the direct spherical geodesic avoids the $\cos(\text{lat})$ distortion of degree-space offsets toward the poles and keeps pair counts constant across short and long lags.

From the resulting curve $\gamma(h)$, `half_sill_distance` computes three quantities:

- **Sill**: the median semivariance across the largest quartile of valid lag bins (`tail_start = max(1, int(0.75 * n_valid))`), representing the plateau variance between distant locations.
- **Half-sill distance (`half_sill_km`)**: the smallest lag where $\gamma(h)$ crosses $0.5 \times \text{sill}$, found by linear interpolation along the $\ln(h)$ axis between the two bracketing lags.
- **Suggested sampling grid (`suggested_grid_km`, `suggested_grid_deg`)**: a Nyquist-style upper bound on grid spacing given by `half_sill_km / 2`, converted to great-circle degrees using `KM_PER_DEGREE = 2 * pi * 6371.0 / 360 ≈ 111.195 km/deg`.

Because embeddings are evaluated without L2 normalization by default (`normalize=False`), `sill` reflects each encoder's output norm and dimensionality, whereas `half_sill_km` is invariant to positive scalar rescaling of the embedding space.

## Measurement Setup

All 35 encoders in [`results/implicit_resolution.csv`](../results/implicit_resolution.csv) were evaluated under one shared protocol:

- **Reference locations**: `16,042` global land points from `worldclim-bio1`.
- **Lag schedule**: `48` logarithmically spaced great-circle lags from `1 km` to `8,000 km` (`min_km=1.0`, `max_km=8000.0`, `n_bins=48`).
- **Estimator**: spherical geodesic offset (`method='geodesic'`) with `pairs_per_lag=2048`, `normalize=False`, `frac=0.5`, and `seed=0`.
- **Temporal inputs**: fixed at `year=2020` and `month=6` for models that accept temporal arguments (`mind`, `mind-3072`, `mind-head`, `terranova-st`, and `climplicit`, where fixing `month=6` produces the 256-d monthly embedding).

## Measured Resolution Across 35 Encoders

Values below are copied verbatim from [`results/implicit_resolution.csv`](../results/implicit_resolution.csv) and ordered from finest to coarsest half-sill distance.

| Model | Dim | Half-Sill (km) | Suggested Grid (km) | Suggested Grid (°) | Sill |
|---|---:|---:|---:|---:|---:|
| `nerf` | 96 | 14.5878 | 7.2939 | 0.065595 | 45.890705 |
| `geoclip` | 512 | 138.8173 | 69.4086 | 0.624207 | 0.092998 |
| `gtloc` | 512 | 177.8062 | 88.9031 | 0.799525 | 0.203478 |
| `mind-3072` | 3072 | 205.648 | 102.824 | 0.924719 | 19890.805902 |
| `rff3072` | 3072 | 239.3966 | 119.6983 | 1.076473 | 1339.265332 |
| `terranova` | 256 | 324.772 | 162.386 | 1.460372 | 251.148706 |
| `sh40` | 1681 | 355.8483 | 177.9242 | 1.60011 | 159.470793 |
| `terranova-st` | 256 | 373.2188 | 186.6094 | 1.678219 | 237.700912 |
| `osmgraphclip-a-l40` | 256 | 421.5197 | 210.7599 | 1.895409 | 3146.35185 |
| `satclip` | 256 | 432.5605 | 216.2802 | 1.945055 | 1573.341152 |
| `satclip-resnet50-l40` | 256 | 432.8409 | 216.4205 | 1.946316 | 1431.084287 |
| `satclip-resnet18-l40` | 256 | 435.0477 | 217.5239 | 1.956239 | 1319.1768 |
| `tte` | 512 | 513.8007 | 256.9003 | 2.31036 | 0.915659 |
| `sled` | 768 | 535.9023 | 267.9512 | 2.409743 | 108.086118 |
| `osmgraphclip` | 256 | 543.4538 | 271.7269 | 2.443699 | 1738.789815 |
| `taxabind` | 512 | 559.7068 | 279.8534 | 2.516782 | 0.076297 |
| `climplicit` | 256 | 574.207 | 287.1035 | 2.581984 | 155.950813 |
| `sled-s2` | 768 | 587.9433 | 293.9716 | 2.64375 | 164.836897 |
| `mind-head` | 64 | 640.7829 | 320.3914 | 2.881349 | 1504.233324 |
| `hydroloc` | 512 | 652.4071 | 326.2036 | 2.933619 | 1092.009936 |
| `mind` | 64 | 978.2283 | 489.1141 | 4.398709 | 1136.3257 |
| `geoclosp` | 384 | 1151.1933 | 575.5967 | 5.176465 | 0.597009 |
| `osmgraphclip-a-l10` | 256 | 1202.9083 | 601.4542 | 5.409007 | 2454.718249 |
| `osmgraphclip-ms-l10` | 256 | 1204.0294 | 602.0147 | 5.414048 | 1044.323851 |
| `satclip-resnet50-l10` | 256 | 1233.8286 | 616.9143 | 5.548044 | 748.023323 |
| `gair` | 768 | 1236.3643 | 618.1821 | 5.559445 | 1.0005 |
| `satclip-resnet18-l10` | 256 | 1245.954 | 622.977 | 5.602567 | 818.114251 |
| `satclip-vit16-l10` | 256 | 1267.5726 | 633.7863 | 5.699777 | 976.267641 |
| `le-sinr` | 256 | 1502.5526 | 751.2763 | 6.75639 | 123.15317 |
| `sinr` | 256 | 1724.1521 | 862.076 | 7.752836 | 88.025017 |
| `csp-fmow` | 256 | 1825.1216 | 912.5608 | 8.206857 | 0.494432 |
| `spherical-harmonics` | 16 | 1915.0397 | 957.5199 | 8.611183 | 0.415495 |
| `csp-inat` | 256 | 1921.4453 | 960.7226 | 8.639986 | 33.088377 |
| `xyz` | 3 | 1970.4887 | 985.2444 | 8.860515 | 0.096334 |
| `sincos` | 4 | 2075.1969 | 1037.5984 | 9.331347 | 0.16723 |

## Observations

- **Matryoshka scale hierarchy in MIND**: The full 3072-d `mind-3072` representation has a half-sill distance of `205.6 km`, whereas the truncated 64-d `mind` slice has `978.2 km` (a `4.76×` coarsening from the same checkpoint), with `mind-head` (64-d) falling between them at `640.8 km`. This ordering shows that coarse global structure is concentrated in the leading dimensions while higher-frequency spatial variation occupies the trailing dimensions.
- **Spherical-harmonic bandwidth (`L=40` vs. `L=10`)**: `osmgraphclip-a-l40` (`421.5 km`) and the three `L=40` SatCLIP checkpoints (`432.6–435.0 km`) cluster just above the analytic `sh40` basis (`355.8 km`), while `osmgraphclip` sits slightly coarser at `543.5 km`. All five `L=10` checkpoints (`osmgraphclip-a-l10`, `osmgraphclip-ms-l10`, and the three `satclip-*-l10` models) lie between `1,202.9 km` and `1,267.6 km`, roughly three times coarser than the `L=40` models.
- **Fine-scale encoders**: The analytic `nerf` positional encoding has the shortest half-sill distance overall at `14.6 km` (suggested grid `7.3 km`). Among pretrained neural encoders, `geoclip` (`138.8 km`), `gtloc` (`177.8 km`), `mind-3072` (`205.6 km`), `terranova` (`324.8 km`), and `terranova-st` (`373.2 km`) resolve sub-400 km variation.
- **Continental-scale encoders**: `le-sinr` (`1,502.6 km`), `sinr` (`1,724.2 km`), `csp-fmow` (`1,825.1 km`), and `csp-inat` (`1,921.4 km`) vary on continental scales close to the low-order analytic baselines `spherical-harmonics` (`1,915.0 km`), `xyz` (`1,970.5 km`), and `sincos` (`2,075.2 km`).

## Intrinsic Dimension and Participation Ratio (100 km Grid)

Half-sill distance measures the spatial lag over which an encoder's embeddings decorrelate, whereas intrinsic dimensionality measures how much of the ambient embedding space $\mathbb{R}^D$ those embeddings occupy. EarthINR implements both linear and nonlinear dimensionality estimators in [`src/earthinr/analysis/intrinsic_dim.py`](../src/earthinr/analysis/intrinsic_dim.py):

- **Participation ratio (`PR`)**: computed in pure NumPy/PyTorch float64 from the eigenvalues $\lambda_i$ of the centered sample covariance matrix,

  $$\text{PR} = \frac{\left(\sum_i \lambda_i\right)^2}{\sum_i \lambda_i^2}.$$

  `PR` counts the effective number of orthogonal linear directions carrying variance across the sampled locations and is invariant to orthogonal rotations and positive scalar rescaling.
- **Nonlinear manifold estimators (`TwoNN`, `MLE`, `MOM`, `TLE`, `FisherS`)**: computed via optional [`torchid`](https://github.com/isaaccorley/torchid) on centered float64 embeddings after removing exact duplicate rows. Following the stability protocol from GeoINRID and MIND, nearest-neighbor queries are routed through `torchid`'s pure-PyTorch float64 k-NN path rather than float32 FAISS so large-norm encoders (such as `csp-inat`) do not suffer squared-distance cancellation. `MLE`, `MOM`, and `TLE` use $k = 20$ neighbors; `FisherS` uses `conditional_number=10.0`.

### 100 km Land-Grid Protocol

All 35 encoders in [`results/intrinsic_dim.csv`](../results/intrinsic_dim.csv) were evaluated on a 100 km equal-area land grid:

- **Base grid**: `generate_equal_area_grid(spacing_km=100.0, jitter_km=100.0, seed=0)` places $N = \operatorname{round}(4\pi R^2 / 100^2) = 51{,}006$ global Fibonacci-sphere points on $R = 6{,}371\text{ km}$ and perturbs each point within a 100 km equal-area spherical disk. Perturbing the lattice is required for Poisson-process k-NN estimators such as `TwoNN`: an unjittered Fibonacci lattice forces $r_2 / r_1 \approx 1$ across smooth maps, which inflates `TwoNN` above 100 even on the 2D unit sphere (`xyz`).
- **Land filtering**: restricting the perturbed 100 km global grid to Natural Earth 110m land polygons retains `14,729` global land points, aligning with the land focus of [`results/implicit_resolution.csv`](../results/implicit_resolution.csv) and GeoINRID (Rao et al., ICLR 2026) so encoders trained primarily over land are evaluated on their intended domain.
- **Temporal inputs**: fixed at `year=2020` and `month=6`, matching the half-sill protocol above.

### Combined Resolution and Dimensionality Table

Values below join `Half-Sill (km)` from [`results/implicit_resolution.csv`](../results/implicit_resolution.csv) with `PR`, `TwoNN`, `MLE`, `TLE`, and `FisherS` from [`results/intrinsic_dim.csv`](../results/intrinsic_dim.csv) (which also records `MOM`), ordered from finest to coarsest half-sill distance.

| Model | Dim | Half-Sill (km) | PR | TwoNN | MLE | TLE | FisherS |
|---|---:|---:|---:|---:|---:|---:|---:|
| `nerf` | 96 | 14.6 | 89.05 | 26.10 | 28.42 | 29.67 | 41.36 |
| `geoclip` | 512 | 138.8 | 56.86 | 12.61 | 14.14 | 14.68 | 7.74 |
| `gtloc` | 512 | 177.8 | 68.52 | 10.86 | 13.82 | 14.79 | 8.45 |
| `mind-3072` | 3072 | 205.6 | 416.17 | 3.07 | 5.51 | 6.30 | 6.62 |
| `rff3072` | 3072 | 239.4 | 172.29 | 3.64 | 7.90 | 8.62 | 5.11 |
| `terranova` | 256 | 324.8 | 43.22 | 3.35 | 4.83 | 5.92 | 10.56 |
| `sh40` | 1681 | 355.8 | 317.06 | 1.97 | 2.19 | 2.65 | 8.00 |
| `terranova-st` | 256 | 373.2 | 42.12 | 3.19 | 4.37 | 5.29 | 10.54 |
| `osmgraphclip-a-l40` | 256 | 421.5 | 19.68 | 2.06 | 2.61 | 3.20 | 8.27 |
| `satclip` | 256 | 432.6 | 21.13 | 2.04 | 2.59 | 3.19 | 8.14 |
| `satclip-resnet50-l40` | 256 | 432.8 | 27.90 | 2.02 | 2.46 | 2.99 | 8.18 |
| `satclip-resnet18-l40` | 256 | 435.0 | 24.24 | 2.02 | 2.50 | 3.04 | 8.06 |
| `tte` | 512 | 513.8 | 40.07 | 2.05 | 2.26 | 2.70 | 7.37 |
| `sled` | 768 | 535.9 | 14.73 | 2.08 | 2.60 | 3.36 | 5.36 |
| `osmgraphclip` | 256 | 543.5 | 12.25 | 2.13 | 3.74 | 4.71 | 7.95 |
| `taxabind` | 512 | 559.7 | 3.50 | 9.84 | 13.80 | 14.64 | 1.61 |
| `climplicit` | 256 | 574.2 | 5.52 | 2.38 | 4.39 | 6.65 | 3.04 |
| `sled-s2` | 768 | 587.9 | 15.63 | 2.10 | 2.66 | 3.44 | 5.28 |
| `mind-head` | 64 | 640.8 | 12.01 | 3.79 | 5.69 | 6.68 | 4.00 |
| `hydroloc` | 512 | 652.4 | 81.06 | 2.00 | 2.34 | 2.81 | 6.45 |
| `mind` | 64 | 978.2 | 20.99 | 2.94 | 4.81 | 5.69 | 4.93 |
| `geoclosp` | 384 | 1151.2 | 5.56 | 1.97 | 2.09 | 2.35 | 5.19 |
| `osmgraphclip-a-l10` | 256 | 1202.9 | 13.19 | 1.97 | 2.06 | 2.32 | 5.75 |
| `osmgraphclip-ms-l10` | 256 | 1204.0 | 10.93 | 1.96 | 2.08 | 2.36 | 5.21 |
| `satclip-resnet50-l10` | 256 | 1233.8 | 15.74 | 1.95 | 2.05 | 2.31 | 5.01 |
| `gair` | 768 | 1236.4 | 3.23 | 42.47 | 34.06 | 32.68 | 1.52 |
| `satclip-resnet18-l10` | 256 | 1246.0 | 12.42 | 1.96 | 2.06 | 2.32 | 4.61 |
| `satclip-vit16-l10` | 256 | 1267.6 | 11.69 | 1.96 | 2.06 | 2.32 | 4.66 |
| `le-sinr` | 256 | 1502.6 | 2.89 | 1.99 | 2.11 | 2.44 | 1.53 |
| `sinr` | 256 | 1724.2 | 1.54 | 2.00 | 2.12 | 2.47 | 0.65 |
| `csp-fmow` | 256 | 1825.1 | 2.66 | 1.96 | 2.00 | 2.18 | 2.61 |
| `spherical-harmonics` | 16 | 1915.0 | 9.26 | 1.96 | 2.03 | 2.18 | 3.94 |
| `csp-inat` | 256 | 1921.4 | 1.16 | 1.97 | 1.99 | 2.17 | 0.92 |
| `xyz` | 3 | 1970.5 | 2.89 | 1.96 | 2.03 | 2.17 | 2.56 |
| `sincos` | 4 | 2075.2 | 3.07 | 1.98 | 2.02 | 2.16 | 2.82 |

### Dimensionality Observations

- **Resolution and dimensionality covary strongly**: Across the 35 encoders, finer half-sill distance correlates negatively with linear participation ratio (Spearman $\rho = -0.857$, $p = 5.1 \times 10^{-11}$) as well as `FisherS` ($\rho = -0.821$), `MLE` ($\rho = -0.755$), `TLE` ($\rho = -0.751$), `MOM` ($\rho = -0.745$), and `TwoNN` ($\rho = -0.678$).
- **Linear capacity vs. local manifold dimension**: High-degree smooth representations occupy many global linear directions while remaining locally close to the 2D surface of the Earth. `sh40` reaches `PR = 317.06` (`half_sill_km = 355.8 km`) while staying at `TwoNN = 1.97` and `MLE = 2.19`; `mind-3072` reaches the highest linear participation ratio overall at `PR = 416.17` (`half_sill_km = 205.6 km`) with `TwoNN = 3.07` and `MLE = 5.51`. All three low-order analytic maps (`xyz`, `sincos`, `spherical-harmonics`) recover `TwoNN = 1.96–1.98` and `MLE = 2.02–2.03`, confirming the 2D spherical baseline on the 100 km land grid.
- **High-frequency folding vs. low-rank global variance**: Encoders whose positional or Fourier features vary at or below 100 km fold the 2D manifold locally, pushing k-NN estimators well above 2: `nerf` (`PR = 89.05`, `TwoNN = 26.10`, `MLE = 28.42`, `FisherS = 41.36`), `geoclip` (`PR = 56.86`, `TwoNN = 12.61`, `MLE = 14.14`), and `gtloc` (`PR = 68.52`, `TwoNN = 10.86`, `MLE = 13.82`). In `gair` (`PR = 3.23`, `FisherS = 1.52`, `TwoNN = 42.47`, `MLE = 34.06`) and `taxabind` (`PR = 3.50`, `FisherS = 1.61`, `TwoNN = 9.84`, `MLE = 13.80`), global variance concentrates in roughly three linear directions while small-amplitude high-frequency residuals scatter local 100 km neighborhoods across many axes.
- **Matryoshka, spherical-harmonic, and continental collapse patterns**: Within MIND, `mind` (`D=64`, `PR = 20.99`) occupies a third of its width linearly, whereas `mind-head` (`D=64`, `PR = 12.01`) compresses linear `PR` while increasing local `TwoNN` from `2.94` to `3.79` and shortening half-sill distance from `978.2 km` to `640.8 km`. The four `L=40` spherical-harmonic checkpoints (`PR = 19.68–27.90`, `FisherS = 8.06–8.27`) occupy roughly twice the linear and Fisher-separability dimensionality of the five `L=10` checkpoints (`PR = 10.93–15.74`, `FisherS = 4.61–5.75`). At the continental end, `csp-inat` (`PR = 1.16`), `sinr` (`PR = 1.54`), `csp-fmow` (`PR = 2.66`), and `le-sinr` (`PR = 2.89`) retain near-2D local manifold estimates (`TwoNN = 1.96–2.00`) while collapsing global variance into 1–3 principal axes.

## Running the Analysis

```python
import numpy as np

from earthinr.analysis import (
    analyze_implicit_resolution,
    analyze_intrinsic_dimension,
    generate_equal_area_grid,
    sample_sphere_points,
)
from earthinr.models import get_model

encoder = get_model('satclip', device='cpu')
lon, lat = sample_sphere_points(n=16_042, seed=0)

result = analyze_implicit_resolution(
    encoder,
    lon,
    lat,
    year=np.full(len(lon), 2020.0),
    month=np.full(len(lon), 6.0),
    n_bins=48,
    min_km=1.0,
    max_km=8000.0,
    pairs_per_lag=2048,
    method='geodesic',
    seed=0,
)
print(
    f'half_sill={result["half_sill_km"]:.1f} km, '
    f'suggested_grid={result["suggested_grid_km"]:.1f} km '
    f'({result["suggested_grid_deg"]:.2f} deg)'
)

# 100 km equal-area Fibonacci grid (pass estimators=() if torchid is not installed)
grid_lon, grid_lat = generate_equal_area_grid(100.0, jitter_km=100.0, seed=0)
id_result = analyze_intrinsic_dimension(
    encoder,
    grid_lon,
    grid_lat,
    year=np.full(len(grid_lon), 2020.0),
    month=np.full(len(grid_lon), 6.0),
    estimators=('twonn', 'mle', 'mom', 'tle', 'fishers'),
    k=20,
    device='cpu',
)
print(
    f'PR={id_result["pr"]:.2f}, '
    f'TwoNN={id_result["twonn"]:.2f}, '
    f'MLE={id_result["mle"]:.2f}'
)
```
