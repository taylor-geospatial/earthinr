# Copyright (c) Taylor Geospatial. All rights reserved.
# Licensed under the MIT License.

"""Intrinsic dimension and participation ratio analysis for location embeddings."""

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor

from .._utils import lazy_import, resolve_device
from .variogram import EARTH_RADIUS_KM

if TYPE_CHECKING:
    from ..models._base import LocationEncoder

#: Estimator names supported via the optional ``torchid`` dependency.
SUPPORTED_ESTIMATORS: tuple[str, ...] = ('fishers', 'mle', 'mom', 'tle', 'twonn')

_ESTIMATOR_SET: frozenset[str] = frozenset(SUPPORTED_ESTIMATORS)


def _to_float64_2d(
    embeddings: NDArray[np.floating[Any]] | Tensor,
) -> NDArray[np.float64]:
    """Validate and convert a 2D embedding array or tensor to float64."""
    if isinstance(embeddings, Tensor):
        arr = embeddings.detach().cpu().to(dtype=torch.float64).numpy()
    else:
        arr = np.asarray(embeddings, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[0] < 2 or arr.shape[1] < 1:
        raise ValueError(
            'embeddings must be a 2D array with at least 2 samples and 1 feature.'
        )
    return arr


def generate_equal_area_grid(
    spacing_km: float = 100.0,
    *,
    lat_min: float = -90.0,
    lat_max: float = 90.0,
    jitter_km: float = 0.0,
    seed: int = 0,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Generate a deterministic equal-area Fibonacci sphere grid at *spacing_km*.

    Places ``N = max(2, round(4 * pi * R^2 / spacing_km^2))`` points globally on a
    sphere of radius ``R = 6371.0 km`` (``N = 51,006`` at ``spacing_km=100.0``) using
    the golden-angle Fibonacci lattice:

    - ``z_i = 1 - (2 * i + 1) / N`` for ``i = 0, ..., N - 1``
    - ``lat_i = arcsin(z_i)`` in degrees
    - ``lon_i = ((degrees(i * pi * (3 - sqrt(5))) + 180) mod 360) - 180``

    Points avoid the exact poles, have zero duplicates, and maintain near-isotropic
    neighbor spacing across all latitudes. When ``jitter_km > 0``, each lattice point
    is displaced along a random bearing by an equal-area disk radius
    ``jitter_km * sqrt(U)`` via the exact spherical geodesic formula; setting
    ``jitter_km=spacing_km`` breaks lattice regularity so Poisson-process k-NN
    estimators such as ``TwoNN`` do not observe degenerate ``r_2 / r_1 ~ 1`` ratios.

    Args:
        spacing_km: Target characteristic spacing between neighboring grid points in
            kilometres (must be finite and positive).
        lat_min: Minimum latitude in degrees (inclusive).
        lat_max: Maximum latitude in degrees (inclusive).
        jitter_km: Optional maximum geodesic disk perturbation radius in kilometres
            (must be finite and non-negative; default ``0.0`` for deterministic grid).
        seed: Random seed used when ``jitter_km > 0``.

    Returns:
        Tuple ``(lon, lat)`` of 1D float64 coordinate arrays in WGS84 degrees.

    Raises:
        ValueError: If *spacing_km* or *jitter_km* is invalid, latitude bounds are
            invalid, or no grid points fall inside ``[lat_min, lat_max]``.
    """
    if not np.isfinite(spacing_km) or spacing_km <= 0.0:
        raise ValueError(f'spacing_km must be finite and positive, got {spacing_km}.')
    if not np.isfinite(jitter_km) or jitter_km < 0.0:
        raise ValueError(f'jitter_km must be finite and non-negative, got {jitter_km}.')
    if not (-90.0 <= lat_min < lat_max <= 90.0):
        raise ValueError(
            f'Expected -90 <= lat_min < lat_max <= 90, got [{lat_min}, {lat_max}].'
        )

    surface_area_km2 = 4.0 * np.pi * EARTH_RADIUS_KM * EARTH_RADIUS_KM
    n_global = max(2, round(surface_area_km2 / (spacing_km * spacing_km)))
    idx = np.arange(n_global, dtype=np.float64)
    z = 1.0 - (2.0 * idx + 1.0) / float(n_global)
    lat = np.degrees(np.arcsin(np.clip(z, -1.0, 1.0)))
    golden_angle = np.pi * (3.0 - np.sqrt(5.0))
    lon = (np.degrees(idx * golden_angle) + 180.0) % 360.0 - 180.0

    if jitter_km > 0.0:
        rng = np.random.default_rng(seed)
        ang = (jitter_km * np.sqrt(rng.uniform(0.0, 1.0, size=n_global))) / (
            EARTH_RADIUS_KM
        )
        bearing = rng.uniform(0.0, 2.0 * np.pi, size=n_global)
        p1 = np.radians(lat)
        l1 = np.radians(lon)
        sin_p2 = np.sin(p1) * np.cos(ang) + np.cos(p1) * np.sin(ang) * np.cos(bearing)
        p2 = np.arcsin(np.clip(sin_p2, -1.0, 1.0))
        l2 = l1 + np.arctan2(
            np.sin(bearing) * np.sin(ang) * np.cos(p1),
            np.cos(ang) - np.sin(p1) * np.sin(p2),
        )
        lon = (np.degrees(l2) + 180.0) % 360.0 - 180.0
        lat = np.degrees(p2)

    mask = (lat >= lat_min) & (lat <= lat_max)
    lon_sel = lon[mask]
    lat_sel = lat[mask]
    if lon_sel.shape[0] == 0:
        raise ValueError(
            f'No Fibonacci grid points fall within [{lat_min}, {lat_max}] '
            f'at spacing_km={spacing_km}.'
        )
    return lon_sel.astype(np.float64), lat_sel.astype(np.float64)


def participation_ratio(
    embeddings: NDArray[np.floating[Any]] | Tensor, *, standardize: bool = False
) -> float:
    """Compute the linear covariance participation ratio in float64.

    Measures the effective number of variance-carrying linear directions from the
    eigenvalues ``lambda_i`` of the centered sample covariance matrix:

    ``PR = (sum_i lambda_i)^2 / sum_i lambda_i^2``

    This quantity equals ``k`` for ``k`` orthogonal equal-variance directions and is
    invariant to orthogonal rotations and positive scalar rescaling when
    ``standardize=False``.

    Args:
        embeddings: 2D array or PyTorch tensor of shape ``(N, D)`` with ``N >= 2``.
        standardize: If ``True``, divide non-constant columns (``std > 1e-8``) by
            their standard deviation before computing the spectrum (correlation-matrix
            participation ratio).

    Returns:
        Participation ratio as a Python float, or ``nan`` if all dimensions have zero
        variance.

    Raises:
        ValueError: If *embeddings* is not a 2D array/tensor with at least 2 samples
            and 1 feature.
    """
    arr = _to_float64_2d(embeddings)
    centered = arr - arr.mean(axis=0, keepdims=True)
    if standardize:
        sd = centered.std(axis=0)
        active = sd > 1e-8
        if not np.any(active):
            return float('nan')
        centered = centered[:, active] / sd[active]

    n_samples, n_features = centered.shape
    if n_samples >= n_features:
        gram = (centered.T @ centered) / float(n_samples)
    else:
        gram = (centered @ centered.T) / float(n_samples)
    eigvals = np.maximum(np.linalg.eigvalsh(gram), 0.0)
    denom = float(np.sum(eigvals * eigvals))
    if denom <= 0.0:
        return float('nan')
    total = float(np.sum(eigvals))
    return (total * total) / denom


def estimate_intrinsic_dimension(
    embeddings: NDArray[np.floating[Any]] | Tensor,
    *,
    estimators: Sequence[str] = ('mle', 'twonn', 'mom', 'tle', 'fishers'),
    k: int = 20,
    conditional_number: float = 10.0,
    standardize_pr: bool = False,
    deduplicate: bool = True,
    device: str | torch.device = 'cpu',
) -> dict[str, float]:
    """Estimate linear participation ratio and nonlinear manifold intrinsic dimension.

    Always computes the pure-NumPy float64 covariance participation ratio (``'pr'``).
    When *estimators* is non-empty, lazily imports ``torchid`` (installed by default
    on Python 3.13+) and evaluates each requested estimator on float64 centered
    embeddings.

    Args:
        embeddings: 2D array or PyTorch tensor of shape ``(N, D)`` with ``N >= 2``.
        estimators: Sequence of ``torchid`` estimator names to run. Supported values
            are ``'mle'``, ``'twonn'``, ``'mom'``, ``'tle'``, and ``'fishers'``
            (case-insensitive). Pass an empty sequence to compute only ``'pr'``
            without importing ``torchid``.
        k: Number of nearest neighbors for neighborhood-based estimators (``'mle'``,
            ``'mom'``, ``'tle'``).
        conditional_number: Conditioning threshold passed to ``FisherS``.
        standardize_pr: If ``True``, standardize non-constant features before
            computing ``'pr'``.
        deduplicate: If ``True``, drop exact duplicate rows prior to running
            ``torchid`` estimators so zero neighbor distances do not corrupt
            distance-based estimates.
        device: PyTorch device used for ``torchid`` estimation (default ``'cpu'``).

    Returns:
        Dictionary mapping ``'pr'`` and each requested lowercased estimator name to
        its float dimension estimate.

    Raises:
        ValueError: If *embeddings*, *k*, *conditional_number*, or any estimator name
            is invalid, or if too few unique rows remain for *k*-NN estimation.
        DependencyNotFoundError: If *estimators* is non-empty and ``torchid`` is not
            installed.
    """
    arr = _to_float64_2d(embeddings)
    if k < 2:
        raise ValueError(f'k must be at least 2, got {k}.')
    if not np.isfinite(conditional_number) or conditional_number <= 0.0:
        raise ValueError(
            f'conditional_number must be finite and positive, got {conditional_number}.'
        )

    normalized: list[str] = []
    for name in estimators:
        key = name.lower()
        if key not in _ESTIMATOR_SET:
            raise ValueError(
                f'Unsupported estimator {name!r}. '
                f'Expected one of {SUPPORTED_ESTIMATORS}.'
            )
        if key not in normalized:
            normalized.append(key)

    results: dict[str, float] = {
        'pr': participation_ratio(arr, standardize=standardize_pr)
    }
    if not normalized:
        return results

    work = np.unique(arr, axis=0) if deduplicate else arr
    min_rows = k + 1 if any(e in ('mle', 'mom', 'tle') for e in normalized) else 3
    if work.shape[0] < min_rows:
        raise ValueError(
            f'Need at least {min_rows} unique embeddings for {normalized}, '
            f'got {work.shape[0]}.'
        )

    torchid_mod = lazy_import('torchid')
    resolved = resolve_device(device)
    centered = work - work.mean(axis=0, keepdims=True)
    tensor = torch.from_numpy(centered).to(device=resolved, dtype=torch.float64)
    est_mod = torchid_mod.estimators

    for key in normalized:
        if key == 'mle':
            est = est_mod.MLE().fit(tensor, n_neighbors=k)
        elif key == 'twonn':
            est = est_mod.TwoNN().fit(tensor)
        elif key == 'mom':
            est = est_mod.MOM(n_neighbors=k).fit(tensor)
        elif key == 'tle':
            est = est_mod.TLE(n_neighbors=k).fit(tensor)
        else:
            est = est_mod.FisherS(conditional_number=conditional_number).fit(tensor)
        results[key] = float(est.dimension_)

    return results


def analyze_intrinsic_dimension(
    encoder: 'LocationEncoder',
    lon: NDArray[np.floating[Any]],
    lat: NDArray[np.floating[Any]],
    *,
    year: NDArray[np.floating[Any]] | None = None,
    month: NDArray[np.floating[Any]] | None = None,
    estimators: Sequence[str] = ('mle', 'twonn', 'mom', 'tle', 'fishers'),
    k: int = 20,
    conditional_number: float = 10.0,
    standardize_pr: bool = False,
    deduplicate: bool = True,
    device: str | torch.device = 'cpu',
) -> dict[str, float]:
    """Encode geographic coordinates and estimate intrinsic dimension and PR.

    Args:
        encoder: Instantiated :class:`~neuralftw.models.LocationEncoder`.
        lon: 1D array of longitudes in degrees.
        lat: 1D array of latitudes in degrees.
        year: Optional 1D array of observation years matching *lon* and *lat*.
        month: Optional 1D array of observation months matching *lon* and *lat*.
        estimators: Sequence of ``torchid`` estimator names to run alongside ``'pr'``.
        k: Number of nearest neighbors for neighborhood-based estimators.
        conditional_number: Conditioning threshold passed to ``FisherS``.
        standardize_pr: If ``True``, standardize non-constant features before
            computing ``'pr'``.
        deduplicate: If ``True``, remove exact duplicate embedding rows before
            estimation.
        device: PyTorch device used for ``torchid`` estimation (default ``'cpu'``).

    Returns:
        Dictionary containing ``'pr'``, each requested estimator value,
        ``'n_points'``, and ``'embed_dim'``.

    Raises:
        ValueError: If *lon* and *lat* are invalid or fewer than 2 points are given.
        DependencyNotFoundError: If *estimators* is non-empty and ``torchid`` is not
            installed.
    """
    lon_arr = np.asarray(lon, dtype=np.float64)
    lat_arr = np.asarray(lat, dtype=np.float64)
    if lon_arr.ndim != 1 or lat_arr.ndim != 1 or lon_arr.shape != lat_arr.shape:
        raise ValueError('lon and lat must be 1D arrays of equal length.')
    n_pts = lon_arr.shape[0]
    if n_pts < 2:
        raise ValueError('lon and lat must contain at least 2 points.')

    year_arr = None if year is None else np.asarray(year, dtype=np.float64)
    month_arr = None if month is None else np.asarray(month, dtype=np.float64)
    emb = encoder.encode(lon_arr, lat_arr, year=year_arr, month=month_arr)
    scores = estimate_intrinsic_dimension(
        emb,
        estimators=estimators,
        k=k,
        conditional_number=conditional_number,
        standardize_pr=standardize_pr,
        deduplicate=deduplicate,
        device=device,
    )
    scores['n_points'] = float(n_pts)
    scores['embed_dim'] = float(emb.shape[1])
    return scores


__all__ = (
    'SUPPORTED_ESTIMATORS',
    'analyze_intrinsic_dimension',
    'estimate_intrinsic_dimension',
    'generate_equal_area_grid',
    'participation_ratio',
)
