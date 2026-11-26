# Copyright (c) Taylor Geospatial. All rights reserved.
# Licensed under the MIT License.

"""Empirical semivariogram and half-sill implicit resolution analysis."""

from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from ..models._base import LocationEncoder

#: Mean volumetric radius of Earth in kilometres.
EARTH_RADIUS_KM: float = 6371.0

#: Approximate kilometres per degree of great-circle arc on Earth.
KM_PER_DEGREE: float = 2.0 * np.pi * EARTH_RADIUS_KM / 360.0


def _haversine_km(
    lon1: NDArray[np.float64],
    lat1: NDArray[np.float64],
    lon2: NDArray[np.float64],
    lat2: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Compute pairwise great-circle Haversine distance in kilometres."""
    rlon1, rlat1 = np.radians(lon1), np.radians(lat1)
    rlon2, rlat2 = np.radians(lon2), np.radians(lat2)
    dlat = rlat2 - rlat1
    dlon = rlon2 - rlon1
    a = (
        np.sin(dlat / 2.0) ** 2
        + np.cos(rlat1) * np.cos(rlat2) * np.sin(dlon / 2.0) ** 2
    )
    return 2.0 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def _prepare_embeddings(
    embeddings: NDArray[np.floating[Any]], normalize: bool
) -> NDArray[np.float64]:
    """Validate and optionally L2-normalize a 2D embedding matrix."""
    emb = np.asarray(embeddings, dtype=np.float64)
    if emb.ndim != 2 or emb.shape[0] == 0 or emb.shape[1] == 0:
        raise ValueError('embeddings must be a non-empty 2D array.')
    if normalize:
        norms = np.linalg.norm(emb, axis=1, keepdims=True)
        emb = emb / np.where(norms > 0.0, norms, 1.0)
    return emb


def sample_sphere_points(
    n: int = 20000,
    *,
    lat_min: float = -90.0,
    lat_max: float = 90.0,
    seed: int = 0,
    land_coords: tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]]]
    | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Sample uniform-area coordinates on the sphere or from a land reference set.

    Area-weighted sampling draws ``sin(lat)`` uniformly rather than ``lat`` so polar
    regions are not over-represented. When *land_coords* is provided, points within
    ``[lat_min, lat_max]`` are subsampled without replacement up to *n*.

    Args:
        n: Number of coordinate points to sample.
        lat_min: Minimum latitude in degrees (inclusive).
        lat_max: Maximum latitude in degrees (inclusive).
        seed: Random seed for reproducible sampling.
        land_coords: Optional ``(lon, lat)`` tuple of reference land coordinates to
            subsample from instead of sampling over the full sphere.

    Returns:
        Tuple ``(lon, lat)`` of 1D float64 coordinate arrays in WGS84 degrees.

    Raises:
        ValueError: If *n* is not positive, latitude bounds are invalid, or
            *land_coords* is malformed or empty within the latitude bounds.
    """
    if n <= 0:
        raise ValueError(f'n must be positive, got {n}.')
    if not (-90.0 <= lat_min < lat_max <= 90.0):
        raise ValueError(
            f'Expected -90 <= lat_min < lat_max <= 90, got [{lat_min}, {lat_max}].'
        )

    rng = np.random.default_rng(seed)
    if land_coords is not None:
        land_lon = np.asarray(land_coords[0], dtype=np.float64)
        land_lat = np.asarray(land_coords[1], dtype=np.float64)
        if land_lon.ndim != 1 or land_lat.ndim != 1 or land_lon.shape != land_lat.shape:
            raise ValueError('land_coords must be a pair of 1D arrays of equal length.')
        mask = (land_lat >= lat_min) & (land_lat <= lat_max)
        land_lon = land_lon[mask]
        land_lat = land_lat[mask]
        if land_lon.shape[0] == 0:
            raise ValueError('No land_coords fall within [lat_min, lat_max].')
        size = min(n, land_lon.shape[0])
        idx = rng.choice(land_lon.shape[0], size=size, replace=False)
        return land_lon[idx], land_lat[idx]

    sin_lo = float(np.sin(np.radians(lat_min)))
    sin_hi = float(np.sin(np.radians(lat_max)))
    lat = np.degrees(np.arcsin(rng.uniform(sin_lo, sin_hi, size=n)))
    lon = rng.uniform(-180.0, 180.0, size=n)
    return lon.astype(np.float64), lat.astype(np.float64)


def geodesic_offset(
    lon: NDArray[np.floating[Any]],
    lat: NDArray[np.floating[Any]],
    dist_km: float,
    *,
    rng: np.random.Generator | None = None,
    seed: int = 0,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Move coordinates along random initial bearings by an exact great-circle lag.

    Solves the direct spherical geodesic problem so the realized separation equals
    *dist_km* at every latitude rather than shrinking by ``cos(lat)`` toward the poles.

    Args:
        lon: 1D array of anchor longitudes in degrees.
        lat: 1D array of anchor latitudes in degrees.
        dist_km: Great-circle offset distance in kilometres (must be non-negative).
        rng: Optional NumPy random generator; when omitted a new generator is seeded
            from *seed*.
        seed: Random seed used when *rng* is ``None``.

    Returns:
        Tuple ``(lon_offset, lat_offset)`` of 1D float64 coordinate arrays in degrees.

    Raises:
        ValueError: If *lon* and *lat* are not 1D arrays of equal length or *dist_km*
            is negative.
    """
    lon_arr = np.asarray(lon, dtype=np.float64)
    lat_arr = np.asarray(lat, dtype=np.float64)
    if lon_arr.ndim != 1 or lat_arr.ndim != 1 or lon_arr.shape != lat_arr.shape:
        raise ValueError('lon and lat must be 1D arrays of equal length.')
    if dist_km < 0.0:
        raise ValueError(f'dist_km must be non-negative, got {dist_km}.')

    gen = np.random.default_rng(seed) if rng is None else rng
    bearing = gen.uniform(0.0, 2.0 * np.pi, size=lon_arr.shape[0])
    ang = dist_km / EARTH_RADIUS_KM
    p1 = np.radians(lat_arr)
    l1 = np.radians(lon_arr)
    sin_p2 = np.sin(p1) * np.cos(ang) + np.cos(p1) * np.sin(ang) * np.cos(bearing)
    p2 = np.arcsin(np.clip(sin_p2, -1.0, 1.0))
    l2 = l1 + np.arctan2(
        np.sin(bearing) * np.sin(ang) * np.cos(p1),
        np.cos(ang) - np.sin(p1) * np.sin(p2),
    )
    lon2 = (np.degrees(l2) + 180.0) % 360.0 - 180.0
    lat2 = np.degrees(p2)
    return lon2.astype(np.float64), lat2.astype(np.float64)


def empirical_semivariogram(
    lon: NDArray[np.floating[Any]],
    lat: NDArray[np.floating[Any]],
    embeddings: NDArray[np.floating[Any]],
    *,
    n_bins: int = 48,
    min_km: float = 1.0,
    max_km: float = 8000.0,
    n_pairs: int = 2_000_000,
    normalize: bool = False,
    seed: int = 0,
) -> dict[str, NDArray[np.float64]]:
    """Estimate the isotropic empirical semivariogram from sampled point pairs.

    Computes ``gamma(h) = 0.5 * E[||f(x_i) - f(x_j)||^2]`` across log-spaced
    great-circle Haversine separation bins up to *max_km*.

    Args:
        lon: 1D array of longitudes in degrees.
        lat: 1D array of latitudes in degrees.
        embeddings: 2D array of shape ``(N, D)`` containing embeddings at each point.
        n_bins: Number of logarithmically spaced separation bins.
        min_km: Lower edge of the smallest separation bin in kilometres.
        max_km: Upper edge of the largest separation bin in kilometres.
        n_pairs: Number of random point pairs sampled to populate the bins.
        normalize: If ``True``, L2-normalize each embedding before computing squared
            differences (bounded cosine semivariance).
        seed: Random seed for reproducible pair sampling.

    Returns:
        Dictionary with float64 arrays ``'bin_centers_km'``, ``'semivariance'``, and
        ``'counts'``, each of shape ``(n_bins,)``. Empty bins have ``nan``
        semivariance and ``0.0`` count.

    Raises:
        ValueError: If coordinate/embedding shapes are invalid or binning parameters
            are out of range.
    """
    lon_arr = np.asarray(lon, dtype=np.float64)
    lat_arr = np.asarray(lat, dtype=np.float64)
    emb = _prepare_embeddings(embeddings, normalize=normalize)
    if lon_arr.ndim != 1 or lat_arr.ndim != 1 or lon_arr.shape != lat_arr.shape:
        raise ValueError('lon and lat must be 1D arrays of equal length.')
    n_pts = lon_arr.shape[0]
    if emb.shape[0] != n_pts or n_pts < 2:
        raise ValueError(
            'embeddings must match lon/lat length and contain at least 2 points.'
        )
    if n_bins < 2:
        raise ValueError(f'n_bins must be at least 2, got {n_bins}.')
    if min_km <= 0.0 or max_km <= min_km:
        raise ValueError(
            f'Expected 0 < min_km < max_km, got min_km={min_km}, max_km={max_km}.'
        )
    if n_pairs <= 0:
        raise ValueError(f'n_pairs must be positive, got {n_pairs}.')

    edges = np.geomspace(min_km, max_km, n_bins + 1)
    centers = np.sqrt(edges[:-1] * edges[1:])
    sums = np.zeros(n_bins, dtype=np.float64)
    counts = np.zeros(n_bins, dtype=np.float64)

    rng = np.random.default_rng(seed)
    chunk_size = max(1024, min(65536, 4_000_000 // max(1, emb.shape[1])))
    remaining = n_pairs
    while remaining > 0:
        batch = min(chunk_size, remaining)
        remaining -= batch
        idx_i = rng.integers(0, n_pts, size=batch)
        idx_j = rng.integers(0, n_pts, size=batch)
        keep = idx_i != idx_j
        idx_i = idx_i[keep]
        idx_j = idx_j[keep]
        if idx_i.shape[0] == 0:
            continue
        dist_km = _haversine_km(
            lon_arr[idx_i], lat_arr[idx_i], lon_arr[idx_j], lat_arr[idx_j]
        )
        bin_idx = np.digitize(dist_km, edges) - 1
        in_range = (bin_idx >= 0) & (bin_idx < n_bins)
        if not np.any(in_range):
            continue
        sel_i = idx_i[in_range]
        sel_j = idx_j[in_range]
        sel_bins = bin_idx[in_range]
        diff = emb[sel_i] - emb[sel_j]
        sq = 0.5 * np.sum(diff * diff, axis=1)
        np.add.at(sums, sel_bins, sq)
        np.add.at(counts, sel_bins, 1.0)

    semivariance = np.full(n_bins, np.nan, dtype=np.float64)
    nonzero = counts > 0.0
    semivariance[nonzero] = sums[nonzero] / counts[nonzero]
    return {'bin_centers_km': centers, 'semivariance': semivariance, 'counts': counts}


def half_sill_distance(
    bin_centers_km: NDArray[np.floating[Any]],
    semivariance: NDArray[np.floating[Any]],
    *,
    frac: float = 0.5,
) -> tuple[float, float]:
    """Compute the half-sill characteristic distance and plateau sill.

    Defines the sill as the median semivariance over the largest quartile of valid
    separation bins, then finds the smallest lag where semivariance crosses
    ``frac * sill`` via linear interpolation on the log-distance axis. Non-crossing
    or degenerate curves return ``nan`` for the characteristic distance.

    Args:
        bin_centers_km: 1D array of positive separation bin centers in kilometres.
        semivariance: 1D array of empirical semivariance values matching
            *bin_centers_km*.
        frac: Target fraction of the sill (default ``0.5`` for half-sill).

    Returns:
        Tuple ``(half_sill_km, sill)`` as Python floats.

    Raises:
        ValueError: If arrays are not 1D of equal length or *frac* is not in ``(0, 1)``.
    """
    lags = np.asarray(bin_centers_km, dtype=np.float64)
    gamma = np.asarray(semivariance, dtype=np.float64)
    if lags.ndim != 1 or gamma.ndim != 1 or lags.shape != gamma.shape:
        raise ValueError(
            'bin_centers_km and semivariance must be 1D arrays of equal length.'
        )
    if not (0.0 < frac < 1.0):
        raise ValueError(f'frac must be in (0, 1), got {frac}.')

    valid = np.isfinite(lags) & np.isfinite(gamma) & (lags > 0.0)
    if int(np.sum(valid)) < 2:
        return (float('nan'), float('nan'))

    lags_v = lags[valid]
    gamma_v = gamma[valid]
    order = np.argsort(lags_v)
    lags_v = lags_v[order]
    gamma_v = gamma_v[order]

    tail_start = max(1, int(0.75 * lags_v.shape[0]))
    sill = float(np.median(gamma_v[tail_start:]))
    if sill <= 0.0:
        return (float('nan'), sill)

    target = frac * sill
    idx = int(np.flatnonzero(gamma_v >= target)[0])
    if idx == 0:
        return (float('nan'), sill)

    g0 = float(gamma_v[idx - 1])
    g1 = float(gamma_v[idx])
    weight = (target - g0) / (g1 - g0)
    log_lag = np.log(lags_v[idx - 1]) + weight * (
        np.log(lags_v[idx]) - np.log(lags_v[idx - 1])
    )
    return (float(np.exp(log_lag)), sill)


def analyze_implicit_resolution(
    encoder: 'LocationEncoder',
    lon: NDArray[np.floating[Any]],
    lat: NDArray[np.floating[Any]],
    *,
    year: NDArray[np.floating[Any]] | None = None,
    month: NDArray[np.floating[Any]] | None = None,
    n_bins: int = 48,
    min_km: float = 1.0,
    max_km: float = 8000.0,
    n_pairs: int = 2_000_000,
    pairs_per_lag: int = 2048,
    method: Literal['geodesic', 'pairs'] = 'geodesic',
    normalize: bool = False,
    frac: float = 0.5,
    seed: int = 0,
) -> dict[str, float]:
    """Measure a location encoder's implicit spatial resolution via its semivariogram.

    Supports two estimators:
    - ``'geodesic'`` (default, matching the MIND paper): offsets *pairs_per_lag*
      anchors along random spherical bearings at each of *n_bins* log-spaced lags,
      giving exact short- and long-lag support even for high-frequency encoders.
    - ``'pairs'``: encodes ``(lon, lat)`` once and bins *n_pairs* random pairs via
      :func:`empirical_semivariogram`.

    Args:
        encoder: Instantiated :class:`~neuralftw.models.LocationEncoder`.
        lon: 1D array of longitudes in degrees.
        lat: 1D array of latitudes in degrees.
        year: Optional 1D array of observation years matching *lon* and *lat*.
        month: Optional 1D array of observation months matching *lon* and *lat*.
        n_bins: Number of log-spaced separation lags/bins.
        min_km: Minimum separation distance in kilometres.
        max_km: Maximum separation distance in kilometres.
        n_pairs: Number of random pairs used when ``method='pairs'``.
        pairs_per_lag: Number of anchor pairs per lag when ``method='geodesic'``.
        method: Variogram estimation method, ``'geodesic'`` or ``'pairs'``.
        normalize: If ``True``, L2-normalize embeddings before computing semivariance.
        frac: Sill fraction used for characteristic distance (default ``0.5``).
        seed: Random seed for reproducible sampling.

    Returns:
        Dictionary containing ``'half_sill_km'``, ``'sill'``, ``'suggested_grid_km'``
        (Nyquist-style ``half_sill_km / 2``), ``'suggested_grid_deg'``, ``'n_points'``,
        and ``'embed_dim'``.

    Raises:
        ValueError: If *method* is unsupported or inputs are invalid.
    """
    if method not in ('geodesic', 'pairs'):
        raise ValueError(
            f"Unsupported method {method!r}. Expected 'geodesic' or 'pairs'."
        )
    lon_arr = np.asarray(lon, dtype=np.float64)
    lat_arr = np.asarray(lat, dtype=np.float64)
    if lon_arr.ndim != 1 or lat_arr.ndim != 1 or lon_arr.shape != lat_arr.shape:
        raise ValueError('lon and lat must be 1D arrays of equal length.')
    n_pts = lon_arr.shape[0]
    if n_pts < 2:
        raise ValueError('lon and lat must contain at least 2 points.')

    year_arr = None if year is None else np.asarray(year, dtype=np.float64)
    month_arr = None if month is None else np.asarray(month, dtype=np.float64)

    base = _prepare_embeddings(
        encoder.encode(lon_arr, lat_arr, year=year_arr, month=month_arr),
        normalize=normalize,
    )
    embed_dim = base.shape[1]

    if method == 'pairs':
        vario = empirical_semivariogram(
            lon_arr,
            lat_arr,
            base,
            n_bins=n_bins,
            min_km=min_km,
            max_km=max_km,
            n_pairs=n_pairs,
            normalize=False,
            seed=seed,
        )
        lags = vario['bin_centers_km']
        gamma = vario['semivariance']
    else:
        if n_bins < 2:
            raise ValueError(f'n_bins must be at least 2, got {n_bins}.')
        if min_km <= 0.0 or max_km <= min_km:
            raise ValueError(
                f'Expected 0 < min_km < max_km, got min_km={min_km}, max_km={max_km}.'
            )
        if pairs_per_lag <= 0:
            raise ValueError(f'pairs_per_lag must be positive, got {pairs_per_lag}.')
        lags = np.geomspace(min_km, max_km, n_bins)
        rng = np.random.default_rng(seed)
        k = min(pairs_per_lag, n_pts)
        selections = [rng.choice(n_pts, size=k, replace=False) for _ in range(n_bins)]
        off_lons: list[NDArray[np.float64]] = []
        off_lats: list[NDArray[np.float64]] = []
        off_years: list[NDArray[np.float64]] = []
        off_months: list[NDArray[np.float64]] = []
        for sel, lag in zip(selections, lags, strict=True):
            o_lon, o_lat = geodesic_offset(
                lon_arr[sel], lat_arr[sel], float(lag), rng=rng
            )
            off_lons.append(o_lon)
            off_lats.append(o_lat)
            if year_arr is not None:
                off_years.append(year_arr[sel])
            if month_arr is not None:
                off_months.append(month_arr[sel])

        moved_emb = _prepare_embeddings(
            encoder.encode(
                np.concatenate(off_lons),
                np.concatenate(off_lats),
                year=None if year_arr is None else np.concatenate(off_years),
                month=None if month_arr is None else np.concatenate(off_months),
            ),
            normalize=normalize,
        )
        gamma = np.empty(n_bins, dtype=np.float64)
        for idx_lag, sel in enumerate(selections):
            diff = base[sel] - moved_emb[idx_lag * k : (idx_lag + 1) * k]
            gamma[idx_lag] = 0.5 * float(np.mean(np.sum(diff * diff, axis=1)))

    hs_km, sill = half_sill_distance(lags, gamma, frac=frac)
    grid_km = hs_km / 2.0
    grid_deg = grid_km / KM_PER_DEGREE
    return {
        'half_sill_km': hs_km,
        'sill': sill,
        'suggested_grid_km': grid_km,
        'suggested_grid_deg': grid_deg,
        'n_points': float(n_pts),
        'embed_dim': float(embed_dim),
    }


__all__ = (
    'EARTH_RADIUS_KM',
    'KM_PER_DEGREE',
    'analyze_implicit_resolution',
    'empirical_semivariogram',
    'geodesic_offset',
    'half_sill_distance',
    'sample_sphere_points',
)
