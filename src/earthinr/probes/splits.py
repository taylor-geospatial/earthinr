# Copyright (c) Taylor Geospatial. All rights reserved.
# Licensed under the MIT License.

"""Coordinate fold splitting, urban/rural tagging, and spatial buffer utilities."""

import io
import os
import urllib.request
import zipfile
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray
from scipy.spatial import KDTree

from .._errors import DatasetNotFoundError
from .._types import Path

EARTH_RADIUS_KM = 6371.0
GEONAMES_CITIES15000_URL = 'https://download.geonames.org/export/dump/cities15000.zip'
RANDOM_FOLD_ALGORITHM = 'torch-randperm-v1'


def _unit_xyz(
    lat: NDArray[np.floating[Any]], lon: NDArray[np.floating[Any]]
) -> NDArray[np.float64]:
    """Convert WGS84 degrees to unit-sphere 3D Cartesian coordinates."""
    rlat = np.radians(np.asarray(lat, dtype=np.float64))
    rlon = np.radians(np.asarray(lon, dtype=np.float64))
    cos_lat = np.cos(rlat)
    return np.stack([cos_lat * np.cos(rlon), cos_lat * np.sin(rlon), np.sin(rlat)], 1)


def _chord_to_km(chord: NDArray[np.floating[Any]]) -> NDArray[np.float64]:
    """Convert unit-sphere chord distances to great-circle distances in km."""
    return 2.0 * EARTH_RADIUS_KM * np.arcsin(np.clip(chord / 2.0, 0.0, 1.0))


def random_fold_ids(n: int, folds: int = 5, seed: int = 0) -> NDArray[np.int64]:
    """Assign *n* samples to deterministic random folds using CPU ``torch.randperm``.

    Uses an isolated CPU :class:`torch.Generator` seeded with *seed*, matching the
    ``torch-randperm-v1`` protocol and shuffling for every seed including ``seed=0``.

    Args:
        n: Number of samples to partition.
        folds: Number of folds (must be at least 2).
        seed: Random seed for permutation.

    Returns:
        Array of fold indices in ``[0, folds - 1]`` with shape ``(n,)`` and dtype
        ``int64``.

    Raises:
        ValueError: If *n* is negative or *folds* is less than 2.
    """
    if n < 0:
        raise ValueError(f'n must be non-negative, got {n}.')
    if folds < 2:
        raise ValueError(f'folds must be at least 2, got {folds}.')
    if n == 0:
        return np.empty(0, dtype=np.int64)
    gen = torch.Generator(device='cpu').manual_seed(seed)
    perm = torch.randperm(n, generator=gen).numpy()
    fold_ids = np.empty(n, dtype=np.int64)
    fold_ids[perm] = np.arange(n, dtype=np.int64) % folds
    return fold_ids


def spatial_fold_ids(
    lat: NDArray[np.floating[Any]],
    lon: NDArray[np.floating[Any]],
    folds: int = 5,
    cell_deg: float = 10.0,
    seed: int = 0,
) -> NDArray[np.int64]:
    """Assign coordinates to spatial block folds on a latitude/longitude grid.

    All points falling inside the same ``cell_deg x cell_deg`` degree cell share a
    fold, so training and test folds are spatially disjoint.

    Args:
        lat: One-dimensional array of latitudes in degrees.
        lon: One-dimensional array of longitudes in degrees.
        folds: Number of folds (must be at least 2).
        cell_deg: Grid cell width in degrees (must be positive).
        seed: Random seed for permuting grid cells across folds.

    Returns:
        Array of fold indices in ``[0, folds - 1]`` with dtype ``int64``.

    Raises:
        ValueError: If *folds* < 2, *cell_deg* <= 0, or *lat*/*lon* shapes mismatch.
    """
    if folds < 2:
        raise ValueError(f'folds must be at least 2, got {folds}.')
    if cell_deg <= 0.0:
        raise ValueError(f'cell_deg must be positive, got {cell_deg}.')
    lat_arr = np.asarray(lat, dtype=np.float64)
    lon_arr = np.asarray(lon, dtype=np.float64)
    if lat_arr.ndim != 1 or lat_arr.shape != lon_arr.shape:
        raise ValueError('lat and lon must be 1D arrays of equal length.')
    if lat_arr.size == 0:
        return np.empty(0, dtype=np.int64)
    cell = np.floor(lat_arr / cell_deg).astype(np.int64) * 100003 + np.floor(
        lon_arr / cell_deg
    ).astype(np.int64)
    uniq, inv = np.unique(cell, return_inverse=True)
    order = np.random.default_rng(seed).permutation(len(uniq))
    fold_of = (order % folds).astype(np.int64)
    return fold_of[inv]


def load_geonames_cities(
    path: Path | None = None, download: bool = False
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Load city latitudes and longitudes from GeoNames ``cities15000.txt``.

    Args:
        path: Local path to ``cities15000.txt``. Defaults to
            ``~/.cache/earthinr/cities15000.txt``.
        download: Whether to download ``cities15000.zip`` from GeoNames if the file
            does not exist locally.

    Returns:
        Tuple ``(cities_lat, cities_lon)`` of float64 arrays in degrees.

    Raises:
        DatasetNotFoundError: If the file is missing and *download* is ``False``.
    """
    if path is None:
        target = os.path.join(
            os.path.expanduser('~'), '.cache', 'earthinr', 'cities15000.txt'
        )
    else:
        target = os.fspath(path)

    if not os.path.isfile(target):
        if not download:
            raise DatasetNotFoundError(
                f'GeoNames cities15000.txt not found at {target}. '
                'Pass download=True or provide a valid path.'
            )
        parent = os.path.dirname(target)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with urllib.request.urlopen(GEONAMES_CITIES15000_URL, timeout=120) as resp:
            raw = resp.read()
        with zipfile.ZipFile(io.BytesIO(raw)) as zf:
            content = zf.read('cities15000.txt')
        with open(target, 'wb') as out:
            out.write(content)

    lats: list[float] = []
    lons: list[float] = []
    with open(target, encoding='utf-8') as fh:
        for line in fh:
            stripped = line.strip()
            if not stripped:
                continue
            parts = stripped.split('\t')
            lats.append(float(parts[4]))
            lons.append(float(parts[5]))
    return np.asarray(lats, dtype=np.float64), np.asarray(lons, dtype=np.float64)


def urban_rural_tags(
    lat: NDArray[np.floating[Any]],
    lon: NDArray[np.floating[Any]],
    cities_lat: NDArray[np.floating[Any]],
    cities_lon: NDArray[np.floating[Any]],
    urban_km: float = 10.0,
    rural_km: float = 50.0,
) -> NDArray[np.int64]:
    """Tag coordinates as urban, rural, or buffer by distance to nearest city.

    Computes exact great-circle distance to the nearest city via unit-sphere chord
    distance on a 3D KDTree: ``0`` for urban (``<= urban_km``), ``1`` for rural
    (``> rural_km``), and ``-1`` for the intermediate buffer zone.

    Args:
        lat: Query latitudes in degrees.
        lon: Query longitudes in degrees.
        cities_lat: City latitudes in degrees.
        cities_lon: City longitudes in degrees.
        urban_km: Maximum distance in km tagged as urban.
        rural_km: Minimum distance in km (strict) tagged as rural.

    Returns:
        Integer tag array of shape ``(N,)`` with values in ``{-1, 0, 1}``.

    Raises:
        ValueError: If thresholds are invalid, arrays mismatch, or cities are empty.
    """
    if urban_km < 0.0:
        raise ValueError(f'urban_km must be non-negative, got {urban_km}.')
    if rural_km <= urban_km:
        raise ValueError(
            f'rural_km ({rural_km}) must be strictly greater than urban_km ({urban_km}).'
        )
    lat_arr = np.asarray(lat, dtype=np.float64)
    lon_arr = np.asarray(lon, dtype=np.float64)
    c_lat = np.asarray(cities_lat, dtype=np.float64)
    c_lon = np.asarray(cities_lon, dtype=np.float64)
    if lat_arr.ndim != 1 or lat_arr.shape != lon_arr.shape:
        raise ValueError('lat and lon must be 1D arrays of equal length.')
    if c_lat.ndim != 1 or c_lat.shape != c_lon.shape or c_lat.size == 0:
        raise ValueError('cities_lat and cities_lon must be non-empty 1D arrays.')
    if lat_arr.size == 0:
        return np.empty(0, dtype=np.int64)

    tree = KDTree(_unit_xyz(c_lat, c_lon))
    chord, _ = tree.query(_unit_xyz(lat_arr, lon_arr))
    dist_km = _chord_to_km(np.asarray(chord, dtype=np.float64))
    return np.where(dist_km <= urban_km, 0, np.where(dist_km > rural_km, 1, -1)).astype(
        np.int64
    )


def buffered_exclusion_mask(
    lat: NDArray[np.floating[Any]],
    lon: NDArray[np.floating[Any]],
    train_idx: NDArray[np.integer[Any]],
    test_idx: NDArray[np.integer[Any]],
    radius_km: float,
) -> NDArray[np.bool_]:
    """Return a boolean mask of training points outside *radius_km* of any test point.

    Args:
        lat: Full dataset latitudes in degrees.
        lon: Full dataset longitudes in degrees.
        train_idx: Integer indices of candidate training points.
        test_idx: Integer indices of held-out test points.
        radius_km: Exclusion buffer radius in km (non-negative).

    Returns:
        Boolean array of shape ``(len(train_idx),)`` where ``True`` marks retained
        training points (distance ``>= radius_km`` from every test point).

    Raises:
        ValueError: If *radius_km* is negative or *lat*/*lon* shapes mismatch.
    """
    if radius_km < 0.0:
        raise ValueError(f'radius_km must be non-negative, got {radius_km}.')
    lat_arr = np.asarray(lat, dtype=np.float64)
    lon_arr = np.asarray(lon, dtype=np.float64)
    if lat_arr.ndim != 1 or lat_arr.shape != lon_arr.shape:
        raise ValueError('lat and lon must be 1D arrays of equal length.')
    tr = np.asarray(train_idx, dtype=np.int64)
    te = np.asarray(test_idx, dtype=np.int64)
    if radius_km == 0.0 or tr.size == 0 or te.size == 0:
        return np.ones(tr.size, dtype=np.bool_)
    xyz_tr = _unit_xyz(lat_arr[tr], lon_arr[tr])
    xyz_te = _unit_xyz(lat_arr[te], lon_arr[te])
    chord, _ = KDTree(xyz_te).query(xyz_tr)
    dist_km = _chord_to_km(np.asarray(chord, dtype=np.float64))
    return np.asarray(dist_km >= radius_km, dtype=np.bool_)


__all__ = (
    'EARTH_RADIUS_KM',
    'GEONAMES_CITIES15000_URL',
    'RANDOM_FOLD_ALGORITHM',
    'buffered_exclusion_mask',
    'load_geonames_cities',
    'random_fold_ids',
    'spatial_fold_ids',
    'urban_rural_tags',
)
