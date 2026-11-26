# Copyright (c) Taylor Geospatial. All rights reserved.
# Licensed under the MIT License.

"""Coordinate grid generation, batched grid inference, and RGB map plotting."""

import functools
import importlib.resources
import os
from collections.abc import Sequence
from typing import Any

import numpy as np
from numpy.typing import NDArray

from ._types import Path
from ._utils import lazy_import
from .models._base import LocationEncoder
from .models.registry import get_model


@functools.lru_cache(maxsize=1)
def _load_land_mask_grid() -> NDArray[np.bool_]:
    """Load the bundled 0.1° global bit-packed land mask as a 2D boolean array."""
    res = importlib.resources.files('neuralftw.data').joinpath('land_mask.npz')
    with importlib.resources.as_file(res) as path, np.load(path) as data:
        packed = data['packed']
        n_lat, n_lon = (int(v) for v in data['shape'])
    return np.unpackbits(packed, axis=1)[:n_lat, :n_lon].astype(np.bool_, copy=False)


def make_land_mask(
    lon: NDArray[np.floating[Any]],
    lat: NDArray[np.floating[Any]],
    *,
    include_antarctica: bool = False,
) -> NDArray[np.bool_]:
    """Look up a global land mask for WGS84 ``(lon, lat)`` coordinates.

    Uses a bundled 0.1° (~11 km) bit-packed land mask derived from GLOBE with
    zero external GIS dependencies.

    Args:
        lon: Longitude array in WGS84 degrees.
        lat: Latitude array in WGS84 degrees with the same shape as *lon*.
        include_antarctica: Whether to include land south of ``60°S``. Defaults
            to ``False`` because land-trained location encoders rarely sample
            Antarctica and polar extrapolation otherwise dominates global PCA.

    Returns:
        Boolean array with the same shape as *lon* (``True`` on land).

    Raises:
        ValueError: If *lon* and *lat* do not have matching shapes.
    """
    lon_arr = np.asarray(lon, dtype=np.float64)
    lat_arr = np.asarray(lat, dtype=np.float64)
    if lon_arr.shape != lat_arr.shape:
        raise ValueError(
            f'lon and lat must have matching shapes, got {lon_arr.shape} and '
            f'{lat_arr.shape}.'
        )
    grid = _load_land_mask_grid()
    n_lat, n_lon = grid.shape
    lon_wrapped = ((lon_arr + 180.0) % 360.0) - 180.0
    lat_clamped = np.clip(lat_arr, -90.0, 90.0)
    row = np.clip(
        np.floor((90.0 - lat_clamped) * (n_lat / 180.0)).astype(np.intp), 0, n_lat - 1
    )
    col = np.clip(
        np.floor((lon_wrapped + 180.0) * (n_lon / 360.0)).astype(np.intp), 0, n_lon - 1
    )
    mask = grid[row, col].copy()
    if not include_antarctica:
        mask &= lat_arr >= -60.0
    return mask


def make_coord_grid(
    bounds: tuple[float, float, float, float] = (-180.0, -90.0, 180.0, 90.0),
    *,
    step_deg: float | None = None,
    width: int | None = None,
    height: int | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Build a regular ``(lon, lat)`` WGS84 coordinate grid over *bounds*.

    Rows are ordered north-to-south (``lat_max`` at row ``0``, ``lat_min`` at
    row ``H - 1``) so the resulting grid aligns directly with ``imshow`` using
    ``origin='upper'`` and ``extent=(lon_min, lon_max, lat_min, lat_max)``.

    Args:
        bounds: Geographic bounding box ``(lon_min, lat_min, lon_max, lat_max)``
            in WGS84 degrees. Defaults to the full globe.
        step_deg: Uniform angular spacing in degrees. Used when *width* and
            *height* are omitted (defaults to ``1.0`` degree).
        width: Optional explicit number of longitude columns (must be paired
            with *height* and mutually exclusive with *step_deg*).
        height: Optional explicit number of latitude rows (must be paired with
            *width* and mutually exclusive with *step_deg*).

    Returns:
        Tuple ``(lon, lat)`` of 2D ``float64`` arrays with shape ``(H, W)``.

    Raises:
        ValueError: If *bounds* or grid dimensions are invalid.
    """
    if len(bounds) != 4:
        raise ValueError(
            f'bounds must be a 4-tuple (lon_min, lat_min, lon_max, lat_max), '
            f'got length {len(bounds)}.'
        )
    lon_min, lat_min, lon_max, lat_max = (float(v) for v in bounds)
    if lon_min < -180.0 or lon_max > 180.0 or lon_min >= lon_max:
        raise ValueError(
            f'Invalid longitude bounds ({lon_min}, {lon_max}); '
            'expected -180 <= lon_min < lon_max <= 180.'
        )
    if lat_min < -90.0 or lat_max > 90.0 or lat_min >= lat_max:
        raise ValueError(
            f'Invalid latitude bounds ({lat_min}, {lat_max}); '
            'expected -90 <= lat_min < lat_max <= 90.'
        )

    has_size = width is not None or height is not None
    if step_deg is not None and has_size:
        raise ValueError('Specify either step_deg or (width, height), not both.')

    if has_size:
        if width is None or height is None:
            raise ValueError('Both width and height must be specified together.')
        if (
            isinstance(width, bool)
            or isinstance(height, bool)
            or width <= 0
            or height <= 0
        ):
            raise ValueError(
                f'width and height must be positive integers, got ({width}, {height}).'
            )
        n_lon, n_lat = int(width), int(height)
    else:
        resolved_step = 1.0 if step_deg is None else float(step_deg)
        if resolved_step <= 0.0:
            raise ValueError(f'step_deg must be positive, got {resolved_step}.')
        n_lon = max(1, round((lon_max - lon_min) / resolved_step))
        n_lat = max(1, round((lat_max - lat_min) / resolved_step))

    dx = (lon_max - lon_min) / n_lon
    dy = (lat_max - lat_min) / n_lat
    lon_1d = lon_min + (np.arange(n_lon, dtype=np.float64) + 0.5) * dx
    lat_1d = lat_max - (np.arange(n_lat, dtype=np.float64) + 0.5) * dy
    lon_grid, lat_grid = np.meshgrid(lon_1d, lat_1d)
    return lon_grid, lat_grid


def encode_grid(
    encoder: LocationEncoder | str,
    bounds: tuple[float, float, float, float] = (-180.0, -90.0, 180.0, 90.0),
    *,
    step_deg: float | None = None,
    width: int | None = None,
    height: int | None = None,
    year: float | None = None,
    month: int | None = None,
    batch_size: int | None = None,
    device: str = 'cpu',
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float32]]:
    """Generate a coordinate grid and run batched inference with *encoder*.

    Args:
        encoder: Loaded :class:`~neuralftw.models.LocationEncoder` instance or
            registered model preset name (e.g. ``'sinr'``).
        bounds: Geographic bounding box ``(lon_min, lat_min, lon_max, lat_max)``
            in WGS84 degrees. Defaults to the full globe.
        step_deg: Uniform angular spacing in degrees (defaults to ``1.0``).
        width: Optional explicit number of longitude columns.
        height: Optional explicit number of latitude rows.
        year: Optional scalar calendar year broadcast across all grid points.
        month: Optional scalar calendar month (``1..12``) broadcast across all
            grid points.
        batch_size: Optional temporary batch-size override for chunked encoding.
        device: Target device used when *encoder* is passed as a preset name.

    Returns:
        Tuple ``(lon, lat, embeddings)`` where *lon* and *lat* have shape
        ``(H, W)`` (``float64``) and *embeddings* has shape ``(H, W, D)``
        (``float32``).

    Raises:
        ValueError: If *batch_size* is non-positive or grid parameters are
            invalid.
    """
    if batch_size is not None and (
        isinstance(batch_size, bool)
        or not isinstance(batch_size, int)
        or batch_size <= 0
    ):
        raise ValueError(f'batch_size must be a positive int, got {batch_size!r}.')

    resolved_encoder = (
        get_model(encoder, device=device) if isinstance(encoder, str) else encoder
    )
    lon_grid, lat_grid = make_coord_grid(
        bounds, step_deg=step_deg, width=width, height=height
    )
    h, w = lon_grid.shape
    n = h * w
    lon_flat = lon_grid.reshape(-1)
    lat_flat = lat_grid.reshape(-1)
    year_arr = None if year is None else np.full(n, float(year), dtype=np.float64)
    month_arr = None if month is None else np.full(n, float(month), dtype=np.float64)

    orig_batch_size = resolved_encoder.batch_size
    if batch_size is not None:
        resolved_encoder.batch_size = batch_size
    try:
        emb_flat = resolved_encoder.encode(
            lon_flat, lat_flat, year=year_arr, month=month_arr
        )
    finally:
        resolved_encoder.batch_size = orig_batch_size

    emb_grid = emb_flat.reshape(h, w, -1).astype(np.float32, copy=False)
    return lon_grid, lat_grid, emb_grid


def embeddings_to_rgb(
    embeddings: NDArray[np.floating[Any]],
    *,
    components: Sequence[int] = (0, 1, 2),
    percentile: tuple[float, float] = (2.0, 98.0),
    mask: NDArray[np.bool_] | None = None,
    weights: NDArray[np.floating[Any]] | None = None,
    background: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> NDArray[np.float32]:
    """Project an ``(H, W, D)`` embedding grid into a ``[0, 1]`` RGB image via PCA.

    Args:
        embeddings: Embedding grid of shape ``(H, W, D)``.
        components: Three zero-based principal component indices mapped to
            ``(R, G, B)`` (e.g. ``(0, 1, 2)`` for global structure or
            ``(3, 4, 5)`` for finer regional variation).
        percentile: Low and high clipping percentiles ``(p_low, p_high)`` in
            ``[0, 100]`` applied independently per channel on valid pixels.
        mask: Optional boolean array of shape ``(H, W)`` restricting PCA
            fitting, contrast stretching, and rendering to ``True`` pixels
            (e.g. from :func:`make_land_mask`).
        weights: Optional non-negative sample-weight array of shape ``(H, W)``
            used for area-weighted PCA mean and covariance (e.g. ``cos(lat)``
            on equirectangular grids).
        background: RGB color in ``[0, 1]`` used to fill masked or non-finite
            pixels.

    Returns:
        Float32 RGB array of shape ``(H, W, 3)`` with values in ``[0.0, 1.0]``.

    Raises:
        ValueError: If *embeddings*, *components*, *percentile*, *mask*,
            *weights*, or *background* is invalid.
    """
    arr = np.asarray(embeddings, dtype=np.float64)
    if arr.ndim != 3 or arr.shape[2] == 0:
        raise ValueError(
            f'embeddings must have shape (H, W, D) with D > 0, got {arr.shape}.'
        )
    if len(components) != 3 or any(c < 0 for c in components):
        raise ValueError(
            f'components must contain 3 non-negative indices, got {components!r}.'
        )
    p_lo, p_hi = float(percentile[0]), float(percentile[1])
    if p_lo < 0.0 or p_hi > 100.0 or p_lo >= p_hi:
        raise ValueError(
            f'percentile must satisfy 0 <= low < high <= 100, got {percentile!r}.'
        )
    bg = np.asarray(background, dtype=np.float32)
    if (
        bg.shape != (3,)
        or not np.isfinite(bg).all()
        or (bg < 0.0).any()
        or (bg > 1.0).any()
    ):
        raise ValueError(
            f'background must be an RGB 3-tuple in [0, 1], got {background!r}.'
        )

    h, w, d = arr.shape
    flat = arr.reshape(h * w, d)
    valid = np.isfinite(flat).all(axis=1)
    if mask is not None:
        mask_arr = np.asarray(mask, dtype=np.bool_)
        if mask_arr.shape != (h, w):
            raise ValueError(
                f'mask shape {mask_arr.shape} does not match grid shape {(h, w)}.'
            )
        valid &= mask_arr.reshape(-1)

    w_arr: NDArray[np.float64] | None = None
    if weights is not None:
        w_arr = np.asarray(weights, dtype=np.float64)
        if w_arr.shape != (h, w):
            raise ValueError(
                f'weights shape {w_arr.shape} does not match grid shape {(h, w)}.'
            )

    rgb = np.empty((h * w, 3), dtype=np.float32)
    rgb[:] = bg
    if not valid.any():
        return rgb.reshape(h, w, 3)

    x = flat[valid]
    if w_arr is not None:
        w_valid = w_arr.reshape(-1)[valid]
        if not np.isfinite(w_valid).all() or (w_valid < 0.0).any():
            raise ValueError('weights must contain finite non-negative values.')
        w_sum = float(w_valid.sum())
        if w_sum <= 0.0:
            raise ValueError('weights must have positive total weight on valid pixels.')
        w_norm = w_valid / w_sum
        mean = (w_norm[:, None] * x).sum(axis=0, keepdims=True)
        centered = x - mean
        _, _, vt = np.linalg.svd(
            centered * np.sqrt(w_norm)[:, None], full_matrices=False
        )
    else:
        mean = x.mean(axis=0, keepdims=True)
        centered = x - mean
        _, _, vt = np.linalg.svd(centered, full_matrices=False)

    max_comp = vt.shape[0]
    proj = np.zeros((x.shape[0], 3), dtype=np.float64)
    for out_idx, comp_idx in enumerate(components):
        if comp_idx < max_comp:
            proj[:, out_idx] = centered @ vt[comp_idx]

    for ch in range(3):
        col = proj[:, ch]
        lo, hi = np.percentile(col, [p_lo, p_hi])
        span = hi - lo
        if span > 1e-12:
            normed = np.clip((col - lo) / span, 0.0, 1.0)
        else:
            normed = np.zeros_like(col)
        rgb[valid, ch] = normed.astype(np.float32)

    return rgb.reshape(h, w, 3)


def plot_embedding_map(
    lon: NDArray[np.floating[Any]],
    lat: NDArray[np.floating[Any]],
    embeddings: NDArray[np.floating[Any]],
    *,
    components: Sequence[int] = (0, 1, 2),
    percentile: tuple[float, float] = (2.0, 98.0),
    mask: NDArray[np.bool_] | None = None,
    land_only: bool = False,
    include_antarctica: bool = False,
    area_weighted: bool = True,
    background: tuple[float, float, float] = (0.96, 0.97, 0.99),
    ax: Any = None,
    title: str | None = None,
    figsize: tuple[float, float] = (10.0, 5.0),
    save_path: Path | None = None,
) -> tuple[Any, Any]:
    """Plot a PCA RGB map of gridded location embeddings with Matplotlib.

    Args:
        lon: 2D longitude grid of shape ``(H, W)`` from :func:`make_coord_grid`
            or :func:`encode_grid`.
        lat: 2D latitude grid of shape ``(H, W)``.
        embeddings: Embedding grid of shape ``(H, W, D)``.
        components: Three zero-based principal component indices mapped to RGB.
        percentile: Low/high percentile contrast stretch per channel.
        mask: Optional boolean mask of shape ``(H, W)`` restricting PCA and
            rendering to ``True`` pixels.
        land_only: If ``True``, fit PCA and render strictly on land pixels
            using :func:`make_land_mask` (recommended for land-trained encoders
            such as SINR, SatCLIP, GeoCLIP, and MIND).
        include_antarctica: Passed to :func:`make_land_mask` when *land_only*
            is ``True`` (defaults to ``False``).
        area_weighted: If ``True`` (default), weight PCA mean and covariance by
            ``cos(lat)`` so high-latitude equirectangular pixels do not bias
            the principal components.
        background: RGB color in ``[0, 1]`` for masked or non-finite pixels.
        ax: Optional existing Matplotlib ``Axes`` to draw onto.
        title: Optional plot title.
        figsize: Figure size ``(width, height)`` in inches when *ax* is omitted.
        save_path: Optional file path to save the figure via ``fig.savefig``.

    Returns:
        Tuple ``(fig, ax)`` of Matplotlib figure and axes objects.

    Raises:
        DependencyNotFoundError: If ``matplotlib`` is not installed.
        ValueError: If *lon*, *lat*, and *embeddings* shapes do not match.
    """
    plt = lazy_import('matplotlib.pyplot')

    lon_arr = np.asarray(lon, dtype=np.float64)
    lat_arr = np.asarray(lat, dtype=np.float64)
    emb_arr = np.asarray(embeddings)
    if lon_arr.ndim != 2 or lat_arr.ndim != 2 or lon_arr.shape != lat_arr.shape:
        raise ValueError(
            f'lon and lat must be matching 2D grids, got {lon_arr.shape} and '
            f'{lat_arr.shape}.'
        )
    if emb_arr.ndim != 3 or emb_arr.shape[:2] != lon_arr.shape:
        raise ValueError(
            f'embeddings spatial shape {emb_arr.shape[:2]} does not match '
            f'grid shape {lon_arr.shape}.'
        )

    resolved_mask: NDArray[np.bool_] | None = (
        None if mask is None else np.asarray(mask, dtype=np.bool_)
    )
    if land_only:
        land_mask = make_land_mask(
            lon_arr, lat_arr, include_antarctica=include_antarctica
        )
        if resolved_mask is None:
            resolved_mask = land_mask
        else:
            if resolved_mask.shape != land_mask.shape:
                raise ValueError(
                    f'mask shape {resolved_mask.shape} does not match '
                    f'grid shape {land_mask.shape}.'
                )
            resolved_mask = resolved_mask & land_mask

    weights = (
        np.clip(np.cos(np.deg2rad(lat_arr)), 1e-6, None) if area_weighted else None
    )
    rgb = embeddings_to_rgb(
        emb_arr,
        components=components,
        percentile=percentile,
        mask=resolved_mask,
        weights=weights,
        background=background,
    )

    h, w = lon_arr.shape
    dx = abs(float(lon_arr[0, 1] - lon_arr[0, 0])) if w > 1 else 1.0
    dy = abs(float(lat_arr[0, 0] - lat_arr[1, 0])) if h > 1 else 1.0
    extent = (
        float(lon_arr.min()) - 0.5 * dx,
        float(lon_arr.max()) + 0.5 * dx,
        float(lat_arr.min()) - 0.5 * dy,
        float(lat_arr.max()) + 0.5 * dy,
    )

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    ax.imshow(rgb, extent=extent, origin='upper', aspect='auto')
    ax.set_xlabel('Longitude (°)')
    ax.set_ylabel('Latitude (°)')
    if title is not None:
        ax.set_title(title)
    if save_path is not None:
        fig.savefig(os.fspath(save_path), dpi=150, bbox_inches='tight')
    return fig, ax


__all__ = (
    'embeddings_to_rgb',
    'encode_grid',
    'make_coord_grid',
    'make_land_mask',
    'plot_embedding_map',
)
