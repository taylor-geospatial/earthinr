# Copyright (c) Taylor Geospatial. All rights reserved.
# Licensed under the MIT License.

"""Coordinate projection, geometry, and checkpoint utilities for models."""

import json
import math
import os
from typing import Any

import torch
from torch import Tensor

from .._errors import ConfigError
from .._types import CoordOrder, Path
from .._utils import resolve_device

# Equal-Earth projection polynomial coefficients and scale factor
# Adapted from https://doi.org/10.1080/13658816.2018.1504949
_EE_A1 = 1.340264
_EE_A2 = -0.081106
_EE_A3 = 0.000893
_EE_A4 = 0.003796
_EE_SF = 66.50336
_SQRT3 = math.sqrt(3.0)


def equal_earth_projection(
    coords: Tensor, *, coord_order: CoordOrder = 'lonlat'
) -> Tensor:
    """Project geographic coordinates in degrees via the Equal-Earth projection.

    Args:
        coords: Coordinate tensor of shape ``(..., 2)`` in degrees.
        coord_order: Ordering of the last dimension, either ``'lonlat'`` or
            ``'latlon'``.

    Returns:
        Projected ``(x, y)`` tensor of shape ``(..., 2)`` scaled by ``SF / 180``.

    Raises:
        ValueError: If *coords* does not have last dimension 2 or *coord_order* is
            invalid.
    """
    if coords.ndim < 1 or coords.shape[-1] != 2:
        raise ValueError(f'coords must have shape (..., 2), got {tuple(coords.shape)}.')
    if coord_order == 'lonlat':
        lon_deg = coords[..., 0]
        lat_deg = coords[..., 1]
    elif coord_order == 'latlon':
        lat_deg = coords[..., 0]
        lon_deg = coords[..., 1]
    else:
        raise ValueError(
            f"coord_order must be 'lonlat' or 'latlon', got {coord_order!r}."
        )

    lon = torch.deg2rad(lon_deg)
    lat = torch.deg2rad(lat_deg)
    theta = torch.asin((_SQRT3 / 2.0) * torch.sin(lat))
    theta2 = theta * theta
    theta6 = theta2 * theta2 * theta2
    theta8 = theta6 * theta2

    denom = 3.0 * (
        9.0 * _EE_A4 * theta8 + 7.0 * _EE_A3 * theta6 + 3.0 * _EE_A2 * theta2 + _EE_A1
    )
    x = (2.0 * _SQRT3 * lon * torch.cos(theta)) / denom
    y = theta * (_EE_A4 * theta8 + _EE_A3 * theta6 + _EE_A2 * theta2 + _EE_A1)
    return torch.stack((x, y), dim=-1) * (_EE_SF / 180.0)


def lonlat_to_xyz(lon: Tensor, lat: Tensor) -> Tensor:
    """Convert longitude and latitude in degrees to unit-sphere 3D coordinates.

    Args:
        lon: Longitude tensor in degrees.
        lat: Latitude tensor in degrees. Finite values must lie in ``[-90, 90]``.

    Returns:
        Unit-sphere coordinates ``(cos(lat)cos(lon), cos(lat)sin(lon), sin(lat))``
        with shape ``(*lon.shape, 3)``.

    Raises:
        ValueError: If *lon* and *lat* have mismatched shapes or any finite *lat*
            lies outside ``[-90, 90]``.
    """
    if lon.shape != lat.shape:
        raise ValueError(
            f'lon and lat must have the same shape, got {tuple(lon.shape)} and '
            f'{tuple(lat.shape)}.'
        )
    finite_lat = lat[torch.isfinite(lat)]
    if finite_lat.numel() > 0 and bool(
        ((finite_lat < -90.0) | (finite_lat > 90.0)).any()
    ):
        raise ValueError('Latitude values must be in [-90, 90] degrees.')

    lon_rad = torch.deg2rad(lon)
    lat_rad = torch.deg2rad(lat)
    cos_lat = torch.cos(lat_rad)
    x = cos_lat * torch.cos(lon_rad)
    y = cos_lat * torch.sin(lon_rad)
    z = torch.sin(lat_rad)
    return torch.stack((x, y, z), dim=-1)


def hf_download(
    repo_id: str,
    filename: str,
    *,
    repo_type: str = 'model',
    revision: str | None = None,
) -> str:
    """Download a file from HuggingFace Hub and return its local cached path.

    Args:
        repo_id: HuggingFace repository identifier (e.g. ``'MVRL/ecogeo'``).
        filename: File path inside the repository.
        repo_type: Repository type (e.g. ``'model'`` or ``'dataset'``).
        revision: Optional Git revision (branch, tag, or commit hash).

    Returns:
        Local filesystem path to the downloaded file.
    """
    import huggingface_hub

    return huggingface_hub.hf_hub_download(
        repo_id=repo_id, filename=filename, repo_type=repo_type, revision=revision
    )


def load_safetensors(
    path: Path, *, device: str | torch.device = 'cpu'
) -> dict[str, Tensor]:
    """Load tensors from a safetensors file onto the resolved device.

    Args:
        path: Path to the ``.safetensors`` file.
        device: Target device for loaded tensors.

    Returns:
        Mapping from tensor names to loaded :class:`torch.Tensor` objects.
    """
    import safetensors.torch

    resolved = resolve_device(device)
    return safetensors.torch.load_file(os.fspath(path), device=str(resolved))


def load_json_config(path: Path) -> dict[str, Any]:
    """Load a JSON configuration object from disk.

    Args:
        path: Path to the JSON configuration file.

    Returns:
        Parsed JSON dictionary.

    Raises:
        ConfigError: If the parsed JSON root value is not a dictionary.
    """
    fpath = os.fspath(path)
    with open(fpath, encoding='utf-8') as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ConfigError(f'Expected JSON object in {fpath}.')
    return data


__all__ = (
    'equal_earth_projection',
    'hf_download',
    'load_json_config',
    'load_safetensors',
    'lonlat_to_xyz',
)
