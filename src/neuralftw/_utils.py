# Copyright (c) Taylor Geospatial. All rights reserved.
# Licensed under the MIT License.

"""Shared runtime utilities for NeuralFTW."""

import importlib
from types import ModuleType

import torch

from ._errors import DependencyNotFoundError


def lazy_import(name: str, extra: str | None = None) -> ModuleType:
    """Import an optional dependency lazily, raising a helpful error if missing.

    Args:
        name: Fully qualified module name to import.
        extra: Optional ``neuralftw`` extra name providing the dependency.

    Returns:
        The imported module object.

    Raises:
        DependencyNotFoundError: If *name* cannot be imported.
    """
    try:
        return importlib.import_module(name)
    except ModuleNotFoundError:
        install_hint = f'neuralftw[{extra}]' if extra is not None else name
        raise DependencyNotFoundError(
            f'{name} is not installed and is required for this feature. '
            f"Install it with 'pip install {install_hint}'."
        ) from None


def resolve_device(device: str | torch.device = 'cpu') -> torch.device:
    """Resolve and validate a PyTorch device specification.

    Explicit ``'cpu'`` requests never query CUDA availability. ``'auto'`` selects the
    current CUDA device when available and otherwise falls back to CPU. Unavailable
    or out-of-range CUDA/MPS requests fail eagerly.

    Args:
        device: Target device string (e.g. ``'cpu'``, ``'auto'``, ``'cuda'``,
            ``'cuda:0'``, ``'mps'``) or :class:`torch.device` instance.

    Returns:
        Validated :class:`torch.device` instance.

    Raises:
        ValueError: If *device* is empty, malformed, unavailable, or out of range.
    """
    if isinstance(device, str):
        raw = device.strip()
        if not raw:
            raise ValueError('Device string must not be empty.')
        if raw == 'cpu':
            return torch.device('cpu')
        if raw == 'auto':
            if torch.cuda.is_available():
                return torch.device('cuda', torch.cuda.current_device())
            return torch.device('cpu')
        try:
            resolved = torch.device(raw)
        except (RuntimeError, ValueError) as err:
            raise ValueError(f'Invalid device: {device!r}.') from err
    else:
        resolved = device

    if resolved.type == 'cpu':
        return resolved
    if resolved.type == 'cuda':
        if not torch.cuda.is_available():
            raise ValueError(
                f'CUDA device {resolved!s} requested, but CUDA is not available.'
            )
        if resolved.index is not None:
            count = torch.cuda.device_count()
            if resolved.index >= count:
                raise ValueError(
                    f'CUDA device index {resolved.index} is out of range for '
                    f'{count} available device(s).'
                )
        return resolved
    if resolved.type == 'mps':
        if not torch.backends.mps.is_available():
            raise ValueError('MPS device requested, but MPS is not available.')
        return resolved
    return resolved


def float64_dtype(device: torch.device) -> torch.dtype:
    """Return ``torch.float32`` on MPS (no float64 support) and ``torch.float64`` elsewhere."""
    return torch.float32 if device.type == 'mps' else torch.float64


def float64_device(device: torch.device) -> torch.device:
    """Return CPU when *device* is MPS (no float64 support), otherwise *device*."""
    return torch.device('cpu') if device.type == 'mps' else device


__all__ = ('float64_device', 'float64_dtype', 'lazy_import', 'resolve_device')
