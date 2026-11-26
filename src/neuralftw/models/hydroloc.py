# Copyright (c) Taylor Geospatial. All rights reserved.
# Licensed under the MIT License.

"""HydroLoc Equal-Earth residual SIREN location encoder."""

import os
import pathlib
from typing import Any

import torch
from torch import Tensor, nn

from .._types import Path
from ._base import LocationEncoder, Weights, WeightsEnum
from ._utils import equal_earth_projection, load_safetensors
from .mind import SIRENLayer

_ZOO_REPO = 'taylor-geospatial/neuralftw-zoo'


def _normalize_hydroloc_state_dict(state_dict: dict[str, Tensor]) -> dict[str, Tensor]:
    """Validate and normalize a HydroLoc checkpoint state dictionary.

    Args:
        state_dict: Raw state dictionary loaded from disk.

    Returns:
        Dictionary containing ``first.linear.*`` and ``blocks.*.linear.*`` tensors.

    Raises:
        KeyError: If required HydroLoc ReSIREN weights are missing.
    """
    filtered = {
        k: v
        for k, v in state_dict.items()
        if k.startswith(('first.linear.', 'blocks.'))
    }
    if 'first.linear.weight' not in filtered:
        raise KeyError(
            "Could not find HydroLoc location encoder weights under 'first.linear.'."
        )
    return filtered


class HydroLoc_Weights(WeightsEnum):  # noqa: N801
    """Pretrained HydroLoc location encoder weights."""

    DEFAULT = Weights(
        repo_id=_ZOO_REPO,
        filename='hydroloc/model.safetensors',
        meta={
            'embed_dim': 512,
            'depth': 4,
            'w0_first': 30.0,
            'w0': 1.0,
            'matryoshka_dims': (64, 128, 256, 512),
            'sha256': (
                '681c7a1ba4ea9bc390589e8344412ed754b78ba26e3676598378702127b8d6a4'
            ),
            'onnx_sha256': (
                'e00e99849a26f1ae7d9e603fa978676b4a49354b384e77f5a0820891f5a4991f'
            ),
            'license': 'Apache-2.0',
        },
    )


class HydroLocLocationEncoder(LocationEncoder):
    """HydroLoc Equal-Earth residual SIREN location encoder.

    Trained on the Hydro dataset (``isaaccorley/hydro``, weights from
    ``isaaccorley/hydroloc``) by distilling frozen DINOv3 ViT-L/16 and OlmoEarth
    v1.2 Base image embeddings with a nested Matryoshka objective over prefix
    widths ``(64, 128, 256, 512)``. Accepts universal ``(lon, lat)`` WGS84
    coordinates in degrees and ignores optional *year* and *month* inputs.
    """

    name: str = 'hydroloc'

    def __init__(
        self,
        embed_dim: int = 512,
        depth: int = 4,
        w0_first: float = 30.0,
        w0: float = 1.0,
        *,
        dim: int | None = None,
        weights: HydroLoc_Weights | str | None = None,
        checkpoint: Path | None = None,
        device: str | torch.device = 'cpu',
        batch_size: int = 8192,
    ) -> None:
        """Initialize a new HydroLocLocationEncoder instance.

        Args:
            embed_dim: Width of the residual SIREN trunk.
            depth: Number of residual SIREN blocks after the first layer.
            w0_first: Frequency scaling for the first SIREN layer.
            w0: Frequency scaling for residual SIREN blocks.
            dim: Optional Matryoshka prefix width truncation in ``[1, embed_dim]``
                (defaults to the full trunk width).
            weights: Pretrained :class:`HydroLoc_Weights` enum member or name.
            checkpoint: Local path to a HydroLoc ``.safetensors`` or ``.pt``
                checkpoint.
            device: Target device for inference.
            batch_size: Batch size used by :meth:`encode`.

        Raises:
            ValueError: If both *weights* and *checkpoint* are specified, or if
                *dim* is out of range.
        """
        super().__init__(device=device, batch_size=batch_size)
        if weights is not None and checkpoint is not None:
            raise ValueError('Specify at most one of weights or checkpoint.')

        raw_state: dict[str, Tensor] | None = None
        if weights is not None:
            resolved = HydroLoc_Weights.verify(weights)
            embed_dim = int(resolved.meta['embed_dim'])
            depth = int(resolved.meta['depth'])
            w0_first = float(resolved.meta['w0_first'])
            w0 = float(resolved.meta['w0'])
            raw_state = load_safetensors(resolved.download(), device='cpu')
        elif checkpoint is not None:
            raw_state = self._load_checkpoint(checkpoint)
            normalized = _normalize_hydroloc_state_dict(raw_state)
            embed_dim = int(normalized['first.linear.weight'].shape[0])
            depth = sum(
                k.startswith('blocks.') and k.endswith('.linear.weight')
                for k in normalized
            )
            raw_state = normalized

        resolved_dim = embed_dim if dim is None else dim
        if (
            isinstance(resolved_dim, bool)
            or not isinstance(resolved_dim, int)
            or resolved_dim < 1
            or resolved_dim > embed_dim
        ):
            raise ValueError(
                f'dim must be an integer in [1, {embed_dim}], got {dim!r}.'
            )

        self.trunk_dim = embed_dim
        self.depth = depth
        self.embed_dim = resolved_dim
        self.first = SIRENLayer(2, embed_dim, w0=w0_first, is_first=True, act='siren')
        self.blocks = nn.ModuleList(
            [SIRENLayer(embed_dim, embed_dim, w0=w0, act='siren') for _ in range(depth)]
        )

        if raw_state is not None:
            self.load_state_dict(_normalize_hydroloc_state_dict(raw_state), strict=True)

        self.to(self.device)
        self.eval()

    @staticmethod
    def _load_checkpoint(path: Path) -> dict[str, Tensor]:
        """Load a local safetensors or PyTorch checkpoint payload."""
        if pathlib.Path(path).suffix == '.safetensors':
            return load_safetensors(path, device='cpu')
        payload = torch.load(os.fspath(path), map_location='cpu', weights_only=True)
        if isinstance(payload, dict) and 'state_dict' in payload:
            return payload['state_dict']
        return payload

    def forward(
        self, coords: Tensor, year: Tensor | None = None, month: Tensor | None = None
    ) -> Tensor:
        """Encode ``(lon, lat)`` coordinates in degrees into HydroLoc embeddings.

        Args:
            coords: Coordinate tensor of shape ``(B, 2)`` in ``(lon, lat)`` WGS84
                degrees.
            year: Unused temporal parameter.
            month: Unused temporal parameter.

        Returns:
            Float32 embedding tensor of shape ``(B, embed_dim)``.
        """
        loc = equal_earth_projection(coords, coord_order='lonlat')
        h = self.first(loc.to(dtype=self.first.linear.weight.dtype))
        for blk in self.blocks:
            h = h + blk(h)
        if self.embed_dim < self.trunk_dim:
            h = h[:, : self.embed_dim]
        return h.to(dtype=torch.float32)


def hydroloc(
    weights: HydroLoc_Weights | str | None = HydroLoc_Weights.DEFAULT,
    *,
    dim: int | None = None,
    device: str | torch.device = 'cpu',
    batch_size: int = 8192,
    **kwargs: Any,
) -> HydroLocLocationEncoder:
    """Create a HydroLoc location encoder.

    HydroLoc is trained on the Hydro dataset (``isaaccorley/hydro``, weights
    released at ``isaaccorley/hydroloc``).

    Args:
        weights: Pretrained weights enum member, member name, or ``None``.
        dim: Optional Matryoshka prefix width truncation (recommended prefixes:
            64, 128, 256, or 512).
        device: Target device for inference.
        batch_size: Batch size used by :meth:`~LocationEncoder.encode`.
        **kwargs: Additional keyword arguments forwarded to
            :class:`HydroLocLocationEncoder`.

    Returns:
        Configured :class:`HydroLocLocationEncoder` instance.
    """
    return HydroLocLocationEncoder(
        weights=weights, dim=dim, device=device, batch_size=batch_size, **kwargs
    )


__all__ = ('HydroLocLocationEncoder', 'HydroLoc_Weights', 'hydroloc')
