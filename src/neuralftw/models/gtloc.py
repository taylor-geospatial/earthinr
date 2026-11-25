# Copyright (c) Taylor Geospatial. All rights reserved.
# Licensed under the MIT License.

"""GT-Loc Equal-Earth multiscale RFF capsule location encoder."""

from collections.abc import Sequence
from typing import Any

import torch
from torch import Tensor

from .._types import Path
from ._base import LocationEncoder, Weights, WeightsEnum
from ._utils import load_safetensors
from .rff_capsule import EqualEarthRFFEncoder

_ZOO_REPO = 'taylor-geospatial/neuralftw-zoo'


def _normalize_gtloc_state_dict(raw_state: dict[str, Tensor]) -> dict[str, Tensor]:
    """Strip an optional ``location_encoder.`` prefix from GT-Loc state dicts.

    Args:
        raw_state: Raw state dictionary loaded from a checkpoint.

    Returns:
        State dictionary matching :class:`EqualEarthRFFEncoder` ``LocEnc*`` keys.
    """
    return {
        key.removeprefix('location_encoder.'): value
        for key, value in raw_state.items()
        if key.removeprefix('location_encoder.').startswith('LocEnc')
    }


class GTLoc_Weights(WeightsEnum):  # noqa: N801
    """Pretrained GT-Loc location encoder weights."""

    DEFAULT = Weights(
        repo_id=_ZOO_REPO,
        filename='gtloc/model.safetensors',
        meta={
            'sigma': (1.0, 16.0, 256.0),
            'encoded_size': 256,
            'hidden_dim': 1024,
            'output_dim': 512,
            'sha256': (
                '6941e5917cada7fad516ec2d32fc56951c026fd5c6c08553e7f2122e4437d0f0'
            ),
            'onnx_sha256': (
                'fc14b71fe55bbedcf3d90f9c020568eaa4bed21271f828265f29e997522e56f5'
            ),
            'license': 'CC-BY-NC-4.0',
        },
    )


class GTLocLocationEncoder(LocationEncoder):
    """GT-Loc 512-d Equal-Earth multiscale RFF location encoder.

    Trained contrastively in a unified image-location-time embedding space
    (Shatwell et al., ICCV 2025 Oral). Accepts ``(lon, lat)`` coordinates in
    WGS84 degrees and converts them internally to GT-Loc's native ``(lat, lon)``
    ordering.
    """

    name: str = 'gtloc'

    def __init__(
        self,
        *,
        weights: GTLoc_Weights | str | None = None,
        checkpoint: Path | None = None,
        sigma: Sequence[float] | None = None,
        encoded_size: int | None = None,
        hidden_dim: int | None = None,
        out_dim: int | None = None,
        device: str | torch.device = 'cpu',
        batch_size: int = 8192,
    ) -> None:
        """Initialize a new GTLocLocationEncoder instance.

        Args:
            weights: Pretrained :class:`GTLoc_Weights` enum member or name.
            checkpoint: Local ``.safetensors`` checkpoint path.
            sigma: Optional override sequence of RFF scales.
            encoded_size: Optional override number of RFF frequency pairs.
            hidden_dim: Optional override hidden MLP width.
            out_dim: Optional override output embedding dimension.
            device: Target device for inference.
            batch_size: Batch size used by :meth:`encode`.

        Raises:
            ValueError: If both *weights* and *checkpoint* are specified.
        """
        super().__init__(device=device, batch_size=batch_size)
        if weights is not None and checkpoint is not None:
            raise ValueError('Specify at most one of weights or checkpoint.')

        raw_state: dict[str, Tensor] | None = None
        res_sigma: tuple[float, ...] = (
            (1.0, 16.0, 256.0) if sigma is None else tuple(sigma)
        )
        res_encoded = 256 if encoded_size is None else encoded_size
        res_hidden = 1024 if hidden_dim is None else hidden_dim
        res_out = 512 if out_dim is None else out_dim

        if weights is not None:
            resolved = GTLoc_Weights.verify(weights)
            meta = resolved.meta
            res_sigma = tuple(meta['sigma'])
            res_encoded = int(meta['encoded_size'])
            res_hidden = int(meta['hidden_dim'])
            res_out = int(meta['output_dim'])
            raw_state = load_safetensors(resolved.download(), device='cpu')
        elif checkpoint is not None:
            raw_state = load_safetensors(checkpoint, device='cpu')

        self.embed_dim = res_out
        self.encoder = EqualEarthRFFEncoder(
            sigma=res_sigma,
            input_size=2,
            encoded_size=res_encoded,
            hidden_dim=res_hidden,
            out_dim=res_out,
            coord_order='latlon',
        )
        if raw_state is not None:
            self.encoder.load_state_dict(
                _normalize_gtloc_state_dict(raw_state), strict=True
            )

        self.to(self.device)
        self.eval()

    def forward(
        self, coords: Tensor, year: Tensor | None = None, month: Tensor | None = None
    ) -> Tensor:
        """Encode ``(lon, lat)`` coordinates in degrees into GT-Loc embeddings.

        Args:
            coords: Coordinate tensor of shape ``(B, 2)`` in ``(lon, lat)`` degrees.
            year: Unused temporal parameter.
            month: Unused temporal parameter.

        Returns:
            Float32 embedding tensor of shape ``(B, embed_dim)``.
        """
        latlon = torch.stack((coords[:, 1], coords[:, 0]), dim=-1)
        return self.encoder(latlon.to(dtype=torch.float32)).to(dtype=torch.float32)


def gtloc(
    weights: GTLoc_Weights | str | None = GTLoc_Weights.DEFAULT,
    *,
    device: str | torch.device = 'cpu',
    batch_size: int = 8192,
    **kwargs: Any,
) -> GTLocLocationEncoder:
    """Create a GT-Loc location encoder.

    If you use this model in your research, please cite the following paper:

    * https://arxiv.org/abs/2507.10473

    Args:
        weights: Pretrained weights enum member, member name, or ``None``.
        device: Target device for inference.
        batch_size: Batch size used by :meth:`~LocationEncoder.encode`.
        **kwargs: Additional keyword arguments forwarded to
            :class:`GTLocLocationEncoder`.

    Returns:
        Configured :class:`GTLocLocationEncoder` instance.
    """
    return GTLocLocationEncoder(
        weights=weights, device=device, batch_size=batch_size, **kwargs
    )


__all__ = ('GTLocLocationEncoder', 'GTLoc_Weights', 'gtloc')
