# Copyright (c) Taylor Geospatial. All rights reserved.
# Licensed under the MIT License.

"""GeoCLOSP-RN spherical-harmonic SIREN location encoder."""

import os
import pathlib
from typing import Any

import torch
from torch import Tensor, nn

from .._types import Path
from ._base import LocationEncoder, Weights, WeightsEnum
from ._utils import load_safetensors
from .satclip import SirenNet, StableAnalyticSphericalHarmonics

_ZOO_REPO = 'taylor-geospatial/earthinr-zoo'


def _normalize_geoclosp_state_dict(state_dict: dict[str, Tensor]) -> dict[str, Tensor]:
    """Normalize GeoCLOSP-RN checkpoint keys to the ``nnet.*`` + ``projection.*`` schema.

    Supports both stripped ``earthinr-zoo`` checkpoints (``nnet.*``,
    ``projection.*``) and full upstream ``DarthReca/GeoCLOSP-RN`` checkpoints
    (``location_encoder.nnet.*``, ``location_projection.*``).

    Args:
        state_dict: Raw state dictionary loaded from disk.

    Returns:
        Normalized state dictionary keyed by ``nnet.*`` and ``projection.*``.

    Raises:
        KeyError: If required SIREN or projection head weights are missing.
    """
    normalized: dict[str, Tensor] = {}
    for key, tensor in state_dict.items():
        if key.startswith('location_encoder.nnet.'):
            normalized[f'nnet.{key.removeprefix("location_encoder.nnet.")}'] = tensor
        elif key.startswith('nnet.'):
            normalized[key] = tensor
        elif key.startswith('location_projection.'):
            normalized[f'projection.{key.removeprefix("location_projection.")}'] = (
                tensor
            )
        elif key.startswith('projection.'):
            normalized[key] = tensor

    has_nnet = any(k.startswith('nnet.') for k in normalized)
    has_proj = 'projection.weight' in normalized and 'projection.bias' in normalized
    if not (has_nnet and has_proj):
        raise KeyError(
            'Could not find GeoCLOSP-RN location encoder and projection weights.'
        )
    return normalized


class GeoCLOSP_Weights(WeightsEnum):  # noqa: N801
    """Pretrained GeoCLOSP-RN location encoder weights."""

    RN50_L10 = Weights(
        repo_id=_ZOO_REPO,
        filename='geoclosp-rn/model.safetensors',
        meta={
            'legendre_polys': 10,
            'capacity': 512,
            'location_dim': 256,
            'embed_dim': 384,
            'num_hidden_layers': 2,
            'sha256': (
                '4979f3a0cd444fd005bfdbf208fe6b35fc282068ad7efa609e93a6461b12272c'
            ),
            'onnx_sha256': (
                '821da765790872bc766e5111abb842a30736cc2f6cdfa90cde1cf324e6d0e5a8'
            ),
            'license': 'CreativeML-OpenRAIL-M',
        },
    )


class GeoCLOSPLocationEncoder(LocationEncoder):
    """GeoCLOSP-RN spherical-harmonic SIREN location encoder with CLOSP projection.

    Combines SatCLIP's ``L=10`` spherical-harmonic SIREN trunk with GeoCLOSP-RN's
    trained 256-to-384 multimodal linear projection head and L2 normalization,
    matching ``CLOSPModel.get_location_features`` (Cambrin et al., 2025). Accepts
    universal ``(lon, lat)`` WGS84 coordinates in degrees and ignores optional
    *year* and *month* inputs.
    """

    name: str = 'geoclosp'

    def __init__(
        self,
        legendre_polys: int = 10,
        capacity: int = 512,
        location_dim: int = 256,
        embed_dim: int = 384,
        num_hidden_layers: int = 2,
        *,
        weights: GeoCLOSP_Weights | str | None = None,
        checkpoint: Path | None = None,
        device: str | torch.device = 'cpu',
        batch_size: int = 8192,
    ) -> None:
        """Initialize a new GeoCLOSPLocationEncoder instance.

        Args:
            legendre_polys: Number of Legendre polynomial degrees ``L``.
            capacity: Hidden layer width of the SIREN network.
            location_dim: Output dimension of the SIREN trunk prior to projection.
            embed_dim: Output embedding dimension after the CLOSP linear projection.
            num_hidden_layers: Number of hidden SIREN layers.
            weights: Pretrained :class:`GeoCLOSP_Weights` enum member or name.
            checkpoint: Local ``.safetensors`` or PyTorch checkpoint path.
            device: Target device for inference.
            batch_size: Batch size used by :meth:`encode`.

        Raises:
            ValueError: If both *weights* and *checkpoint* are specified.
        """
        super().__init__(device=device, batch_size=batch_size)
        if weights is not None and checkpoint is not None:
            raise ValueError('Specify at most one of weights or checkpoint.')

        raw_state: dict[str, Tensor] | None = None
        if weights is not None:
            resolved = GeoCLOSP_Weights.verify(weights)
            legendre_polys = int(resolved.meta['legendre_polys'])
            capacity = int(resolved.meta['capacity'])
            location_dim = int(resolved.meta['location_dim'])
            embed_dim = int(resolved.meta['embed_dim'])
            num_hidden_layers = int(resolved.meta['num_hidden_layers'])
            raw_state = load_safetensors(resolved.download(), device='cpu')
        elif checkpoint is not None:
            raw_state = self._load_checkpoint(checkpoint)

        self.legendre_polys = legendre_polys
        self.location_dim = location_dim
        self.embed_dim = embed_dim
        self.posenc = StableAnalyticSphericalHarmonics(
            legendre_polys=legendre_polys,
            compute_dtype=torch.float64,
            output_dtype=torch.float32,
        )
        self.nnet = SirenNet(
            dim_in=self.posenc.embedding_dim,
            dim_hidden=capacity,
            dim_out=location_dim,
            num_layers=num_hidden_layers,
        )
        self.projection = nn.Linear(location_dim, embed_dim)

        if raw_state is not None:
            normalized = _normalize_geoclosp_state_dict(raw_state)
            self.load_state_dict(normalized, strict=True)

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
        """Encode ``(lon, lat)`` coordinates in degrees into GeoCLOSP embeddings.

        Args:
            coords: Coordinate tensor of shape ``(B, 2)`` in ``(lon, lat)`` WGS84
                degrees.
            year: Unused temporal parameter.
            month: Unused temporal parameter.

        Returns:
            L2-normalized float32 embedding tensor of shape ``(B, embed_dim)``.
        """
        encoded = self.posenc(coords).to(dtype=self.nnet.last_layer.weight.dtype)
        features = self.nnet(encoded)
        projected = self.projection(features)
        return nn.functional.normalize(projected, p=2, dim=-1).to(dtype=torch.float32)


def geoclosp(
    weights: GeoCLOSP_Weights | str | None = GeoCLOSP_Weights.RN50_L10,
    *,
    device: str | torch.device = 'cpu',
    batch_size: int = 8192,
    **kwargs: Any,
) -> GeoCLOSPLocationEncoder:
    """Create a GeoCLOSP-RN location encoder.

    If you use this model in your research, please cite the following paper:

    * https://arxiv.org/abs/2507.10403

    Args:
        weights: Pretrained weights enum member, member name, or ``None``.
        device: Target device for inference.
        batch_size: Batch size used by :meth:`~LocationEncoder.encode`.
        **kwargs: Additional keyword arguments forwarded to
            :class:`GeoCLOSPLocationEncoder`.

    Returns:
        Configured :class:`GeoCLOSPLocationEncoder` instance.
    """
    return GeoCLOSPLocationEncoder(
        weights=weights, device=device, batch_size=batch_size, **kwargs
    )


__all__ = ('GeoCLOSPLocationEncoder', 'GeoCLOSP_Weights', 'geoclosp')
