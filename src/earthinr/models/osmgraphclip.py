# Copyright (c) Taylor Geospatial. All rights reserved.
# Licensed under the MIT License.

"""OSMGraphCLIP spherical-harmonic SIREN location encoder."""

from typing import Any

import torch
from torch import Tensor

from .._types import Path
from ._base import LocationEncoder, Weights, WeightsEnum
from ._utils import load_safetensors
from .satclip import (
    SatCLIPLocationEncoder,
    SirenNet,
    StableAnalyticSphericalHarmonics,
    _normalize_satclip_state_dict,
)

_ZOO_REPO = 'taylor-geospatial/earthinr-zoo'


class OSMGraphCLIP_Weights(WeightsEnum):  # noqa: N801
    """Pretrained OSMGraphCLIP location encoder weights."""

    MS_L40 = Weights(
        repo_id=_ZOO_REPO,
        filename='osmgraphclip-ms-l40/model.safetensors',
        meta={
            'legendre_polys': 40,
            'capacity': 512,
            'embed_dim': 256,
            'num_hidden_layers': 2,
            'multiscale': True,
            'sha256': 'bd058d2dce2067c28c34e7315904a0c16b4243b2d63ebe9810f2a79640880f39',
            'onnx_sha256': (
                'eff1a2b1811b0c95cbcd41bdc312312f8bbc2768559cb842dd36113e8ad5b53a'
            ),
            'license': 'Apache-2.0',
        },
    )
    MS_L10 = Weights(
        repo_id=_ZOO_REPO,
        filename='osmgraphclip-ms-l10/model.safetensors',
        meta={
            'legendre_polys': 10,
            'capacity': 512,
            'embed_dim': 256,
            'num_hidden_layers': 2,
            'multiscale': True,
            'sha256': '70cd9052c97eaeb6d7bd2703291834e03e18f9585329bf866c9a007c75b110b6',
            'onnx_sha256': (
                '25363e510ef2f9668780df6591cfc37f69f57275013c8df24c06155b96147743'
            ),
            'license': 'Apache-2.0',
        },
    )
    A_L40 = Weights(
        repo_id=_ZOO_REPO,
        filename='osmgraphclip-a-l40/model.safetensors',
        meta={
            'legendre_polys': 40,
            'capacity': 512,
            'embed_dim': 256,
            'num_hidden_layers': 2,
            'multiscale': False,
            'sha256': '8d8675baaef1d9e132871a77af2b08cdcd9cdd1f620f1a71d72178f72c19c57c',
            'onnx_sha256': (
                '5e67082d0d4edf2b43347458dca7e7be7fc12daad347338e2614905242ba1f29'
            ),
            'license': 'Apache-2.0',
        },
    )
    A_L10 = Weights(
        repo_id=_ZOO_REPO,
        filename='osmgraphclip-a-l10/model.safetensors',
        meta={
            'legendre_polys': 10,
            'capacity': 512,
            'embed_dim': 256,
            'num_hidden_layers': 2,
            'multiscale': False,
            'sha256': 'c10d24b759eeec5da04c3aef80118acec0f46621545d2dfacd0616ee06a39560',
            'onnx_sha256': (
                '1abca3d8e5e0472600c7b6a77ebeb23d3059bdf78a5a00baa1549a6e3955e98d'
            ),
            'license': 'Apache-2.0',
        },
    )


class OSMGraphCLIPLocationEncoder(LocationEncoder):
    """OSMGraphCLIP spherical-harmonic SIREN location encoder.

    Trained contrastively against heterogeneous OpenStreetMap graph embeddings
    and multiscale concentric band features (Michail et al., SIGSPATIAL 2026).
    Accepts ``(lon, lat)`` coordinates in WGS84 degrees natively and ignores
    optional *year* and *month* inputs.
    """

    name: str = 'osmgraphclip'

    def __init__(
        self,
        legendre_polys: int = 40,
        capacity: int = 512,
        embed_dim: int = 256,
        num_hidden_layers: int = 2,
        *,
        weights: OSMGraphCLIP_Weights | str | None = None,
        checkpoint: Path | None = None,
        device: str | torch.device = 'cpu',
        batch_size: int = 8192,
    ) -> None:
        """Initialize a new OSMGraphCLIPLocationEncoder instance.

        Args:
            legendre_polys: Number of Legendre polynomial degrees ``L``.
            capacity: Hidden layer width of the SIREN network.
            embed_dim: Output embedding dimension.
            num_hidden_layers: Number of hidden SIREN layers.
            weights: Pretrained :class:`OSMGraphCLIP_Weights` enum member or name.
            checkpoint: Local ``.safetensors`` or PyTorch ``.ckpt`` checkpoint path.
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
            resolved = OSMGraphCLIP_Weights.verify(weights)
            legendre_polys = int(resolved.meta['legendre_polys'])
            capacity = int(resolved.meta['capacity'])
            embed_dim = int(resolved.meta['embed_dim'])
            num_hidden_layers = int(resolved.meta['num_hidden_layers'])
            raw_state = load_safetensors(resolved.download(), device='cpu')
        elif checkpoint is not None:
            raw_state, ckpt_meta = SatCLIPLocationEncoder._load_checkpoint(checkpoint)
            if ckpt_meta is not None:
                legendre_polys = int(ckpt_meta['legendre_polys'])
                capacity = int(ckpt_meta['capacity'])
                embed_dim = int(ckpt_meta['embed_dim'])
                num_hidden_layers = int(ckpt_meta['num_hidden_layers'])

        self.legendre_polys = legendre_polys
        self.embed_dim = embed_dim
        self.posenc = StableAnalyticSphericalHarmonics(
            legendre_polys=legendre_polys,
            compute_dtype=torch.float64,
            output_dtype=torch.float32,
        )
        self.nnet = SirenNet(
            dim_in=self.posenc.embedding_dim,
            dim_hidden=capacity,
            dim_out=embed_dim,
            num_layers=num_hidden_layers,
        )
        if raw_state is not None:
            normalized = _normalize_satclip_state_dict(raw_state)
            self.load_state_dict(normalized, strict=True)

        self.to(self.device)
        self.eval()

    def forward(
        self, coords: Tensor, year: Tensor | None = None, month: Tensor | None = None
    ) -> Tensor:
        """Encode ``(lon, lat)`` coordinates in degrees into OSMGraphCLIP embeddings.

        Args:
            coords: Coordinate tensor of shape ``(B, 2)`` in ``(lon, lat)`` degrees.
            year: Unused temporal parameter.
            month: Unused temporal parameter.

        Returns:
            Float32 embedding tensor of shape ``(B, embed_dim)``.
        """
        encoded = self.posenc(coords).to(dtype=self.nnet.last_layer.weight.dtype)
        return self.nnet(encoded).to(dtype=torch.float32)


def osmgraphclip(
    weights: OSMGraphCLIP_Weights | str | None = OSMGraphCLIP_Weights.MS_L40,
    *,
    device: str | torch.device = 'cpu',
    batch_size: int = 8192,
    **kwargs: Any,
) -> OSMGraphCLIPLocationEncoder:
    """Create an OSMGraphCLIP location encoder.

    If you use this model in your research, please cite the following paper:

    * https://doi.org/10.1145/3841645.3842950
    * https://arxiv.org/abs/2606.08046

    Args:
        weights: Pretrained weights enum member, member name, or ``None``.
        device: Target device for inference.
        batch_size: Batch size used by :meth:`~LocationEncoder.encode`.
        **kwargs: Additional keyword arguments forwarded to
            :class:`OSMGraphCLIPLocationEncoder`.

    Returns:
        Configured :class:`OSMGraphCLIPLocationEncoder` instance.
    """
    return OSMGraphCLIPLocationEncoder(
        weights=weights, device=device, batch_size=batch_size, **kwargs
    )


__all__ = ('OSMGraphCLIPLocationEncoder', 'OSMGraphCLIP_Weights', 'osmgraphclip')
