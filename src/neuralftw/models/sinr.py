# Copyright (c) Taylor Geospatial. All rights reserved.
# Licensed under the MIT License.

"""SINR residual MLP location encoder."""

import math
from typing import Any

import torch
from torch import Tensor, nn

from .._types import Path
from ._base import LocationEncoder, Weights, WeightsEnum
from ._utils import load_safetensors

_ZOO_REPO = 'taylor-geospatial/neuralftw-zoo'


class ResLayer(nn.Module):
    """Residual fully connected block matching SINR checkpoint key layout."""

    def __init__(self, linear_size: int) -> None:
        """Initialize a new ResLayer instance.

        Args:
            linear_size: Hidden dimension for both linear layers.
        """
        super().__init__()
        self.l_size = linear_size
        self.nonlin1 = nn.ReLU()
        self.nonlin2 = nn.ReLU()
        self.dropout1 = nn.Dropout()
        self.w1 = nn.Linear(linear_size, linear_size)
        self.w2 = nn.Linear(linear_size, linear_size)

    def forward(self, x: Tensor) -> Tensor:
        """Apply two-layer residual MLP block.

        Args:
            x: Input tensor of shape ``(B, linear_size)``.

        Returns:
            Output tensor of shape ``(B, linear_size)``.
        """
        y = self.w1(x)
        y = self.nonlin1(y)
        y = self.dropout1(y)
        y = self.w2(y)
        y = self.nonlin2(y)
        return x + y


class SINRNet(nn.Module):
    """SINR sinusoidal coordinate preprocessing + residual MLP network.

    Accepts raw ``(lon, lat)`` coordinates in degrees without mutating the input
    tensor in-place.
    """

    def __init__(
        self, num_inputs: int = 4, num_filts: int = 256, depth: int = 4
    ) -> None:
        """Initialize a new SINRNet instance.

        Args:
            num_inputs: Preprocessed coordinate feature dimension (4 for
                ``[sin(pi*lon/180), sin(pi*lat/90), cos(pi*lon/180),
                cos(pi*lat/90)]``).
            num_filts: Hidden and output embedding width.
            depth: Number of :class:`ResLayer` blocks.
        """
        super().__init__()
        self.num_inputs = num_inputs
        self.num_filts = num_filts
        self.depth = depth
        layers: list[nn.Module] = [nn.Linear(num_inputs, num_filts), nn.ReLU()]
        layers.extend(ResLayer(num_filts) for _ in range(depth))
        self.feats = nn.Sequential(*layers)

    @staticmethod
    def preprocess_locs(coords: Tensor) -> Tensor:
        """Map ``(lon, lat)`` degrees to 4-d wrapped sin/cos features out-of-place.

        Args:
            coords: Coordinate tensor of shape ``(B, 2)`` in ``(lon, lat)`` degrees.

        Returns:
            Feature tensor of shape ``(B, 4)``.
        """
        scaled = torch.stack((coords[:, 0] / 180.0, coords[:, 1] / 90.0), dim=-1)
        pi_scaled = math.pi * scaled
        return torch.cat((torch.sin(pi_scaled), torch.cos(pi_scaled)), dim=-1)

    def forward(self, coords: Tensor) -> Tensor:
        """Encode raw ``(lon, lat)`` coordinates in degrees into SINR features.

        Args:
            coords: Coordinate tensor of shape ``(B, 2)`` in ``(lon, lat)`` degrees.

        Returns:
            Embedding tensor of shape ``(B, num_filts)``.
        """
        return self.feats(self.preprocess_locs(coords))


class SINR_Weights(WeightsEnum):  # noqa: N801
    """Pretrained SINR location encoder weights."""

    INAT_2023 = Weights(
        repo_id=_ZOO_REPO,
        filename='sinr/model.safetensors',
        meta={
            'num_inputs': 4,
            'num_filts': 256,
            'depth': 4,
            'embed_dim': 256,
            'sha256': '43095b2f1f1ba695a96132af6c0df14d9b5aad699d6651fbddc7c90aa1e120d5',
            'license': 'MIT',
        },
    )


class SINRLocationEncoder(LocationEncoder):
    """SINR 256-d location encoder accepting ``(lon, lat)`` WGS84 degrees."""

    name: str = 'sinr'

    def __init__(
        self,
        num_inputs: int = 4,
        num_filts: int = 256,
        depth: int = 4,
        *,
        weights: SINR_Weights | str | None = None,
        checkpoint: Path | None = None,
        device: str | torch.device = 'cpu',
        batch_size: int = 8192,
    ) -> None:
        """Initialize a new SINRLocationEncoder instance.

        Args:
            num_inputs: Preprocessed coordinate feature dimension.
            num_filts: Embedding dimension of the residual MLP.
            depth: Number of residual blocks.
            weights: Pretrained :class:`SINR_Weights` enum member or name.
            checkpoint: Local ``.safetensors`` checkpoint path.
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
            resolved = SINR_Weights.verify(weights)
            num_inputs = int(resolved.meta['num_inputs'])
            num_filts = int(resolved.meta['num_filts'])
            depth = int(resolved.meta['depth'])
            raw_state = load_safetensors(resolved.download(), device='cpu')
        elif checkpoint is not None:
            raw_state = load_safetensors(checkpoint, device='cpu')

        self.embed_dim = num_filts
        self.model = SINRNet(num_inputs=num_inputs, num_filts=num_filts, depth=depth)
        if raw_state is not None:
            # Drop optional species classification head (class_emb.*) if present
            filtered = {
                k: v for k, v in raw_state.items() if not k.startswith('class_emb.')
            }
            self.model.load_state_dict(filtered, strict=True)

        self.to(self.device)
        self.eval()

    def forward(
        self, coords: Tensor, year: Tensor | None = None, month: Tensor | None = None
    ) -> Tensor:
        """Encode ``(lon, lat)`` coordinates in degrees into SINR embeddings.

        Args:
            coords: Coordinate tensor of shape ``(B, 2)`` in ``(lon, lat)`` degrees.
            year: Unused temporal parameter.
            month: Unused temporal parameter.

        Returns:
            Float32 embedding tensor of shape ``(B, embed_dim)``.
        """
        return self.model(coords.to(dtype=torch.float32)).to(dtype=torch.float32)


def sinr(
    weights: SINR_Weights | str | None = SINR_Weights.INAT_2023,
    *,
    device: str | torch.device = 'cpu',
    batch_size: int = 8192,
    **kwargs: Any,
) -> SINRLocationEncoder:
    """Create a SINR location encoder.

    If you use this model in your research, please cite the following paper:

    * https://arxiv.org/abs/2306.02564

    Args:
        weights: Pretrained weights enum member, member name, or ``None``.
        device: Target device for inference.
        batch_size: Batch size used by :meth:`~LocationEncoder.encode`.
        **kwargs: Additional keyword arguments forwarded to
            :class:`SINRLocationEncoder`.

    Returns:
        Configured :class:`SINRLocationEncoder` instance.
    """
    return SINRLocationEncoder(
        weights=weights, device=device, batch_size=batch_size, **kwargs
    )


__all__ = ('ResLayer', 'SINRLocationEncoder', 'SINRNet', 'SINR_Weights', 'sinr')
