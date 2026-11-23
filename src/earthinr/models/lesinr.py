# Copyright (c) Taylor Geospatial. All rights reserved.
# Licensed under the MIT License.

"""LE-SINR language-enhanced residual MLP location encoder."""

from typing import Any

import torch
from torch import Tensor, nn

from .._types import Path
from ._base import LocationEncoder, Weights, WeightsEnum
from ._utils import load_safetensors
from .sinr import SINRNet

_ZOO_REPO = 'taylor-geospatial/earthinr-zoo'


def _normalize_lesinr_state_dict(
    raw_state: dict[str, Tensor],
) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
    """Split a LE-SINR state dict into backbone (``feats.*``) and projection tensors.

    Accepts either stripped ``earthinr-zoo`` keys (``feats.*`` +
    ``class_emb.weight``) or upstream ``HyperNet`` keys prefixed with
    ``pos_enc.``.

    Args:
        raw_state: Raw state dictionary loaded from a checkpoint.

    Returns:
        Tuple ``(backbone_state, proj_state)`` ready for ``load_state_dict``.
    """
    backbone_state: dict[str, Tensor] = {}
    proj_state: dict[str, Tensor] = {}
    for key, value in raw_state.items():
        clean = key.removeprefix('pos_enc.')
        if clean.startswith('feats.'):
            backbone_state[clean] = value
        elif clean.startswith('class_emb.'):
            proj_state[clean.removeprefix('class_emb.')] = value
    return backbone_state, proj_state


class LESINR_Weights(WeightsEnum):  # noqa: N801
    """Pretrained LE-SINR location encoder weights."""

    DEFAULT = Weights(
        repo_id=_ZOO_REPO,
        filename='le-sinr/model.safetensors',
        meta={
            'num_inputs': 4,
            'num_filts': 256,
            'depth': 4,
            'embed_dim': 256,
            'sha256': (
                'c7f3c15358d173f973c4cca8dfe1cf3c12181b0dcb7c4c1987294610cc7c9f4e'
            ),
            'onnx_sha256': (
                '800f69f68e23bffbec288f3eb1ccb4e511024891886dea5311d9da9dd12990ac'
            ),
            'license': 'MIT',
        },
    )


class LESINRLocationEncoder(LocationEncoder):
    """LE-SINR 256-d location encoder accepting ``(lon, lat)`` WGS84 degrees.

    Combines the SINR sinusoidal coordinate preprocessing and residual MLP
    backbone (:class:`~earthinr.models.sinr.SINRNet`) with the bias-free linear
    projection head (``class_emb``) aligned with Wikipedia species text
    embeddings (Hamilton et al., NeurIPS 2024).
    """

    name: str = 'le-sinr'

    def __init__(
        self,
        num_inputs: int = 4,
        num_filts: int = 256,
        depth: int = 4,
        embed_dim: int = 256,
        *,
        weights: LESINR_Weights | str | None = None,
        checkpoint: Path | None = None,
        device: str | torch.device = 'cpu',
        batch_size: int = 8192,
    ) -> None:
        """Initialize a new LESINRLocationEncoder instance.

        Args:
            num_inputs: Preprocessed coordinate feature dimension.
            num_filts: Hidden width of the residual MLP backbone.
            depth: Number of residual blocks.
            embed_dim: Output embedding dimension of the linear projection head.
            weights: Pretrained :class:`LESINR_Weights` enum member or name.
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
            resolved = LESINR_Weights.verify(weights)
            num_inputs = int(resolved.meta['num_inputs'])
            num_filts = int(resolved.meta['num_filts'])
            depth = int(resolved.meta['depth'])
            embed_dim = int(resolved.meta['embed_dim'])
            raw_state = load_safetensors(resolved.download(), device='cpu')
        elif checkpoint is not None:
            raw_state = load_safetensors(checkpoint, device='cpu')

        self.embed_dim = embed_dim
        self.backbone = SINRNet(num_inputs=num_inputs, num_filts=num_filts, depth=depth)
        self.class_emb = nn.Linear(num_filts, embed_dim, bias=False)
        if raw_state is not None:
            backbone_state, proj_state = _normalize_lesinr_state_dict(raw_state)
            self.backbone.load_state_dict(backbone_state, strict=True)
            self.class_emb.load_state_dict(proj_state, strict=True)

        self.to(self.device)
        self.eval()

    def forward(
        self, coords: Tensor, year: Tensor | None = None, month: Tensor | None = None
    ) -> Tensor:
        """Encode ``(lon, lat)`` coordinates in degrees into LE-SINR embeddings.

        Args:
            coords: Coordinate tensor of shape ``(B, 2)`` in ``(lon, lat)`` degrees.
            year: Unused temporal parameter.
            month: Unused temporal parameter.

        Returns:
            Float32 embedding tensor of shape ``(B, embed_dim)``.
        """
        feats = self.backbone(coords.to(dtype=torch.float32))
        return self.class_emb(feats).to(dtype=torch.float32)


def lesinr(
    weights: LESINR_Weights | str | None = LESINR_Weights.DEFAULT,
    *,
    device: str | torch.device = 'cpu',
    batch_size: int = 8192,
    **kwargs: Any,
) -> LESINRLocationEncoder:
    """Create a LE-SINR location encoder.

    If you use this model in your research, please cite the following paper:

    * https://arxiv.org/abs/2410.10931

    Args:
        weights: Pretrained weights enum member, member name, or ``None``.
        device: Target device for inference.
        batch_size: Batch size used by :meth:`~LocationEncoder.encode`.
        **kwargs: Additional keyword arguments forwarded to
            :class:`LESINRLocationEncoder`.

    Returns:
        Configured :class:`LESINRLocationEncoder` instance.
    """
    return LESINRLocationEncoder(
        weights=weights, device=device, batch_size=batch_size, **kwargs
    )


__all__ = ('LESINRLocationEncoder', 'LESINR_Weights', 'lesinr')
