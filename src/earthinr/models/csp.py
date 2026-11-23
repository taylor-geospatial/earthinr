# Copyright (c) Taylor Geospatial. All rights reserved.
# Licensed under the MIT License.

"""Contrastive Spatial Pre-Training (CSP) grid-cell location encoder."""

import math
from typing import Any

import torch
from torch import Tensor, nn

from .._types import Path
from ._base import LocationEncoder, Weights, WeightsEnum
from ._utils import load_safetensors

_EARTHINR_ZOO_REPO = 'taylor-geospatial/earthinr-zoo'


def _make_activation(name: str) -> nn.Module:
    """Construct a CSP activation module by name.

    Args:
        name: Activation identifier (``'leakyrelu'``, ``'gelu'``, ``'relu'``,
            ``'sigmoid'``, or ``'tanh'``).

    Returns:
        Instantiated activation module.

    Raises:
        ValueError: If *name* is not a supported activation.
    """
    key = name.lower()
    if key == 'leakyrelu':
        return nn.LeakyReLU(negative_slope=0.2)
    if key == 'gelu':
        return nn.GELU()
    if key == 'relu':
        return nn.ReLU()
    if key == 'sigmoid':
        return nn.Sigmoid()
    if key == 'tanh':
        return nn.Tanh()
    raise ValueError(f'Unsupported CSP activation: {name!r}.')


class SingleFeedForwardNN(nn.Module):
    """Single linear + activation + optional dropout/skip/LayerNorm block."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        *,
        activation: str = 'leakyrelu',
        dropout_rate: float = 0.0,
        use_layernorm: bool = False,
        skip_connection: bool = False,
    ) -> None:
        """Initialize a new SingleFeedForwardNN instance.

        Args:
            input_dim: Input feature dimensionality.
            output_dim: Output feature dimensionality.
            activation: Nonlinearity name.
            dropout_rate: Dropout probability applied after activation.
            use_layernorm: Whether to apply LayerNorm at the end of the layer.
            skip_connection: Whether to add a residual connection when dimensions match.
        """
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.linear = nn.Linear(input_dim, output_dim)
        self.act = _make_activation(activation)
        self.dropout: nn.Dropout | None = (
            nn.Dropout(p=dropout_rate) if dropout_rate > 0.0 else None
        )
        self.skip_connection = skip_connection and (input_dim == output_dim)
        self.layernorm: nn.LayerNorm | None = (
            nn.LayerNorm(output_dim) if use_layernorm else None
        )

    def forward(self, x: Tensor) -> Tensor:
        """Apply the feed-forward layer.

        Args:
            x: Input tensor of shape ``(B, input_dim)``.

        Returns:
            Output tensor of shape ``(B, output_dim)``.
        """
        out = self.act(self.linear(x))
        if self.dropout is not None:
            out = self.dropout(out)
        if self.skip_connection:
            out = out + x
        if self.layernorm is not None:
            out = self.layernorm(out)
        return out


class MultiLayerFeedForwardNN(nn.Module):
    """Multi-layer feed-forward network matching the CSP checkpoint schema."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        *,
        num_hidden_layers: int = 1,
        hidden_dim: int = 512,
        activation: str = 'leakyrelu',
        dropout_rate: float = 0.0,
        use_layernorm: bool = True,
        skip_connection: bool = True,
    ) -> None:
        """Initialize a new MultiLayerFeedForwardNN instance.

        Args:
            input_dim: Input feature dimensionality.
            output_dim: Output embedding dimensionality.
            num_hidden_layers: Number of hidden layers preceding the output layer.
            hidden_dim: Width of each hidden layer.
            activation: Nonlinearity name applied to every layer including the output.
            dropout_rate: Dropout probability applied after activation.
            use_layernorm: Whether to apply LayerNorm on hidden layers.
            skip_connection: Whether to enable skip connections when dims match.
        """
        super().__init__()
        self.layers = nn.ModuleList()
        if num_hidden_layers <= 0:
            self.layers.append(
                SingleFeedForwardNN(
                    input_dim,
                    output_dim,
                    activation=activation,
                    dropout_rate=dropout_rate,
                    use_layernorm=False,
                    skip_connection=False,
                )
            )
        else:
            self.layers.append(
                SingleFeedForwardNN(
                    input_dim,
                    hidden_dim,
                    activation=activation,
                    dropout_rate=dropout_rate,
                    use_layernorm=use_layernorm,
                    skip_connection=skip_connection,
                )
            )
            for _ in range(num_hidden_layers - 1):
                self.layers.append(
                    SingleFeedForwardNN(
                        hidden_dim,
                        hidden_dim,
                        activation=activation,
                        dropout_rate=dropout_rate,
                        use_layernorm=use_layernorm,
                        skip_connection=skip_connection,
                    )
                )
            self.layers.append(
                SingleFeedForwardNN(
                    hidden_dim,
                    output_dim,
                    activation=activation,
                    dropout_rate=dropout_rate,
                    use_layernorm=False,
                    skip_connection=False,
                )
            )

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through all feed-forward layers.

        Args:
            x: Input tensor of shape ``(B, input_dim)``.

        Returns:
            Output tensor of shape ``(B, output_dim)``.
        """
        out = x
        for layer in self.layers:
            out = layer(out)
        return out


class GridCellSpatialRelationEncoder(nn.Module):
    """Pure-PyTorch multi-scale sinusoidal grid-cell encoder from CSP.

    Encodes ``(lon, lat)`` coordinates in degrees using geometric frequencies and a
    :class:`MultiLayerFeedForwardNN` projection head.
    """

    freqs: Tensor

    def __init__(
        self,
        *,
        frequency_num: int = 32,
        max_radius: float = 360.0,
        min_radius: float = 0.1,
        num_hidden_layers: int = 1,
        hidden_dim: int = 512,
        output_dim: int = 256,
        activation: str = 'leakyrelu',
        use_layernorm: bool = True,
        skip_connection: bool = True,
        normalize_coords: bool = True,
    ) -> None:
        """Initialize a new GridCellSpatialRelationEncoder instance.

        Args:
            frequency_num: Number of geometric frequency bands per axis.
            max_radius: Maximum spatial scale wavelength.
            min_radius: Minimum spatial scale wavelength.
            num_hidden_layers: Number of hidden layers in the FFN head.
            hidden_dim: Hidden width of the FFN head.
            output_dim: Output embedding dimensionality.
            activation: Nonlinearity name (``'leakyrelu'`` for iNat, ``'gelu'`` for
                fMoW).
            use_layernorm: Whether hidden FFN layers use LayerNorm.
            skip_connection: Whether matching-width hidden FFN layers use residuals.
            normalize_coords: Whether to divide ``(lon, lat)`` by ``(180, 90)``.

        Raises:
            ValueError: If *frequency_num* is not positive or radius bounds are invalid.
        """
        super().__init__()
        if frequency_num <= 0:
            raise ValueError(f'frequency_num must be positive, got {frequency_num}.')
        if min_radius <= 0.0 or max_radius < min_radius:
            raise ValueError(
                f'Invalid radius bounds: min_radius={min_radius}, '
                f'max_radius={max_radius}.'
            )

        self.frequency_num = frequency_num
        self.output_dim = output_dim
        self.normalize_coords = normalize_coords

        if frequency_num == 1:
            freqs = torch.tensor([1.0 / min_radius], dtype=torch.float32)
        else:
            log_inc = math.log(max_radius / min_radius) / (frequency_num - 1)
            scales = min_radius * torch.exp(
                torch.arange(frequency_num, dtype=torch.float64) * log_inc
            )
            freqs = (1.0 / scales).to(dtype=torch.float32)
        self.register_buffer('freqs', freqs, persistent=False)

        self.ffn = MultiLayerFeedForwardNN(
            input_dim=4 * frequency_num,
            output_dim=output_dim,
            num_hidden_layers=num_hidden_layers,
            hidden_dim=hidden_dim,
            activation=activation,
            use_layernorm=use_layernorm,
            skip_connection=skip_connection,
        )

    def forward(self, coords: Tensor) -> Tensor:
        """Encode ``(lon, lat)`` coordinates in degrees.

        Args:
            coords: Coordinate tensor of shape ``(B, 2)`` in ``(lon, lat)`` degrees.

        Returns:
            Float32 embedding tensor of shape ``(B, output_dim)``.
        """
        if self.normalize_coords:
            scaled = torch.stack((coords[:, 0] / 180.0, coords[:, 1] / 90.0), dim=-1)
        else:
            scaled = coords
        angles = scaled.unsqueeze(-1) * self.freqs
        pe = torch.stack((torch.sin(angles), torch.cos(angles)), dim=-1).flatten(1)
        return self.ffn(pe)


class CSP_Weights(WeightsEnum):  # noqa: N801
    """Pretrained weights for Contrastive Spatial Pre-Training (CSP).

    .. note::

       Upstream CSP checkpoints from ``gengchenmai/csp`` (Mai et al., ICML 2023) do
       not declare a software or weights license; verify permissions with the upstream
       authors before redistributing or using in restricted settings.
    """

    INAT = Weights(
        repo_id=_EARTHINR_ZOO_REPO,
        filename='csp-inat/model.safetensors',
        meta={
            'embed_dim': 256,
            'activation': 'leakyrelu',
            'frequency_num': 32,
            'max_radius': 360.0,
            'min_radius': 0.1,
            'num_hidden_layers': 1,
            'hidden_dim': 512,
            'output_dim': 256,
            'use_layernorm': True,
            'skip_connection': True,
            'license': 'unlicensed (contact upstream authors; Mai et al., ICML 2023)',
            'source_repo': 'gengchenmai/csp',
            'upstream_url': 'https://github.com/gengchenmai/csp',
            'sha256': (
                'ab102d01b3e8e3f81b880be487de23e413dfc3e9623700201e309382667935be'
            ),
        },
    )
    FMOW = Weights(
        repo_id=_EARTHINR_ZOO_REPO,
        filename='csp-fmow/model.safetensors',
        meta={
            'embed_dim': 256,
            'activation': 'gelu',
            'frequency_num': 32,
            'max_radius': 360.0,
            'min_radius': 0.1,
            'num_hidden_layers': 1,
            'hidden_dim': 512,
            'output_dim': 256,
            'use_layernorm': True,
            'skip_connection': True,
            'license': 'unlicensed (contact upstream authors; Mai et al., ICML 2023)',
            'source_repo': 'gengchenmai/csp',
            'upstream_url': 'https://github.com/gengchenmai/csp',
            'sha256': (
                'ad55993b8b40d3ac2b9279f16835442586263b946106641c73f77c2309bb9306'
            ),
        },
    )


class CSPLocationEncoder(LocationEncoder):
    """Contrastive Spatial Pre-Training (CSP) location encoder.

    Natively accepts ``(lon, lat)`` WGS84 degrees and produces 256-d embeddings.

    .. note::

       Upstream CSP checkpoints (Mai et al., ICML 2023) are currently unlicensed
       pending author confirmation.
    """

    name = 'csp'

    def __init__(
        self,
        weights: CSP_Weights | str | None = CSP_Weights.INAT,
        *,
        checkpoint_path: Path | None = None,
        activation: str | None = None,
        frequency_num: int = 32,
        max_radius: float = 360.0,
        min_radius: float = 0.1,
        num_hidden_layers: int = 1,
        hidden_dim: int = 512,
        output_dim: int = 256,
        use_layernorm: bool = True,
        skip_connection: bool = True,
        normalize_coords: bool = True,
        device: str | torch.device = 'cpu',
        batch_size: int = 8192,
    ) -> None:
        """Initialize a new CSPLocationEncoder instance.

        Args:
            weights: Pretrained :class:`CSP_Weights` member, name, or ``None``.
            checkpoint_path: Optional local ``.safetensors`` checkpoint path.
            activation: FFN activation name. Defaults to the weights preset activation
                (``'leakyrelu'`` for ``INAT``, ``'gelu'`` for ``FMOW``) or
                ``'leakyrelu'`` when *weights* is ``None``.
            frequency_num: Number of geometric frequency bands per axis.
            max_radius: Maximum spatial scale wavelength.
            min_radius: Minimum spatial scale wavelength.
            num_hidden_layers: Number of hidden layers in the FFN head.
            hidden_dim: Hidden width of the FFN head.
            output_dim: Output embedding dimensionality.
            use_layernorm: Whether hidden FFN layers apply LayerNorm.
            skip_connection: Whether matching-width hidden FFN layers use residuals.
            normalize_coords: Whether to divide ``(lon, lat)`` by ``(180, 90)``.
            device: Target computation device.
            batch_size: Batch size used by :meth:`encode`.
        """
        super().__init__(device=device, batch_size=batch_size)
        resolved_weights = CSP_Weights.verify(weights) if weights is not None else None
        act_name = (
            activation
            if activation is not None
            else (
                str(resolved_weights.meta['activation'])
                if resolved_weights is not None
                else 'leakyrelu'
            )
        )

        self.embed_dim = output_dim
        self.model = GridCellSpatialRelationEncoder(
            frequency_num=frequency_num,
            max_radius=max_radius,
            min_radius=min_radius,
            num_hidden_layers=num_hidden_layers,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            activation=act_name,
            use_layernorm=use_layernorm,
            skip_connection=skip_connection,
            normalize_coords=normalize_coords,
        )

        ckpt_path: Path | None = checkpoint_path
        if ckpt_path is None and resolved_weights is not None:
            ckpt_path = resolved_weights.download()
        if ckpt_path is not None:
            state = load_safetensors(ckpt_path, device='cpu')
            self.model.load_state_dict(state, strict=True)

        self.model.to(self.device)
        self.model.eval()

    def forward(
        self, coords: Tensor, year: Tensor | None = None, month: Tensor | None = None
    ) -> Tensor:
        """Encode geographic ``(lon, lat)`` coordinates into CSP embeddings.

        Args:
            coords: Coordinate tensor of shape ``(B, 2)`` in ``(lon, lat)`` degrees.
            year: Unused temporal input kept for :class:`LocationEncoder` parity.
            month: Unused temporal input kept for :class:`LocationEncoder` parity.

        Returns:
            Float32 embedding tensor of shape ``(B, output_dim)``.
        """
        return self.model(coords)


def csp(
    weights: CSP_Weights | str | None = CSP_Weights.INAT, **kwargs: Any
) -> CSPLocationEncoder:
    """Create a Contrastive Spatial Pre-Training (CSP) location encoder.

    If you use this model in your research, please cite the following paper:

    * https://arxiv.org/abs/2305.01118

    Args:
        weights: Pretrained :class:`CSP_Weights` member, name, or ``None``.
        **kwargs: Additional keyword arguments passed to :class:`CSPLocationEncoder`.

    Returns:
        Configured :class:`CSPLocationEncoder` instance.
    """
    return CSPLocationEncoder(weights=weights, **kwargs)


__all__ = (
    'CSPLocationEncoder',
    'CSP_Weights',
    'GridCellSpatialRelationEncoder',
    'MultiLayerFeedForwardNN',
    'SingleFeedForwardNN',
    'csp',
)
