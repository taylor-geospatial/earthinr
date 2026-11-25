# Copyright (c) Taylor Geospatial. All rights reserved.
# Licensed under the MIT License.

"""TerraNova hybrid WIRE-UNet and multi-resolution hash location encoder."""

import itertools
import math
from collections.abc import Sequence
from typing import Any, Literal

import torch
from torch import Tensor, nn
from torch.nn import functional as fn

from .._types import Path
from ._base import LocationEncoder, Weights, WeightsEnum
from ._utils import load_safetensors

_NEURALFTW_ZOO_REPO = 'taylor-geospatial/neuralftw-zoo'

_DEFAULT_WIRE_WIDTHS: tuple[int, ...] = (512, 1024, 2048, 4096, 2048, 1024, 512)
_DEFAULT_HASH_HIDDEN: tuple[int, ...] = (256, 256, 256, 256)


def _radius_normalize(x: Tensor, dim: int) -> Tensor:
    """Rescale rows of *x* onto the sphere of radius ``sqrt(dim)``."""
    return fn.normalize(x, dim=-1) * math.sqrt(dim)


def _make_activation(name: str) -> nn.Module:
    """Construct an activation module for TerraNova residual blocks."""
    if name == 'silu':
        return nn.SiLU()
    if name == 'gelu':
        return nn.GELU()
    raise ValueError(f'Unsupported TerraNova activation: {name!r}.')


class _ResidualBlock(nn.Module):
    """Pre-norm residual block used in TerraNova hash and time MLPs."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int | None = None,
        hidden_dim: int | None = None,
        *,
        dropout: float = 0.0,
        activation: str = 'silu',
    ) -> None:
        """Initialize a new _ResidualBlock instance.

        Args:
            in_dim: Input feature dimension.
            out_dim: Output feature dimension (defaults to *in_dim*).
            hidden_dim: Hidden bottleneck dimension (defaults to ``4 * in_dim``).
            dropout: Dropout probability.
            activation: Nonlinearity name (``'silu'`` or ``'gelu'``).
        """
        super().__init__()
        resolved_out = in_dim if out_dim is None else out_dim
        resolved_hidden = in_dim * 4 if hidden_dim is None else hidden_dim
        self.norm = nn.LayerNorm(in_dim)
        self.fc1 = nn.Linear(in_dim, resolved_hidden)
        self.activation = _make_activation(activation)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(resolved_hidden, resolved_out)
        self.residual_proj: nn.Module = (
            nn.Linear(in_dim, resolved_out, bias=False)
            if in_dim != resolved_out
            else nn.Identity()
        )

    def forward(self, x: Tensor) -> Tensor:
        """Apply the pre-norm residual block.

        Args:
            x: Input tensor of shape ``(B, in_dim)``.

        Returns:
            Output tensor of shape ``(B, out_dim)``.
        """
        residual = self.residual_proj(x)
        h = self.dropout(self.activation(self.fc1(self.norm(x))))
        h = self.dropout(self.fc2(h))
        return h + residual


class WrapNormalized(nn.Module):
    """Seam-free 5-d geographic coordinate lift for ``(lon, lat)`` in degrees."""

    def __init__(self) -> None:
        """Initialize a new WrapNormalized instance."""
        super().__init__()
        self.embedding_dim = 5

    def forward(self, coords: Tensor) -> Tensor:
        """Lift ``(lon, lat)`` degrees to 5-d unit-sphere plus phase-shifted coords.

        Args:
            coords: Coordinate tensor of shape ``(B, 2)`` in ``(lon, lat)`` degrees.

        Returns:
            Tensor of shape ``(B, 5)``.
        """
        lon = torch.deg2rad(coords[:, 0:1])
        lat = torch.deg2rad(coords[:, 1:2])
        cos_lat = torch.cos(lat)
        x1 = cos_lat * torch.cos(lon)
        y1 = cos_lat * torch.sin(lon)
        z1 = torch.sin(lat)
        lon_shifted = lon + (math.pi / 2.0)
        x2 = cos_lat * torch.cos(lon_shifted)
        y2 = cos_lat * torch.sin(lon_shifted)
        return torch.cat((x1, y1, z1, x2, y2), dim=-1)


class _GaborActivation(nn.Module):
    """Real-valued Gabor wavelet activation ``exp(-0.5*(sigma_0*x)^2)*cos(omega_0*x)``."""

    def __init__(self, omega_0: float = 10.0, sigma_0: float = 10.0) -> None:
        """Initialize a new _GaborActivation instance.

        Args:
            omega_0: Carrier frequency.
            sigma_0: Inverse Gaussian envelope width.
        """
        super().__init__()
        self.omega_0 = omega_0
        self.sigma_0 = sigma_0

    def forward(self, x: Tensor) -> Tensor:
        """Apply the real Gabor activation.

        Args:
            x: Pre-activation tensor.

        Returns:
            Activated tensor of the same shape.
        """
        return torch.exp(-0.5 * (self.sigma_0 * x) ** 2) * torch.cos(self.omega_0 * x)


class WireLayer(nn.Module):
    """Single WIRE layer with Gabor wavelet activation and optional linear skip."""

    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        *,
        omega_0: float = 10.0,
        sigma_0: float = 10.0,
        is_first: bool = False,
        is_residual: bool = False,
        activation: nn.Module | None = None,
    ) -> None:
        """Initialize a new WireLayer instance.

        Args:
            dim_in: Input feature dimensionality.
            dim_out: Output feature dimensionality.
            omega_0: Gabor carrier frequency.
            sigma_0: Gabor envelope inverse width.
            is_first: Whether this is the first layer of the network.
            is_residual: Whether to add a learned-weight residual projection.
            activation: Optional custom activation (defaults to Gabor wavelet).
        """
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.is_first = is_first
        self.is_residual = is_residual

        self.linear = nn.Linear(dim_in, dim_out)
        self.activation: nn.Module = (
            _GaborActivation(omega_0=omega_0, sigma_0=sigma_0)
            if activation is None
            else activation
        )

        if is_residual:
            self.residual: nn.Module = (
                nn.Linear(dim_in, dim_out)
                if (dim_in != dim_out or not is_first)
                else nn.Identity()
            )
            self.skip_weight: nn.Parameter | None = nn.Parameter(torch.ones(1))
        else:
            self.residual = nn.Identity()
            self.skip_weight = None

    def forward(self, x: Tensor) -> Tensor:
        """Apply the WIRE layer.

        Args:
            x: Input tensor of shape ``(B, dim_in)``.

        Returns:
            Output tensor of shape ``(B, dim_out)``.
        """
        out = self.activation(self.linear(x))
        if self.skip_weight is not None:
            out = out + self.residual(x) * self.skip_weight
        return out


class WireUNet(nn.Module):
    """U-Net of :class:`WireLayer` blocks with additive skip connections."""

    def __init__(
        self,
        dim_in: int,
        dim_hidden: Sequence[int],
        dim_out: int,
        *,
        omega_0: float = 10.0,
        sigma_0: float = 10.0,
        omega_0_initial: float = 30.0,
    ) -> None:
        """Initialize a new WireUNet instance.

        Args:
            dim_in: Input dimensionality (5 for :class:`WrapNormalized`).
            dim_hidden: Sequence of hidden widths defining the U-Net bottleneck.
            dim_out: Output dimensionality.
            omega_0: Hidden-layer Gabor carrier frequency.
            sigma_0: Gabor inverse envelope width.
            omega_0_initial: First-layer Gabor carrier frequency.

        Raises:
            ValueError: If *dim_hidden* is empty.
        """
        super().__init__()
        widths = list(dim_hidden)
        if not widths:
            raise ValueError('dim_hidden must be non-empty.')
        self.dim_out = dim_out

        bottleneck_idx = widths.index(max(widths))
        self.encoder_dims = widths[: bottleneck_idx + 1]
        self.decoder_dims = widths[bottleneck_idx + 1 :]
        self.skip_weights = nn.Parameter(torch.ones(len(self.decoder_dims)))

        encoder_layers: list[nn.Module] = []
        prev_dim = dim_in
        for i, hidden_dim in enumerate(self.encoder_dims):
            is_first = i == 0
            encoder_layers.append(
                WireLayer(
                    prev_dim,
                    hidden_dim,
                    omega_0=omega_0_initial if is_first else omega_0,
                    sigma_0=sigma_0,
                    is_first=is_first,
                    is_residual=True,
                )
            )
            prev_dim = hidden_dim
        self.encoder = nn.ModuleList(encoder_layers)

        decoder_layers: list[nn.Module] = []
        for i, hidden_dim in enumerate(self.decoder_dims):
            layer_dim_in = self.decoder_dims[i - 1] if i > 0 else self.encoder_dims[-1]
            decoder_layers.append(
                WireLayer(
                    layer_dim_in,
                    hidden_dim,
                    omega_0=omega_0,
                    sigma_0=sigma_0,
                    is_first=False,
                    is_residual=True,
                )
            )
        self.decoder = nn.ModuleList(decoder_layers)

        last_in = self.decoder_dims[-1] if self.decoder_dims else self.encoder_dims[-1]
        self.last_layer = WireLayer(
            last_in,
            dim_out,
            omega_0=omega_0,
            sigma_0=sigma_0,
            is_first=False,
            is_residual=False,
            activation=nn.Identity(),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Apply the WIRE U-Net.

        Args:
            x: Positional features of shape ``(B, dim_in)``.

        Returns:
            Output tensor of shape ``(B, dim_out)``.
        """
        encoder_features: list[Tensor] = []
        for layer in self.encoder:
            x = layer(x)
            encoder_features.append(x)
        encoder_features = encoder_features[:-1]

        for ind, layer in enumerate(self.decoder):
            if 0 < ind < len(encoder_features):
                skip = encoder_features[-ind]
                if x.shape[-1] == skip.shape[-1]:
                    x = x + self.skip_weights[ind] * skip
            x = layer(x)

        return self.last_layer(x)


class MultiResolutionHashEncoding(nn.Module):
    """Pure-PyTorch Instant-NGP multi-resolution geographic hash grid."""

    _offsets: Tensor
    _primes: Tensor

    def __init__(
        self,
        n_levels: int = 16,
        n_features_per_level: int = 4,
        log2_hashmap_size: int = 16,
        base_resolution: int = 16,
        finest_resolution: int = 1024,
    ) -> None:
        """Initialize a new MultiResolutionHashEncoding instance.

        Args:
            n_levels: Number of spatial resolution levels.
            n_features_per_level: Feature channels per level.
            log2_hashmap_size: Base-2 logarithm of entries per hash table.
            base_resolution: Coarsest grid resolution.
            finest_resolution: Finest grid resolution.
        """
        super().__init__()
        self.n_levels = n_levels
        self.n_features_per_level = n_features_per_level
        self.hashmap_size = 2**log2_hashmap_size
        self.base_resolution = base_resolution
        self.finest_resolution = finest_resolution
        self.embedding_dim = n_levels * n_features_per_level

        if n_levels > 1:
            self.growth_factor = math.exp(
                (math.log(finest_resolution) - math.log(base_resolution))
                / (n_levels - 1)
            )
        else:
            self.growth_factor = 1.0

        self.hash_tables = nn.ModuleList(
            [
                nn.Embedding(self.hashmap_size, n_features_per_level)
                for _ in range(n_levels)
            ]
        )
        self.register_buffer(
            '_offsets', torch.tensor([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=torch.long)
        )
        self.register_buffer('_primes', torch.tensor([1, 2654435761], dtype=torch.long))

    def forward(self, coords: Tensor) -> Tensor:
        """Encode ``(lon, lat)`` degrees with periodic-longitude hash interpolation.

        Args:
            coords: Coordinate tensor of shape ``(B, 2)`` in ``(lon, lat)`` degrees.

        Returns:
            Concatenated multi-resolution features of shape
            ``(B, n_levels * n_features_per_level)``.
        """
        lon_norm = torch.remainder(coords[:, 0] + 180.0, 360.0) / 360.0
        lat_norm = torch.clamp((coords[:, 1] + 90.0) / 180.0, 0.0, 1.0 - 1e-6)
        coords_norm = torch.stack((lon_norm, lat_norm), dim=1)

        outputs: list[Tensor] = []
        for level in range(self.n_levels):
            resolution = int(self.base_resolution * (self.growth_factor**level))
            scaled = coords_norm * resolution
            floor_coords = torch.floor(scaled).long()
            frac = scaled - floor_coords.float()

            corners = floor_coords.unsqueeze(1) + self._offsets.unsqueeze(0)
            corners_flat = corners.reshape(-1, 2)
            c_lon = torch.remainder(corners_flat[:, 0], resolution)
            c_lat = corners_flat[:, 1]
            indices = (
                (c_lon * self._primes[0]) ^ (c_lat * self._primes[1])
            ) % self.hashmap_size

            features = self.hash_tables[level](indices).reshape(
                -1, 4, self.n_features_per_level
            )

            fx = frac[:, 0]
            fy = frac[:, 1]
            weights = torch.stack(
                ((1.0 - fx) * (1.0 - fy), fx * (1.0 - fy), (1.0 - fx) * fy, fx * fy),
                dim=1,
            ).unsqueeze(-1)
            outputs.append((features * weights).sum(dim=1))

        return torch.cat(outputs, dim=-1)


class ResidualMLPDecoder(nn.Module):
    """Pre-norm GELU residual MLP decoding hash-grid features."""

    def __init__(
        self,
        dim_in: int,
        hidden_dims: Sequence[int],
        dim_out: int,
        *,
        dropout: float = 0.0,
    ) -> None:
        """Initialize a new ResidualMLPDecoder instance.

        Args:
            dim_in: Input hash feature dimensionality.
            hidden_dims: Sequence of output dimensions for each residual block.
            dim_out: Final output dimensionality.
            dropout: Dropout probability inside residual blocks.
        """
        super().__init__()
        dims = [dim_in, *hidden_dims]
        blocks = [
            _ResidualBlock(d_in, d_out, dropout=dropout, activation='gelu')
            for d_in, d_out in itertools.pairwise(dims)
        ]
        self.blocks = nn.Sequential(*blocks)
        self.out_proj = nn.Linear(dims[-1], dim_out)

    def forward(self, x: Tensor) -> Tensor:
        """Decode hash features.

        Args:
            x: Tensor of shape ``(B, dim_in)``.

        Returns:
            Decoded tensor of shape ``(B, dim_out)``.
        """
        return self.out_proj(self.blocks(x))


class _HybridWireHash(nn.Module):
    """Gated combination of :class:`WireUNet` and hash-grid decoder."""

    def __init__(
        self,
        *,
        dim_in: int,
        wire_widths: Sequence[int],
        dim_out: int,
        omega_0: float,
        sigma_0: float,
        omega_0_initial: float,
        hash_n_levels: int,
        hash_n_features: int,
        hash_log2_size: int,
        hash_base_res: int,
        hash_finest_res: int,
        hash_hidden: Sequence[int],
        gate_init_bias: float,
    ) -> None:
        """Initialize a new _HybridWireHash instance."""
        super().__init__()
        self.wire = WireUNet(
            dim_in=dim_in,
            dim_hidden=wire_widths,
            dim_out=dim_out,
            omega_0=omega_0,
            sigma_0=sigma_0,
            omega_0_initial=omega_0_initial,
        )
        self.secondary_encoding = MultiResolutionHashEncoding(
            n_levels=hash_n_levels,
            n_features_per_level=hash_n_features,
            log2_hashmap_size=hash_log2_size,
            base_resolution=hash_base_res,
            finest_resolution=hash_finest_res,
        )
        self.secondary_decoder = ResidualMLPDecoder(
            dim_in=self.secondary_encoding.embedding_dim,
            hidden_dims=hash_hidden,
            dim_out=dim_out,
        )
        self.gate_alpha = nn.Parameter(torch.full((dim_out,), gate_init_bias))
        self.combine_proj = nn.Linear(dim_out, dim_out)

    def forward(self, pe_input: Tensor, coords: Tensor) -> Tensor:
        """Combine WIRE and hash branches via a clamped per-channel gate."""
        wire_out = self.wire(pe_input)
        sec_out = self.secondary_decoder(self.secondary_encoding(coords))
        gate = torch.sigmoid(self.gate_alpha).clamp(0.15, 0.85)
        return self.combine_proj(gate * wire_out + (1.0 - gate) * sec_out)


class TerraNovaLocationNet(nn.Module):
    """TerraNova spatial encoder matching ``location_encoder.*`` checkpoint keys."""

    def __init__(
        self,
        *,
        location_dim: int = 256,
        wire_widths: Sequence[int] = _DEFAULT_WIRE_WIDTHS,
        wire_omega_0: float = 10.0,
        wire_sigma_0: float = 10.0,
        wire_omega_0_initial: float = 30.0,
        hash_n_levels: int = 16,
        hash_n_features: int = 4,
        hash_log2_size: int = 16,
        hash_base_res: int = 16,
        hash_finest_res: int = 1024,
        hash_hidden: Sequence[int] = _DEFAULT_HASH_HIDDEN,
        gate_init_bias: float = 1.386,
    ) -> None:
        """Initialize a new TerraNovaLocationNet instance.

        Args:
            location_dim: Output spatial embedding dimensionality.
            wire_widths: Hidden widths for :class:`WireUNet`.
            wire_omega_0: Hidden-layer Gabor carrier frequency.
            wire_sigma_0: Gabor inverse envelope width.
            wire_omega_0_initial: First-layer Gabor carrier frequency.
            hash_n_levels: Number of hash grid resolution levels.
            hash_n_features: Hash features per resolution level.
            hash_log2_size: Base-2 log hash table size.
            hash_base_res: Coarsest hash grid resolution.
            hash_finest_res: Finest hash grid resolution.
            hash_hidden: Hidden widths for :class:`ResidualMLPDecoder`.
            gate_init_bias: Initial logit for the WIRE/hash blend gate.
        """
        super().__init__()
        self.output_dim = location_dim
        self.positional_encoder = WrapNormalized()
        self.neural_network = _HybridWireHash(
            dim_in=self.positional_encoder.embedding_dim,
            wire_widths=wire_widths,
            dim_out=location_dim,
            omega_0=wire_omega_0,
            sigma_0=wire_sigma_0,
            omega_0_initial=wire_omega_0_initial,
            hash_n_levels=hash_n_levels,
            hash_n_features=hash_n_features,
            hash_log2_size=hash_log2_size,
            hash_base_res=hash_base_res,
            hash_finest_res=hash_finest_res,
            hash_hidden=hash_hidden,
            gate_init_bias=gate_init_bias,
        )

    def forward(self, coords: Tensor) -> Tensor:
        """Encode ``(lon, lat)`` degrees to radius-normalized spatial embeddings.

        Args:
            coords: Coordinate tensor of shape ``(B, 2)`` in ``(lon, lat)`` degrees.

        Returns:
            Radius-normalized spatial embeddings of shape ``(B, location_dim)``.
        """
        out = self.neural_network(self.positional_encoder(coords), coords)
        return _radius_normalize(out, self.output_dim)


class TerraNovaTimeNet(nn.Module):
    """TerraNova calendar-year encoder matching ``time_encoder.*`` checkpoint keys."""

    fourier_freqs: Tensor

    def __init__(
        self,
        *,
        time_dim: int = 32,
        time_width: int = 256,
        time_depth: int = 4,
        time_fourier: int = 32,
        year_min: float = 1900.0,
        year_max: float = 2035.0,
        dropout: float = 0.1,
    ) -> None:
        """Initialize a new TerraNovaTimeNet instance.

        Args:
            time_dim: Output temporal embedding dimensionality.
            time_width: Hidden width of residual blocks.
            time_depth: Number of residual blocks.
            time_fourier: Number of log-spaced Fourier frequency components.
            year_min: Minimum calendar year for normalization.
            year_max: Maximum calendar year for normalization.
            dropout: Dropout probability inside residual blocks.
        """
        super().__init__()
        self.output_dim = time_dim
        self.year_min = year_min
        self.year_max = year_max
        max_freq = max(time_fourier / 2.0, 1.0)
        freqs = torch.exp(
            torch.linspace(math.log(1.0), math.log(max_freq), time_fourier)
        )
        self.register_buffer('fourier_freqs', freqs)
        self.time_scale = nn.Parameter(torch.ones(1))
        layers: list[nn.Module] = [nn.Linear(time_fourier * 2, time_width)]
        for _ in range(time_depth):
            layers.append(
                _ResidualBlock(time_width, dropout=dropout, activation='silu')
            )
        layers.append(nn.Linear(time_width, time_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, years: Tensor) -> Tensor:
        """Encode calendar years into radius-normalized temporal embeddings.

        Args:
            years: 1D tensor of calendar years with shape ``(B,)``.

        Returns:
            Radius-normalized temporal embeddings of shape ``(B, time_dim)``.
        """
        t = (years.float() - self.year_min) / (self.year_max - self.year_min)
        scaled = (
            t.unsqueeze(-1) * self.time_scale * self.fourier_freqs * (2.0 * math.pi)
        )
        feats = torch.cat((torch.sin(scaled), torch.cos(scaled)), dim=-1)
        return _radius_normalize(self.network(feats), self.output_dim)


class _STBlock(nn.Module):
    """Post-norm cross-attention + SiLU FFN block for spatiotemporal fusion."""

    def __init__(self, width: int, num_heads: int, dropout: float) -> None:
        """Initialize a new _STBlock instance."""
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            width, num_heads, dropout=dropout, batch_first=True
        )
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(width)
        self.ffn = nn.Sequential(
            nn.Linear(width, width * 4),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(width * 4, width),
        )
        self.norm2 = nn.LayerNorm(width)

    def forward(self, q: Tensor, kv: Tensor) -> Tensor:
        """Apply cross-attention from spatial query *q* to temporal *kv*."""
        out, _ = self.cross_attn(q, kv, kv, need_weights=False)
        q = self.norm1(q + self.dropout(out))
        return self.norm2(q + self.dropout(self.ffn(q)))


class _TransformerCrossModalFusion(nn.Module):
    """Cross-modal transformer stack matching ``st_fusion.strategy.*`` keys."""

    def __init__(
        self,
        spatial_dim: int,
        time_dim: int,
        output_dim: int,
        *,
        depth: int,
        width: int,
        num_heads: int,
        dropout: float,
    ) -> None:
        """Initialize a new _TransformerCrossModalFusion instance."""
        super().__init__()
        self.spatial_proj = nn.Linear(spatial_dim, width)
        self.time_proj = nn.Linear(time_dim, width)
        self.blocks = nn.ModuleList(
            [_STBlock(width, num_heads, dropout) for _ in range(depth)]
        )
        self.output_proj = nn.Linear(width, output_dim)

    def forward(self, spatial: Tensor, time: Tensor) -> Tensor:
        """Compute cross-modal residual delta from spatial and temporal embeddings."""
        q = self.spatial_proj(spatial).unsqueeze(1)
        kv = self.time_proj(time).unsqueeze(1)
        for block in self.blocks:
            q = block(q, kv)
        return self.output_proj(q.squeeze(1))


class TerraNovaSTFusion(nn.Module):
    """TerraNova spatiotemporal fusion module matching ``st_fusion.*`` keys."""

    def __init__(
        self,
        *,
        location_dim: int = 256,
        time_dim: int = 32,
        st_width: int = 1024,
        st_heads: int = 16,
        st_depth: int = 8,
        alpha_residual_init: float = -1.1,
        dropout: float = 0.1,
    ) -> None:
        """Initialize a new TerraNovaSTFusion instance.

        Args:
            location_dim: Spatial and fused output embedding dimensionality.
            time_dim: Temporal embedding dimensionality.
            st_width: Cross-attention transformer hidden width.
            st_heads: Number of attention heads.
            st_depth: Number of cross-attention blocks.
            alpha_residual_init: Initial scalar residual gate logit.
            dropout: Dropout probability in attention and FFN blocks.
        """
        super().__init__()
        self.output_dim = location_dim
        self.spatial_proj = nn.Linear(location_dim, location_dim)
        self.alpha_residual = nn.Parameter(
            torch.full((), alpha_residual_init, dtype=torch.float32)
        )
        self.residual_scaler_act = nn.Sigmoid()
        self.strategy = _TransformerCrossModalFusion(
            spatial_dim=location_dim,
            time_dim=time_dim,
            output_dim=location_dim,
            depth=st_depth,
            width=st_width,
            num_heads=st_heads,
            dropout=dropout,
        )

    def forward(self, spatial: Tensor, time: Tensor) -> Tensor:
        """Fuse spatial and temporal embeddings onto the radius-``sqrt(D)`` sphere.

        Args:
            spatial: Spatial embeddings of shape ``(B, location_dim)``.
            time: Temporal embeddings of shape ``(B, time_dim)``.

        Returns:
            Radius-normalized spatiotemporal embeddings of shape ``(B, location_dim)``.
        """
        s = self.spatial_proj(spatial)
        delta = self.strategy(s, time)
        gate = 0.05 + 0.9 * self.residual_scaler_act(self.alpha_residual)
        return _radius_normalize(s + gate * delta, self.output_dim)


class TerraNova_Weights(WeightsEnum):  # noqa: N801
    """Pretrained weights for TerraNova (spatial and spatiotemporal)."""

    SPATIAL = Weights(
        repo_id=_NEURALFTW_ZOO_REPO,
        filename='terranova-spatial/model.safetensors',
        meta={
            'variant': 'spatial',
            'embed_dim': 256,
            'license': 'CC-BY-4.0',
            'source_repo': 'crp94/terranova',
            'upstream_url': 'https://github.com/crp94/terranova-model',
            'sha256': (
                '4e936fd2dc16ee426fffc954409ee8464d89af5bf271a44b31ac96bf0077222f'
            ),
        },
    )
    SPATIOTEMPORAL = Weights(
        repo_id=_NEURALFTW_ZOO_REPO,
        filename='terranova-st/model.safetensors',
        meta={
            'variant': 'spatiotemporal',
            'embed_dim': 256,
            'default_year': 2020,
            'license': 'CC-BY-4.0',
            'source_repo': 'crp94/terranova',
            'upstream_url': 'https://github.com/crp94/terranova-model',
            'sha256': (
                '250bb994210538976eeb495b441199cd0169483a0fb32006b1a8490ecf502fb8'
            ),
        },
    )


def _filter_terranova_state(
    state: dict[str, Tensor], space: Literal['spatial', 'spatiotemporal']
) -> dict[str, Tensor]:
    """Keep only tensors belonging to the active TerraNova spatial/ST submodules."""
    prefixes = (
        ('location_encoder.',)
        if space == 'spatial'
        else ('location_encoder.', 'time_encoder.', 'st_fusion.')
    )
    return {k: v for k, v in state.items() if k.startswith(prefixes)}


class TerraNovaLocationEncoder(LocationEncoder):
    """TerraNova geographic and spatiotemporal location encoder.

    Natively accepts ``(lon, lat)`` WGS84 degrees (wrapping longitude into
    ``[-180, 180)``) and outputs 256-d embeddings normalized to radius
    ``sqrt(256) = 16``. In ``space='spatiotemporal'`` mode, an optional ``year``
    tensor is fused with the spatial representation (defaulting to ``2020`` when
    omitted).
    """

    name = 'terranova'

    def __init__(
        self,
        weights: TerraNova_Weights | str | None = TerraNova_Weights.SPATIAL,
        *,
        space: Literal['spatial', 'spatiotemporal'] | None = None,
        checkpoint_path: Path | None = None,
        default_year: float = 2020.0,
        location_dim: int = 256,
        wire_widths: Sequence[int] = _DEFAULT_WIRE_WIDTHS,
        wire_omega_0: float = 10.0,
        wire_sigma_0: float = 10.0,
        wire_omega_0_initial: float = 30.0,
        hash_n_levels: int = 16,
        hash_n_features: int = 4,
        hash_log2_size: int = 16,
        hash_base_res: int = 16,
        hash_finest_res: int = 1024,
        hash_hidden: Sequence[int] = _DEFAULT_HASH_HIDDEN,
        gate_init_bias: float = 1.386,
        time_dim: int = 32,
        time_width: int = 256,
        time_depth: int = 4,
        time_fourier: int = 32,
        year_min: float = 1900.0,
        year_max: float = 2035.0,
        st_width: int = 1024,
        st_heads: int = 16,
        st_depth: int = 8,
        alpha_residual_init: float = -1.1,
        dropout: float = 0.1,
        device: str | torch.device = 'cpu',
        batch_size: int = 8192,
    ) -> None:
        """Initialize a new TerraNovaLocationEncoder instance.

        Args:
            weights: Pretrained :class:`TerraNova_Weights` member, name, or ``None``.
            space: Embedding mode, either ``'spatial'`` or ``'spatiotemporal'``.
                Inferred from *weights* when omitted, or defaults to ``'spatial'``.
            checkpoint_path: Optional local ``.safetensors`` checkpoint path.
            default_year: Calendar year used in spatiotemporal mode when *year* is
                ``None``.
            location_dim: Output embedding dimensionality.
            wire_widths: Hidden widths for :class:`WireUNet`.
            wire_omega_0: Hidden-layer Gabor carrier frequency.
            wire_sigma_0: Gabor inverse envelope width.
            wire_omega_0_initial: First-layer Gabor carrier frequency.
            hash_n_levels: Number of hash grid resolution levels.
            hash_n_features: Hash features per resolution level.
            hash_log2_size: Base-2 log hash table size.
            hash_base_res: Coarsest hash grid resolution.
            hash_finest_res: Finest hash grid resolution.
            hash_hidden: Hidden widths for :class:`ResidualMLPDecoder`.
            gate_init_bias: Initial logit for the WIRE/hash blend gate.
            time_dim: Temporal embedding dimensionality.
            time_width: Temporal MLP hidden width.
            time_depth: Number of temporal residual blocks.
            time_fourier: Number of temporal Fourier frequencies.
            year_min: Minimum year for temporal normalization.
            year_max: Maximum year for temporal normalization.
            st_width: Spatiotemporal fusion attention width.
            st_heads: Spatiotemporal fusion attention heads.
            st_depth: Spatiotemporal fusion block depth.
            alpha_residual_init: Initial spatiotemporal fusion gate logit.
            dropout: Dropout probability in time and fusion blocks.
            device: Target computation device.
            batch_size: Batch size used by :meth:`encode`.

        Raises:
            ValueError: If *space* is invalid or conflicts with *weights*.
        """
        super().__init__(device=device, batch_size=batch_size)
        resolved_weights = (
            TerraNova_Weights.verify(weights) if weights is not None else None
        )

        if space is None:
            resolved_space: Literal['spatial', 'spatiotemporal'] = (
                'spatiotemporal'
                if (
                    resolved_weights is not None
                    and resolved_weights.meta['variant'] == 'spatiotemporal'
                )
                else 'spatial'
            )
        elif space in ('spatial', 'spatiotemporal'):
            resolved_space = space
            if (
                resolved_weights is not None
                and resolved_weights.meta['variant'] != resolved_space
            ):
                raise ValueError(
                    f'Weights {resolved_weights.name} are for '
                    f'{resolved_weights.meta["variant"]!r}, not {resolved_space!r}.'
                )
        else:
            raise ValueError(
                f"space must be 'spatial' or 'spatiotemporal', got {space!r}."
            )

        self.space = resolved_space
        self.default_year = float(default_year)
        self.embed_dim = location_dim

        self.location_encoder = TerraNovaLocationNet(
            location_dim=location_dim,
            wire_widths=wire_widths,
            wire_omega_0=wire_omega_0,
            wire_sigma_0=wire_sigma_0,
            wire_omega_0_initial=wire_omega_0_initial,
            hash_n_levels=hash_n_levels,
            hash_n_features=hash_n_features,
            hash_log2_size=hash_log2_size,
            hash_base_res=hash_base_res,
            hash_finest_res=hash_finest_res,
            hash_hidden=hash_hidden,
            gate_init_bias=gate_init_bias,
        )
        if self.space == 'spatiotemporal':
            self.time_encoder: TerraNovaTimeNet | None = TerraNovaTimeNet(
                time_dim=time_dim,
                time_width=time_width,
                time_depth=time_depth,
                time_fourier=time_fourier,
                year_min=year_min,
                year_max=year_max,
                dropout=dropout,
            )
            self.st_fusion: TerraNovaSTFusion | None = TerraNovaSTFusion(
                location_dim=location_dim,
                time_dim=time_dim,
                st_width=st_width,
                st_heads=st_heads,
                st_depth=st_depth,
                alpha_residual_init=alpha_residual_init,
                dropout=dropout,
            )
        else:
            self.time_encoder = None
            self.st_fusion = None

        ckpt_path: Path | None = checkpoint_path
        if ckpt_path is None and resolved_weights is not None:
            ckpt_path = resolved_weights.download()
        if ckpt_path is not None:
            state = _filter_terranova_state(
                load_safetensors(ckpt_path, device='cpu'), self.space
            )
            self.load_state_dict(state, strict=True)

        self.to(self.device)
        self.eval()

    def forward(
        self, coords: Tensor, year: Tensor | None = None, month: Tensor | None = None
    ) -> Tensor:
        """Encode ``(lon, lat)`` coordinates and optional calendar year.

        Args:
            coords: Coordinate tensor of shape ``(B, 2)`` in ``(lon, lat)`` degrees.
            year: Optional tensor of calendar years with shape ``(B,)``. Used in
                ``'spatiotemporal'`` mode (defaults to ``self.default_year``).
            month: Unused temporal input kept for :class:`LocationEncoder` parity.

        Returns:
            Radius-normalized float32 embedding tensor of shape ``(B, location_dim)``.
        """
        lon = torch.remainder(coords[:, 0] + 180.0, 360.0) - 180.0
        wrapped = torch.stack((lon, coords[:, 1]), dim=-1)
        spatial = self.location_encoder(wrapped)
        if self.time_encoder is None or self.st_fusion is None:
            return spatial
        year_tensor = (
            coords.new_full((coords.shape[0],), self.default_year)
            if year is None
            else year
        )
        time_emb = self.time_encoder(year_tensor)
        return self.st_fusion(spatial, time_emb)


def terranova(
    weights: TerraNova_Weights | str | None = TerraNova_Weights.SPATIAL, **kwargs: Any
) -> TerraNovaLocationEncoder:
    """Create a TerraNova spatial or spatiotemporal location encoder.

    If you use this model in your research, please cite the following paper:

    * https://arxiv.org/abs/2607.29527

    Args:
        weights: Pretrained :class:`TerraNova_Weights` member, name, or ``None``.
        **kwargs: Additional keyword arguments passed to
            :class:`TerraNovaLocationEncoder`.

    Returns:
        Configured :class:`TerraNovaLocationEncoder` instance.
    """
    return TerraNovaLocationEncoder(weights=weights, **kwargs)


__all__ = (
    'MultiResolutionHashEncoding',
    'ResidualMLPDecoder',
    'TerraNovaLocationEncoder',
    'TerraNovaLocationNet',
    'TerraNovaSTFusion',
    'TerraNovaTimeNet',
    'TerraNova_Weights',
    'WireLayer',
    'WireUNet',
    'WrapNormalized',
    'terranova',
)
