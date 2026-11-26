# Copyright (c) Taylor Geospatial. All rights reserved.
# Licensed under the MIT License.

"""MIND Equal-Earth residual SIREN location encoder."""

import math
import os
import pathlib
from typing import Literal

import torch
from torch import Tensor, nn

from .._types import Path
from .._utils import resolve_device
from ._base import LocationEncoder, Weights, WeightsEnum
from ._utils import equal_earth_projection, load_safetensors

_VALID_ACTS = ('siren', 'finer', 'hsiren')


class SIRENLayer(nn.Module):
    """Periodic-activation linear layer supporting SIREN, FINER, and H-SIREN.

    Uses ``sin(w0 * z)`` for ``'siren'``, ``sin(w0 * (|z| + 1) * z)`` for
    ``'finer'``, and ``sin(w0 * sinh(z))`` on the first layer for ``'hsiren'``.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        w0: float = 1.0,
        is_first: bool = False,
        act: str = 'siren',
        finer_k: float = 10.0,
    ) -> None:
        """Initialize a new SIRENLayer instance.

        Args:
            in_features: Input feature dimension.
            out_features: Output feature dimension.
            w0: Frequency scaling factor.
            is_first: Whether this is the input layer of the network.
            act: Activation variant, one of ``'siren'``, ``'finer'``, or
                ``'hsiren'``.
            finer_k: Uniform bias initialization half-range for first-layer FINER.

        Raises:
            ValueError: If *act* is not one of ``'siren'``, ``'finer'``, or
                ``'hsiren'``.
        """
        super().__init__()
        if act not in _VALID_ACTS:
            raise ValueError(f'Unsupported SIREN activation: {act!r}.')
        self.linear = nn.Linear(in_features, out_features)
        self.w0 = w0
        # H-SIREN applies sinh only on the first layer; deeper layers stay plain SIREN
        self.act = act if (is_first or act != 'hsiren') else 'siren'
        with torch.no_grad():
            bound = 1.0 / in_features if is_first else math.sqrt(6.0 / in_features) / w0
            self.linear.weight.uniform_(-bound, bound)
            if self.act == 'finer' and is_first:
                self.linear.bias.uniform_(-finer_k, finer_k)
            else:
                nn.init.zeros_(self.linear.bias)

    def forward(self, x: Tensor) -> Tensor:
        """Apply linear transform and periodic activation.

        Args:
            x: Input tensor of shape ``(..., in_features)``.

        Returns:
            Activated tensor of shape ``(..., out_features)``.
        """
        z = self.linear(x)
        if self.act == 'finer':
            z = (z.abs() + 1.0) * z
        elif self.act == 'hsiren':
            z = torch.sinh(z)
        return torch.sin(self.w0 * z)


class ReSIRENLocationEncoder(nn.Module):
    """MIND Equal-Earth residual SIREN network.

    Accepts ``(lat, lon)`` coordinates in degrees (matching upstream MIND weights)
    with shape ``(B, 2)`` or ``(B, K, 2)`` (mean-pooled across ``K`` points).
    """

    year_freqs: Tensor

    def __init__(
        self,
        embed_dim: int = 3072,
        out_dim: int | None = 64,
        depth: int = 12,
        use_year: bool = False,
        w0_first: float = 30.0,
        w0: float = 1.0,
        variant: str = 'siren',
        finer_k: float = 10.0,
        year_frequencies: int = 8,
        year_ref: float = 2021.0,
        year_scale: float = 4.0,
    ) -> None:
        """Initialize a new ReSIRENLocationEncoder instance.

        Args:
            embed_dim: Width of the residual SIREN trunk.
            out_dim: Output dimension of the linear projection head, or ``None``
                for an identity head.
            depth: Number of residual SIREN blocks after the first layer.
            use_year: If True, concatenate Fourier-encoded year features to inputs.
            w0_first: Frequency scaling for the first layer.
            w0: Frequency scaling for residual layers.
            variant: Activation variant (``'siren'``, ``'finer'``, or ``'hsiren'``).
            finer_k: First-layer bias bound for ``'finer'`` variant.
            year_frequencies: Number of dyadic Fourier frequencies for year inputs.
            year_ref: Reference year subtracted prior to scaling.
            year_scale: Scale divisor applied to ``year - year_ref``.
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.out_dim = out_dim
        self.use_year = use_year
        self.year_ref = year_ref
        self.year_scale = year_scale
        self.register_buffer(
            'year_freqs', 2.0 ** torch.arange(year_frequencies, dtype=torch.float32)
        )
        in_dim = 2 + (2 * year_frequencies if use_year else 0)
        self.first = SIRENLayer(
            in_dim, embed_dim, w0=w0_first, is_first=True, act=variant, finer_k=finer_k
        )
        self.blocks = nn.ModuleList(
            [
                SIRENLayer(embed_dim, embed_dim, w0=w0, act=variant, finer_k=finer_k)
                for _ in range(depth)
            ]
        )
        self.head: nn.Module = (
            nn.Identity()
            if (out_dim is None or out_dim == embed_dim)
            else nn.Linear(embed_dim, out_dim)
        )

    def _year_feats(self, year: Tensor) -> Tensor:
        """Compute Fourier features for observation years."""
        freqs = self.year_freqs.to(device=year.device, dtype=torch.float32)
        yn = ((year.to(dtype=torch.float32) - self.year_ref) / self.year_scale).reshape(
            -1, 1
        )
        ang = yn * freqs.unsqueeze(0)
        return torch.cat((torch.sin(ang), torch.cos(ang)), dim=-1)

    def forward(
        self, latlon: Tensor, year: Tensor | None = None, return_features: bool = False
    ) -> Tensor:
        """Encode ``(lat, lon)`` coordinates into pooled trunk or head outputs.

        Args:
            latlon: Coordinate tensor of shape ``(B, 2)`` or ``(B, K, 2)`` in
                ``(lat, lon)`` degrees.
            year: Optional tensor of shape ``(B,)`` containing observation years.
            return_features: If True, return the pooled trunk embedding instead of
                passing through the projection head.

        Returns:
            Embedding tensor of shape ``(B, embed_dim)`` or ``(B, out_dim)``.

        Raises:
            ValueError: If *use_year* is True and *year* is ``None``.
        """
        if latlon.ndim == 2:
            latlon = latlon.unsqueeze(1)
        b, k, _ = latlon.shape
        loc = equal_earth_projection(latlon.reshape(b * k, 2), coord_order='latlon')
        if self.use_year:
            if year is None:
                raise ValueError('year is required when use_year=True.')
            yf = (
                self._year_feats(year)
                .to(dtype=loc.dtype)
                .unsqueeze(1)
                .expand(b, k, -1)
                .reshape(b * k, -1)
            )
            x = torch.cat((loc, yf), dim=-1)
        else:
            x = loc

        h = self.first(x.to(dtype=self.first.linear.weight.dtype))
        for blk in self.blocks:
            h = h + blk(h)
        pooled = h.reshape(b, k, self.embed_dim).mean(dim=1)
        return pooled if return_features else self.head(pooled)


class MIND_Weights(WeightsEnum):  # noqa: N801
    """Pretrained MIND location encoder weights."""

    MIND = Weights(
        repo_id='taylor-geospatial/MIND',
        filename='mind.safetensors',
        meta={
            'embed_dim': 3072,
            'out_dim': 64,
            'depth': 12,
            'use_year': False,
            'license': 'MIT',
        },
    )


def load_mind(
    ckpt_path: Path,
    *,
    device: str | torch.device = 'cpu',
    variant: str = 'siren',
    w0_first: float = 30.0,
) -> ReSIRENLocationEncoder:
    """Load a MIND ReSIREN checkpoint from ``.safetensors`` or PyTorch format.

    Architecture width, depth, head dimension, and year conditioning are inferred
    directly from the checkpoint tensors.

    Args:
        ckpt_path: Path to a ``.safetensors`` or ``.pt`` checkpoint file.
        device: Target device for the loaded model.
        variant: Activation variant (``'siren'``, ``'finer'``, or ``'hsiren'``).
        w0_first: First-layer SIREN frequency scaling factor.

    Returns:
        Loaded :class:`ReSIRENLocationEncoder` in evaluation mode.

    Raises:
        ValueError: If *ckpt_path* is not a valid MIND checkpoint or *variant* is
            unsupported.
    """
    if pathlib.Path(ckpt_path).suffix == '.safetensors':
        state = load_safetensors(ckpt_path, device='cpu')
    else:
        payload = torch.load(
            os.fspath(ckpt_path), map_location='cpu', weights_only=True
        )
        if isinstance(payload, dict) and 'state_dict' in payload:
            args = payload.get('meta', {}).get('args', {})
            encoder = args.get('encoder', variant)
            variant = 'siren' if encoder == 'resiren' else encoder
            w0_first = float(args.get('w0_first', w0_first))
            state = payload['state_dict']
        else:
            state = payload

    if 'first.linear.weight' not in state:
        raise ValueError('Expected a MIND ReSIREN checkpoint.')
    if variant not in _VALID_ACTS:
        raise ValueError(f'Unsupported ReSIREN activation: {variant!r}.')

    width, inputs = state['first.linear.weight'].shape
    depth = sum(k.startswith('blocks.') and k.endswith('.linear.weight') for k in state)
    out_dim = state['head.weight'].shape[0] if 'head.weight' in state else width
    frequencies = int(state['year_freqs'].shape[0])
    if inputs not in (2, 2 + 2 * frequencies):
        raise ValueError(f'Unexpected coordinate input dimension: {inputs}.')

    model = ReSIRENLocationEncoder(
        embed_dim=width,
        out_dim=out_dim,
        depth=depth,
        use_year=inputs != 2,
        year_frequencies=frequencies,
        variant=variant,
        w0_first=w0_first,
    )
    if 'head.weight' in state and isinstance(model.head, nn.Identity):
        model.head = nn.Linear(width, out_dim)
    model.load_state_dict(state, strict=True)
    return model.to(resolve_device(device)).eval()


class MINDLocationEncoder(LocationEncoder):
    """MIND location encoder with universal ``(lon, lat)`` WGS84 coordinate order.

    Swaps incoming ``(lon, lat)`` coordinates to ``(lat, lon)`` internally to match
    upstream MIND weights and supports Matryoshka prefix truncation via *dim*.
    """

    name: str = 'mind'

    def __init__(
        self,
        *,
        weights: MIND_Weights | str | None = None,
        checkpoint: Path | None = None,
        model: ReSIRENLocationEncoder | None = None,
        feature: Literal['pooled', 'head'] = 'pooled',
        dim: int | None = None,
        default_year: int = 2021,
        device: str | torch.device = 'cpu',
        batch_size: int = 8192,
    ) -> None:
        """Initialize a new MINDLocationEncoder instance.

        Args:
            weights: Pretrained :class:`MIND_Weights` enum member or name.
            checkpoint: Local path to a MIND ``.safetensors`` or ``.pt`` checkpoint.
            model: Optional pre-instantiated :class:`ReSIRENLocationEncoder`.
            feature: Output branch to return, either ``'pooled'`` (Matryoshka
                trunk) or ``'head'`` (projection head).
            dim: Optional prefix dimension truncation (defaults to full width of the
                selected *feature*).
            default_year: Default observation year used when a year-conditioned
                checkpoint is called with ``year=None``.
            device: Target device for inference.
            batch_size: Batch size used by :meth:`encode`.

        Raises:
            ValueError: If multiple model sources are provided, *feature* is
                invalid, or *dim* is out of range.
        """
        super().__init__(device=device, batch_size=batch_size)
        sources = sum(src is not None for src in (weights, checkpoint, model))
        if sources > 1:
            raise ValueError('Specify at most one of weights, checkpoint, or model.')
        if feature not in ('pooled', 'head'):
            raise ValueError(f"feature must be 'pooled' or 'head', got {feature!r}.")

        if weights is not None:
            resolved = MIND_Weights.verify(weights)
            backbone = load_mind(resolved.download(), device=self.device)
        elif checkpoint is not None:
            backbone = load_mind(checkpoint, device=self.device)
        elif model is not None:
            backbone = model.to(self.device).eval()
        else:
            backbone = ReSIRENLocationEncoder().to(self.device).eval()

        max_width = (
            backbone.embed_dim
            if feature == 'pooled'
            else (backbone.out_dim or backbone.embed_dim)
        )
        resolved_dim = max_width if dim is None else dim
        if (
            isinstance(resolved_dim, bool)
            or not isinstance(resolved_dim, int)
            or resolved_dim < 1
            or resolved_dim > max_width
        ):
            raise ValueError(
                f'dim must be an integer in [1, {max_width}], got {dim!r}.'
            )

        self.model = backbone
        self.feature: Literal['pooled', 'head'] = feature
        self.embed_dim = resolved_dim
        self.default_year = default_year

    def forward(
        self, coords: Tensor, year: Tensor | None = None, month: Tensor | None = None
    ) -> Tensor:
        """Encode ``(lon, lat)`` coordinates in degrees into MIND embeddings.

        Args:
            coords: Coordinate tensor of shape ``(B, 2)`` or ``(B, K, 2)`` in
                ``(lon, lat)`` WGS84 degrees.
            year: Optional tensor of shape ``(B,)`` containing observation years.
            month: Unused temporal parameter.

        Returns:
            Float32 embedding tensor of shape ``(B, embed_dim)``.
        """
        latlon = torch.stack((coords[..., 1], coords[..., 0]), dim=-1)
        resolved_year = year
        if self.model.use_year and resolved_year is None:
            resolved_year = torch.full(
                (coords.shape[0],),
                float(self.default_year),
                dtype=torch.float32,
                device=coords.device,
            )
        out = self.model(
            latlon, year=resolved_year, return_features=self.feature == 'pooled'
        )
        return out[:, : self.embed_dim].to(dtype=torch.float32)


def mind(
    weights: MIND_Weights | str | None = MIND_Weights.MIND,
    *,
    feature: Literal['pooled', 'head'] = 'pooled',
    dim: int | None = None,
    default_year: int = 2021,
    device: str | torch.device = 'cpu',
    batch_size: int = 8192,
) -> MINDLocationEncoder:
    """Create a MIND location encoder.

    Args:
        weights: Pretrained weights enum member, member name, or ``None``.
        feature: Output branch, either ``'pooled'`` or ``'head'``.
        dim: Optional Matryoshka prefix width truncation.
        default_year: Default year for year-conditioned checkpoints when omitted.
        device: Target device for inference.
        batch_size: Batch size used by :meth:`~LocationEncoder.encode`.

    Returns:
        Configured :class:`MINDLocationEncoder` instance.
    """
    return MINDLocationEncoder(
        weights=weights,
        feature=feature,
        dim=dim,
        default_year=default_year,
        device=device,
        batch_size=batch_size,
    )


__all__ = (
    'MINDLocationEncoder',
    'MIND_Weights',
    'ReSIRENLocationEncoder',
    'SIRENLayer',
    'load_mind',
    'mind',
)
