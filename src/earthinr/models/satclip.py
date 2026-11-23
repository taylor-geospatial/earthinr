# Copyright (c) Taylor Geospatial. All rights reserved.
# Licensed under the MIT License.

"""SatCLIP spherical-harmonic SIREN location encoder."""

import math
import os
import pathlib
from typing import Any

import torch
from torch import Tensor, nn

from .._types import Path
from ._base import LocationEncoder, Weights, WeightsEnum
from ._utils import load_safetensors

# Clamp latitudes slightly away from exact poles for numerical stability of sqrt(1 - x^2)
POLE_LATITUDE_EPS_DEGREES = 0.05

_ZOO_REPO = 'taylor-geospatial/earthinr-zoo'


def _normalize_satclip_state_dict(state_dict: dict[str, Tensor]) -> dict[str, Tensor]:
    """Normalize SatCLIP checkpoint keys to the ``nnet.*`` schema.

    Supports ``nnet.*`` (earthinr-zoo), ``location.nnet.*`` (MVRL HF),
    ``model.location.nnet.*`` (Microsoft Lightning ``.ckpt``), and bare
    ``layers.*`` / ``last_layer.*`` keys.

    Args:
        state_dict: Raw state dictionary loaded from disk.

    Returns:
        Normalized state dictionary keyed by ``nnet.*``.

    Raises:
        KeyError: If no SatCLIP SIREN weights are present in *state_dict*.
    """
    nnet_keys = {
        f'nnet.{k.split("nnet.", 1)[1]}': v
        for k, v in state_dict.items()
        if 'nnet.' in k
    }
    if nnet_keys:
        return nnet_keys
    bare_keys = {
        f'nnet.{k}': v
        for k, v in state_dict.items()
        if k.startswith(('layers.', 'last_layer.'))
    }
    if bare_keys:
        return bare_keys
    raise KeyError("Could not find SatCLIP location encoder weights under 'nnet.'.")


class StableAnalyticSphericalHarmonics(nn.Module):
    """Numerically stable real spherical harmonics matching SatCLIP's analytic basis.

    Uses a normalized three-term associated Legendre recurrence in float64 instead
    of expanded SymPy monomials, avoiding catastrophic cancellation at high degrees
    (``L >= 33``) while preserving SatCLIP's ``m = 0`` extra ``pi`` factor and
    absence of Condon-Shortley phase.
    """

    orders: Tensor
    diag_coeffs: Tensor
    subdiag_coeffs: Tensor
    alpha: Tensor
    beta: Tensor

    def __init__(
        self,
        legendre_polys: int = 40,
        compute_dtype: torch.dtype = torch.float64,
        output_dtype: torch.dtype | None = None,
    ) -> None:
        """Initialize a new StableAnalyticSphericalHarmonics instance.

        Args:
            legendre_polys: Number of Legendre polynomial degrees ``L`` (outputs
                ``L ** 2`` harmonics for degrees ``0 .. L - 1``).
            compute_dtype: Floating-point dtype used for the Legendre recurrence.
            output_dtype: Optional dtype to cast the output tensor to.

        Raises:
            TypeError: If *legendre_polys* is not an integer.
            ValueError: If *legendre_polys* is less than 1.
        """
        super().__init__()
        if isinstance(legendre_polys, bool) or not isinstance(legendre_polys, int):
            raise TypeError(f'legendre_polys must be an int, got {legendre_polys!r}.')
        if legendre_polys < 1:
            raise ValueError(f'legendre_polys must be >= 1, got {legendre_polys}.')
        self.legendre_polys = legendre_polys
        self.embedding_dim = legendre_polys * legendre_polys
        self.compute_dtype = compute_dtype
        self.output_dtype = output_dtype
        self._init_constants()

    def _init_constants(self) -> None:
        """Precompute recurrence coefficients as non-persistent buffers."""
        n_polys = self.legendre_polys
        orders = torch.arange(n_polys, dtype=torch.int64)
        diag_coeffs = torch.ones(n_polys, dtype=torch.float64)
        if n_polys > 1:
            diag_terms = torch.sqrt(1.0 + 1.0 / (2.0 * orders[1:].to(torch.float64)))
            diag_coeffs[1:] = torch.cumprod(diag_terms, dim=0)

        subdiag_coeffs = torch.sqrt(2.0 * orders[:-1].to(torch.float64) + 3.0)
        alpha = torch.zeros((n_polys, n_polys), dtype=torch.float64)
        beta = torch.zeros((n_polys, n_polys), dtype=torch.float64)

        for degree in range(2, n_polys):
            m = torch.arange(degree - 1, dtype=torch.float64)
            num_scale = (2.0 * degree + 1.0) / (2.0 * degree - 3.0)
            denom = degree * degree - m * m
            alpha[degree, : degree - 1] = torch.sqrt(
                num_scale * ((4.0 * (degree - 1) * (degree - 1) - 1.0) / denom)
            )
            beta[degree, : degree - 1] = torch.sqrt(
                num_scale * ((((degree - 1) * (degree - 1)) - m * m) / denom)
            )

        self.register_buffer('orders', orders, persistent=False)
        self.register_buffer('diag_coeffs', diag_coeffs, persistent=False)
        self.register_buffer('subdiag_coeffs', subdiag_coeffs, persistent=False)
        self.register_buffer('alpha', alpha, persistent=False)
        self.register_buffer('beta', beta, persistent=False)

    def forward(self, coords: Tensor) -> Tensor:
        """Evaluate real spherical harmonics on ``(lon, lat)`` coordinates in degrees.

        Args:
            coords: Coordinate tensor of shape ``(B, 2)`` in ``(lon, lat)`` degrees.

        Returns:
            Spherical harmonic feature tensor of shape ``(B, legendre_polys ** 2)``.
        """
        n_polys = self.legendre_polys
        lonlat = coords.to(dtype=self.compute_dtype)
        lon = lonlat[:, 0]
        lat = lonlat[:, 1].clamp(
            min=-90.0 + POLE_LATITUDE_EPS_DEGREES, max=90.0 - POLE_LATITUDE_EPS_DEGREES
        )

        phi = torch.deg2rad(lon + 180.0)
        theta = torch.deg2rad(lat + 90.0)
        x = torch.cos(theta)
        u = torch.sqrt(torch.clamp(1.0 - x * x, min=0.0))

        dtype = lonlat.dtype
        device = lonlat.device
        batch_size = lonlat.shape[0]
        order_values = self.orders.to(device=device, dtype=dtype)

        p00 = lonlat.new_full((batch_size,), math.sqrt(1.0 / (4.0 * math.pi)))
        diag_terms: list[Tensor] = [p00]
        if n_polys > 1:
            diag_coeffs = self.diag_coeffs.to(device=device, dtype=dtype)
            diag_values = (
                p00.unsqueeze(1)
                * diag_coeffs[1:].unsqueeze(0)
                * u.unsqueeze(1).pow(order_values[1:].unsqueeze(0))
            )
            diag_terms.extend(diag_values.unbind(dim=1))

        subdiag_coeffs = self.subdiag_coeffs.to(device=device, dtype=dtype)
        alpha = self.alpha.to(device=device, dtype=dtype)
        beta = self.beta.to(device=device, dtype=dtype)

        if n_polys == 1:
            rows: list[Tensor] = [diag_terms[0].unsqueeze(1)]
        else:
            rows = [
                torch.cat(
                    (
                        diag_terms[0].unsqueeze(1),
                        lonlat.new_zeros((batch_size, n_polys - 1)),
                    ),
                    dim=1,
                )
            ]
            row1_terms: list[Tensor] = [
                (subdiag_coeffs[0] * x * diag_terms[0]).unsqueeze(1),
                diag_terms[1].unsqueeze(1),
            ]
            if n_polys > 2:
                row1_terms.append(lonlat.new_zeros((batch_size, n_polys - 2)))
            rows.append(torch.cat(row1_terms, dim=1))

        for degree in range(2, n_polys):
            recurrence = (
                alpha[degree, : degree - 1].unsqueeze(0)
                * x.unsqueeze(1)
                * rows[degree - 1][:, : degree - 1]
                - beta[degree, : degree - 1].unsqueeze(0)
                * rows[degree - 2][:, : degree - 1]
            )
            row_terms: list[Tensor] = [
                recurrence,
                (subdiag_coeffs[degree - 1] * x * diag_terms[degree - 1]).unsqueeze(1),
                diag_terms[degree].unsqueeze(1),
            ]
            if degree < n_polys - 1:
                row_terms.append(lonlat.new_zeros((batch_size, n_polys - degree - 1)))
            rows.append(torch.cat(row_terms, dim=1))

        plm = torch.stack(rows, dim=1)
        phases = phi.unsqueeze(1) * order_values.unsqueeze(0)
        sin_terms = torch.sin(phases)
        cos_terms = torch.cos(phases)

        degree_blocks: list[Tensor] = []
        for degree in range(n_polys):
            center = (math.pi * plm[:, degree, 0]).unsqueeze(1)
            if degree == 0:
                degree_blocks.append(center)
                continue
            base = math.sqrt(2.0) * plm[:, degree, 1 : degree + 1]
            negative = torch.flip(base * sin_terms[:, 1 : degree + 1], dims=(1,))
            positive = base * cos_terms[:, 1 : degree + 1]
            degree_blocks.append(torch.cat((negative, center, positive), dim=1))

        output = torch.cat(degree_blocks, dim=1)
        if self.output_dtype is not None:
            output = output.to(dtype=self.output_dtype)
        return output


class _Sine(nn.Module):
    """Scaled sine activation."""

    def __init__(self, w0: float = 1.0) -> None:
        """Initialize a new _Sine instance.

        Args:
            w0: Frequency scaling factor.
        """
        super().__init__()
        self.w0 = w0

    def forward(self, x: Tensor) -> Tensor:
        """Apply scaled sine activation.

        Args:
            x: Input tensor.

        Returns:
            Activated tensor ``sin(w0 * x)``.
        """
        return torch.sin(self.w0 * x)


class _SirenLayer(nn.Module):
    """Single SIREN layer matching SatCLIP checkpoint key layout."""

    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        w0: float = 1.0,
        c: float = 6.0,
        is_first: bool = False,
        use_bias: bool = True,
        activation: nn.Module | None = None,
        dropout: bool = False,
    ) -> None:
        """Initialize a new _SirenLayer instance.

        Args:
            dim_in: Input feature dimension.
            dim_out: Output feature dimension.
            w0: Sine frequency scaling factor.
            c: SIREN initialization variance constant.
            is_first: Whether this is the first layer of the network.
            use_bias: Whether to include a learnable bias vector.
            activation: Optional override activation module.
            dropout: Whether to apply dropout during training.
        """
        super().__init__()
        self.dim_in = dim_in
        self.is_first = is_first
        self.dropout = dropout

        weight = torch.zeros(dim_out, dim_in)
        bias = torch.zeros(dim_out) if use_bias else None
        scale = (1.0 / dim_in) if is_first else (math.sqrt(c / dim_in) / w0)
        weight.uniform_(-scale, scale)
        if bias is not None:
            bias.uniform_(-scale, scale)

        self.weight = nn.Parameter(weight)
        self.bias: nn.Parameter | None = (
            nn.Parameter(bias) if bias is not None else None
        )
        self.activation: nn.Module = _Sine(w0) if activation is None else activation

    def forward(self, x: Tensor) -> Tensor:
        """Apply linear projection, optional dropout, and activation.

        Args:
            x: Input tensor of shape ``(B, dim_in)``.

        Returns:
            Output tensor of shape ``(B, dim_out)``.
        """
        out = nn.functional.linear(x, self.weight, self.bias)
        if self.dropout:
            out = nn.functional.dropout(out, training=self.training)
        return self.activation(out)


class SirenNet(nn.Module):
    """Multi-layer SIREN network matching SatCLIP's ``nnet`` architecture."""

    def __init__(
        self,
        dim_in: int,
        dim_hidden: int,
        dim_out: int,
        num_layers: int,
        w0: float = 1.0,
        w0_initial: float = 30.0,
        use_bias: bool = True,
        final_activation: nn.Module | None = None,
    ) -> None:
        """Initialize a new SirenNet instance.

        Args:
            dim_in: Input feature dimension.
            dim_hidden: Hidden layer width.
            dim_out: Output embedding dimension.
            num_layers: Number of hidden SIREN layers.
            w0: Frequency scaling for hidden layers after the first.
            w0_initial: Frequency scaling for the first hidden layer.
            use_bias: Whether to include bias parameters.
            final_activation: Optional activation module for the final linear layer.
        """
        super().__init__()
        self.layers = nn.ModuleList(
            [
                _SirenLayer(
                    dim_in=dim_in if index == 0 else dim_hidden,
                    dim_out=dim_hidden,
                    w0=w0_initial if index == 0 else w0,
                    use_bias=use_bias,
                    is_first=index == 0,
                    dropout=True,
                )
                for index in range(num_layers)
            ]
        )
        self.last_layer = _SirenLayer(
            dim_in=dim_hidden,
            dim_out=dim_out,
            w0=w0,
            use_bias=use_bias,
            activation=nn.Identity() if final_activation is None else final_activation,
            dropout=False,
        )

    def forward(self, x: Tensor) -> Tensor:
        """Run the SIREN MLP forward pass.

        Args:
            x: Input tensor of shape ``(B, dim_in)``.

        Returns:
            Output tensor of shape ``(B, dim_out)``.
        """
        for layer in self.layers:
            x = layer(x)
        return self.last_layer(x)


class SatCLIP_Weights(WeightsEnum):  # noqa: N801
    """Pretrained SatCLIP location encoder weights."""

    VIT16_L40 = Weights(
        repo_id=_ZOO_REPO,
        filename='satclip-vit16-l40/model.safetensors',
        meta={
            'legendre_polys': 40,
            'capacity': 512,
            'embed_dim': 256,
            'num_hidden_layers': 2,
            'sha256': '68f110d2dc8a3f2c668c3f052f5503588d4d84300cabf6097edcf903526d6cb2',
            'license': 'MIT',
        },
    )
    VIT16_L10 = Weights(
        repo_id=_ZOO_REPO,
        filename='satclip-vit16-l10/model.safetensors',
        meta={
            'legendre_polys': 10,
            'capacity': 512,
            'embed_dim': 256,
            'num_hidden_layers': 2,
            'sha256': 'b5363fa899ddbfa99c7fd75b4038d3ac45d100c3f743c9e44e3f28cb20bd247a',
            'license': 'MIT',
        },
    )
    RESNET18_L40 = Weights(
        repo_id=_ZOO_REPO,
        filename='satclip-resnet18-l40/model.safetensors',
        meta={
            'legendre_polys': 40,
            'capacity': 512,
            'embed_dim': 256,
            'num_hidden_layers': 2,
            'sha256': '9bebcb6e977eb34d46c8687129134d87d186cb5c91b126704625c4deb879f0cd',
            'license': 'MIT',
        },
    )
    RESNET18_L10 = Weights(
        repo_id=_ZOO_REPO,
        filename='satclip-resnet18-l10/model.safetensors',
        meta={
            'legendre_polys': 10,
            'capacity': 512,
            'embed_dim': 256,
            'num_hidden_layers': 2,
            'sha256': '2467b5c453e1d28711fdcc29d158398d05d389c700d5a6eef875ef969fdbe9de',
            'license': 'MIT',
        },
    )
    RESNET50_L40 = Weights(
        repo_id=_ZOO_REPO,
        filename='satclip-resnet50-l40/model.safetensors',
        meta={
            'legendre_polys': 40,
            'capacity': 512,
            'embed_dim': 256,
            'num_hidden_layers': 2,
            'sha256': 'cd37c7529c9d8b6087492a14cf40d4ead3c1b542b9254641fecff5a866a94da6',
            'license': 'MIT',
        },
    )
    RESNET50_L10 = Weights(
        repo_id=_ZOO_REPO,
        filename='satclip-resnet50-l10/model.safetensors',
        meta={
            'legendre_polys': 10,
            'capacity': 512,
            'embed_dim': 256,
            'num_hidden_layers': 2,
            'sha256': '3f70603ae1747622e2afc9fbacee698fd9e48aeee6d2ad590b34625683a9655e',
            'license': 'MIT',
        },
    )


class SatCLIPLocationEncoder(LocationEncoder):
    """SatCLIP spherical-harmonic SIREN location encoder.

    Accepts ``(lon, lat)`` coordinates in WGS84 degrees natively. Ignores optional
    *year* and *month* inputs.
    """

    name: str = 'satclip'

    def __init__(
        self,
        legendre_polys: int = 40,
        capacity: int = 512,
        embed_dim: int = 256,
        num_hidden_layers: int = 2,
        *,
        weights: SatCLIP_Weights | str | None = None,
        checkpoint: Path | None = None,
        device: str | torch.device = 'cpu',
        batch_size: int = 8192,
    ) -> None:
        """Initialize a new SatCLIPLocationEncoder instance.

        Args:
            legendre_polys: Number of Legendre polynomial degrees ``L``.
            capacity: Hidden layer width of the SIREN network.
            embed_dim: Output embedding dimension.
            num_hidden_layers: Number of hidden SIREN layers.
            weights: Pretrained :class:`SatCLIP_Weights` enum member or name.
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
            resolved = SatCLIP_Weights.verify(weights)
            legendre_polys = int(resolved.meta['legendre_polys'])
            capacity = int(resolved.meta['capacity'])
            embed_dim = int(resolved.meta['embed_dim'])
            num_hidden_layers = int(resolved.meta['num_hidden_layers'])
            raw_state = load_safetensors(resolved.download(), device='cpu')
        elif checkpoint is not None:
            raw_state, ckpt_meta = self._load_checkpoint(checkpoint)
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

    @staticmethod
    def _load_checkpoint(path: Path) -> tuple[dict[str, Tensor], dict[str, Any] | None]:
        """Load a local safetensors or Lightning checkpoint payload."""
        if pathlib.Path(path).suffix == '.safetensors':
            return load_safetensors(path, device='cpu'), None
        payload = torch.load(os.fspath(path), map_location='cpu', weights_only=True)
        if isinstance(payload, dict) and 'state_dict' in payload:
            hparams = payload.get('hyper_parameters')
            meta = hparams if isinstance(hparams, dict) else None
            return payload['state_dict'], meta
        return payload, None

    def forward(
        self, coords: Tensor, year: Tensor | None = None, month: Tensor | None = None
    ) -> Tensor:
        """Encode ``(lon, lat)`` coordinates in degrees into SatCLIP embeddings.

        Args:
            coords: Coordinate tensor of shape ``(B, 2)`` in ``(lon, lat)`` degrees.
            year: Unused temporal parameter.
            month: Unused temporal parameter.

        Returns:
            Float32 embedding tensor of shape ``(B, embed_dim)``.
        """
        encoded = self.posenc(coords).to(dtype=self.nnet.last_layer.weight.dtype)
        return self.nnet(encoded).to(dtype=torch.float32)


def satclip(
    weights: SatCLIP_Weights | str | None = SatCLIP_Weights.VIT16_L40,
    *,
    device: str | torch.device = 'cpu',
    batch_size: int = 8192,
    **kwargs: Any,
) -> SatCLIPLocationEncoder:
    """Create a SatCLIP location encoder.

    If you use this model in your research, please cite the following paper:

    * https://arxiv.org/abs/2311.17179

    Args:
        weights: Pretrained weights enum member, member name, or ``None``.
        device: Target device for inference.
        batch_size: Batch size used by :meth:`~LocationEncoder.encode`.
        **kwargs: Additional keyword arguments forwarded to
            :class:`SatCLIPLocationEncoder`.

    Returns:
        Configured :class:`SatCLIPLocationEncoder` instance.
    """
    return SatCLIPLocationEncoder(
        weights=weights, device=device, batch_size=batch_size, **kwargs
    )


__all__ = (
    'POLE_LATITUDE_EPS_DEGREES',
    'SatCLIPLocationEncoder',
    'SatCLIP_Weights',
    'SirenNet',
    'StableAnalyticSphericalHarmonics',
    'satclip',
)
