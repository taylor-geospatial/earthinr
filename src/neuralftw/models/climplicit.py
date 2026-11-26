# Copyright (c) Taylor Geospatial. All rights reserved.
# Licensed under the MIT License.

"""Climplicit ReSIREN / H-SIREN climate location encoder."""

import math
from collections.abc import Sequence
from typing import Any

import torch
from torch import Tensor, nn

from .._types import Path
from .._utils import float64_dtype
from ._base import LocationEncoder, Weights, WeightsEnum
from ._utils import load_safetensors

_ZOO_REPO = 'taylor-geospatial/neuralftw-zoo'

CHELSA_VAR_NAMES: tuple[str, ...] = (
    'cmi',
    'clt',
    'hurs',
    'pet',
    'pr',
    'rsds',
    'sfcWind',
    'tas',
    'tasmax',
    'tasmin',
    'vpd',
)

# Exact CHELSA variable normalization statistics from ecovision-uzh/climplicit
CHELSA_MEAN: tuple[float, ...] = (
    -264.1493656,
    3912.44628016,
    5921.65964573,
    9385.47468266,
    697.03653109,
    15219.37926928,
    3498.8511804,
    2819.56006368,
    2864.08583811,
    2773.46759638,
    8039.37322797,
)

CHELSA_STD: tuple[float, ...] = (
    1042.67560332,
    1767.94018571,
    1185.91587823,
    6639.79069994,
    883.56243405,
    7843.49167037,
    1637.09237995,
    174.43791946,
    181.69448751,
    167.07485901,
    7516.98198719,
)

DEFAULT_MONTHS: tuple[int, ...] = (3, 6, 9, 12)


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
        """Compute ``sin(w0 * x)``.

        Args:
            x: Input tensor.

        Returns:
            Activated tensor.
        """
        return torch.sin(self.w0 * x)


class _ClimplicitSirenLayer(nn.Module):
    """ReSIREN / H-SIREN layer matching Climplicit's pre-activation residual math."""

    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        w0: float = 1.0,
        c: float = 6.0,
        is_first: bool = False,
        activation: nn.Module | None = None,
        residual_connections: bool = False,
        h_siren: bool = False,
    ) -> None:
        """Initialize a new _ClimplicitSirenLayer instance.

        Args:
            dim_in: Input feature dimension.
            dim_out: Output feature dimension.
            w0: Sine frequency scaling factor.
            c: SIREN weight initialization constant.
            is_first: Whether this is the first layer of the network.
            activation: Optional override activation module.
            residual_connections: If True, average pre-activation with the previous
                layer's pre-activation when shapes match.
            h_siren: If True and *is_first*, apply ``sinh(2 * out)`` before sine.
        """
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.is_first = is_first
        self.residual_connections = residual_connections
        self.h_siren = h_siren

        weight = torch.zeros(dim_out, dim_in)
        bias = torch.zeros(dim_out)
        scale = (1.0 / dim_in) if is_first else (math.sqrt(c / dim_in) / w0)
        weight.uniform_(-scale, scale)
        bias.uniform_(-scale, scale)

        self.weight = nn.Parameter(weight)
        self.bias = nn.Parameter(bias)
        self.activation: nn.Module = _Sine(w0) if activation is None else activation

    def forward(
        self, x: Tensor, prev_gaussian: Tensor | None = None
    ) -> tuple[Tensor, Tensor]:
        """Compute activated output and pre-activation residual tensor.

        Args:
            x: Input tensor of shape ``(B, dim_in)``.
            prev_gaussian: Pre-activation tensor from the previous layer.

        Returns:
            Tuple ``(activated_output, pre_activation_gaussian)``.
        """
        out = nn.functional.linear(x, self.weight, self.bias)
        if (
            self.residual_connections
            and prev_gaussian is not None
            and out.shape == prev_gaussian.shape
        ):
            out = (out + prev_gaussian) / 2.0
        # Save pre-activation before H-SIREN sinh (matches ecovision-uzh/climplicit)
        gaussian = out
        if self.h_siren and self.is_first:
            out = torch.sinh(2.0 * out)
        return self.activation(out), gaussian


class _ClimplicitSirenNet(nn.Module):
    """16-layer ReSIREN + H-SIREN backbone matching Climplicit checkpoint keys."""

    def __init__(
        self,
        dim_in: int = 4,
        dim_hidden: int = 512,
        dim_out: int = 256,
        num_layers: int = 16,
        w0: float = 1.0,
        w0_initial: float = 30.0,
        h_siren: bool = True,
        residual_connections: bool = True,
    ) -> None:
        """Initialize a new _ClimplicitSirenNet instance.

        Args:
            dim_in: Input dimension (4 for normalized ``lon, lat, sin(m), cos(m)``).
            dim_hidden: Hidden layer width.
            dim_out: Output embedding dimension per month.
            num_layers: Number of hidden ReSIREN layers.
            w0: Frequency scaling for layers after the first.
            w0_initial: Frequency scaling for the first layer.
            h_siren: Whether to apply H-SIREN on layer 0.
            residual_connections: Whether to average pre-activations across hidden
                layers.
        """
        super().__init__()
        self.layers = nn.ModuleList(
            [
                _ClimplicitSirenLayer(
                    dim_in=dim_in if idx == 0 else dim_hidden,
                    dim_out=dim_hidden,
                    w0=w0_initial if idx == 0 else w0,
                    is_first=idx == 0,
                    residual_connections=residual_connections,
                    h_siren=h_siren,
                )
                for idx in range(num_layers)
            ]
        )
        self.last_layer = _ClimplicitSirenLayer(
            dim_in=dim_hidden,
            dim_out=dim_out,
            w0=w0,
            activation=nn.Identity(),
            residual_connections=False,
            h_siren=False,
        )

    def forward(self, x: Tensor) -> Tensor:
        """Run the Climplicit ReSIREN backbone forward pass.

        Args:
            x: Input tensor of shape ``(B, dim_in)``.

        Returns:
            Output tensor of shape ``(B, dim_out)``.
        """
        gaussian: Tensor | None = None
        for layer in self.layers:
            x, gaussian = layer(x, gaussian)
        out, _ = self.last_layer(x, gaussian)
        return out


class ClimplicitNet(nn.Module):
    """Pure-PyTorch Climplicit network with exact checkpoint key parity.

    Registers ``CHELSA_MEAN`` and ``CHELSA_STD`` as non-persistent buffers so
    ``return_chelsa=True`` works cleanly on CPU, CUDA, and MPS devices.
    """

    chelsa_mean: Tensor
    chelsa_std: Tensor

    def __init__(
        self,
        dim_in: int = 4,
        dim_hidden: int = 512,
        dim_out: int = 256,
        num_layers: int = 16,
        w0: float = 1.0,
        w0_initial: float = 30.0,
        h_siren: bool = True,
        residual_connections: bool = True,
        num_chelsa_vars: int = 11,
        return_chelsa: bool = False,
        default_months: Sequence[int] = DEFAULT_MONTHS,
    ) -> None:
        """Initialize a new ClimplicitNet instance.

        Args:
            dim_in: Input coordinate + cyclical month feature dimension.
            dim_hidden: Hidden layer width of the ReSIREN backbone.
            dim_out: Per-month embedding dimension.
            num_layers: Number of hidden ReSIREN layers.
            w0: Frequency scaling for hidden layers after layer 0.
            w0_initial: Frequency scaling for layer 0.
            h_siren: Whether to apply H-SIREN on layer 0.
            residual_connections: Whether to use ReSIREN pre-activation residuals.
            num_chelsa_vars: Number of CHELSA climate variables in the regressor.
            return_chelsa: If True, return un-normalized CHELSA reconstructions
                instead of implicit climate embeddings.
            default_months: Months concatenated when *month* is omitted.

        Raises:
            ValueError: If *default_months* is empty.
        """
        super().__init__()
        if len(default_months) == 0:
            raise ValueError('default_months must be non-empty.')
        self.dim_out = dim_out
        self.num_chelsa_vars = num_chelsa_vars
        self.return_chelsa = return_chelsa
        self.default_months = tuple(int(m) for m in default_months)

        self.location_encoder = _ClimplicitSirenNet(
            dim_in=dim_in,
            dim_hidden=dim_hidden,
            dim_out=dim_out,
            num_layers=num_layers,
            w0=w0,
            w0_initial=w0_initial,
            h_siren=h_siren,
            residual_connections=residual_connections,
        )
        self.chelsa_regressor = nn.Linear(dim_out, num_chelsa_vars)
        self.register_buffer(
            'chelsa_mean',
            torch.tensor(CHELSA_MEAN[:num_chelsa_vars], dtype=torch.float32).unsqueeze(
                0
            ),
            persistent=False,
        )
        self.register_buffer(
            'chelsa_std',
            torch.tensor(CHELSA_STD[:num_chelsa_vars], dtype=torch.float32).unsqueeze(
                0
            ),
            persistent=False,
        )

    def _encode_month_features(
        self, loc: Tensor, sin_m: Tensor, cos_m: Tensor
    ) -> Tensor:
        """Encode normalized ``(lon, lat)`` concatenated with month sin/cos."""
        x = self.location_encoder(torch.cat((loc, sin_m, cos_m), dim=-1))
        if self.return_chelsa:
            mean = self.chelsa_mean.to(device=x.device, dtype=x.dtype)
            std = self.chelsa_std.to(device=x.device, dtype=x.dtype)
            x = self.chelsa_regressor(x) * std + mean
        return x

    def forward(self, coords: Tensor, month: Tensor | None = None) -> Tensor:
        """Encode ``(lon, lat)`` coordinates with optional calendar month.

        Args:
            coords: Coordinate tensor of shape ``(B, 2)`` in ``(lon, lat)`` degrees.
            month: Optional tensor of shape ``(B,)`` with calendar months ``1..12``.
                When ``None``, embeddings for ``default_months`` (``[3, 6, 9, 12]``)
                are concatenated along the feature axis.

        Returns:
            Float32 tensor of shape ``(B, 1024)`` (or ``(B, 256)`` with *month*;
            ``(B, 44)`` / ``(B, 11)`` when ``return_chelsa=True``).
        """
        loc = torch.stack((coords[:, 0] / 180.0, coords[:, 1] / 90.0), dim=-1).to(
            dtype=torch.float32
        )
        if month is None:
            ones = torch.ones_like(loc[:, :1])
            outputs = [
                self._encode_month_features(
                    loc,
                    ones * math.sin((m / 12.0) * 2.0 * math.pi),
                    ones * math.cos((m / 12.0) * 2.0 * math.pi),
                )
                for m in self.default_months
            ]
            return torch.cat(outputs, dim=-1)

        angle = month.to(dtype=float64_dtype(loc.device), device=loc.device).unsqueeze(
            -1
        ) * (2.0 * math.pi / 12.0)
        return self._encode_month_features(
            loc,
            torch.sin(angle).to(dtype=loc.dtype),
            torch.cos(angle).to(dtype=loc.dtype),
        )


class Climplicit_Weights(WeightsEnum):  # noqa: N801
    """Pretrained Climplicit location encoder weights (CC-BY-4.0)."""

    CHELSA = Weights(
        repo_id=_ZOO_REPO,
        filename='climplicit/model.safetensors',
        meta={
            'dim_in': 4,
            'dim_hidden': 512,
            'dim_out': 256,
            'num_layers': 16,
            'num_chelsa_vars': 11,
            'default_months': DEFAULT_MONTHS,
            'embed_dim': 1024,
            'sha256': '307842c88b6265eb90adc5d2035ae26acc4ca9d74cf723a963aed147c8b231cc',
            'license': 'CC-BY-4.0',
        },
    )


class ClimplicitLocationEncoder(LocationEncoder):
    """Climplicit climate location encoder accepting ``(lon, lat)`` WGS84 degrees.

    When *month* is omitted, embeddings across months ``[3, 6, 9, 12]`` are
    concatenated to yield 1024-d embeddings (or 44-d when ``return_chelsa=True``).
    When *month* is provided, returns 256-d embeddings (or 11-d CHELSA variables).
    """

    name: str = 'climplicit'

    def __init__(
        self,
        *,
        weights: Climplicit_Weights | str | None = None,
        checkpoint: Path | None = None,
        dim_hidden: int = 512,
        dim_out: int = 256,
        num_layers: int = 16,
        return_chelsa: bool = False,
        default_months: Sequence[int] = DEFAULT_MONTHS,
        device: str | torch.device = 'cpu',
        batch_size: int = 8192,
    ) -> None:
        """Initialize a new ClimplicitLocationEncoder instance.

        Args:
            weights: Pretrained :class:`Climplicit_Weights` enum member or name.
            checkpoint: Local ``.safetensors`` checkpoint path.
            dim_hidden: Hidden layer width of the ReSIREN backbone.
            dim_out: Per-month embedding dimension.
            num_layers: Number of hidden ReSIREN layers.
            return_chelsa: If True, output un-normalized CHELSA variables instead
                of implicit embeddings.
            default_months: Calendar months concatenated when *month* is omitted.
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
            resolved = Climplicit_Weights.verify(weights)
            dim_hidden = int(resolved.meta['dim_hidden'])
            dim_out = int(resolved.meta['dim_out'])
            num_layers = int(resolved.meta['num_layers'])
            raw_state = load_safetensors(resolved.download(), device='cpu')
        elif checkpoint is not None:
            raw_state = load_safetensors(checkpoint, device='cpu')

        self.return_chelsa = return_chelsa
        self.model = ClimplicitNet(
            dim_in=4,
            dim_hidden=dim_hidden,
            dim_out=dim_out,
            num_layers=num_layers,
            return_chelsa=return_chelsa,
            default_months=default_months,
        )
        self.embed_dim = len(self.model.default_months) * (
            self.model.num_chelsa_vars if return_chelsa else dim_out
        )
        if raw_state is not None:
            self.model.load_state_dict(raw_state, strict=True)

        self.to(self.device)
        self.eval()

    def forward(
        self, coords: Tensor, year: Tensor | None = None, month: Tensor | None = None
    ) -> Tensor:
        """Encode ``(lon, lat)`` coordinates with optional calendar month.

        Args:
            coords: Coordinate tensor of shape ``(B, 2)`` in ``(lon, lat)`` degrees.
            year: Unused temporal parameter.
            month: Optional tensor of shape ``(B,)`` containing months ``1..12``.

        Returns:
            Float32 embedding or CHELSA reconstruction tensor.
        """
        return self.model(coords, month=month).to(dtype=torch.float32)


def climplicit(
    weights: Climplicit_Weights | str | None = Climplicit_Weights.CHELSA,
    *,
    return_chelsa: bool = False,
    device: str | torch.device = 'cpu',
    batch_size: int = 8192,
    **kwargs: Any,
) -> ClimplicitLocationEncoder:
    """Create a Climplicit location encoder.

    If you use this model in your research, please cite the following paper:

    * https://arxiv.org/abs/2504.14223

    Args:
        weights: Pretrained weights enum member, member name, or ``None``.
        return_chelsa: If True, return un-normalized CHELSA reconstructions.
        device: Target device for inference.
        batch_size: Batch size used by :meth:`~LocationEncoder.encode`.
        **kwargs: Additional keyword arguments forwarded to
            :class:`ClimplicitLocationEncoder`.

    Returns:
        Configured :class:`ClimplicitLocationEncoder` instance.
    """
    return ClimplicitLocationEncoder(
        weights=weights,
        return_chelsa=return_chelsa,
        device=device,
        batch_size=batch_size,
        **kwargs,
    )


__all__ = (
    'CHELSA_MEAN',
    'CHELSA_STD',
    'CHELSA_VAR_NAMES',
    'DEFAULT_MONTHS',
    'ClimplicitLocationEncoder',
    'ClimplicitNet',
    'Climplicit_Weights',
    'climplicit',
)
