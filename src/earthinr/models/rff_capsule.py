# Copyright (c) Taylor Geospatial. All rights reserved.
# Licensed under the MIT License.

"""Equal-Earth multiscale RFF capsule encoders (GeoCLIP, TaxaBind, SLED, GAIR)."""

import math
import os
import pathlib
from collections.abc import Sequence
from typing import Any, cast

import torch
from torch import Tensor, nn

from .._types import CoordOrder, Path
from ._base import LocationEncoder, Weights, WeightsEnum
from ._utils import equal_earth_projection, load_safetensors

_ZOO_REPO = 'taylor-geospatial/earthinr-zoo'


def _extract_gair_state_dict(payload: dict[str, Any]) -> dict[str, Tensor]:
    """Extract ``LocEnc*`` tensors from a legacy GAIR ``.pth`` checkpoint."""
    state: dict[str, Any] = payload
    for key in ('state_dict', 'model', 'model_state_dict'):
        if key in state and isinstance(state[key], dict):
            state = state[key]
            break

    marker = next((k for k in state if 'LocEnc0.capsule.1.weight' in k), None)
    if marker is None:
        raise KeyError(
            "Could not locate 'LocEnc0.capsule.1.weight' in GAIR checkpoint."
        )
    prefix = marker.split('LocEnc0', 1)[0]
    return {
        k[len(prefix) :]: v
        for k, v in state.items()
        if k.startswith(prefix) and k[len(prefix) :].startswith('LocEnc')
    }


class GaussianEncoding(nn.Module):
    """Random Fourier feature layer with a frozen Gaussian frequency matrix."""

    def __init__(
        self,
        sigma: float | None = None,
        input_size: int | None = None,
        encoded_size: int | None = None,
        b: Tensor | None = None,
    ) -> None:
        """Initialize a new GaussianEncoding instance.

        Args:
            sigma: Standard deviation for sampling the frequency matrix.
            input_size: Input coordinate dimensionality.
            encoded_size: Number of sin/cos frequency pairs (output is
                ``2 * encoded_size``).
            b: Optional pre-sampled frequency matrix of shape
                ``(encoded_size, input_size)``.

        Raises:
            ValueError: If *b* and sampling arguments are both provided or both
                omitted.
        """
        super().__init__()
        if b is None:
            if sigma is None or input_size is None or encoded_size is None:
                raise ValueError(
                    'sigma, input_size, and encoded_size are required when b is None.'
                )
            b = torch.randn((encoded_size, input_size)) * sigma
        elif sigma is not None or input_size is not None or encoded_size is not None:
            raise ValueError(
                'Specify either b or (sigma, input_size, encoded_size), not both.'
            )
        self.b = nn.Parameter(b, requires_grad=False)

    def forward(self, v: Tensor) -> Tensor:
        """Map projected coordinates through random Fourier features.

        Args:
            v: Input tensor of shape ``(..., input_size)``.

        Returns:
            Concatenated ``[cos(2*pi*v@b^T), sin(2*pi*v@b^T)]`` tensor of shape
            ``(..., 2 * encoded_size)``.
        """
        vp = 2.0 * math.pi * (v @ self.b.T)
        return torch.cat((torch.cos(vp), torch.sin(vp)), dim=-1)


class RFFCapsule(nn.Module):
    """Single-scale RFF capsule MLP matching GeoCLIP/TaxaBind/SLED/GAIR weights."""

    def __init__(
        self,
        sigma: float,
        input_size: int = 2,
        encoded_size: int = 256,
        hidden_dim: int = 1024,
        out_dim: int = 512,
    ) -> None:
        """Initialize a new RFFCapsule instance.

        Args:
            sigma: Gaussian frequency standard deviation for this capsule.
            input_size: Projected coordinate dimension.
            encoded_size: Number of RFF frequency pairs.
            hidden_dim: Hidden layer width of the 3-layer ReLU MLP.
            out_dim: Output embedding dimension of the linear head.
        """
        super().__init__()
        self.sigma = sigma
        rff_dim = 2 * encoded_size
        self.capsule = nn.Sequential(
            GaussianEncoding(
                sigma=sigma, input_size=input_size, encoded_size=encoded_size
            ),
            nn.Linear(rff_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.head = nn.Sequential(nn.Linear(hidden_dim, out_dim))

    def forward(self, x: Tensor) -> Tensor:
        """Encode Equal-Earth projected coordinates through this scale capsule.

        Args:
            x: Projected coordinates of shape ``(B, input_size)``.

        Returns:
            Capsule embedding of shape ``(B, out_dim)``.
        """
        return self.head(self.capsule(x))


class EqualEarthRFFEncoder(nn.Module):
    """Multi-scale Equal-Earth RFF capsule encoder with ``LocEnc{i}`` key parity."""

    def __init__(
        self,
        sigma: Sequence[float] = (1.0, 16.0, 256.0),
        input_size: int = 2,
        encoded_size: int = 256,
        hidden_dim: int = 1024,
        out_dim: int = 512,
        coord_order: CoordOrder = 'latlon',
    ) -> None:
        """Initialize a new EqualEarthRFFEncoder instance.

        Args:
            sigma: Sequence of Gaussian frequency scales, one capsule per scale.
            input_size: Projected coordinate dimension.
            encoded_size: Number of RFF frequency pairs per capsule.
            hidden_dim: Hidden width per capsule MLP.
            out_dim: Output embedding dimension summed across capsules.
            coord_order: Native coordinate order expected by this raw module.

        Raises:
            ValueError: If *sigma* is empty.
        """
        super().__init__()
        if len(sigma) == 0:
            raise ValueError('sigma must be a non-empty sequence.')
        self.sigma = tuple(float(s) for s in sigma)
        self.num_capsules = len(self.sigma)
        self.out_dim = out_dim
        self.coord_order: CoordOrder = coord_order

        for i, s in enumerate(self.sigma):
            self.add_module(
                f'LocEnc{i}',
                RFFCapsule(
                    sigma=s,
                    input_size=input_size,
                    encoded_size=encoded_size,
                    hidden_dim=hidden_dim,
                    out_dim=out_dim,
                ),
            )

    def forward(self, coords: Tensor) -> Tensor:
        """Encode geographic coordinates by summing all multi-scale RFF capsules.

        Args:
            coords: Coordinate tensor of shape ``(B, 2)`` in degrees ordered by
                ``self.coord_order``.

        Returns:
            Summed embedding tensor of shape ``(B, out_dim)``.
        """
        projected = equal_earth_projection(coords, coord_order=self.coord_order)
        first_capsule = cast(nn.Module, self._modules['LocEnc0'])
        out: Tensor = first_capsule(projected)
        for i in range(1, self.num_capsules):
            capsule = cast(nn.Module, self._modules[f'LocEnc{i}'])
            out = out + capsule(projected)
        return out


class _RFFCapsuleLocationEncoder(LocationEncoder):
    """Shared LocationEncoder base for Equal-Earth RFF capsule checkpoints."""

    _weights_cls: type[WeightsEnum]
    _default_sigma: tuple[float, ...]
    _default_encoded_size: int
    _default_hidden_dim: int
    _default_out_dim: int
    _native_coord_order: CoordOrder
    _allow_legacy_pth: bool = False

    def __init__(
        self,
        *,
        weights: WeightsEnum | str | None = None,
        checkpoint: Path | None = None,
        sigma: Sequence[float] | None = None,
        encoded_size: int | None = None,
        hidden_dim: int | None = None,
        out_dim: int | None = None,
        device: str | torch.device = 'cpu',
        batch_size: int = 8192,
    ) -> None:
        """Initialize an Equal-Earth RFF capsule location encoder.

        Args:
            weights: Pretrained weights enum member or member name string.
            checkpoint: Local checkpoint path (``.safetensors`` or GAIR ``.pth``).
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
        res_sigma = self._default_sigma if sigma is None else tuple(sigma)
        res_encoded = (
            self._default_encoded_size if encoded_size is None else encoded_size
        )
        res_hidden = self._default_hidden_dim if hidden_dim is None else hidden_dim
        res_out = self._default_out_dim if out_dim is None else out_dim

        if weights is not None:
            resolved = self._weights_cls.verify(weights)
            meta = resolved.meta
            res_sigma = tuple(meta['sigma'])
            res_encoded = int(meta['encoded_size'])
            res_hidden = int(meta['hidden_dim'])
            res_out = int(meta['output_dim'])
            raw_state = load_safetensors(resolved.download(), device='cpu')
        elif checkpoint is not None:
            if pathlib.Path(checkpoint).suffix == '.safetensors':
                raw_state = load_safetensors(checkpoint, device='cpu')
            elif self._allow_legacy_pth:
                payload = torch.load(
                    os.fspath(checkpoint), map_location='cpu', weights_only=False
                )
                raw_state = _extract_gair_state_dict(payload)
            else:
                raise ValueError(
                    f'Expected a .safetensors checkpoint, got {os.fspath(checkpoint)!r}.'
                )

        self.embed_dim = res_out
        self.encoder = EqualEarthRFFEncoder(
            sigma=res_sigma,
            input_size=2,
            encoded_size=res_encoded,
            hidden_dim=res_hidden,
            out_dim=res_out,
            coord_order=self._native_coord_order,
        )
        if raw_state is not None:
            self.encoder.load_state_dict(raw_state, strict=True)

        self.to(self.device)
        self.eval()

    def forward(
        self, coords: Tensor, year: Tensor | None = None, month: Tensor | None = None
    ) -> Tensor:
        """Encode ``(lon, lat)`` coordinates in WGS84 degrees.

        Models whose upstream checkpoints expect ``(lat, lon)`` perform the swap
        internally.

        Args:
            coords: Coordinate tensor of shape ``(B, 2)`` in ``(lon, lat)`` degrees.
            year: Unused temporal parameter.
            month: Unused temporal parameter.

        Returns:
            Float32 embedding tensor of shape ``(B, embed_dim)``.
        """
        inp = (
            torch.stack((coords[:, 1], coords[:, 0]), dim=-1)
            if self._native_coord_order == 'latlon'
            else coords
        )
        return self.encoder(inp).to(dtype=torch.float32)


class GeoCLIP_Weights(WeightsEnum):  # noqa: N801
    """Pretrained GeoCLIP location encoder weights."""

    DEFAULT = Weights(
        repo_id=_ZOO_REPO,
        filename='geoclip/model.safetensors',
        meta={
            'sigma': (1.0, 16.0, 256.0),
            'encoded_size': 256,
            'hidden_dim': 1024,
            'output_dim': 512,
            'sha256': '6a0f4f85c70bf8a0cd82f1c427e548fc4a5aeb17cac023211ad7aa1a830092a9',
            'license': 'MIT',
        },
    )


class GeoCLIPLocationEncoder(_RFFCapsuleLocationEncoder):
    """GeoCLIP 512-d Equal-Earth RFF location encoder."""

    name: str = 'geoclip'
    _weights_cls = GeoCLIP_Weights
    _default_sigma = (1.0, 16.0, 256.0)
    _default_encoded_size = 256
    _default_hidden_dim = 1024
    _default_out_dim = 512
    _native_coord_order: CoordOrder = 'latlon'


def geoclip(
    weights: GeoCLIP_Weights | str | None = GeoCLIP_Weights.DEFAULT,
    *,
    device: str | torch.device = 'cpu',
    batch_size: int = 8192,
    **kwargs: Any,
) -> GeoCLIPLocationEncoder:
    """Create a GeoCLIP location encoder.

    If you use this model in your research, please cite the following paper:

    * https://arxiv.org/abs/2309.16020

    Args:
        weights: Pretrained weights enum member, member name, or ``None``.
        device: Target device for inference.
        batch_size: Batch size used by :meth:`~LocationEncoder.encode`.
        **kwargs: Additional keyword arguments forwarded to
            :class:`GeoCLIPLocationEncoder`.

    Returns:
        Configured :class:`GeoCLIPLocationEncoder` instance.
    """
    return GeoCLIPLocationEncoder(
        weights=weights, device=device, batch_size=batch_size, **kwargs
    )


class TaxaBind_Weights(WeightsEnum):  # noqa: N801
    """Pretrained TaxaBind (EcoGeo) location encoder weights."""

    ECOGEO = Weights(
        repo_id=_ZOO_REPO,
        filename='taxabind/model.safetensors',
        meta={
            'sigma': (1.0, 16.0, 256.0),
            'encoded_size': 256,
            'hidden_dim': 1024,
            'output_dim': 512,
            'sha256': 'f9f8986c0c55057921a0175f45dfe5c767e3932075f755a0420286d132dfdee1',
            'license': 'Apache-2.0',
        },
    )


class TaxaBindLocationEncoder(_RFFCapsuleLocationEncoder):
    """TaxaBind (EcoGeo) 512-d Equal-Earth RFF location encoder."""

    name: str = 'taxabind'
    _weights_cls = TaxaBind_Weights
    _default_sigma = (1.0, 16.0, 256.0)
    _default_encoded_size = 256
    _default_hidden_dim = 1024
    _default_out_dim = 512
    _native_coord_order: CoordOrder = 'latlon'


def taxabind(
    weights: TaxaBind_Weights | str | None = TaxaBind_Weights.ECOGEO,
    *,
    device: str | torch.device = 'cpu',
    batch_size: int = 8192,
    **kwargs: Any,
) -> TaxaBindLocationEncoder:
    """Create a TaxaBind (EcoGeo) location encoder.

    If you use this model in your research, please cite the following paper:

    * https://arxiv.org/abs/2411.00683

    Args:
        weights: Pretrained weights enum member, member name, or ``None``.
        device: Target device for inference.
        batch_size: Batch size used by :meth:`~LocationEncoder.encode`.
        **kwargs: Additional keyword arguments forwarded to
            :class:`TaxaBindLocationEncoder`.

    Returns:
        Configured :class:`TaxaBindLocationEncoder` instance.
    """
    return TaxaBindLocationEncoder(
        weights=weights, device=device, batch_size=batch_size, **kwargs
    )


class SLED_Weights(WeightsEnum):  # noqa: N801
    """Pretrained SLED location encoder weights."""

    S1_S2_LS = Weights(
        repo_id=_ZOO_REPO,
        filename='sled-s1-s2-ls/model.safetensors',
        meta={
            'sigma': (2.0, 4.0),
            'encoded_size': 384,
            'hidden_dim': 1536,
            'output_dim': 768,
            'sha256': 'c26a8462dbe34ab3b449a0b79156cb9466ebd00bd81fa04290f471e6d860966a',
            'license': 'Apache-2.0',
        },
    )
    S2_LS = Weights(
        repo_id=_ZOO_REPO,
        filename='sled-s2-ls/model.safetensors',
        meta={
            'sigma': (2.0, 4.0),
            'encoded_size': 384,
            'hidden_dim': 1536,
            'output_dim': 768,
            'sha256': 'fba66a7358937f588b7213ef5ff1d3a46392518cfa53c66b7572bc2a35f7c22a',
            'license': 'Apache-2.0',
        },
    )


class SLEDLocationEncoder(_RFFCapsuleLocationEncoder):
    """SLED 768-d Equal-Earth RFF location encoder."""

    name: str = 'sled'
    _weights_cls = SLED_Weights
    _default_sigma = (2.0, 4.0)
    _default_encoded_size = 384
    _default_hidden_dim = 1536
    _default_out_dim = 768
    _native_coord_order: CoordOrder = 'latlon'


def sled(
    weights: SLED_Weights | str | None = SLED_Weights.S1_S2_LS,
    *,
    device: str | torch.device = 'cpu',
    batch_size: int = 8192,
    **kwargs: Any,
) -> SLEDLocationEncoder:
    """Create a SLED location encoder.

    If you use this model in your research, please cite the following paper:

    * https://arxiv.org/abs/2608.06612

    Args:
        weights: Pretrained weights enum member, member name, or ``None``.
        device: Target device for inference.
        batch_size: Batch size used by :meth:`~LocationEncoder.encode`.
        **kwargs: Additional keyword arguments forwarded to
            :class:`SLEDLocationEncoder`.

    Returns:
        Configured :class:`SLEDLocationEncoder` instance.
    """
    return SLEDLocationEncoder(
        weights=weights, device=device, batch_size=batch_size, **kwargs
    )


class GAIR_Weights(WeightsEnum):  # noqa: N801
    """Pretrained GAIR location encoder weights."""

    DEFAULT = Weights(
        repo_id=_ZOO_REPO,
        filename='gair/model.safetensors',
        meta={
            'sigma': (1.0, 16.0, 256.0),
            'encoded_size': 256,
            'hidden_dim': 1024,
            'output_dim': 768,
            'sha256': '0a9cbb46b6b090fcadbb4efa52c5b996b66ef06fe98a7dd01b5573c8a8fc08f9',
            'license': 'Apache-2.0',
        },
    )


class GAIRLocationEncoder(_RFFCapsuleLocationEncoder):
    """GAIR 768-d Equal-Earth RFF location encoder (native ``(lon, lat)``)."""

    name: str = 'gair'
    _weights_cls = GAIR_Weights
    _default_sigma = (1.0, 16.0, 256.0)
    _default_encoded_size = 256
    _default_hidden_dim = 1024
    _default_out_dim = 768
    _native_coord_order: CoordOrder = 'lonlat'
    _allow_legacy_pth: bool = True


def gair(
    weights: GAIR_Weights | str | None = GAIR_Weights.DEFAULT,
    *,
    device: str | torch.device = 'cpu',
    batch_size: int = 8192,
    **kwargs: Any,
) -> GAIRLocationEncoder:
    """Create a GAIR location encoder.

    If you use this model in your research, please cite the following paper:

    * https://arxiv.org/abs/2503.16683

    Args:
        weights: Pretrained weights enum member, member name, or ``None``.
        device: Target device for inference.
        batch_size: Batch size used by :meth:`~LocationEncoder.encode`.
        **kwargs: Additional keyword arguments forwarded to
            :class:`GAIRLocationEncoder`.

    Returns:
        Configured :class:`GAIRLocationEncoder` instance.
    """
    return GAIRLocationEncoder(
        weights=weights, device=device, batch_size=batch_size, **kwargs
    )


__all__ = (
    'EqualEarthRFFEncoder',
    'GAIRLocationEncoder',
    'GAIR_Weights',
    'GaussianEncoding',
    'GeoCLIPLocationEncoder',
    'GeoCLIP_Weights',
    'RFFCapsule',
    'SLEDLocationEncoder',
    'SLED_Weights',
    'TaxaBindLocationEncoder',
    'TaxaBind_Weights',
    'gair',
    'geoclip',
    'sled',
    'taxabind',
)
