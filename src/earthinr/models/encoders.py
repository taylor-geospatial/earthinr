# Copyright (c) Taylor Geospatial. All rights reserved.
# Licensed under the MIT License.

"""Parameter-free and analytical coordinate encoders."""

import math

import numpy as np
import torch
from torch import Tensor

from ._base import LocationEncoder
from ._utils import lonlat_to_xyz
from .satclip import StableAnalyticSphericalHarmonics

_EARTH_RADIUS_KM = 6371.0


class SinCosLocationEncoder(LocationEncoder):
    """Four-dimensional sin/cos coordinate baseline encoder.

    Outputs ``[sin(lat), cos(lat), sin(lon), cos(lon)]`` from ``(lon, lat)``
    coordinates in WGS84 degrees.
    """

    name: str = 'sincos'
    embed_dim: int = 4

    def forward(
        self, coords: Tensor, year: Tensor | None = None, month: Tensor | None = None
    ) -> Tensor:
        """Encode ``(lon, lat)`` coordinates into sin/cos features.

        Args:
            coords: Coordinate tensor of shape ``(B, 2)`` in ``(lon, lat)`` degrees.
            year: Unused temporal parameter.
            month: Unused temporal parameter.

        Returns:
            Float32 feature tensor of shape ``(B, 4)``.
        """
        lon_r = torch.deg2rad(coords[:, 0])
        lat_r = torch.deg2rad(coords[:, 1])
        return torch.stack(
            (torch.sin(lat_r), torch.cos(lat_r), torch.sin(lon_r), torch.cos(lon_r)),
            dim=-1,
        ).to(dtype=torch.float32)


class XYZLocationEncoder(LocationEncoder):
    """Three-dimensional unit-sphere Cartesian coordinate encoder."""

    name: str = 'xyz'
    embed_dim: int = 3

    def forward(
        self, coords: Tensor, year: Tensor | None = None, month: Tensor | None = None
    ) -> Tensor:
        """Encode ``(lon, lat)`` coordinates onto the 3D unit sphere.

        Args:
            coords: Coordinate tensor of shape ``(B, 2)`` in ``(lon, lat)`` degrees.
            year: Unused temporal parameter.
            month: Unused temporal parameter.

        Returns:
            Float32 unit-sphere coordinate tensor of shape ``(B, 3)``.
        """
        return lonlat_to_xyz(coords[:, 0], coords[:, 1]).to(dtype=torch.float32)


class NeRFLocationEncoder(LocationEncoder):
    """Multi-frequency NeRF positional encoding on 3D unit-sphere coordinates."""

    name: str = 'nerf'
    freqs: Tensor

    def __init__(
        self,
        num_frequencies: int = 10,
        *,
        include_xyz: bool = False,
        device: str | torch.device = 'cpu',
        batch_size: int = 8192,
    ) -> None:
        """Initialize a new NeRFLocationEncoder instance.

        Args:
            num_frequencies: Number of dyadic frequency octaves ``2^0 .. 2^(F-1)``.
            include_xyz: If True, prepend raw 3D unit-sphere coordinates.
            device: Target device for inference.
            batch_size: Batch size used by :meth:`encode`.

        Raises:
            TypeError: If *num_frequencies* is not an int or *include_xyz* is not a
                bool.
            ValueError: If *num_frequencies* is less than 1.
        """
        super().__init__(device=device, batch_size=batch_size)
        if isinstance(num_frequencies, bool) or not isinstance(num_frequencies, int):
            raise TypeError(f'num_frequencies must be an int, got {num_frequencies!r}.')
        if num_frequencies < 1:
            raise ValueError(f'num_frequencies must be >= 1, got {num_frequencies}.')
        if not isinstance(include_xyz, bool):
            raise TypeError(f'include_xyz must be a bool, got {include_xyz!r}.')

        self.num_frequencies = num_frequencies
        self.include_xyz = include_xyz
        self.embed_dim = 6 * num_frequencies + (3 if include_xyz else 0)
        freqs = (2.0 ** torch.arange(num_frequencies, dtype=torch.float32)) * math.pi
        self.register_buffer('freqs', freqs, persistent=False)
        self.to(self.device)

    def forward(
        self, coords: Tensor, year: Tensor | None = None, month: Tensor | None = None
    ) -> Tensor:
        """Encode ``(lon, lat)`` coordinates with dyadic Fourier features on XYZ.

        Args:
            coords: Coordinate tensor of shape ``(B, 2)`` in ``(lon, lat)`` degrees.
            year: Unused temporal parameter.
            month: Unused temporal parameter.

        Returns:
            Float32 feature tensor of shape ``(B, embed_dim)``.
        """
        xyz = lonlat_to_xyz(coords[:, 0], coords[:, 1]).to(dtype=torch.float32)
        angles = xyz.unsqueeze(-1) * self.freqs.to(device=xyz.device, dtype=xyz.dtype)
        batch_size = xyz.shape[0]
        sin_feats = torch.sin(angles).reshape(batch_size, 3 * self.num_frequencies)
        cos_feats = torch.cos(angles).reshape(batch_size, 3 * self.num_frequencies)
        if self.include_xyz:
            return torch.cat((xyz, sin_feats, cos_feats), dim=-1)
        return torch.cat((sin_feats, cos_feats), dim=-1)


class SphericalHarmonicLocationEncoder(LocationEncoder):
    """Low-degree real orthonormal spherical harmonic encoder (degrees 0 to 3)."""

    name: str = 'spherical-harmonics'

    def __init__(
        self,
        degree: int = 3,
        *,
        device: str | torch.device = 'cpu',
        batch_size: int = 8192,
    ) -> None:
        """Initialize a new SphericalHarmonicLocationEncoder instance.

        Args:
            degree: Maximum spherical harmonic degree in ``{0, 1, 2, 3}``.
            device: Target device for inference.
            batch_size: Batch size used by :meth:`encode`.

        Raises:
            TypeError: If *degree* is not an int.
            ValueError: If *degree* is outside ``[0, 3]``.
        """
        super().__init__(device=device, batch_size=batch_size)
        if isinstance(degree, bool) or not isinstance(degree, int):
            raise TypeError(f'degree must be an int, got {degree!r}.')
        if degree < 0 or degree > 3:
            raise ValueError(f'degree must be between 0 and 3, got {degree}.')

        self.degree = degree
        self.embed_dim = (degree + 1) ** 2

    def forward(
        self, coords: Tensor, year: Tensor | None = None, month: Tensor | None = None
    ) -> Tensor:
        """Encode ``(lon, lat)`` coordinates into real orthonormal harmonics.

        Args:
            coords: Coordinate tensor of shape ``(B, 2)`` in ``(lon, lat)`` degrees.
            year: Unused temporal parameter.
            month: Unused temporal parameter.

        Returns:
            Float32 feature tensor of shape ``(B, (degree + 1) ** 2)``.
        """
        xyz = lonlat_to_xyz(coords[:, 0], coords[:, 1]).to(dtype=torch.float32)
        x = xyz[:, 0]
        y = xyz[:, 1]
        z = xyz[:, 2]

        cols: list[Tensor] = [torch.full_like(x, 0.28209479177387814)]
        if self.degree >= 1:
            cols.extend(
                (
                    -0.4886025119029199 * y,
                    0.4886025119029199 * z,
                    -0.4886025119029199 * x,
                )
            )
        if self.degree >= 2:
            cols.extend(
                (
                    1.0925484305920792 * x * y,
                    -1.0925484305920792 * y * z,
                    0.31539156525252005 * (3.0 * z * z - 1.0),
                    -1.0925484305920792 * x * z,
                    0.5462742152960396 * (x * x - y * y),
                )
            )
        if self.degree >= 3:
            cols.extend(
                (
                    -0.5900435899266435 * y * (3.0 * x * x - y * y),
                    2.890611442640554 * x * y * z,
                    -0.4570457994644658 * y * (5.0 * z * z - 1.0),
                    0.3731763325901154 * z * (5.0 * z * z - 3.0),
                    -0.4570457994644658 * x * (5.0 * z * z - 1.0),
                    1.445305721320277 * z * (x * x - y * y),
                    -0.5900435899266435 * x * (x * x - 3.0 * y * y),
                )
            )
        return torch.stack(cols, dim=-1)


class AnalyticSphericalHarmonicLocationEncoder(LocationEncoder):
    """High-degree analytic spherical harmonic encoder using stable recurrence."""

    name: str = 'sh40'

    def __init__(
        self,
        legendre_polys: int = 41,
        *,
        device: str | torch.device = 'cpu',
        batch_size: int = 8192,
    ) -> None:
        """Initialize a new AnalyticSphericalHarmonicLocationEncoder instance.

        Args:
            legendre_polys: Number of Legendre polynomial degrees ``L`` (``L=41``
                covers degrees 0 through 40 with 1,681 features).
            device: Target device for inference.
            batch_size: Batch size used by :meth:`encode`.
        """
        super().__init__(device=device, batch_size=batch_size)
        self.posenc = StableAnalyticSphericalHarmonics(
            legendre_polys=legendre_polys,
            compute_dtype=torch.float64,
            output_dtype=torch.float32,
        )
        self.legendre_polys = legendre_polys
        self.embed_dim = self.posenc.embedding_dim
        self.to(self.device)

    def forward(
        self, coords: Tensor, year: Tensor | None = None, month: Tensor | None = None
    ) -> Tensor:
        """Encode ``(lon, lat)`` coordinates into high-degree spherical harmonics.

        Args:
            coords: Coordinate tensor of shape ``(B, 2)`` in ``(lon, lat)`` degrees.
            year: Unused temporal parameter.
            month: Unused temporal parameter.

        Returns:
            Float32 harmonic feature tensor of shape ``(B, legendre_polys ** 2)``.
        """
        return self.posenc(coords).to(dtype=torch.float32)


class FrequencyRFFLocationEncoder(LocationEncoder):
    """Frequency-ordered multiscale random Fourier features on the 3D Earth sphere.

    Draws deterministic Gaussian frequencies whose physical length scales descend
    geometrically from *max_scale_km* to *min_scale_km*, yielding coarse-to-fine
    interleaved ``[cos, sin]`` feature pairs.
    """

    name: str = 'rff3072'
    weights_matrix: Tensor

    def __init__(
        self,
        num_frequencies: int = 1536,
        *,
        max_scale_km: float = 5000.0,
        min_scale_km: float = 20.0,
        seed: int = 0,
        device: str | torch.device = 'cpu',
        batch_size: int = 8192,
    ) -> None:
        """Initialize a new FrequencyRFFLocationEncoder instance.

        Args:
            num_frequencies: Number of sin/cos frequency pairs (output dimension is
                ``2 * num_frequencies``).
            max_scale_km: Coarsest physical length scale in kilometers.
            min_scale_km: Finest physical length scale in kilometers.
            seed: Deterministic NumPy RNG seed for drawing frequency vectors.
            device: Target device for inference.
            batch_size: Batch size used by :meth:`encode`.

        Raises:
            TypeError: If *num_frequencies* or *seed* is not an int.
            ValueError: If *num_frequencies* < 1 or scales are non-positive.
        """
        super().__init__(device=device, batch_size=batch_size)
        if isinstance(num_frequencies, bool) or not isinstance(num_frequencies, int):
            raise TypeError(f'num_frequencies must be an int, got {num_frequencies!r}.')
        if num_frequencies < 1:
            raise ValueError(f'num_frequencies must be >= 1, got {num_frequencies}.')
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise TypeError(f'seed must be an int, got {seed!r}.')
        if max_scale_km <= 0.0 or min_scale_km <= 0.0:
            raise ValueError('max_scale_km and min_scale_km must be positive.')

        self.num_frequencies = num_frequencies
        self.max_scale_km = max_scale_km
        self.min_scale_km = min_scale_km
        self.seed = seed
        self.embed_dim = 2 * num_frequencies

        scales = np.geomspace(max_scale_km, min_scale_km, num_frequencies)
        rng = np.random.default_rng(seed)
        w = rng.normal(0.0, 1.0, size=(3, num_frequencies)) / scales[None, :]
        self.register_buffer(
            'weights_matrix', torch.from_numpy(w.astype(np.float32)), persistent=False
        )
        self.to(self.device)

    def forward(
        self, coords: Tensor, year: Tensor | None = None, month: Tensor | None = None
    ) -> Tensor:
        """Encode ``(lon, lat)`` coordinates into coarse-to-fine spherical RFFs.

        Args:
            coords: Coordinate tensor of shape ``(B, 2)`` in ``(lon, lat)`` degrees.
            year: Unused temporal parameter.
            month: Unused temporal parameter.

        Returns:
            Float32 feature tensor of shape ``(B, 2 * num_frequencies)``.
        """
        xyz_km = (
            lonlat_to_xyz(coords[:, 0], coords[:, 1]).to(dtype=torch.float32)
            * _EARTH_RADIUS_KM
        )
        proj = xyz_km @ self.weights_matrix.to(device=xyz_km.device, dtype=xyz_km.dtype)
        return torch.stack((torch.cos(proj), torch.sin(proj)), dim=-1).reshape(
            xyz_km.shape[0], self.embed_dim
        )


__all__ = (
    'AnalyticSphericalHarmonicLocationEncoder',
    'FrequencyRFFLocationEncoder',
    'NeRFLocationEncoder',
    'SinCosLocationEncoder',
    'SphericalHarmonicLocationEncoder',
    'XYZLocationEncoder',
)
