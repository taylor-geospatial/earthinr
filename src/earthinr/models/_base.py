# Copyright (c) Taylor Geospatial. All rights reserved.
# Licensed under the MIT License.

"""Base classes for location encoders and pretrained weights."""

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Self

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor, nn

from .._utils import resolve_device
from ._utils import hf_download


class Weights:
    """Container for pretrained model checkpoint coordinates and metadata."""

    def __init__(
        self,
        *,
        repo_id: str | None = None,
        filename: str | None = None,
        url: str | None = None,
        revision: str | None = None,
        meta: dict[str, Any] | None = None,
    ) -> None:
        """Initialize a new Weights instance.

        Args:
            repo_id: HuggingFace Hub repository identifier.
            filename: Checkpoint filename inside *repo_id*.
            url: Direct download URL if not hosted via HuggingFace Hub file lookup.
            revision: Optional Git revision on HuggingFace Hub.
            meta: Optional metadata dictionary describing architecture and provenance.

        Raises:
            ValueError: If neither *repo_id* nor *url* is specified.
        """
        if repo_id is None and url is None:
            raise ValueError('Weights must specify at least one of repo_id or url.')
        self.repo_id = repo_id
        self.filename = filename
        self.url = url
        self.revision = revision
        self.meta: dict[str, Any] = {} if meta is None else dict(meta)

    def __repr__(self) -> str:
        """Return a developer-friendly string representation."""
        return (
            f'Weights(repo_id={self.repo_id!r}, filename={self.filename!r}, '
            f'url={self.url!r}, revision={self.revision!r}, meta={self.meta!r})'
        )


class WeightsEnum(Enum):
    """Base enumeration for pretrained model weights."""

    @property
    def repo_id(self) -> str | None:
        """HuggingFace repository identifier."""
        return self.value.repo_id

    @property
    def filename(self) -> str | None:
        """Checkpoint filename within the HuggingFace repository."""
        return self.value.filename

    @property
    def url(self) -> str | None:
        """Direct download URL for the checkpoint."""
        return self.value.url

    @property
    def revision(self) -> str | None:
        """Optional HuggingFace repository revision."""
        return self.value.revision

    @property
    def meta(self) -> dict[str, Any]:
        """Metadata dictionary for this weights entry."""
        return self.value.meta

    @classmethod
    def verify(cls, obj: Any) -> Self:
        """Validate and resolve a weights member or member name string.

        Args:
            obj: Enum member or member name string.

        Returns:
            Resolved enum member.

        Raises:
            ValueError: If *obj* is not a valid member or member name.
        """
        if isinstance(obj, cls):
            return obj
        if isinstance(obj, str) and obj in cls.__members__:
            return cls[obj]
        valid = ', '.join(cls.__members__)
        raise ValueError(f'Invalid weights {obj!r}. Expected one of: {valid}.')

    def download(self) -> str:
        """Download the checkpoint file from HuggingFace Hub.

        Returns:
            Local filesystem path to the downloaded checkpoint.

        Raises:
            ValueError: If *repo_id* or *filename* is missing on this weights entry.
        """
        if self.repo_id is None or self.filename is None:
            raise ValueError(
                'Weights must specify both repo_id and filename to download from '
                'HuggingFace Hub.'
            )
        return hf_download(self.repo_id, self.filename, revision=self.revision)


class LocationEncoder(nn.Module, ABC):
    """Abstract base class for coordinate location encoders."""

    #: Human-readable identifier for this encoder.
    name: str = 'location_encoder'

    #: Output embedding dimensionality.
    embed_dim: int = 0

    def __init__(
        self, device: str | torch.device = 'cpu', batch_size: int = 8192
    ) -> None:
        """Initialize a new LocationEncoder instance.

        Args:
            device: Device used when encoding NumPy coordinate arrays.
            batch_size: Maximum number of coordinates encoded per batch.

        Raises:
            TypeError: If *batch_size* is not an integer.
            ValueError: If *batch_size* is not positive or *device* is invalid.
        """
        super().__init__()
        if isinstance(batch_size, bool) or not isinstance(batch_size, int):
            raise TypeError(f'batch_size must be an int, got {batch_size!r}.')
        if batch_size <= 0:
            raise ValueError(f'batch_size must be positive, got {batch_size}.')
        self.device = resolve_device(device)
        self.batch_size = batch_size

    @abstractmethod
    def forward(
        self, coords: Tensor, year: Tensor | None = None, month: Tensor | None = None
    ) -> Tensor:
        """Encode geographic coordinates into embeddings.

        Args:
            coords: Coordinate tensor of shape ``(B, 2)`` in ``(lon, lat)`` WGS84
                degrees.
            year: Optional tensor of shape ``(B,)`` containing observation years.
            month: Optional tensor of shape ``(B,)`` containing observation months
                (1--12).

        Returns:
            Float32 embedding tensor of shape ``(B, D)``.
        """

    def _encode(
        self,
        lon: NDArray[np.floating[Any]],
        lat: NDArray[np.floating[Any]],
        year: NDArray[np.floating[Any]] | None,
        month: NDArray[np.floating[Any]] | None = None,
    ) -> NDArray[np.float32]:
        """Encode a single batch of NumPy coordinates.

        Subclasses backed by non-tensor stores (e.g. Zarr embedding products) may
        override this method directly.

        Args:
            lon: 1D longitude array in degrees.
            lat: 1D latitude array in degrees.
            year: Optional 1D array of years matching *lon* and *lat*.
            month: Optional 1D array of months (1--12) matching *lon* and *lat*.

        Returns:
            Float32 NumPy array of shape ``(B, D)``.
        """
        coords = torch.as_tensor(
            np.stack((lon, lat), axis=-1), dtype=torch.float32, device=self.device
        )
        year_tensor = (
            None
            if year is None
            else torch.as_tensor(year, dtype=torch.float32, device=self.device)
        )
        month_tensor = (
            None
            if month is None
            else torch.as_tensor(month, dtype=torch.float32, device=self.device)
        )
        with torch.inference_mode():
            out = self(coords, year=year_tensor, month=month_tensor)
        return out.detach().cpu().to(dtype=torch.float32).numpy()

    def encode(
        self,
        lon: NDArray[np.floating[Any]],
        lat: NDArray[np.floating[Any]],
        year: NDArray[np.floating[Any]] | None = None,
        month: NDArray[np.floating[Any]] | None = None,
    ) -> NDArray[np.float32]:
        """Encode 1D longitude and latitude arrays in batches.

        Args:
            lon: 1D array of longitudes in degrees.
            lat: 1D array of latitudes in degrees.
            year: Optional 1D array of observation years.
            month: Optional 1D array of observation months (1--12).

        Returns:
            Float32 embedding array of shape ``(N, D)``, or empty shape ``(0, 0)``
            when given empty inputs.

        Raises:
            ValueError: If *lon*, *lat*, *year*, or *month* are not 1D or differ in
                length.
        """
        lon_arr = np.asarray(lon, dtype=np.float64)
        lat_arr = np.asarray(lat, dtype=np.float64)
        if lon_arr.ndim != 1 or lat_arr.ndim != 1:
            raise ValueError('lon and lat must be 1D arrays.')
        if lon_arr.shape[0] != lat_arr.shape[0]:
            raise ValueError(
                f'lon and lat must have the same length, got {lon_arr.shape[0]} and '
                f'{lat_arr.shape[0]}.'
            )

        year_arr: NDArray[np.float64] | None = None
        if year is not None:
            year_arr = np.asarray(year, dtype=np.float64)
            if year_arr.ndim != 1 or year_arr.shape[0] != lon_arr.shape[0]:
                raise ValueError('year must be a 1D array matching lon and lat length.')

        month_arr: NDArray[np.float64] | None = None
        if month is not None:
            month_arr = np.asarray(month, dtype=np.float64)
            if month_arr.ndim != 1 or month_arr.shape[0] != lon_arr.shape[0]:
                raise ValueError(
                    'month must be a 1D array matching lon and lat length.'
                )

        n = lon_arr.shape[0]
        if n == 0:
            return np.empty((0, 0), dtype=np.float32)

        chunks: list[NDArray[np.float32]] = []
        for start in range(0, n, self.batch_size):
            stop = start + self.batch_size
            year_chunk = None if year_arr is None else year_arr[start:stop]
            month_chunk = None if month_arr is None else month_arr[start:stop]
            emb = np.asarray(
                self._encode(
                    lon_arr[start:stop], lat_arr[start:stop], year_chunk, month_chunk
                ),
                dtype=np.float32,
            )
            chunks.append(emb)
        return np.concatenate(chunks, axis=0)


__all__ = ('LocationEncoder', 'Weights', 'WeightsEnum')
