# Copyright (c) Taylor Geospatial. All rights reserved.
# Licensed under the MIT License.

"""Optional ONNX Runtime inference backend for location encoders."""

import os
from collections.abc import Sequence
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor

from .._types import Path
from .._utils import lazy_import
from ._base import LocationEncoder
from ._utils import hf_download

_ZOO_REPO_ID = 'taylor-geospatial/earthinr-zoo'

#: Mapping from preset name to ``(repo_id, filename, embed_dim, accepts_year, default_year)``.
ONNX_PRESET_SPECS: dict[str, tuple[str, str, int, bool, int]] = {
    'satclip': (_ZOO_REPO_ID, 'satclip-vit16-l40/model.onnx', 256, False, 2020),
    'satclip-vit16-l10': (
        _ZOO_REPO_ID,
        'satclip-vit16-l10/model.onnx',
        256,
        False,
        2020,
    ),
    'satclip-resnet18-l40': (
        _ZOO_REPO_ID,
        'satclip-resnet18-l40/model.onnx',
        256,
        False,
        2020,
    ),
    'satclip-resnet18-l10': (
        _ZOO_REPO_ID,
        'satclip-resnet18-l10/model.onnx',
        256,
        False,
        2020,
    ),
    'satclip-resnet50-l40': (
        _ZOO_REPO_ID,
        'satclip-resnet50-l40/model.onnx',
        256,
        False,
        2020,
    ),
    'satclip-resnet50-l10': (
        _ZOO_REPO_ID,
        'satclip-resnet50-l10/model.onnx',
        256,
        False,
        2020,
    ),
    'geoclip': (_ZOO_REPO_ID, 'geoclip/model.onnx', 512, False, 2020),
    'taxabind': (_ZOO_REPO_ID, 'taxabind/model.onnx', 512, False, 2020),
    'sled': (_ZOO_REPO_ID, 'sled-s1-s2-ls/model.onnx', 768, False, 2020),
    'sled-s2': (_ZOO_REPO_ID, 'sled-s2-ls/model.onnx', 768, False, 2020),
    'gair': (_ZOO_REPO_ID, 'gair/model.onnx', 768, False, 2020),
    'sinr': (_ZOO_REPO_ID, 'sinr/model.onnx', 256, False, 2020),
    'climplicit': (_ZOO_REPO_ID, 'climplicit/model.onnx', 1024, False, 2020),
    'tte': (_ZOO_REPO_ID, 'tte/model.onnx', 512, False, 2020),
    'terranova': (_ZOO_REPO_ID, 'terranova-spatial/model.onnx', 256, False, 2020),
    'terranova-st': (_ZOO_REPO_ID, 'terranova-st/model.onnx', 256, True, 2020),
    'csp-inat': (_ZOO_REPO_ID, 'csp-inat/model.onnx', 256, False, 2020),
    'csp-fmow': (_ZOO_REPO_ID, 'csp-fmow/model.onnx', 256, False, 2020),
    'osmgraphclip': (_ZOO_REPO_ID, 'osmgraphclip-ms-l40/model.onnx', 256, False, 2020),
    'osmgraphclip-ms-l10': (
        _ZOO_REPO_ID,
        'osmgraphclip-ms-l10/model.onnx',
        256,
        False,
        2020,
    ),
    'osmgraphclip-a-l40': (
        _ZOO_REPO_ID,
        'osmgraphclip-a-l40/model.onnx',
        256,
        False,
        2020,
    ),
    'osmgraphclip-a-l10': (
        _ZOO_REPO_ID,
        'osmgraphclip-a-l10/model.onnx',
        256,
        False,
        2020,
    ),
    'le-sinr': (_ZOO_REPO_ID, 'le-sinr/model.onnx', 256, False, 2020),
    'gtloc': (_ZOO_REPO_ID, 'gtloc/model.onnx', 512, False, 2020),
    'hydroloc': (_ZOO_REPO_ID, 'hydroloc/model.onnx', 512, False, 2020),
    'geoclosp': (_ZOO_REPO_ID, 'geoclosp-rn/model.onnx', 384, False, 2020),
}


class ONNXLocationEncoder(LocationEncoder):
    """ONNX Runtime inference backend for exported location encoders.

    Runs self-contained dynamic-batch ``.onnx`` graphs through
    ``onnxruntime.InferenceSession`` while preserving the full
    :class:`LocationEncoder` interface.

    .. note::

       Requires the ``onnx`` extra (``pip install earthinr[onnx]``).
    """

    name: str = 'onnx'
    embed_dim: int = 0

    def __init__(
        self,
        model_path: Path | None = None,
        *,
        repo_id: str | None = None,
        filename: str | None = None,
        embed_dim: int | None = None,
        accepts_year: bool = False,
        default_year: int = 2020,
        providers: Sequence[str] | None = None,
        device: str | torch.device = 'cpu',
        batch_size: int = 8192,
        name: str = 'onnx',
    ) -> None:
        """Initialize a new ONNXLocationEncoder instance.

        Args:
            model_path: Local path to an exported ``.onnx`` model file.
            repo_id: HuggingFace Hub repository identifier containing the model.
            filename: File path of the ``.onnx`` model inside *repo_id*.
            embed_dim: Output embedding dimensionality. Inferred from the ONNX
                graph output shape when omitted.
            accepts_year: Whether the ONNX graph expects a second ``year`` input.
            default_year: Default calendar year broadcast when *accepts_year* is
                True and no *year* input is supplied.
            providers: Optional explicit ONNX Runtime execution providers. When
                omitted, selects CUDA+CPU for CUDA devices and CPU otherwise.
            device: Target device for provider selection and :meth:`forward`
                tensor outputs.
            batch_size: Maximum number of coordinates encoded per batch.
            name: Human-readable encoder identifier.

        Raises:
            DependencyNotFoundError: If ``onnxruntime`` is not installed.
            ValueError: If neither *model_path* nor both *repo_id* and *filename*
                are provided.
        """
        super().__init__(device=device, batch_size=batch_size)
        ort = lazy_import('onnxruntime', extra='onnx')

        if model_path is not None:
            resolved_path = os.fspath(model_path)
        elif repo_id is not None and filename is not None:
            resolved_path = hf_download(repo_id, filename)
        else:
            raise ValueError(
                'Either model_path or both repo_id and filename must be provided.'
            )

        if providers is not None:
            resolved_providers = list(providers)
        elif self.device.type == 'cuda':
            resolved_providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        else:
            resolved_providers = ['CPUExecutionProvider']

        self.session: Any = ort.InferenceSession(
            resolved_path, providers=resolved_providers
        )
        self.model_path = resolved_path
        self.providers = tuple(resolved_providers)
        self.accepts_year = accepts_year
        self.default_year = default_year
        self.name = name

        inputs = self.session.get_inputs()
        outputs = self.session.get_outputs()
        self._coords_name: str = inputs[0].name
        self._year_name: str = inputs[1].name if accepts_year else 'year'
        self._output_name: str = outputs[0].name
        self.embed_dim = int(outputs[0].shape[-1]) if embed_dim is None else embed_dim

    def _encode(
        self,
        lon: NDArray[np.floating[Any]],
        lat: NDArray[np.floating[Any]],
        year: NDArray[np.floating[Any]] | None = None,
        month: NDArray[np.floating[Any]] | None = None,
    ) -> NDArray[np.float32]:
        """Encode a single batch of NumPy coordinates via ONNX Runtime.

        Args:
            lon: 1D longitude array in degrees.
            lat: 1D latitude array in degrees.
            year: Optional 1D array of years matching *lon* and *lat*.
            month: Unused temporal input kept for :class:`LocationEncoder` parity.

        Returns:
            Float32 NumPy array of shape ``(B, D)``.
        """
        coords = np.stack((lon, lat), axis=-1).astype(np.float32, copy=False)
        feed: dict[str, NDArray[np.float32]] = {self._coords_name: coords}
        if self.accepts_year:
            if year is None:
                year_arr = np.full(
                    lon.shape[0], float(self.default_year), dtype=np.float32
                )
            else:
                year_arr = np.asarray(year, dtype=np.float32)
            feed[self._year_name] = year_arr
        out = self.session.run([self._output_name], feed)[0]
        return np.asarray(out, dtype=np.float32)

    def forward(
        self, coords: Tensor, year: Tensor | None = None, month: Tensor | None = None
    ) -> Tensor:
        """Encode geographic coordinates into embeddings via ONNX Runtime.

        Args:
            coords: Coordinate tensor of shape ``(B, 2)`` in ``(lon, lat)`` WGS84
                degrees.
            year: Optional tensor of shape ``(B,)`` containing observation years.
            month: Unused temporal input kept for :class:`LocationEncoder` parity.

        Returns:
            Float32 embedding tensor of shape ``(B, D)`` on *coords*'s device.
        """
        coords_np = coords.detach().cpu().to(dtype=torch.float32).numpy()
        year_np = (
            None
            if year is None
            else year.detach().cpu().to(dtype=torch.float32).numpy()
        )
        out = self._encode(coords_np[:, 0], coords_np[:, 1], year=year_np)
        return torch.from_numpy(out).to(device=coords.device, dtype=torch.float32)


def onnx_encoder(
    preset_or_path: Path,
    *,
    embed_dim: int | None = None,
    accepts_year: bool | None = None,
    default_year: int | None = None,
    providers: Sequence[str] | None = None,
    device: str | torch.device = 'cpu',
    batch_size: int = 8192,
) -> ONNXLocationEncoder:
    """Instantiate an :class:`ONNXLocationEncoder` from a preset name or local file.

    Args:
        preset_or_path: Registered ONNX preset name (a key in
            :data:`ONNX_PRESET_SPECS`) or local path to an ``.onnx`` model file.
        embed_dim: Optional override for output embedding dimensionality.
        accepts_year: Optional override indicating whether the model accepts a
            ``year`` input tensor.
        default_year: Optional override for the default calendar year.
        providers: Optional explicit ONNX Runtime execution providers.
        device: Target device for provider selection and tensor outputs.
        batch_size: Maximum number of coordinates encoded per batch.

    Returns:
        Instantiated :class:`ONNXLocationEncoder`.
    """
    if isinstance(preset_or_path, str) and preset_or_path in ONNX_PRESET_SPECS:
        repo_id, filename, spec_dim, spec_accepts_year, spec_default_year = (
            ONNX_PRESET_SPECS[preset_or_path]
        )
        return ONNXLocationEncoder(
            repo_id=repo_id,
            filename=filename,
            embed_dim=spec_dim if embed_dim is None else embed_dim,
            accepts_year=(spec_accepts_year if accepts_year is None else accepts_year),
            default_year=(spec_default_year if default_year is None else default_year),
            providers=providers,
            device=device,
            batch_size=batch_size,
            name=preset_or_path,
        )
    return ONNXLocationEncoder(
        model_path=preset_or_path,
        embed_dim=embed_dim,
        accepts_year=False if accepts_year is None else accepts_year,
        default_year=2020 if default_year is None else default_year,
        providers=providers,
        device=device,
        batch_size=batch_size,
    )


__all__ = ('ONNX_PRESET_SPECS', 'ONNXLocationEncoder', 'onnx_encoder')
