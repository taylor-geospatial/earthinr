# Copyright (c) Taylor Geospatial. All rights reserved.
# Licensed under the MIT License.

"""Optional ONNX Runtime inference backend for location encoders."""

import os
from collections.abc import Sequence
from typing import Any, Literal

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor

from .._types import Path
from .._utils import lazy_import
from ._base import LocationEncoder
from ._utils import hf_download

ONNXPrecision = Literal['fp32', 'fp16', 'int8', 'int4']

_ZOO_REPO_ID = 'taylor-geospatial/neuralftw-zoo'

_PRECISION_FILENAMES: dict[str, str] = {
    'fp32': 'model.onnx',
    'fp16': 'model_fp16.onnx',
    'int8': 'model_int8.onnx',
    'int4': 'model_int4_bs32.onnx',
}

_ALL_PRECISIONS: tuple[str, ...] = ('fp32', 'fp16', 'int8', 'int4')
_FP32_ONLY: tuple[str, ...] = ('fp32',)
_MIND_PRECISIONS: tuple[str, ...] = ('int8', 'int4')

#: Mapping from preset name to
#: ``(repo_id, filename, embed_dim, accepts_year, default_year)``.
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
    'mind': (_ZOO_REPO_ID, 'mind/model_int8.onnx', 64, False, 2021),
    'mind-3072': (_ZOO_REPO_ID, 'mind/model_int8.onnx', 3072, False, 2021),
}

#: Available quantized ONNX precisions per preset in ``taylor-geospatial/neuralftw-zoo``.
ONNX_PRESET_PRECISIONS: dict[str, tuple[str, ...]] = {
    preset: (
        _MIND_PRECISIONS
        if preset in {'mind', 'mind-3072'}
        else _FP32_ONLY
        if preset in {'terranova', 'terranova-st'}
        else _ALL_PRECISIONS
    )
    for preset in ONNX_PRESET_SPECS
}

_PRESET_SLICE_DIMS: dict[str, int] = {'mind': 64}
_PRESET_DEFAULT_MONTHS: dict[str, tuple[int, ...]] = {'climplicit': (3, 6, 9, 12)}


class ONNXLocationEncoder(LocationEncoder):
    """ONNX Runtime inference backend for exported location encoders.

    Runs self-contained dynamic-batch ``.onnx`` graphs through
    ``onnxruntime.InferenceSession`` while preserving the full
    :class:`LocationEncoder` interface.
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
        slice_dim: int | None = None,
        accepts_year: bool = False,
        default_year: int = 2020,
        accepts_month: bool = False,
        default_months: Sequence[int] = (3, 6, 9, 12),
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
            embed_dim: Output embedding dimensionality. Inferred from *slice_dim*
                or the ONNX graph output shape when omitted.
            slice_dim: Optional leading Matryoshka prefix width sliced from the
                ONNX graph output (e.g. ``64`` for ``mind``).
            accepts_year: Whether the ONNX graph expects a ``year`` input.
            default_year: Default calendar year broadcast when *accepts_year* is
                True and no *year* input is supplied.
            accepts_month: Whether the ONNX graph expects a ``month`` input
                (e.g. ``climplicit``, returning 256-d for a single month or
                concatenating *default_months* when *month* is omitted).
            default_months: Calendar months concatenated when *accepts_month* is
                True and *month* is omitted.
            providers: Optional explicit ONNX Runtime execution providers. When
                omitted, selects CUDA+CPU for CUDA devices and CPU otherwise.
            device: Target device for provider selection and :meth:`forward`
                tensor outputs.
            batch_size: Maximum number of coordinates encoded per batch.
            name: Human-readable encoder identifier.

        Raises:
            DependencyNotFoundError: If ``onnxruntime`` is not installed.
            ValueError: If neither *model_path* nor both *repo_id* and *filename*
                are provided, if *slice_dim* is not positive, or if
                *default_months* is empty when *accepts_month* is True.
        """
        super().__init__(device=device, batch_size=batch_size)
        ort = lazy_import('onnxruntime')

        if slice_dim is not None and (
            isinstance(slice_dim, bool)
            or not isinstance(slice_dim, int)
            or slice_dim <= 0
        ):
            raise ValueError(f'slice_dim must be a positive int, got {slice_dim!r}.')
        resolved_months = tuple(int(m) for m in default_months)
        if accepts_month and len(resolved_months) == 0:
            raise ValueError(
                'default_months must be non-empty when accepts_month is True.'
            )

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
        self.slice_dim = slice_dim
        self.accepts_year = accepts_year
        self.default_year = default_year
        self.accepts_month = accepts_month
        self.default_months = resolved_months
        self.name = name

        inputs = self.session.get_inputs()
        outputs = self.session.get_outputs()
        self._coords_name: str = inputs[0].name
        next_idx = 1
        if accepts_year:
            self._year_name: str = inputs[next_idx].name
            next_idx += 1
        else:
            self._year_name = 'year'
        if accepts_month:
            self._month_name: str = inputs[next_idx].name
        else:
            self._month_name = 'month'
        self._output_name: str = outputs[0].name

        raw_dim = int(outputs[0].shape[-1])
        if embed_dim is not None:
            self.embed_dim = embed_dim
        elif slice_dim is not None:
            self.embed_dim = slice_dim
        elif accepts_month:
            self.embed_dim = raw_dim * len(resolved_months)
        else:
            self.embed_dim = raw_dim

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
            month: Optional 1D array of calendar months (``1..12``). When
                ``accepts_month`` is True and *month* is ``None``, evaluates
                every month in ``default_months`` and concatenates along the
                embedding axis.

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

        if self.accepts_month:
            if month is None:
                parts = []
                for m in self.default_months:
                    feed[self._month_name] = np.full(
                        lon.shape[0], float(m), dtype=np.float32
                    )
                    parts.append(
                        np.asarray(
                            self.session.run([self._output_name], feed)[0],
                            dtype=np.float32,
                        )
                    )
                out = np.concatenate(parts, axis=-1)
            else:
                feed[self._month_name] = np.asarray(month, dtype=np.float32)
                out = np.asarray(
                    self.session.run([self._output_name], feed)[0], dtype=np.float32
                )
        else:
            out = np.asarray(
                self.session.run([self._output_name], feed)[0], dtype=np.float32
            )

        if self.slice_dim is not None:
            out = np.ascontiguousarray(out[:, : self.slice_dim])
        return out

    def forward(
        self, coords: Tensor, year: Tensor | None = None, month: Tensor | None = None
    ) -> Tensor:
        """Encode geographic coordinates into embeddings via ONNX Runtime.

        Args:
            coords: Coordinate tensor of shape ``(B, 2)`` in ``(lon, lat)`` WGS84
                degrees.
            year: Optional tensor of shape ``(B,)`` containing observation years.
            month: Optional tensor of shape ``(B,)`` containing observation months
                (``1..12``).

        Returns:
            Float32 embedding tensor of shape ``(B, D)`` on *coords*'s device.
        """
        coords_np = coords.detach().cpu().to(dtype=torch.float32).numpy()
        year_np = (
            None
            if year is None
            else year.detach().cpu().to(dtype=torch.float32).numpy()
        )
        month_np = (
            None
            if month is None
            else month.detach().cpu().to(dtype=torch.float32).numpy()
        )
        out = self._encode(
            coords_np[:, 0], coords_np[:, 1], year=year_np, month=month_np
        )
        return torch.from_numpy(out).to(device=coords.device, dtype=torch.float32)


def onnx_encoder(
    preset_or_path: Path,
    *,
    precision: ONNXPrecision | None = None,
    dim: int | None = None,
    embed_dim: int | None = None,
    slice_dim: int | None = None,
    accepts_year: bool | None = None,
    default_year: int | None = None,
    accepts_month: bool | None = None,
    default_months: Sequence[int] | None = None,
    providers: Sequence[str] | None = None,
    device: str | torch.device = 'cpu',
    batch_size: int = 8192,
) -> ONNXLocationEncoder:
    """Instantiate an :class:`ONNXLocationEncoder` from a preset name or local file.

    Args:
        preset_or_path: Registered ONNX preset name (a key in
            :data:`ONNX_PRESET_SPECS`) or local path to an ``.onnx`` model file.
        precision: Optional ONNX weight precision for zoo presets
            (``'fp32'``, ``'fp16'``, ``'int8'``, or ``'int4'``). Defaults to
            ``'fp32'`` for standard zoo models and ``'int8'`` for ``mind``
            presets.
        dim: Optional Matryoshka prefix width alias for *slice_dim* (matching
            PyTorch ``mind(..., dim=...)`` and ``hydroloc(..., dim=...)``).
        embed_dim: Optional override for output embedding dimensionality.
        slice_dim: Optional leading Matryoshka prefix width sliced from outputs.
        accepts_year: Optional override indicating whether the model accepts a
            ``year`` input tensor.
        default_year: Optional override for the default calendar year.
        accepts_month: Optional override indicating whether the model accepts a
            ``month`` input tensor.
        default_months: Optional override for default months concatenated when
            *month* is omitted.
        providers: Optional explicit ONNX Runtime execution providers.
        device: Target device for provider selection and tensor outputs.
        batch_size: Maximum number of coordinates encoded per batch.

    Returns:
        Instantiated :class:`ONNXLocationEncoder`.

    Raises:
        ValueError: If *precision* is unsupported or conflicting *dim* /
            *slice_dim* values are supplied.
    """
    if dim is not None and slice_dim is not None and dim != slice_dim:
        raise ValueError(
            f'Specify at most one of dim ({dim}) or slice_dim ({slice_dim}).'
        )
    explicit_slice = dim if slice_dim is None else slice_dim

    if isinstance(preset_or_path, str) and preset_or_path in ONNX_PRESET_SPECS:
        repo_id, default_filename, spec_dim, spec_accepts_year, spec_default_year = (
            ONNX_PRESET_SPECS[preset_or_path]
        )
        if precision is not None:
            allowed = ONNX_PRESET_PRECISIONS[preset_or_path]
            if precision not in allowed:
                raise ValueError(
                    f'Precision {precision!r} is not available for ONNX preset '
                    f'{preset_or_path!r}. Available precisions: {", ".join(allowed)}.'
                )
            model_dir = default_filename.split('/', 1)[0]
            filename = f'{model_dir}/{_PRECISION_FILENAMES[precision]}'
        else:
            filename = default_filename

        resolved_slice = (
            _PRESET_SLICE_DIMS.get(preset_or_path)
            if explicit_slice is None
            else explicit_slice
        )
        spec_months = _PRESET_DEFAULT_MONTHS.get(preset_or_path)
        resolved_accepts_month = (
            spec_months is not None if accepts_month is None else accepts_month
        )
        resolved_months: Sequence[int] = (
            default_months
            if default_months is not None
            else (spec_months if spec_months is not None else (3, 6, 9, 12))
        )
        resolved_embed_dim = (
            embed_dim
            if embed_dim is not None
            else (resolved_slice if explicit_slice is not None else spec_dim)
        )
        return ONNXLocationEncoder(
            repo_id=repo_id,
            filename=filename,
            embed_dim=resolved_embed_dim,
            slice_dim=resolved_slice,
            accepts_year=(spec_accepts_year if accepts_year is None else accepts_year),
            default_year=(spec_default_year if default_year is None else default_year),
            accepts_month=resolved_accepts_month,
            default_months=resolved_months,
            providers=providers,
            device=device,
            batch_size=batch_size,
            name=preset_or_path,
        )
    return ONNXLocationEncoder(
        model_path=preset_or_path,
        embed_dim=embed_dim,
        slice_dim=explicit_slice,
        accepts_year=False if accepts_year is None else accepts_year,
        default_year=2020 if default_year is None else default_year,
        accepts_month=False if accepts_month is None else accepts_month,
        default_months=(3, 6, 9, 12) if default_months is None else default_months,
        providers=providers,
        device=device,
        batch_size=batch_size,
    )


__all__ = (
    'ONNX_PRESET_PRECISIONS',
    'ONNX_PRESET_SPECS',
    'ONNXLocationEncoder',
    'ONNXPrecision',
    'onnx_encoder',
)
