#!/usr/bin/env python3
# Copyright (c) Taylor Geospatial. All rights reserved.
# Licensed under the MIT License.

"""Export, quantize, and verify dynamic-batch ONNX models for EarthINR.

Exports all 26 location encoders hosted in ``taylor-geospatial/earthinr-zoo``
(plus ``mind`` as ``(lon, lat) -> (B, 3072)``) to single-file, self-contained
ONNX graphs (opset 18 by default) with dynamic batch dimension 0, optionally
applies mixed-FP16 or weight-only INT8/INT4 ``MatMulNBits`` quantization while
keeping positional encodings and first ``w0=30`` SIREN/H-SIREN layers in FP32,
verifies numerical parity against PyTorch via ``onnxruntime`` across multiple
batch sizes and diverse global coordinates, optionally checks
``taylor-geospatial/MIND/mind.onnx`` parity, and optionally uploads staged ONNX
files to Hugging Face Hub.

Run with:
    uv run --with onnx --with onnxruntime --with onnxscript \
        python scripts/export_onnx.py --staging-dir /path/to/staging
"""

from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
import math
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import onnx
import onnxruntime as ort
import torch
from huggingface_hub import HfApi, hf_hub_download
from onnx import TensorProto, helper, numpy_helper
from torch import Tensor, nn

from earthinr.models._base import LocationEncoder
from earthinr.models.climplicit import Climplicit_Weights, climplicit
from earthinr.models.csp import CSP_Weights, csp
from earthinr.models.geoclosp import GeoCLOSP_Weights, geoclosp
from earthinr.models.gtloc import GTLoc_Weights, gtloc
from earthinr.models.hydroloc import HydroLoc_Weights, hydroloc
from earthinr.models.lesinr import LESINR_Weights, lesinr
from earthinr.models.mind import MIND_Weights, mind
from earthinr.models.osmgraphclip import OSMGraphCLIP_Weights, osmgraphclip
from earthinr.models.rff_capsule import (
    GAIR_Weights,
    GeoCLIP_Weights,
    SLED_Weights,
    TaxaBind_Weights,
    gair,
    geoclip,
    sled,
    taxabind,
)
from earthinr.models.satclip import (
    POLE_LATITUDE_EPS_DEGREES,
    SatCLIP_Weights,
    StableAnalyticSphericalHarmonics,
    satclip,
)
from earthinr.models.sinr import SINR_Weights, sinr
from earthinr.models.terranova import TerraNova_Weights, terranova
from earthinr.models.tte import TTE_Weights, TTELocationEncoder, tte

ZOO_REPO_ID = 'taylor-geospatial/earthinr-zoo'
MIND_REPO_ID = 'taylor-geospatial/MIND'
DEFAULT_OPSET = 18
DEFAULT_MAX_ABS_TOL = 1e-4
DEFAULT_MIN_COS_SIM = 0.99999
DEFAULT_BLOCK_SIZE = 128
QUANTIZE_CHOICES: tuple[str, ...] = ('none', 'fp16', 'int8', 'int4')
QuantizeMode = Literal['none', 'fp16', 'int8', 'int4']
_DEFAULT_PARITY_THRESHOLDS: dict[str, tuple[float, float]] = {
    'none': (DEFAULT_MAX_ABS_TOL, DEFAULT_MIN_COS_SIM),
    'fp16': (0.25, 0.999),
    'int8': (2.0, 0.99),
    'int4': (15.0, 0.70),
}
TEST_BATCH_SIZES = (1, 7, 32)
_TEMP_OUTPUT_NAME = '_earthinr_embedding_output'


@dataclass(frozen=True)
class ModelSpec:
    """Specification for an EarthINR model to export to ONNX."""

    model_id: str
    builder: Callable[[], LocationEncoder]
    output_dim: int
    temporal: bool = False
    zoo: bool = True


MODEL_SPECS: tuple[ModelSpec, ...] = (
    ModelSpec(
        'satclip-vit16-l40',
        lambda: satclip(weights=SatCLIP_Weights.VIT16_L40, device='cpu'),
        256,
    ),
    ModelSpec(
        'satclip-vit16-l10',
        lambda: satclip(weights=SatCLIP_Weights.VIT16_L10, device='cpu'),
        256,
    ),
    ModelSpec(
        'satclip-resnet18-l40',
        lambda: satclip(weights=SatCLIP_Weights.RESNET18_L40, device='cpu'),
        256,
    ),
    ModelSpec(
        'satclip-resnet18-l10',
        lambda: satclip(weights=SatCLIP_Weights.RESNET18_L10, device='cpu'),
        256,
    ),
    ModelSpec(
        'satclip-resnet50-l40',
        lambda: satclip(weights=SatCLIP_Weights.RESNET50_L40, device='cpu'),
        256,
    ),
    ModelSpec(
        'satclip-resnet50-l10',
        lambda: satclip(weights=SatCLIP_Weights.RESNET50_L10, device='cpu'),
        256,
    ),
    ModelSpec(
        'geoclip', lambda: geoclip(weights=GeoCLIP_Weights.DEFAULT, device='cpu'), 512
    ),
    ModelSpec(
        'taxabind', lambda: taxabind(weights=TaxaBind_Weights.ECOGEO, device='cpu'), 512
    ),
    ModelSpec(
        'sled-s1-s2-ls', lambda: sled(weights=SLED_Weights.S1_S2_LS, device='cpu'), 768
    ),
    ModelSpec(
        'sled-s2-ls', lambda: sled(weights=SLED_Weights.S2_LS, device='cpu'), 768
    ),
    ModelSpec('gair', lambda: gair(weights=GAIR_Weights.DEFAULT, device='cpu'), 768),
    ModelSpec('sinr', lambda: sinr(weights=SINR_Weights.INAT_2023, device='cpu'), 256),
    ModelSpec(
        'climplicit',
        lambda: climplicit(weights=Climplicit_Weights.CHELSA, device='cpu'),
        1024,
    ),
    ModelSpec('tte', lambda: tte(weights=TTE_Weights.DEFAULT, device='cpu'), 512),
    ModelSpec(
        'terranova-spatial',
        lambda: terranova(weights=TerraNova_Weights.SPATIAL, device='cpu'),
        256,
    ),
    ModelSpec(
        'terranova-st',
        lambda: terranova(weights=TerraNova_Weights.SPATIOTEMPORAL, device='cpu'),
        256,
        temporal=True,
    ),
    ModelSpec('csp-inat', lambda: csp(weights=CSP_Weights.INAT, device='cpu'), 256),
    ModelSpec('csp-fmow', lambda: csp(weights=CSP_Weights.FMOW, device='cpu'), 256),
    ModelSpec(
        'osmgraphclip-ms-l40',
        lambda: osmgraphclip(weights=OSMGraphCLIP_Weights.MS_L40, device='cpu'),
        256,
    ),
    ModelSpec(
        'osmgraphclip-ms-l10',
        lambda: osmgraphclip(weights=OSMGraphCLIP_Weights.MS_L10, device='cpu'),
        256,
    ),
    ModelSpec(
        'osmgraphclip-a-l40',
        lambda: osmgraphclip(weights=OSMGraphCLIP_Weights.A_L40, device='cpu'),
        256,
    ),
    ModelSpec(
        'osmgraphclip-a-l10',
        lambda: osmgraphclip(weights=OSMGraphCLIP_Weights.A_L10, device='cpu'),
        256,
    ),
    ModelSpec(
        'le-sinr', lambda: lesinr(weights=LESINR_Weights.DEFAULT, device='cpu'), 256
    ),
    ModelSpec('gtloc', lambda: gtloc(weights=GTLoc_Weights.DEFAULT, device='cpu'), 512),
    ModelSpec(
        'hydroloc',
        lambda: hydroloc(weights=HydroLoc_Weights.DEFAULT, device='cpu'),
        512,
    ),
    ModelSpec(
        'geoclosp-rn',
        lambda: geoclosp(weights=GeoCLOSP_Weights.RN50_L10, device='cpu'),
        384,
    ),
    ModelSpec(
        'mind',
        lambda: mind(
            weights=MIND_Weights.MIND, feature='pooled', dim=3072, device='cpu'
        ),
        3072,
        zoo=False,
    ),
)

MODEL_SPECS_BY_ID: dict[str, ModelSpec] = {spec.model_id: spec for spec in MODEL_SPECS}
ZOO_MODEL_IDS: tuple[str, ...] = tuple(
    spec.model_id for spec in MODEL_SPECS if spec.zoo
)


def _tree_cat(tensors: list[Tensor], *, dim: int = 1, max_arity: int = 4) -> Tensor:
    """Concatenate *tensors* via a balanced tree so every ONNX ``Concat`` has <= *max_arity* inputs.

    WebGPU limits compute shaders to 8 storage buffers per stage (inputs + output),
    so a single wide ``Concat`` (e.g., 10 or 40 inputs) fails pipeline validation.
    """
    current = list(tensors)
    while len(current) > 1:
        next_level: list[Tensor] = []
        for idx in range(0, len(current), max_arity):
            chunk = current[idx : idx + max_arity]
            next_level.append(
                chunk[0] if len(chunk) == 1 else torch.cat(chunk, dim=dim)
            )
        current = next_level
    return current[0]


class _WebGPUAnalyticSphericalHarmonics(nn.Module):
    """Float32, WebGPU/WASM-compatible drop-in for ``StableAnalyticSphericalHarmonics``.

    Replaces the default ``float64`` Legendre recurrence (which emits ``Cos(7)`` /
    ``Sin(7)`` on ``DOUBLE``, unsupported in ``onnxruntime-web``), replaces
    ``torch.flip`` (which emits negative-step ``Slice``, silently miscomputed by
    ``onnxruntime-web`` WebGPU) with ``Gather`` reversal, and keeps every ``Concat``
    node at arity ``<= 4`` (within WebGPU's 8-storage-buffer shader limit), while
    preserving near-pole precision via ``x = -sin(lat)``, ``u = cos(lat)`` and
    degree-reduced azimuths.
    """

    orders: Tensor
    diag_coeffs: Tensor
    subdiag_coeffs: Tensor
    alpha: Tensor
    beta: Tensor

    def __init__(self, source: StableAnalyticSphericalHarmonics) -> None:
        """Initialize from a ``StableAnalyticSphericalHarmonics`` instance."""
        super().__init__()
        self.legendre_polys = source.legendre_polys
        self.embedding_dim = source.embedding_dim
        self.register_buffer(
            'orders', source.orders.to(dtype=torch.float32), persistent=False
        )
        self.register_buffer(
            'diag_coeffs', source.diag_coeffs.to(dtype=torch.float32), persistent=False
        )
        self.register_buffer(
            'subdiag_coeffs',
            source.subdiag_coeffs.to(dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            'alpha', source.alpha.to(dtype=torch.float32), persistent=False
        )
        self.register_buffer(
            'beta', source.beta.to(dtype=torch.float32), persistent=False
        )
        for degree in range(1, self.legendre_polys):
            self.register_buffer(
                f'_rev_{degree}',
                torch.arange(degree - 1, -1, -1, dtype=torch.int64),
                persistent=False,
            )

    def forward(self, coords: Tensor) -> Tensor:
        """Evaluate real spherical harmonics in pure float32 for WebGPU/WASM."""
        n_polys = self.legendre_polys
        lonlat = coords.to(dtype=torch.float32)
        lon = lonlat[:, 0]
        lat = lonlat[:, 1].clamp(
            min=-90.0 + POLE_LATITUDE_EPS_DEGREES, max=90.0 - POLE_LATITUDE_EPS_DEGREES
        )

        # x = cos(deg2rad(lat + 90)) = -sin(deg2rad(lat)), u = sin(deg2rad(lat + 90)) = cos(deg2rad(lat))
        # Avoids float32 cancellation in sqrt(1 - x^2) near poles.
        lat_rad = torch.deg2rad(lat)
        x = -torch.sin(lat_rad)
        u = torch.cos(lat_rad)

        batch_size = lonlat.shape[0]
        order_values = self.orders
        p00 = lonlat.new_full((batch_size,), math.sqrt(1.0 / (4.0 * math.pi)))
        diag_terms: list[Tensor] = [p00]
        if n_polys > 1:
            diag_values = (
                p00.unsqueeze(1)
                * self.diag_coeffs[1:].unsqueeze(0)
                * u.unsqueeze(1).pow(order_values[1:].unsqueeze(0))
            )
            diag_terms.extend(diag_values.unbind(dim=1))

        # Keep each Legendre degree row unpadded at width (degree + 1) to avoid
        # zero-padding allocations and high-arity torch.stack Concat nodes.
        rows: list[Tensor] = [diag_terms[0].unsqueeze(1)]
        if n_polys > 1:
            rows.append(
                torch.cat(
                    (
                        (self.subdiag_coeffs[0] * x * diag_terms[0]).unsqueeze(1),
                        diag_terms[1].unsqueeze(1),
                    ),
                    dim=1,
                )
            )
        for degree in range(2, n_polys):
            recurrence = (
                self.alpha[degree, : degree - 1].unsqueeze(0)
                * x.unsqueeze(1)
                * rows[degree - 1][:, : degree - 1]
                - self.beta[degree, : degree - 1].unsqueeze(0)
                * rows[degree - 2][:, : degree - 1]
            )
            rows.append(
                torch.cat(
                    (
                        recurrence,
                        (
                            self.subdiag_coeffs[degree - 1] * x * diag_terms[degree - 1]
                        ).unsqueeze(1),
                        diag_terms[degree].unsqueeze(1),
                    ),
                    dim=1,
                )
            )

        phases_deg = torch.remainder(
            (lon + 180.0).unsqueeze(1) * order_values.unsqueeze(0), 360.0
        )
        phases = torch.deg2rad(phases_deg)
        sin_terms = torch.sin(phases)
        cos_terms = torch.cos(phases)

        degree_blocks: list[Tensor] = []
        for degree in range(n_polys):
            row = rows[degree]
            center = (math.pi * row[:, 0]).unsqueeze(1)
            if degree == 0:
                degree_blocks.append(center)
                continue
            base = math.sqrt(2.0) * row[:, 1 : degree + 1]
            rev_idx = getattr(self, f'_rev_{degree}')
            negative = torch.index_select(
                base * sin_terms[:, 1 : degree + 1], dim=1, index=rev_idx
            )
            positive = base * cos_terms[:, 1 : degree + 1]
            degree_blocks.append(torch.cat((negative, center, positive), dim=1))

        return _tree_cat(degree_blocks, dim=1, max_arity=4)


class _WebGPUTTEOnnxWrapper(nn.Module):
    """Spatial ONNX wrapper for TTE that swaps ``(lon, lat)`` without negative-step Slice."""

    def __init__(self, encoder: TTELocationEncoder) -> None:
        """Initialize the WebGPU-safe TTE wrapper."""
        super().__init__()
        self.model = encoder.model

    def forward(self, coords: Tensor) -> Tensor:
        """Encode ``(lon, lat)`` degrees by swapping columns via ``torch.stack``."""
        latlon = torch.stack((coords[:, 1], coords[:, 0]), dim=-1)
        return self.model(latlon)


def _prepare_export_wrapper(
    encoder: LocationEncoder, *, temporal: bool
) -> tuple[nn.Module, bool]:
    """Build a WebGPU/WASM-compatible ONNX export wrapper without mutating *encoder*.

    Replaces any ``StableAnalyticSphericalHarmonics`` submodules on a shallow copy
    with :class:`_WebGPUAnalyticSphericalHarmonics` (float32, no ``DOUBLE`` ops,
    no negative-step ``Slice``) and avoids ``coords.flip(-1)`` for TTE.

    Returns:
        Tuple ``(export_wrapper, adapted)`` where ``adapted`` is True when the
        export graph uses a float32/WebGPU-safe rewrite distinct from *encoder*.
    """
    if isinstance(encoder, TTELocationEncoder):
        wrapper: nn.Module = _WebGPUTTEOnnxWrapper(encoder)
        wrapper.eval()
        return wrapper, True

    has_sh = any(
        isinstance(m, StableAnalyticSphericalHarmonics) for m in encoder.modules()
    )
    if has_sh:
        adapted_encoder = copy.deepcopy(encoder)
        for parent in adapted_encoder.modules():
            for child_name, child in list(parent.named_children()):
                if isinstance(child, StableAnalyticSphericalHarmonics):
                    setattr(
                        parent, child_name, _WebGPUAnalyticSphericalHarmonics(child)
                    )
        adapted_encoder.eval()
        wrapper = (
            _SpatiotemporalOnnxWrapper(adapted_encoder)
            if temporal
            else _SpatialOnnxWrapper(adapted_encoder)
        )
        wrapper.eval()
        return wrapper, True

    wrapper = (
        _SpatiotemporalOnnxWrapper(encoder)
        if temporal
        else _SpatialOnnxWrapper(encoder)
    )
    wrapper.eval()
    return wrapper, False


class _SpatialOnnxWrapper(nn.Module):
    """Thin wrapper exposing a single ``coords -> embedding`` signature."""

    def __init__(self, encoder: LocationEncoder) -> None:
        """Initialize the spatial ONNX wrapper.

        Args:
            encoder: Loaded location encoder in evaluation mode.
        """
        super().__init__()
        self.encoder = encoder

    def forward(self, coords: Tensor) -> Tensor:
        """Encode ``(lon, lat)`` coordinates in WGS84 degrees.

        Args:
            coords: Coordinate tensor of shape ``(B, 2)``.

        Returns:
            Float32 embedding tensor of shape ``(B, D)``.
        """
        return self.encoder(coords)


class _SpatiotemporalOnnxWrapper(nn.Module):
    """Thin wrapper exposing a ``(coords, year) -> embedding`` signature."""

    def __init__(self, encoder: LocationEncoder) -> None:
        """Initialize the spatiotemporal ONNX wrapper.

        Args:
            encoder: Loaded spatiotemporal location encoder in evaluation mode.
        """
        super().__init__()
        self.encoder = encoder

    def forward(self, coords: Tensor, year: Tensor) -> Tensor:
        """Encode ``(lon, lat)`` coordinates and calendar years.

        Args:
            coords: Coordinate tensor of shape ``(B, 2)``.
            year: Calendar year tensor of shape ``(B,)``.

        Returns:
            Float32 embedding tensor of shape ``(B, D)``.
        """
        return self.encoder(coords, year=year)


def sha256_file(path: Path) -> str:
    """Compute the hex SHA-256 digest of a file on disk.

    Args:
        path: Path to the file.

    Returns:
        Lowercase hexadecimal SHA-256 digest.
    """
    digest = hashlib.sha256()
    with path.open('rb') as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _rename_graph_output_ssa_safe(
    model_proto: onnx.ModelProto, target_output_name: str = 'embedding'
) -> None:
    """Rename the single graph output without colliding with internal node outputs.

    PyTorch dynamo export names internal ``nn.Embedding`` node outputs
    ``'embedding'``, which causes an ONNX SSA validation error if the graph
    output is also named ``'embedding'`` directly. This helper renames any
    colliding internal value first, then renames the graph output cleanly.

    Args:
        model_proto: Loaded ONNX model protobuf modified in-place.
        target_output_name: Desired graph output tensor name.
    """
    graph = model_proto.graph
    if len(graph.output) != 1:
        raise ValueError(f'Expected 1 ONNX graph output, found {len(graph.output)}.')

    orig_out = graph.output[0].name
    if orig_out == target_output_name:
        return

    used_names: set[str] = set()
    for node in graph.node:
        used_names.update(node.output)
    used_names.update(inp.name for inp in graph.input)
    used_names.update(init.name for init in graph.initializer)

    if target_output_name in used_names:
        replacement = f'_{target_output_name}_internal'
        while replacement in used_names:
            replacement += '_'
        for node in graph.node:
            for idx, name in enumerate(node.input):
                if name == target_output_name:
                    node.input[idx] = replacement
            for idx, name in enumerate(node.output):
                if name == target_output_name:
                    node.output[idx] = replacement
        for value_info in graph.value_info:
            if value_info.name == target_output_name:
                value_info.name = replacement

    for node in graph.node:
        for idx, name in enumerate(node.input):
            if name == orig_out:
                node.input[idx] = target_output_name
        for idx, name in enumerate(node.output):
            if name == orig_out:
                node.output[idx] = target_output_name
    for value_info in graph.value_info:
        if value_info.name == orig_out:
            value_info.name = target_output_name
    graph.output[0].name = target_output_name


def make_test_batch(batch_size: int) -> tuple[np.ndarray, np.ndarray]:
    """Construct diverse global ``(lon, lat)`` coordinates and calendar years.

    Covers the equator, near-poles (``+/-89.9`` deg), antimeridian (``+/-180``
    deg), major global cities, rural/oceanic locations, and deterministic random
    global samples across years ``2017..2025``.

    Args:
        batch_size: Number of test samples to generate.

    Returns:
        Tuple ``(coords, years)`` of float32 arrays with shapes ``(B, 2)`` and
        ``(B,)``.
    """
    anchor_coords = np.array(
        [
            [-122.4194, 37.7749],  # San Francisco
            [0.0, 0.0],  # Equator / Prime Meridian
            [0.0, 89.9],  # Near North Pole
            [0.0, -89.9],  # Near South Pole
            [180.0, 12.5],  # +Antimeridian
            [-180.0, -33.8688],  # -Antimeridian / Sydney latitude
            [139.6917, 35.6895],  # Tokyo
            [-58.3816, -34.6037],  # Buenos Aires
            [11.5, 23.5],  # Rural Sahara
            [-62.2159, -3.4653],  # Amazon rainforest
            [36.8219, -1.2921],  # Nairobi
            [28.9784, 41.0082],  # Istanbul
            [-73.9857, 40.7484],  # New York
            [151.2093, -33.8688],  # Sydney
            [-149.9003, 61.2181],  # Anchorage
            [180.0, 89.9],  # Antimeridian + North Pole corner
            [-180.0, -89.9],  # Antimeridian + South Pole corner
            [77.2090, 28.6139],  # New Delhi
        ],
        dtype=np.float32,
    )
    anchor_years = np.array(
        [2020.0, 2017.0, 2018.0, 2019.0, 2021.0, 2022.0, 2025.0], dtype=np.float32
    )

    if batch_size <= len(anchor_coords):
        coords = anchor_coords[:batch_size].copy()
    else:
        rng = np.random.default_rng(seed=2026 + batch_size)
        extra = batch_size - len(anchor_coords)
        extra_lon = rng.uniform(-180.0, 180.0, size=(extra, 1))
        extra_lat = rng.uniform(-89.9, 89.9, size=(extra, 1))
        coords = np.concatenate(
            [anchor_coords, np.concatenate([extra_lon, extra_lat], axis=1)], axis=0
        ).astype(np.float32)

    if batch_size <= len(anchor_years):
        years = anchor_years[:batch_size].copy()
    else:
        years = np.linspace(2017.0, 2025.0, batch_size, dtype=np.float32)
    return coords, years


def _verify_io_schema(
    sess: ort.InferenceSession, spec: ModelSpec
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Validate ONNX Runtime input/output names, dtypes, and dynamic batch axes."""
    inputs = sess.get_inputs()
    outputs = sess.get_outputs()

    expected_inputs = ['coords', 'year'] if spec.temporal else ['coords']
    actual_inputs = [inp.name for inp in inputs]
    if actual_inputs != expected_inputs:
        raise ValueError(
            f'{spec.model_id}: expected inputs {expected_inputs}, got {actual_inputs}'
        )

    for inp in inputs:
        if inp.type != 'tensor(float)':
            raise ValueError(
                f'{spec.model_id}: input {inp.name!r} has type {inp.type!r}, '
                "expected 'tensor(float)'"
            )
        if not isinstance(inp.shape[0], str):
            raise TypeError(
                f'{spec.model_id}: input {inp.name!r} dim 0 is not dynamic: {inp.shape}'
            )
    if inputs[0].shape[1] != 2:
        raise ValueError(
            f'{spec.model_id}: coords input shape expected [batch, 2], '
            f'got {inputs[0].shape}'
        )
    if spec.temporal and len(inputs[1].shape) != 1:
        raise ValueError(
            f'{spec.model_id}: year input shape expected [batch], got {inputs[1].shape}'
        )

    if len(outputs) != 1 or outputs[0].name != 'embedding':
        raise ValueError(
            f'{spec.model_id}: expected single output "embedding", '
            f'got {[out.name for out in outputs]}'
        )
    out = outputs[0]
    if out.type != 'tensor(float)':
        raise ValueError(
            f'{spec.model_id}: output type {out.type!r}, expected "tensor(float)"'
        )
    if not isinstance(out.shape[0], str) or out.shape[1] != spec.output_dim:
        raise ValueError(
            f'{spec.model_id}: output shape expected [batch, {spec.output_dim}], '
            f'got {out.shape}'
        )

    in_meta = [{'name': i.name, 'type': i.type, 'shape': i.shape} for i in inputs]
    out_meta = [{'name': o.name, 'type': o.type, 'shape': o.shape} for o in outputs]
    return in_meta, out_meta


def _cosine_similarities(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compute per-row cosine similarity in float64."""
    a64 = a.astype(np.float64)
    b64 = b.astype(np.float64)
    dots = np.sum(a64 * b64, axis=-1)
    norms = np.linalg.norm(a64, axis=-1) * np.linalg.norm(b64, axis=-1)
    return np.where(norms > 0.0, dots / np.maximum(norms, 1e-12), 1.0)


def _verify_webgpu_compatible(onnx_path: Path, model_id: str) -> None:
    """Assert that an ONNX graph contains no ``DOUBLE`` ops or negative-step ``Slice``."""
    proto = onnx.load(str(onnx_path), load_external_data=False)
    graph = proto.graph

    double_items: list[str] = []
    for tensor in graph.initializer:
        if tensor.data_type == TensorProto.DOUBLE:
            double_items.append(f'initializer:{tensor.name}')
    for vi in [*graph.input, *graph.output, *graph.value_info]:
        if (
            vi.type.HasField('tensor_type')
            and vi.type.tensor_type.elem_type == TensorProto.DOUBLE
        ):
            double_items.append(f'value:{vi.name}')
    for node in graph.node:
        for attr in node.attribute:
            if (
                node.op_type == 'Cast'
                and attr.name == 'to'
                and attr.i == TensorProto.DOUBLE
            ):
                double_items.append(f'Cast:{node.name}')
            if attr.HasField('t') and attr.t.data_type == TensorProto.DOUBLE:
                double_items.append(f'{node.op_type}:{node.name}')
    if double_items:
        raise AssertionError(
            f'{model_id}: found {len(double_items)} DOUBLE graph elements '
            f'(incompatible with onnxruntime-web), e.g. {double_items[:5]}'
        )

    inits = {t.name: numpy_helper.to_array(t) for t in graph.initializer}
    neg_slices: list[str] = []
    for node in graph.node:
        if node.op_type == 'Slice' and len(node.input) >= 5:
            steps_name = node.input[4]
            if steps_name in inits and bool((inits[steps_name] < 0).any()):
                neg_slices.append(node.name)
    if neg_slices:
        raise AssertionError(
            f'{model_id}: found {len(neg_slices)} negative-step Slice nodes '
            f'(miscomputed by onnxruntime-web WebGPU), e.g. {neg_slices[:5]}'
        )

    wide_concats = [
        (node.name, len(node.input))
        for node in graph.node
        if node.op_type == 'Concat' and len(node.input) > 6
    ]
    if wide_concats:
        raise AssertionError(
            f'{model_id}: found {len(wide_concats)} Concat nodes with >6 inputs '
            f'(exceeds WebGPU maxStorageBuffersPerShaderStage=8), e.g. {wide_concats[:5]}'
        )


def verify_onnx_parity(
    onnx_path: Path,
    wrapper: nn.Module,
    spec: ModelSpec,
    *,
    max_abs_tol: float = DEFAULT_MAX_ABS_TOL,
    min_cos_sim: float = DEFAULT_MIN_COS_SIM,
    reference_wrapper: nn.Module | None = None,
) -> dict[str, Any]:
    """Verify ONNX graph validity, dynamic schema, and PyTorch/ORT numerical parity.

    Args:
        onnx_path: Path to the exported self-contained ``model.onnx`` file.
        wrapper: PyTorch wrapper module matching the ONNX input/output contract.
        spec: Model specification.
        max_abs_tol: Maximum allowed absolute difference across all test batches.
        min_cos_sim: Minimum allowed per-sample cosine similarity.
        reference_wrapper: Optional unmodified reference PyTorch wrapper (e.g.,
            float64 spherical harmonics) checked for cosine parity alongside *wrapper*.

    Returns:
        Dictionary of parity metrics and I/O schema metadata.
    """
    onnx.checker.check_model(str(onnx_path))
    _verify_webgpu_compatible(onnx_path, spec.model_id)

    external_data_path = Path(f'{onnx_path}.data')
    if external_data_path.exists():
        raise RuntimeError(
            f'{spec.model_id}: unexpected external data file {external_data_path}'
        )

    sess = ort.InferenceSession(str(onnx_path), providers=['CPUExecutionProvider'])
    in_meta, out_meta = _verify_io_schema(sess, spec)

    max_abs_diff = 0.0
    sum_abs_diff = 0.0
    total_elements = 0
    min_cosine_sim = 1.0
    ref_max_abs_diff = 0.0
    ref_min_cosine_sim = 1.0
    per_batch: dict[str, dict[str, float]] = {}

    wrapper.eval()
    if reference_wrapper is not None:
        reference_wrapper.eval()
    for batch_size in TEST_BATCH_SIZES:
        coords_np, years_np = make_test_batch(batch_size)
        coords_t = torch.from_numpy(coords_np)
        years_t = torch.from_numpy(years_np)

        with torch.inference_mode():
            pt_out = (
                wrapper(coords_t, years_t).detach().cpu().numpy()
                if spec.temporal
                else wrapper(coords_t).detach().cpu().numpy()
            )
            ref_out = (
                (
                    reference_wrapper(coords_t, years_t).detach().cpu().numpy()
                    if spec.temporal
                    else reference_wrapper(coords_t).detach().cpu().numpy()
                )
                if reference_wrapper is not None
                else None
            )

        feed: dict[str, np.ndarray] = {'coords': coords_np}
        if spec.temporal:
            feed['year'] = years_np
        ort_out = sess.run(['embedding'], feed)[0]

        if pt_out.shape != (batch_size, spec.output_dim):
            raise ValueError(
                f'{spec.model_id} (B={batch_size}): PyTorch output shape {pt_out.shape}'
            )
        if ort_out.shape != pt_out.shape:
            raise ValueError(
                f'{spec.model_id} (B={batch_size}): ORT output shape {ort_out.shape} '
                f'!= PyTorch {pt_out.shape}'
            )
        if not np.isfinite(ort_out).all():
            raise ValueError(
                f'{spec.model_id} (B={batch_size}): non-finite values in ORT output'
            )

        abs_diff = np.abs(pt_out - ort_out)
        b_max = float(np.max(abs_diff))
        b_mean = float(np.mean(abs_diff))
        b_cos = float(np.min(_cosine_similarities(pt_out, ort_out)))

        max_abs_diff = max(max_abs_diff, b_max)
        sum_abs_diff += float(np.sum(abs_diff))
        total_elements += abs_diff.size
        min_cosine_sim = min(min_cosine_sim, b_cos)
        per_batch[f'B={batch_size}'] = {
            'max_abs_diff': b_max,
            'mean_abs_diff': b_mean,
            'min_cosine_sim': b_cos,
        }
        if ref_out is not None:
            ref_abs = np.abs(ref_out - ort_out)
            ref_max_abs_diff = max(ref_max_abs_diff, float(np.max(ref_abs)))
            ref_min_cosine_sim = min(
                ref_min_cosine_sim,
                float(np.min(_cosine_similarities(ref_out, ort_out))),
            )

    mean_abs_diff = sum_abs_diff / max(total_elements, 1)

    if math.isnan(max_abs_diff) or max_abs_diff > max_abs_tol:
        raise AssertionError(
            f'{spec.model_id}: max_abs_diff={max_abs_diff:.3e} exceeds '
            f'tolerance {max_abs_tol:.3e}'
        )
    if math.isnan(min_cosine_sim) or min_cosine_sim < min_cos_sim:
        raise AssertionError(
            f'{spec.model_id}: min_cosine_sim={min_cosine_sim:.8f} below '
            f'threshold {min_cos_sim:.8f}'
        )
    if reference_wrapper is not None and (
        math.isnan(ref_min_cosine_sim) or ref_min_cosine_sim < min_cos_sim
    ):
        raise AssertionError(
            f'{spec.model_id}: ref_min_cosine_sim={ref_min_cosine_sim:.8f} below '
            f'threshold {min_cos_sim:.8f}'
        )

    result: dict[str, Any] = {
        'inputs': in_meta,
        'outputs': out_meta,
        'max_abs_diff': max_abs_diff,
        'mean_abs_diff': mean_abs_diff,
        'min_cosine_sim': min_cosine_sim,
        'per_batch': per_batch,
    }
    if reference_wrapper is not None:
        result['ref_max_abs_diff'] = ref_max_abs_diff
        result['ref_min_cosine_sim'] = ref_min_cosine_sim
    return result


def _quantized_filename(quantize: str, block_size: int) -> str:
    """Return the staged ONNX filename for a quantization mode and block size."""
    if quantize == 'none':
        return 'model.onnx'
    if quantize == 'fp16':
        return 'model_fp16.onnx'
    if block_size == DEFAULT_BLOCK_SIZE:
        return f'model_{quantize}.onnx'
    return f'model_{quantize}_bs{block_size}.onnx'


def _get_protected_linear_nodes(model_id: str, model: onnx.ModelProto) -> set[str]:
    """Return ``Gemm``/``MatMul`` node names that must remain in FP32.

    Protects all coordinate positional-encoding ``MatMul`` nodes (e.g., RFF
    frequency projections), dynamic-weight matmuls, raw low-dimensional input
    layers, and the first ``w0=30`` SIREN/H-SIREN layer across ``mind``,
    ``climplicit``, ``satclip-*``, and ``osmgraphclip-*``.
    """
    inits = {t.name: tuple(t.dims) for t in model.graph.initializer}
    protected: set[str] = set()

    for node in model.graph.node:
        if node.op_type not in ('Gemm', 'MatMul'):
            continue
        if node.op_type == 'MatMul':
            protected.add(node.name)
            continue

        w_name = node.input[1]
        shape = inits.get(w_name)
        if shape is None:
            protected.add(node.name)
            continue

        is_protected_siren_first = (
            (model_id == 'mind' and (shape == (3072, 2) or 'first' in w_name))
            or (
                model_id == 'climplicit'
                and ('layers.0.' in w_name or shape == (512, 4))
            )
            or (
                model_id.startswith(('satclip-', 'osmgraphclip-'))
                and 'layers.0.' in w_name
            )
            or (len(shape) == 2 and min(shape) <= 4)
        )
        if is_protected_siren_first:
            protected.add(node.name)

    return protected


def _convert_hidden_gemm_to_fp16(
    fp32_path: Path, out_path: Path, exclude_gemm_names: set[str]
) -> None:
    """Convert only non-excluded hidden ``Gemm`` nodes to FP16.

    Wraps each selected ``Gemm`` in ``Cast(FP32->FP16) -> Gemm(FP16) ->
    Cast(FP16->FP32)`` and stores its weight/bias initializers in ``float16``,
    leaving positional encodings, trigonometric operations, and first ``w0=30``
    SIREN layers strictly untouched in FP32 (avoiding the inter-blocked-node
    FP16 cast regression in ``onnxconverter_common.float16``).
    """
    model = onnx.load(str(fp32_path), load_external_data=True)
    graph = model.graph
    inits = {t.name: t for t in graph.initializer}
    fp16_inits: dict[str, str] = {}
    new_nodes: list[onnx.NodeProto] = []

    for node in graph.node:
        if node.op_type == 'Gemm' and node.name not in exclude_gemm_names:
            x_in = node.input[0]
            x_16 = f'{node.name}__x_fp16'
            new_nodes.append(
                helper.make_node(
                    'Cast',
                    [x_in],
                    [x_16],
                    to=TensorProto.FLOAT16,
                    name=f'{node.name}__cast_in',
                )
            )
            new_inputs = [x_16]
            for idx in range(1, len(node.input)):
                inp = node.input[idx]
                if inp in inits:
                    if inp not in fp16_inits:
                        arr = numpy_helper.to_array(inits[inp]).astype(np.float16)
                        name16 = f'{inp}__fp16'
                        graph.initializer.append(
                            numpy_helper.from_array(arr, name=name16)
                        )
                        fp16_inits[inp] = name16
                    new_inputs.append(fp16_inits[inp])
                else:
                    new_inputs.append(inp)
            y_16 = f'{node.name}__y_fp16'
            g_node = helper.make_node('Gemm', new_inputs, [y_16], name=node.name)
            g_node.attribute.extend(node.attribute)
            new_nodes.append(g_node)
            new_nodes.append(
                helper.make_node(
                    'Cast',
                    [y_16],
                    [node.output[0]],
                    to=TensorProto.FLOAT,
                    name=f'{node.name}__cast_out',
                )
            )
        else:
            new_nodes.append(node)

    graph.ClearField('node')
    graph.node.extend(new_nodes)
    used = {inp for n in graph.node for inp in n.input}
    kept = [t for t in graph.initializer if t.name in used]
    graph.ClearField('initializer')
    graph.initializer.extend(kept)
    onnx.save_model(model, str(out_path), save_as_external_data=False)
    del model
    gc.collect()


def _gemm_to_matmul_add(
    model: onnx.ModelProto, exclude_nodes: set[str]
) -> tuple[onnx.ModelProto, list[str]]:
    """Rewrite non-excluded ``Gemm(A, B, C)`` nodes into ``MatMul(A, B^T) + Add(C)``.

    Deduplicates transposed weight initializers when multiple ``Gemm`` nodes
    share the same weight tensor (required for ``climplicit``'s four monthly
    passes) and prunes unreferenced original weights.
    """
    rewritten = copy.deepcopy(model)
    graph = rewritten.graph
    init_map = {t.name: t for t in graph.initializer}
    transposed_cache: dict[str, str] = {}
    referenced_orig_weights: set[str] = set()
    replaced_orig_weights: set[str] = set()
    new_nodes: list[onnx.NodeProto] = []
    target_matmul_names: list[str] = []

    for node in graph.node:
        if node.op_type != 'Gemm' or node.name in exclude_nodes:
            for inp in node.input:
                if inp in init_map:
                    referenced_orig_weights.add(inp)
            new_nodes.append(node)
            continue

        attrs = {a.name: helper.get_attribute_value(a) for a in node.attribute}
        alpha = float(attrs.get('alpha', 1.0))
        beta = float(attrs.get('beta', 1.0))
        trans_a = int(attrs.get('transA', 0))
        trans_b = int(attrs.get('transB', 0))
        w_name = node.input[1]
        if (
            trans_a != 0
            or abs(alpha - 1.0) > 1e-6
            or abs(beta - 1.0) > 1e-6
            or w_name not in init_map
        ):
            for inp in node.input:
                if inp in init_map:
                    referenced_orig_weights.add(inp)
            new_nodes.append(node)
            continue

        if w_name not in transposed_cache:
            w_arr = numpy_helper.to_array(init_map[w_name])
            w_t = np.ascontiguousarray(w_arr.T if trans_b == 1 else w_arr)
            t_name = f'{w_name}__matmul_T'
            graph.initializer.append(numpy_helper.from_array(w_t, name=t_name))
            transposed_cache[w_name] = t_name
        replaced_orig_weights.add(w_name)
        t_name = transposed_cache[w_name]

        mm_name = f'{node.name}__matmul'
        target_matmul_names.append(mm_name)
        if len(node.input) >= 3 and node.input[2]:
            mm_out = f'{node.output[0]}__mm_out'
            mm_node = helper.make_node(
                'MatMul', [node.input[0], t_name], [mm_out], name=mm_name
            )
            add_node = helper.make_node(
                'Add',
                [mm_out, node.input[2]],
                [node.output[0]],
                name=f'{node.name}__add',
            )
            referenced_orig_weights.add(node.input[2])
            new_nodes.extend([mm_node, add_node])
        else:
            mm_node = helper.make_node(
                'MatMul', [node.input[0], t_name], [node.output[0]], name=mm_name
            )
            new_nodes.append(mm_node)

    graph.ClearField('node')
    graph.node.extend(new_nodes)

    dead = replaced_orig_weights - referenced_orig_weights
    if dead:
        kept_inits = [t for t in graph.initializer if t.name not in dead]
        graph.ClearField('initializer')
        graph.initializer.extend(kept_inits)

    return rewritten, target_matmul_names


def _deduplicate_initializers(model: onnx.ModelProto) -> int:
    """Deduplicate identical constant initializers (e.g., shared ``MatMulNBits`` weights)."""
    graph = model.graph
    seen: dict[tuple[int, tuple[int, ...], bytes], str] = {}
    rename: dict[str, str] = {}
    kept: list[onnx.TensorProto] = []
    for tensor in graph.initializer:
        arr = numpy_helper.to_array(tensor)
        key = (tensor.data_type, tuple(tensor.dims), arr.tobytes())
        if key in seen:
            rename[tensor.name] = seen[key]
        else:
            seen[key] = tensor.name
            kept.append(tensor)
    if not rename:
        return 0
    for node in graph.node:
        for idx, inp in enumerate(node.input):
            if inp in rename:
                node.input[idx] = rename[inp]
    graph.ClearField('initializer')
    graph.initializer.extend(kept)
    return len(rename)


def _quantize_matmul_nbits(
    fp32_path: Path,
    out_path: Path,
    exclude_nodes: set[str],
    *,
    bits: int,
    block_size: int = DEFAULT_BLOCK_SIZE,
) -> dict[str, int]:
    """Quantize non-excluded hidden linear layers to weight-only ``MatMulNBits``."""
    from onnxruntime.quantization import matmul_nbits_quantizer

    fp32_model = onnx.load(str(fp32_path), load_external_data=True)
    mm_model, target_mm_names = _gemm_to_matmul_add(fp32_model, exclude_nodes)
    target_mm_set = set(target_mm_names)
    all_mm_exclude = {
        node.name
        for node in mm_model.graph.node
        if node.op_type == 'MatMul' and node.name not in target_mm_set
    } | exclude_nodes

    quant = matmul_nbits_quantizer.MatMulNBitsQuantizer(
        model=mm_model,
        bits=bits,
        block_size=block_size,
        is_symmetric=False,
        nodes_to_exclude=sorted(all_mm_exclude),
    )
    quant.process()
    q_model = quant.model.model
    dedup_count = _deduplicate_initializers(q_model)
    onnx.save_model(q_model, str(out_path), save_as_external_data=False)
    nbits_nodes = sum(1 for node in q_model.graph.node if node.op_type == 'MatMulNBits')
    del fp32_model, mm_model, quant, q_model
    gc.collect()
    return {'nbits_nodes': nbits_nodes, 'dedup_inits': dedup_count}


def export_model(
    spec: ModelSpec,
    staging_dir: Path,
    *,
    opset: int = DEFAULT_OPSET,
    quantize: str = 'none',
    block_size: int = DEFAULT_BLOCK_SIZE,
    max_abs_tol: float | None = None,
    min_cos_sim: float | None = None,
    skip_existing: bool = False,
) -> dict[str, Any]:
    """Export one EarthINR location encoder to ONNX and verify parity.

    Args:
        spec: Model specification to export.
        staging_dir: Root directory where ``<model_id>/<filename>`` is written.
        opset: Target ONNX opset version.
        quantize: Post-export quantization mode (``'none'``, ``'fp16'``,
            ``'int8'``, or ``'int4'``).
        block_size: Block size for ``MatMulNBits`` weight-only quantization
            (default ``128``; use ``32`` for browser WebGPU INT4).
        max_abs_tol: Maximum allowed absolute difference vs PyTorch (defaults
            to a mode-appropriate threshold when omitted).
        min_cos_sim: Minimum allowed cosine similarity vs PyTorch (defaults to
            a mode-appropriate threshold when omitted).
        skip_existing: If True and the target ONNX file already exists, skip
            export and run parity verification directly on the existing file.

    Returns:
        Manifest dictionary with file size, SHA-256, opset, and parity metrics.
    """
    if quantize not in _DEFAULT_PARITY_THRESHOLDS:
        raise ValueError(
            f'Unsupported quantize mode {quantize!r}; expected one of {QUANTIZE_CHOICES}'
        )
    default_abs_tol, default_cos_sim = _DEFAULT_PARITY_THRESHOLDS[quantize]
    resolved_abs_tol = default_abs_tol if max_abs_tol is None else max_abs_tol
    resolved_cos_sim = default_cos_sim if min_cos_sim is None else min_cos_sim

    out_dir = staging_dir / spec.model_id
    out_dir.mkdir(parents=True, exist_ok=True)
    onnx_filename = _quantized_filename(quantize, block_size)
    onnx_path = out_dir / onnx_filename
    ext_data_path = Path(f'{onnx_path}.data')

    encoder = spec.builder()
    encoder.eval()
    ref_wrapper: nn.Module = (
        _SpatiotemporalOnnxWrapper(encoder)
        if spec.temporal
        else _SpatialOnnxWrapper(encoder)
    )
    ref_wrapper.eval()
    wrapper, adapted = _prepare_export_wrapper(encoder, temporal=spec.temporal)

    quant_meta: dict[str, Any] | None = None
    if not (skip_existing and onnx_path.exists()):
        # Use dummy batch size > 1 so torch.export does not specialize dim 0 to 1.
        dummy_coords_np, dummy_years_np = make_test_batch(4)
        dummy_coords = torch.from_numpy(dummy_coords_np)
        dummy_years = torch.from_numpy(dummy_years_np)
        batch_dim = torch.export.Dim('batch', min=1)

        if spec.temporal:
            args: tuple[Tensor, ...] = (dummy_coords, dummy_years)
            input_names = ['coords', 'year']
            dynamic_shapes: dict[str, Any] = {
                'coords': {0: batch_dim},
                'year': {0: batch_dim},
            }
        else:
            args = (dummy_coords,)
            input_names = ['coords']
            dynamic_shapes = {'coords': {0: batch_dim}}

        fp32_path = onnx_path if quantize == 'none' else out_dir / '.tmp_fp32.onnx'
        fp32_ext_path = Path(f'{fp32_path}.data')
        for p in (ext_data_path, fp32_ext_path):
            if p.exists():
                p.unlink()

        try:
            with torch.no_grad():
                torch.onnx.export(
                    wrapper,
                    args,
                    str(fp32_path),
                    opset_version=opset,
                    input_names=input_names,
                    output_names=[_TEMP_OUTPUT_NAME],
                    dynamic_shapes=dynamic_shapes,
                    external_data=False,
                )

            proto = onnx.load(str(fp32_path), load_external_data=True)
            _rename_graph_output_ssa_safe(proto, target_output_name='embedding')
            onnx.save_model(proto, str(fp32_path), save_as_external_data=False)
            if fp32_ext_path.exists():
                fp32_ext_path.unlink()

            if quantize != 'none':
                protected = _get_protected_linear_nodes(spec.model_id, proto)
                del proto
                gc.collect()
                if quantize == 'fp16':
                    _convert_hidden_gemm_to_fp16(fp32_path, onnx_path, protected)
                    quant_meta = {'protected_nodes': sorted(protected)}
                else:
                    bits = 8 if quantize == 'int8' else 4
                    nbits_meta = _quantize_matmul_nbits(
                        fp32_path,
                        onnx_path,
                        protected,
                        bits=bits,
                        block_size=block_size,
                    )
                    quant_meta = {
                        'bits': bits,
                        'block_size': block_size,
                        'protected_nodes': sorted(protected),
                        **nbits_meta,
                    }
            else:
                del proto
                gc.collect()
        finally:
            if quantize != 'none':
                for tmp in (fp32_path, fp32_ext_path):
                    if tmp.exists():
                        tmp.unlink()

    proto_meta = onnx.load(str(onnx_path), load_external_data=False)
    opsets = {imp.domain: imp.version for imp in proto_meta.opset_import}
    actual_opset = int(opsets.get('', opset))
    del proto_meta

    parity = verify_onnx_parity(
        onnx_path,
        wrapper,
        spec,
        max_abs_tol=resolved_abs_tol,
        min_cos_sim=resolved_cos_sim,
        reference_wrapper=ref_wrapper if adapted else None,
    )

    size_bytes = onnx_path.stat().st_size
    digest = sha256_file(onnx_path)

    del wrapper, ref_wrapper, encoder
    gc.collect()

    record: dict[str, Any] = {
        'model_id': spec.model_id,
        'path': str(onnx_path),
        'opset': actual_opset,
        'quantize': quantize,
        'size_bytes': size_bytes,
        'sha256': digest,
        **parity,
    }
    if quant_meta is not None:
        record['quantization'] = quant_meta
    return record


def check_mind_onnx() -> dict[str, Any]:
    """Verify numerical parity between ``earthinr`` MIND and upstream ``mind.onnx``.

    Note that ``taylor-geospatial/MIND/mind.onnx`` takes native ``(lat, lon)``
    input named ``latlon`` and outputs ``embedding`` of shape ``(B, 3072)``,
    while ``earthinr`` accepts universal ``(lon, lat)`` coordinates and flips
    them internally.

    Returns:
        Dictionary summarizing schema and PyTorch-vs-ORT parity for MIND.
    """
    onnx_path = Path(hf_hub_download(MIND_REPO_ID, 'mind.onnx'))
    hf_hub_download(MIND_REPO_ID, 'mind.onnx.data')

    encoder = mind(weights=MIND_Weights.MIND, feature='pooled', device='cpu')
    encoder.eval()
    sess = ort.InferenceSession(str(onnx_path), providers=['CPUExecutionProvider'])

    max_abs_diff = 0.0
    sum_abs_diff = 0.0
    total_elements = 0
    min_cosine_sim = 1.0

    for batch_size in TEST_BATCH_SIZES:
        coords_lonlat, _ = make_test_batch(batch_size)
        coords_latlon = coords_lonlat[:, ::-1].copy()
        with torch.inference_mode():
            pt_out = encoder(torch.from_numpy(coords_lonlat)).detach().cpu().numpy()
        ort_out = sess.run(['embedding'], {'latlon': coords_latlon})[0]
        abs_diff = np.abs(pt_out - ort_out)
        max_abs_diff = max(max_abs_diff, float(np.max(abs_diff)))
        sum_abs_diff += float(np.sum(abs_diff))
        total_elements += abs_diff.size
        min_cosine_sim = min(
            min_cosine_sim, float(np.min(_cosine_similarities(pt_out, ort_out)))
        )

    return {
        'repo_id': MIND_REPO_ID,
        'file': 'mind.onnx (+ mind.onnx.data)',
        'inputs': [
            {'name': i.name, 'type': i.type, 'shape': i.shape}
            for i in sess.get_inputs()
        ],
        'outputs': [
            {'name': o.name, 'type': o.type, 'shape': o.shape}
            for o in sess.get_outputs()
        ],
        'max_abs_diff': max_abs_diff,
        'mean_abs_diff': sum_abs_diff / max(total_elements, 1),
        'min_cosine_sim': min_cosine_sim,
    }


def upload_onnx_models(
    staging_dir: Path,
    repo_id: str,
    model_ids: list[str],
    *,
    opset: int = DEFAULT_OPSET,
    filename: str = 'model.onnx',
) -> tuple[str, dict[str, dict[str, Any]]]:
    """Upload staged ``<model_id>/<filename>`` files to Hugging Face Hub.

    Uses ``allow_patterns=[f'*/{filename}']`` so no existing ``README.md``,
    ``model.safetensors``, ``config.json``, or concurrent model directories are
    modified.

    Args:
        staging_dir: Directory containing ``<model_id>/<filename>`` files.
        repo_id: Target Hugging Face Hub model repository.
        model_ids: List of expected model IDs to verify after upload.
        opset: ONNX opset version recorded in the Hub commit message.
        filename: ONNX filename within each model directory.

    Returns:
        Tuple ``(commit_sha, remote_info)`` where ``remote_info`` maps each
        ``model_id`` to its remote size and LFS SHA-256 on the Hub.
    """
    api = HfApi()
    commit_info = api.upload_folder(
        repo_id=repo_id,
        repo_type='model',
        folder_path=str(staging_dir),
        allow_patterns=[f'*/{filename}'],
        commit_message=(
            f'Add dynamic-batch ONNX exports ({filename}, opset {opset}) for '
            f'{len(model_ids)} models'
        ),
    )
    commit_sha = str(commit_info.oid)

    tree = api.list_repo_tree(
        repo_id=repo_id, repo_type='model', revision=commit_sha, recursive=True
    )
    remote_info: dict[str, dict[str, Any]] = {}
    target_paths = {f'{mid}/{filename}': mid for mid in model_ids}
    for entry in tree:
        if entry.path in target_paths:
            mid = target_paths[entry.path]
            lfs = getattr(entry, 'lfs', None)
            remote_info[mid] = {
                'size': getattr(entry, 'size', None),
                'lfs_sha256': getattr(lfs, 'sha256', None) if lfs is not None else None,
            }
    return commit_sha, remote_info


def main() -> None:
    """CLI entry point for exporting, verifying, and uploading ONNX models."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--staging-dir',
        type=Path,
        required=True,
        help='Directory where <model_id>/<model*.onnx> files and manifest are staged.',
    )
    parser.add_argument(
        '--models',
        nargs='+',
        default=None,
        choices=list(MODEL_SPECS_BY_ID),
        help=(
            'Subset of model IDs to export (defaults to all 26 earthinr-zoo '
            'models; pass "mind" to export MIND).'
        ),
    )
    parser.add_argument(
        '--opset',
        type=int,
        default=DEFAULT_OPSET,
        help=f'Target ONNX opset version (default: {DEFAULT_OPSET}).',
    )
    parser.add_argument(
        '--quantize',
        choices=list(QUANTIZE_CHOICES),
        default='none',
        help=(
            'Post-export quantization mode: "none" (FP32), "fp16" (hidden-only '
            'mixed FP16), "int8" (weight-only MatMulNBits), or "int4" '
            '(weight-only MatMulNBits). Default: none.'
        ),
    )
    parser.add_argument(
        '--block-size',
        type=int,
        default=DEFAULT_BLOCK_SIZE,
        help=(
            'Block size for int8/int4 MatMulNBits quantization (default: '
            f'{DEFAULT_BLOCK_SIZE}; use 32 for browser WebGPU int4).'
        ),
    )
    parser.add_argument(
        '--max-abs-tol',
        type=float,
        default=None,
        help=(
            'Maximum allowed abs diff vs PyTorch (defaults to '
            f'{DEFAULT_MAX_ABS_TOL} for none, relaxed for quantized modes).'
        ),
    )
    parser.add_argument(
        '--min-cos-sim',
        type=float,
        default=None,
        help=(
            'Minimum allowed cosine similarity vs PyTorch (defaults to '
            f'{DEFAULT_MIN_COS_SIM} for none, relaxed for quantized modes).'
        ),
    )
    parser.add_argument(
        '--skip-existing',
        action='store_true',
        help='Skip export if the target ONNX file already exists and only verify.',
    )
    parser.add_argument(
        '--check-mind',
        action='store_true',
        help='Also verify parity of earthinr MIND against taylor-geospatial/MIND.',
    )
    parser.add_argument(
        '--upload',
        action='store_true',
        help='Upload staged ONNX files to Hugging Face Hub.',
    )
    parser.add_argument(
        '--repo-id',
        default=ZOO_REPO_ID,
        help=f'Target Hugging Face Hub repo (default: {ZOO_REPO_ID}).',
    )
    args = parser.parse_args()

    if args.block_size < 16 or args.block_size % 16 != 0:
        parser.error('--block-size must be a positive multiple of 16 (e.g. 32 or 128).')

    staging_dir: Path = args.staging_dir
    staging_dir.mkdir(parents=True, exist_ok=True)

    selected_ids = args.models if args.models is not None else list(ZOO_MODEL_IDS)
    records: list[dict[str, Any]] = []

    for model_id in selected_ids:
        spec = MODEL_SPECS_BY_ID[model_id]
        quant_label = (
            f'{args.quantize}, bs={args.block_size}'
            if args.quantize in ('int8', 'int4')
            else args.quantize
        )
        print(
            f'Exporting & verifying {model_id} (opset {args.opset}, '
            f'quantize={quant_label})...',
            flush=True,
        )
        rec = export_model(
            spec,
            staging_dir,
            opset=args.opset,
            quantize=args.quantize,
            block_size=args.block_size,
            max_abs_tol=args.max_abs_tol,
            min_cos_sim=args.min_cos_sim,
            skip_existing=args.skip_existing,
        )
        records.append(rec)
        size_mb = rec['size_bytes'] / (1024 * 1024)
        ref_str = (
            f' | ref_max_diff={rec["ref_max_abs_diff"]:.2e}'
            f' | ref_min_cos={rec["ref_min_cosine_sim"]:.8f}'
            if 'ref_max_abs_diff' in rec
            else ''
        )
        print(
            f'  OK {model_id:22s} | {size_mb:7.2f} MiB | opset={rec["opset"]} | '
            f'max_diff={rec["max_abs_diff"]:.2e} | mean_diff={rec["mean_abs_diff"]:.2e} | '
            f'min_cos={rec["min_cosine_sim"]:.8f}{ref_str} | sha256={rec["sha256"][:16]}...',
            flush=True,
        )

    mind_report: dict[str, Any] | None = None
    if args.check_mind:
        print('\nChecking taylor-geospatial/MIND/mind.onnx parity...', flush=True)
        mind_report = check_mind_onnx()
        print(
            f'  MIND | max_diff={mind_report["max_abs_diff"]:.2e} | '
            f'mean_diff={mind_report["mean_abs_diff"]:.2e} | '
            f'min_cos={mind_report["min_cosine_sim"]:.8f}',
            flush=True,
        )

    manifest: dict[str, Any] = {
        'repo_id': args.repo_id,
        'opset': args.opset,
        'quantize': args.quantize,
        'block_size': args.block_size if args.quantize in ('int8', 'int4') else None,
        'models': records,
        'mind_check': mind_report,
    }

    if args.upload:
        onnx_filename = _quantized_filename(args.quantize, args.block_size)
        print(
            f'\nUploading {len(records)} ONNX models ({onnx_filename}) '
            f'to {args.repo_id}...',
            flush=True,
        )
        commit_sha, remote_info = upload_onnx_models(
            staging_dir,
            args.repo_id,
            [r['model_id'] for r in records],
            opset=args.opset,
            filename=onnx_filename,
        )
        manifest['hub_commit_sha'] = commit_sha
        for rec in records:
            mid = rec['model_id']
            if mid in remote_info:
                rec['hub_size'] = remote_info[mid]['size']
                rec['hub_lfs_sha256'] = remote_info[mid]['lfs_sha256']
        print(f'Uploaded cleanly at Hub commit {commit_sha}', flush=True)

    manifest_path = staging_dir / 'onnx_manifest.json'
    manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')

    total_mb = sum(r['size_bytes'] for r in records) / (1024 * 1024)
    print(
        f'\nCompleted {len(records)} ONNX models ({total_mb:.2f} MiB total). '
        f'Manifest written to {manifest_path}',
        flush=True,
    )


if __name__ == '__main__':
    main()
