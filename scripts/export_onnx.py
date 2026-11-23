#!/usr/bin/env python3
# Copyright (c) Taylor Geospatial. All rights reserved.
# Licensed under the MIT License.

"""Export and verify dynamic-batch ONNX models for taylor-geospatial/earthinr-zoo.

Exports all 18 location encoders hosted in ``taylor-geospatial/earthinr-zoo`` to
single-file, self-contained ONNX graphs (opset 18 by default) with dynamic batch
dimension 0, verifies strict numerical parity against PyTorch via
``onnxruntime`` across multiple batch sizes and diverse global coordinates,
optionally checks ``taylor-geospatial/MIND/mind.onnx`` parity, and optionally
uploads ``<model_id>/model.onnx`` files to Hugging Face Hub.

Run with:
    uv run --with onnx --with onnxruntime --with onnxscript \
        python scripts/export_onnx.py --staging-dir /path/to/staging
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort
import torch
from huggingface_hub import HfApi, hf_hub_download
from torch import Tensor, nn

from earthinr.models._base import LocationEncoder
from earthinr.models.climplicit import Climplicit_Weights, climplicit
from earthinr.models.csp import CSP_Weights, csp
from earthinr.models.mind import MIND_Weights, mind
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
from earthinr.models.satclip import SatCLIP_Weights, satclip
from earthinr.models.sinr import SINR_Weights, sinr
from earthinr.models.terranova import TerraNova_Weights, terranova
from earthinr.models.tte import TTE_Weights, tte

ZOO_REPO_ID = 'taylor-geospatial/earthinr-zoo'
MIND_REPO_ID = 'taylor-geospatial/MIND'
DEFAULT_OPSET = 18
DEFAULT_MAX_ABS_TOL = 1e-4
DEFAULT_MIN_COS_SIM = 0.99999
TEST_BATCH_SIZES = (1, 7, 32)
_TEMP_OUTPUT_NAME = '_earthinr_embedding_output'


@dataclass(frozen=True)
class ModelSpec:
    """Specification for an ``earthinr-zoo`` model to export to ONNX."""

    model_id: str
    builder: Callable[[], LocationEncoder]
    output_dim: int
    temporal: bool = False


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
)

MODEL_SPECS_BY_ID: dict[str, ModelSpec] = {spec.model_id: spec for spec in MODEL_SPECS}


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


def verify_onnx_parity(
    onnx_path: Path,
    wrapper: nn.Module,
    spec: ModelSpec,
    *,
    max_abs_tol: float = DEFAULT_MAX_ABS_TOL,
    min_cos_sim: float = DEFAULT_MIN_COS_SIM,
) -> dict[str, Any]:
    """Verify ONNX graph validity, dynamic schema, and PyTorch/ORT numerical parity.

    Args:
        onnx_path: Path to the exported self-contained ``model.onnx`` file.
        wrapper: PyTorch wrapper module matching the ONNX input/output contract.
        spec: Model specification.
        max_abs_tol: Maximum allowed absolute difference across all test batches.
        min_cos_sim: Minimum allowed per-sample cosine similarity.

    Returns:
        Dictionary of parity metrics and I/O schema metadata.
    """
    onnx.checker.check_model(str(onnx_path))

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
    per_batch: dict[str, dict[str, float]] = {}

    wrapper.eval()
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

    return {
        'inputs': in_meta,
        'outputs': out_meta,
        'max_abs_diff': max_abs_diff,
        'mean_abs_diff': mean_abs_diff,
        'min_cosine_sim': min_cosine_sim,
        'per_batch': per_batch,
    }


def export_model(
    spec: ModelSpec,
    staging_dir: Path,
    *,
    opset: int = DEFAULT_OPSET,
    max_abs_tol: float = DEFAULT_MAX_ABS_TOL,
    min_cos_sim: float = DEFAULT_MIN_COS_SIM,
    skip_existing: bool = False,
) -> dict[str, Any]:
    """Export one ``earthinr-zoo`` location encoder to ONNX and verify parity.

    Args:
        spec: Model specification to export.
        staging_dir: Root directory where ``<model_id>/model.onnx`` is written.
        opset: Target ONNX opset version.
        max_abs_tol: Maximum allowed absolute difference vs PyTorch.
        min_cos_sim: Minimum allowed cosine similarity vs PyTorch.
        skip_existing: If True and ``model.onnx`` already exists, skip export
            and run parity verification directly on the existing file.

    Returns:
        Manifest dictionary with file size, SHA-256, opset, and parity metrics.
    """
    out_dir = staging_dir / spec.model_id
    out_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = out_dir / 'model.onnx'
    ext_data_path = out_dir / 'model.onnx.data'

    encoder = spec.builder()
    encoder.eval()
    wrapper: nn.Module = (
        _SpatiotemporalOnnxWrapper(encoder)
        if spec.temporal
        else _SpatialOnnxWrapper(encoder)
    )
    wrapper.eval()

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

        if ext_data_path.exists():
            ext_data_path.unlink()

        with torch.no_grad():
            torch.onnx.export(
                wrapper,
                args,
                str(onnx_path),
                opset_version=opset,
                input_names=input_names,
                output_names=[_TEMP_OUTPUT_NAME],
                dynamic_shapes=dynamic_shapes,
                external_data=False,
            )

        proto = onnx.load(str(onnx_path), load_external_data=True)
        _rename_graph_output_ssa_safe(proto, target_output_name='embedding')
        onnx.save_model(proto, str(onnx_path), save_as_external_data=False)
        if ext_data_path.exists():
            ext_data_path.unlink()
        del proto
        gc.collect()

    proto_meta = onnx.load(str(onnx_path), load_external_data=False)
    opsets = {imp.domain: imp.version for imp in proto_meta.opset_import}
    actual_opset = int(opsets.get('', opset))
    del proto_meta

    parity = verify_onnx_parity(
        onnx_path, wrapper, spec, max_abs_tol=max_abs_tol, min_cos_sim=min_cos_sim
    )

    size_bytes = onnx_path.stat().st_size
    digest = sha256_file(onnx_path)

    del wrapper, encoder
    gc.collect()

    return {
        'model_id': spec.model_id,
        'path': str(onnx_path),
        'opset': actual_opset,
        'size_bytes': size_bytes,
        'sha256': digest,
        **parity,
    }


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
    staging_dir: Path, repo_id: str, model_ids: list[str], *, opset: int = DEFAULT_OPSET
) -> tuple[str, dict[str, dict[str, Any]]]:
    """Upload staged ``<model_id>/model.onnx`` files to Hugging Face Hub.

    Uses ``allow_patterns=['*/model.onnx']`` so no existing ``README.md``,
    ``model.safetensors``, ``config.json``, or concurrent model directories are
    modified.

    Args:
        staging_dir: Directory containing ``<model_id>/model.onnx`` files.
        repo_id: Target Hugging Face Hub model repository.
        model_ids: List of expected model IDs to verify after upload.
        opset: ONNX opset version recorded in the Hub commit message.

    Returns:
        Tuple ``(commit_sha, remote_info)`` where ``remote_info`` maps each
        ``model_id`` to its remote size and LFS SHA-256 on the Hub.
    """
    api = HfApi()
    commit_info = api.upload_folder(
        repo_id=repo_id,
        repo_type='model',
        folder_path=str(staging_dir),
        allow_patterns=['*/model.onnx'],
        commit_message=(
            f'Add dynamic-batch ONNX exports (opset {opset}) for '
            f'{len(model_ids)} earthinr-zoo models'
        ),
    )
    commit_sha = str(commit_info.oid)

    tree = api.list_repo_tree(
        repo_id=repo_id, repo_type='model', revision=commit_sha, recursive=True
    )
    remote_info: dict[str, dict[str, Any]] = {}
    target_paths = {f'{mid}/model.onnx': mid for mid in model_ids}
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
        help='Directory where <model_id>/model.onnx files and manifest are staged.',
    )
    parser.add_argument(
        '--models',
        nargs='+',
        default=None,
        choices=list(MODEL_SPECS_BY_ID),
        help='Subset of model IDs to export (defaults to all 18 zoo models).',
    )
    parser.add_argument(
        '--opset',
        type=int,
        default=DEFAULT_OPSET,
        help=f'Target ONNX opset version (default: {DEFAULT_OPSET}).',
    )
    parser.add_argument(
        '--max-abs-tol',
        type=float,
        default=DEFAULT_MAX_ABS_TOL,
        help=f'Maximum allowed abs diff vs PyTorch (default: {DEFAULT_MAX_ABS_TOL}).',
    )
    parser.add_argument(
        '--min-cos-sim',
        type=float,
        default=DEFAULT_MIN_COS_SIM,
        help=f'Minimum allowed cosine similarity (default: {DEFAULT_MIN_COS_SIM}).',
    )
    parser.add_argument(
        '--skip-existing',
        action='store_true',
        help='Skip export if <model_id>/model.onnx already exists and only verify.',
    )
    parser.add_argument(
        '--check-mind',
        action='store_true',
        help='Also verify parity of earthinr MIND against taylor-geospatial/MIND.',
    )
    parser.add_argument(
        '--upload',
        action='store_true',
        help='Upload staged */model.onnx files to Hugging Face Hub.',
    )
    parser.add_argument(
        '--repo-id',
        default=ZOO_REPO_ID,
        help=f'Target Hugging Face Hub repo (default: {ZOO_REPO_ID}).',
    )
    args = parser.parse_args()

    staging_dir: Path = args.staging_dir
    staging_dir.mkdir(parents=True, exist_ok=True)

    selected_ids = args.models if args.models is not None else list(MODEL_SPECS_BY_ID)
    records: list[dict[str, Any]] = []

    for model_id in selected_ids:
        spec = MODEL_SPECS_BY_ID[model_id]
        print(f'Exporting & verifying {model_id} (opset {args.opset})...', flush=True)
        rec = export_model(
            spec,
            staging_dir,
            opset=args.opset,
            max_abs_tol=args.max_abs_tol,
            min_cos_sim=args.min_cos_sim,
            skip_existing=args.skip_existing,
        )
        records.append(rec)
        size_mb = rec['size_bytes'] / (1024 * 1024)
        print(
            f'  OK {model_id:22s} | {size_mb:7.2f} MiB | opset={rec["opset"]} | '
            f'max_diff={rec["max_abs_diff"]:.2e} | mean_diff={rec["mean_abs_diff"]:.2e} | '
            f'min_cos={rec["min_cosine_sim"]:.8f} | sha256={rec["sha256"][:16]}...',
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
        'models': records,
        'mind_check': mind_report,
    }

    if args.upload:
        print(
            f'\nUploading {len(records)} ONNX models to {args.repo_id}...', flush=True
        )
        commit_sha, remote_info = upload_onnx_models(
            staging_dir,
            args.repo_id,
            [r['model_id'] for r in records],
            opset=args.opset,
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
