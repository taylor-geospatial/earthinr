# Copyright (c) Taylor Geospatial. All rights reserved.
# Licensed under the MIT License.

"""CoordBench end-to-end evaluation runner and atomic CSV result persistence."""

import csv
import os
import tempfile
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import torch
from filelock import FileLock
from numpy.typing import NDArray

from .._errors import ConfigError
from .._types import Path
from ..models._base import LocationEncoder
from ..models.registry import build_model, resolve_model_preset
from ..probes.classifiers import knn_probe_score, logreg_probe_score, mlp_probe_score
from ..probes.linear import chunked_probe_score, linear_probe_score
from ..probes.metrics import classification_metrics_from_cm, confusion_matrix
from ..probes.spatial import idw_probe_score, kriging_probe_score
from ..probes.splits import (
    RANDOM_FOLD_ALGORITHM,
    load_geonames_cities,
    spatial_fold_ids,
    urban_rural_tags,
)
from .config import CoordConfig
from .datasets import CoordBenchmark, load_benchmarks

SPATIAL_FOLD_ALGORITHM = 'spatial-grid-v1'
URBAN_RURAL_HOLDOUT_ALGORITHM = 'urban-rural-holdout-v1'
MIN_URBAN_RURAL_SAMPLES = 20

CSV_COLUMNS: tuple[str, ...] = (
    'dataset',
    'task',
    'task_type',
    'method',
    'split',
    'metric_name',
    'metric_value',
    'ci_lower',
    'ci_upper',
    'n_folds',
    'cell_deg',
    'feature_dim',
    'n_samples',
    'n_test',
    'seed',
    'model_name',
    'model_target',
    'fold_algorithm',
    'macro_f1',
    'balanced_accuracy',
    'cohen_kappa',
)

RESUME_KEY_COLS: tuple[str, ...] = (
    'dataset',
    'task',
    'method',
    'model_name',
    'split',
    'cell_deg',
    'fold_algorithm',
)

_SPATIAL_METHODS: frozenset[str] = frozenset({'idw', 'kriging'})
_CLASSIFICATION_ONLY_METHODS: frozenset[str] = frozenset({'knn', 'logreg'})


class CoordResult:
    """Single CoordBench evaluation metric row."""

    def __init__(
        self,
        *,
        dataset: str,
        task: str,
        task_type: str,
        method: str,
        split: str,
        metric_name: str,
        metric_value: float,
        ci_lower: float,
        ci_upper: float,
        n_folds: int,
        cell_deg: float,
        feature_dim: int,
        n_samples: int,
        n_test: int,
        seed: int,
        model_name: str,
        model_target: str,
        fold_algorithm: str,
        macro_f1: float | None = None,
        balanced_accuracy: float | None = None,
        cohen_kappa: float | None = None,
    ) -> None:
        """Initialize a new CoordResult instance.

        Args:
            dataset: CoordBench benchmark name.
            task: Target task name within *dataset*.
            task_type: Either ``'regression'`` or ``'classification'``.
            method: Probe method label (e.g. ``'linear'``, ``'knn5'``, ``'kriging'``).
            split: Split label (e.g. ``'random'``, ``'spatial'``, ``'official'``).
            metric_name: Primary metric name (``'r2'`` or ``'accuracy'``).
            metric_value: Mean primary metric value across evaluated folds.
            ci_lower: Lower dispersion bound (mean minus fold standard deviation).
            ci_upper: Upper dispersion bound (mean plus fold standard deviation).
            n_folds: Number of evaluated folds.
            cell_deg: Spatial block grid size in degrees (``0.0`` for non-spatial).
            feature_dim: Input feature dimensionality (``0`` for encoder-free methods).
            n_samples: Total valid sample count for the task.
            n_test: Number of valid test samples evaluated.
            seed: Deterministic seed used for the run.
            model_name: Model preset or display name.
            model_target: Dotted Python target path of the model or spatial baseline.
            fold_algorithm: Identifier describing the fold generation algorithm.
            macro_f1: Mean macro-F1 across folds (classification only).
            balanced_accuracy: Mean balanced accuracy across folds (classification).
            cohen_kappa: Mean Cohen's kappa across folds (classification only).
        """
        self.dataset = dataset
        self.task = task
        self.task_type = task_type
        self.method = method
        self.split = split
        self.metric_name = metric_name
        self.metric_value = float(metric_value)
        self.ci_lower = float(ci_lower)
        self.ci_upper = float(ci_upper)
        self.n_folds = int(n_folds)
        self.cell_deg = float(cell_deg)
        self.feature_dim = int(feature_dim)
        self.n_samples = int(n_samples)
        self.n_test = int(n_test)
        self.seed = int(seed)
        self.model_name = model_name
        self.model_target = model_target
        self.fold_algorithm = fold_algorithm
        self.macro_f1 = None if macro_f1 is None else float(macro_f1)
        self.balanced_accuracy = (
            None if balanced_accuracy is None else float(balanced_accuracy)
        )
        self.cohen_kappa = None if cohen_kappa is None else float(cohen_kappa)

    def to_row(self) -> dict[str, Any]:
        """Return a flat dictionary matching :data:`CSV_COLUMNS`."""
        return {
            'dataset': self.dataset,
            'task': self.task,
            'task_type': self.task_type,
            'method': self.method,
            'split': self.split,
            'metric_name': self.metric_name,
            'metric_value': self.metric_value,
            'ci_lower': self.ci_lower,
            'ci_upper': self.ci_upper,
            'n_folds': self.n_folds,
            'cell_deg': self.cell_deg,
            'feature_dim': self.feature_dim,
            'n_samples': self.n_samples,
            'n_test': self.n_test,
            'seed': self.seed,
            'model_name': self.model_name,
            'model_target': self.model_target,
            'fold_algorithm': self.fold_algorithm,
            'macro_f1': self.macro_f1,
            'balanced_accuracy': self.balanced_accuracy,
            'cohen_kappa': self.cohen_kappa,
        }

    def __repr__(self) -> str:
        """Return a concise summary string for debugging."""
        return (
            f'CoordResult(dataset={self.dataset!r}, task={self.task!r}, '
            f'method={self.method!r}, split={self.split!r}, '
            f'{self.metric_name}={self.metric_value:.4f})'
        )


def _normalize_cell_deg(value: Any) -> str:
    """Format *cell_deg* canonically for resume-key comparison."""
    if value is None or value == '':
        return '0'
    return f'{float(value):.6g}'


def _resume_key_from_mapping(row: Mapping[str, Any]) -> tuple[str, ...]:
    """Extract a normalized resume-key tuple from a row mapping."""
    return tuple(
        _normalize_cell_deg(row.get(col, 0.0))
        if col == 'cell_deg'
        else str(row.get(col, ''))
        for col in RESUME_KEY_COLS
    )


def _write_csv_atomic(
    filepath: str, fieldnames: Sequence[str], rows: Sequence[Mapping[str, Any]]
) -> None:
    """Write *rows* to a temp file in the target directory and atomically swap."""
    directory = os.path.dirname(os.path.abspath(filepath))
    fd, tmp_path = tempfile.mkstemp(dir=directory, prefix='.neuralftw_', suffix='.tmp')
    try:
        with os.fdopen(fd, 'w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)
        os.replace(tmp_path, filepath)
    except BaseException:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise


def append_rows_atomic(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    """Append result rows to a CSV file safely under a cross-process file lock.

    When new columns appear relative to an existing CSV header, rewrites the full
    table via an atomic temporary-file swap so no readers ever observe partial data.

    Args:
        path: Output CSV filesystem path.
        rows: Sequence of row dictionaries to append.
    """
    if not rows:
        return
    filepath = os.fspath(path)
    parent = os.path.dirname(filepath)
    if parent:
        os.makedirs(parent, exist_ok=True)

    lock = FileLock(f'{filepath}.lock')
    with lock:
        row_cols: list[str] = []
        for row in rows:
            for key in row:
                if key not in row_cols:
                    row_cols.append(key)

        if not os.path.exists(filepath) or os.path.getsize(filepath) == 0:
            _write_csv_atomic(filepath, row_cols, rows)
            return

        with open(filepath, encoding='utf-8', newline='') as f:
            reader = csv.reader(f)
            existing_cols = next(reader, [])

        merged_cols = list(existing_cols)
        for col in row_cols:
            if col not in merged_cols:
                merged_cols.append(col)

        if merged_cols == existing_cols:
            with open(filepath, 'a', encoding='utf-8', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=existing_cols)
                for row in rows:
                    writer.writerow(row)
            return

        with open(filepath, encoding='utf-8', newline='') as f:
            existing_rows = list(csv.DictReader(f))
        _write_csv_atomic(filepath, merged_cols, [*existing_rows, *rows])


def _load_completed_keys(path: Path) -> set[tuple[str, ...]]:
    """Read already-completed resume keys from an existing results CSV."""
    filepath = os.fspath(path)
    if not os.path.exists(filepath) or os.path.getsize(filepath) == 0:
        return set()
    lock = FileLock(f'{filepath}.lock')
    with lock, open(filepath, encoding='utf-8', newline='') as f:
        return {_resume_key_from_mapping(row) for row in csv.DictReader(f)}


def _method_label(method: str, knn_k: int) -> str:
    """Format the CSV ``method`` column value for a probe method."""
    return f'knn{knn_k}' if method == 'knn' else method


def _valid_label_mask(labels: NDArray[Any], task_type: str) -> NDArray[np.bool_]:
    """Return boolean mask of finite / non-null labels."""
    arr = np.asarray(labels)
    if task_type == 'regression':
        return np.asarray(np.isfinite(arr.astype(np.float64)), dtype=np.bool_)
    return np.array(
        [v is not None and (isinstance(v, str) or np.isfinite(v)) for v in arr],
        dtype=np.bool_,
    )


def _valid_row_mask(
    labels: NDArray[Any], task_type: str, features: NDArray[np.float32] | None
) -> NDArray[np.bool_]:
    """Return boolean mask of rows with valid labels and finite features (if given)."""
    valid_y = _valid_label_mask(labels, task_type)
    if features is None:
        return valid_y
    return np.asarray(valid_y & np.isfinite(features).all(axis=1), dtype=np.bool_)


def _mean_clf_metrics(
    confusions: Sequence[NDArray[np.int64]],
) -> tuple[float | None, float | None, float | None]:
    """Average macro-F1, balanced accuracy, and Cohen's kappa across fold CMs."""
    if not confusions:
        return None, None, None
    stats = [classification_metrics_from_cm(cm) for cm in confusions]
    macro_f1 = float(np.mean([s['macro_f1'] for s in stats]))
    bal_acc = float(np.mean([s['balanced_accuracy'] for s in stats]))
    kappa = float(np.mean([s['cohen_kappa'] for s in stats]))
    return macro_f1, bal_acc, kappa


def _score_bounds(score: float, fold_scores: Sequence[float]) -> tuple[float, float]:
    """Compute ``(score - std, score + std)`` across fold scores."""
    std = float(np.std(fold_scores)) if len(fold_scores) > 1 else 0.0
    return score - std, score + std


def _r2_score_np(y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
    """Compute scalar regression :math:`R^2` with denominator floor ``1e-12``."""
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = max(float(np.sum((y_true - np.mean(y_true)) ** 2)), 1e-12)
    return 1.0 - ss_res / ss_tot


def _encode_labels_to_int(
    labels: NDArray[Any], valid: NDArray[np.bool_]
) -> tuple[NDArray[np.int64], int]:
    """Map valid classification labels to contiguous ``0..C-1`` indices."""
    uniq, inv = np.unique(np.asarray(labels)[valid], return_inverse=True)
    full = np.full(len(labels), -1, dtype=np.int64)
    full[valid] = inv.astype(np.int64)
    return full, len(uniq)


def _cell_deg_sequence(cfg: CoordConfig) -> tuple[float, ...]:
    """Normalize ``cfg.evaluation.cell_deg`` to a tuple of floats."""
    raw = cfg.evaluation.cell_deg
    return raw if isinstance(raw, tuple) else (raw,)


def _planned_split_specs(
    bench: CoordBenchmark, cfg: CoordConfig
) -> list[tuple[str, float, str]]:
    """Return ``(split_label, cell_deg, fold_algorithm)`` entries for *bench*."""
    if bench.test_mask is not None and cfg.evaluation.respect_official_splits:
        return [('official', 0.0, 'official')]
    split = cfg.evaluation.split
    if split == 'random':
        return [('random', 0.0, RANDOM_FOLD_ALGORITHM)]
    if split == 'spatial':
        return [('spatial', c, SPATIAL_FOLD_ALGORITHM) for c in _cell_deg_sequence(cfg)]
    if split == 'both':
        return [
            ('random', 0.0, RANDOM_FOLD_ALGORITHM),
            *(('spatial', c, SPATIAL_FOLD_ALGORITHM) for c in _cell_deg_sequence(cfg)),
        ]
    if split == 'urban_rural_stratified':
        return [
            ('urban_stratified', 0.0, RANDOM_FOLD_ALGORITHM),
            ('rural_stratified', 0.0, RANDOM_FOLD_ALGORITHM),
        ]
    return [
        ('train_rural_test_urban', 0.0, URBAN_RURAL_HOLDOUT_ALGORITHM),
        ('train_urban_test_rural', 0.0, URBAN_RURAL_HOLDOUT_ALGORITHM),
    ]


def _applicable_methods(bench: CoordBenchmark, cfg: CoordConfig) -> list[str]:
    """Return probe methods valid for *bench*'s task type."""
    if bench.task_type == 'regression':
        return [
            m for m in cfg.evaluation.methods if m not in _CLASSIFICATION_ONLY_METHODS
        ]
    return list(cfg.evaluation.methods)


def _benchmark_fully_completed(
    bench: CoordBenchmark,
    cfg: CoordConfig,
    model_name: str,
    completed: set[tuple[str, ...]],
) -> bool:
    """Return True if every expected result key for *bench* is already recorded."""
    if not completed:
        return False
    methods = _applicable_methods(bench, cfg)
    specs = _planned_split_specs(bench, cfg)
    for task in bench.tasks:
        for method in methods:
            m_label = _method_label(method, cfg.evaluation.knn_k)
            for split_label, cell_deg, fold_algo in specs:
                key = (
                    bench.name,
                    task,
                    m_label,
                    model_name,
                    split_label,
                    _normalize_cell_deg(cell_deg),
                    fold_algo,
                )
                if key not in completed:
                    return False
    return True


def _run_single_probe(
    *,
    method: str,
    features: NDArray[np.float32] | None,
    lat: NDArray[np.float64],
    lon: NDArray[np.float64],
    labels: NDArray[Any],
    task_type: str,
    cfg: CoordConfig,
    test_mask: NDArray[np.bool_] | None = None,
    fold_assign: NDArray[np.int64] | None = None,
    cell_deg: float = 0.0,
    confusions: list[NDArray[np.int64]] | None = None,
) -> tuple[float, list[float]]:
    """Dispatch one single-target probe evaluation."""
    if method == 'idw':
        return idw_probe_score(
            lat,
            lon,
            labels,
            task_type=task_type,
            folds=cfg.evaluation.folds,
            seed=cfg.runtime.seed,
            test_mask=test_mask,
            fold_assign=fold_assign,
            confusions=confusions,
        )
    if method == 'kriging':
        return kriging_probe_score(
            lat,
            lon,
            labels,
            task_type=task_type,
            folds=cfg.evaluation.folds,
            seed=cfg.runtime.seed,
            cell_deg=cell_deg,
            test_mask=test_mask,
            fold_assign=fold_assign,
            confusions=confusions,
        )
    assert features is not None
    if method == 'linear':
        score, fold_scores = linear_probe_score(
            features,
            labels,
            task_type=task_type,
            folds=cfg.evaluation.folds,
            seed=cfg.runtime.seed,
            device=cfg.runtime.device,
            test_mask=test_mask,
            fold_assign=fold_assign,
            confusions=confusions,
        )
        return float(score), [float(s) for s in fold_scores]
    if method == 'chunked':
        score, fold_scores = chunked_probe_score(
            features,
            labels,
            task_type=task_type,
            folds=cfg.evaluation.folds,
            seed=cfg.runtime.seed,
            device=cfg.runtime.device,
            test_mask=test_mask,
            fold_assign=fold_assign,
            confusions=confusions,
        )
        return float(score), [float(s) for s in fold_scores]
    if method == 'knn':
        return knn_probe_score(
            features,
            labels,
            k=cfg.evaluation.knn_k,
            folds=cfg.evaluation.folds,
            seed=cfg.runtime.seed,
            device=cfg.evaluation.knn_device,
            test_mask=test_mask,
            fold_assign=fold_assign,
            confusions=confusions,
        )
    if method == 'logreg':
        return logreg_probe_score(
            features,
            labels,
            folds=cfg.evaluation.folds,
            seed=cfg.runtime.seed,
            device=cfg.runtime.device,
            test_mask=test_mask,
            fold_assign=fold_assign,
            confusions=confusions,
        )
    return mlp_probe_score(
        features,
        labels,
        task_type=task_type,
        folds=cfg.evaluation.folds,
        seed=cfg.runtime.seed,
        device=cfg.runtime.device,
        test_mask=test_mask,
        fold_assign=fold_assign,
        confusions=confusions,
    )


def _group_regression_tasks(
    bench: CoordBenchmark, task_names: Sequence[str]
) -> list[list[str]]:
    """Group regression task names that share an identical finite-label mask."""
    groups: dict[bytes, list[str]] = {}
    for task in task_names:
        mask_key = np.isfinite(bench.tasks[task].astype(np.float64)).tobytes()
        groups.setdefault(mask_key, []).append(task)
    return list(groups.values())


def _eval_standard_split(
    *,
    bench: CoordBenchmark,
    features: NDArray[np.float32] | None,
    method: str,
    split_label: str,
    cell_deg: float,
    fold_algo: str,
    cfg: CoordConfig,
    model_name: str,
    model_target: str,
    completed: set[tuple[str, ...]],
) -> list[CoordResult]:
    """Evaluate official, random, or spatial split for one benchmark and method."""
    m_label = _method_label(method, cfg.evaluation.knn_k)
    metric_name = 'r2' if bench.task_type == 'regression' else 'accuracy'
    feat_dim = (
        0 if method in _SPATIAL_METHODS or features is None else features.shape[1]
    )
    pending_tasks = [
        task
        for task in bench.tasks
        if (
            bench.name,
            task,
            m_label,
            model_name,
            split_label,
            _normalize_cell_deg(cell_deg),
            fold_algo,
        )
        not in completed
    ]
    if not pending_tasks:
        return []

    test_mask = (
        np.asarray(bench.test_mask, dtype=np.bool_)
        if split_label == 'official' and bench.test_mask is not None
        else None
    )
    fold_assign = (
        spatial_fold_ids(
            bench.lat,
            bench.lon,
            folds=cfg.evaluation.folds,
            cell_deg=cell_deg,
            seed=cfg.runtime.seed,
        )
        if split_label == 'spatial'
        else None
    )

    results: list[CoordResult] = []
    if bench.task_type == 'regression' and method == 'linear':
        assert features is not None
        for group in _group_regression_tasks(bench, pending_tasks):
            if len(group) > 1:
                y_mat = np.column_stack(
                    [bench.tasks[t].astype(np.float64) for t in group]
                )
                valid = _valid_row_mask(bench.tasks[group[0]], 'regression', features)
                n_samples = int(valid.sum())
                n_test = (
                    int((valid & test_mask).sum())
                    if test_mask is not None
                    else n_samples
                )
                scores_arr, fold_scores_list = linear_probe_score(
                    features,
                    y_mat,
                    task_type='regression',
                    folds=cfg.evaluation.folds,
                    seed=cfg.runtime.seed,
                    device=cfg.runtime.device,
                    test_mask=test_mask,
                    fold_assign=fold_assign,
                )
                scores_vec = np.asarray(scores_arr, dtype=np.float64)
                fold_mat = np.stack(
                    [np.asarray(fs, dtype=np.float64) for fs in fold_scores_list],
                    axis=0,
                )
                for idx, task in enumerate(group):
                    task_folds = [float(v) for v in fold_mat[:, idx]]
                    score = float(scores_vec[idx])
                    ci_lo, ci_hi = _score_bounds(score, task_folds)
                    results.append(
                        CoordResult(
                            dataset=bench.name,
                            task=task,
                            task_type='regression',
                            method=m_label,
                            split=split_label,
                            metric_name=metric_name,
                            metric_value=score,
                            ci_lower=ci_lo,
                            ci_upper=ci_hi,
                            n_folds=len(task_folds),
                            cell_deg=cell_deg,
                            feature_dim=feat_dim,
                            n_samples=n_samples,
                            n_test=n_test,
                            seed=cfg.runtime.seed,
                            model_name=model_name,
                            model_target=model_target,
                            fold_algorithm=fold_algo,
                        )
                    )
                continue
            pending_single = group
            for task in pending_single:
                labels = bench.tasks[task]
                valid = _valid_row_mask(labels, 'regression', features)
                n_samples = int(valid.sum())
                n_test = (
                    int((valid & test_mask).sum())
                    if test_mask is not None
                    else n_samples
                )
                score, fold_scores = _run_single_probe(
                    method=method,
                    features=features,
                    lat=bench.lat,
                    lon=bench.lon,
                    labels=labels,
                    task_type='regression',
                    cfg=cfg,
                    test_mask=test_mask,
                    fold_assign=fold_assign,
                    cell_deg=cell_deg,
                )
                ci_lo, ci_hi = _score_bounds(score, fold_scores)
                results.append(
                    CoordResult(
                        dataset=bench.name,
                        task=task,
                        task_type='regression',
                        method=m_label,
                        split=split_label,
                        metric_name=metric_name,
                        metric_value=score,
                        ci_lower=ci_lo,
                        ci_upper=ci_hi,
                        n_folds=len(fold_scores),
                        cell_deg=cell_deg,
                        feature_dim=feat_dim,
                        n_samples=n_samples,
                        n_test=n_test,
                        seed=cfg.runtime.seed,
                        model_name=model_name,
                        model_target=model_target,
                        fold_algorithm=fold_algo,
                    )
                )
        return results

    for task in pending_tasks:
        labels = bench.tasks[task]
        probe_feats = None if method in _SPATIAL_METHODS else features
        valid = _valid_row_mask(labels, bench.task_type, probe_feats)
        n_samples = int(valid.sum())
        n_test = int((valid & test_mask).sum()) if test_mask is not None else n_samples
        confusions: list[NDArray[np.int64]] | None = (
            [] if bench.task_type == 'classification' else None
        )
        score, fold_scores = _run_single_probe(
            method=method,
            features=probe_feats,
            lat=bench.lat,
            lon=bench.lon,
            labels=labels,
            task_type=bench.task_type,
            cfg=cfg,
            test_mask=test_mask,
            fold_assign=fold_assign,
            cell_deg=cell_deg,
            confusions=confusions,
        )
        ci_lo, ci_hi = _score_bounds(score, fold_scores)
        macro_f1, bal_acc, kappa = _mean_clf_metrics(confusions or [])
        results.append(
            CoordResult(
                dataset=bench.name,
                task=task,
                task_type=bench.task_type,
                method=m_label,
                split=split_label,
                metric_name=metric_name,
                metric_value=score,
                ci_lower=ci_lo,
                ci_upper=ci_hi,
                n_folds=len(fold_scores),
                cell_deg=cell_deg,
                feature_dim=feat_dim,
                n_samples=n_samples,
                n_test=n_test,
                seed=cfg.runtime.seed,
                model_name=model_name,
                model_target=model_target,
                fold_algorithm=fold_algo,
                macro_f1=macro_f1,
                balanced_accuracy=bal_acc,
                cohen_kappa=kappa,
            )
        )
    return results


def _eval_urban_rural_stratified(
    *,
    bench: CoordBenchmark,
    features: NDArray[np.float32] | None,
    tags: NDArray[np.int64],
    method: str,
    cfg: CoordConfig,
    model_name: str,
    model_target: str,
    completed: set[tuple[str, ...]],
) -> list[CoordResult]:
    """Evaluate random k-fold OOF scored separately on urban and rural rows."""
    m_label = _method_label(method, cfg.evaluation.knn_k)
    metric_name = 'r2' if bench.task_type == 'regression' else 'accuracy'
    feat_dim = (
        0 if method in _SPATIAL_METHODS or features is None else features.shape[1]
    )
    strata: tuple[tuple[str, NDArray[np.bool_]], ...] = (
        ('urban_stratified', tags == 0),
        ('rural_stratified', tags == 1),
    )

    results: list[CoordResult] = []
    for task, labels in bench.tasks.items():
        needed = [
            (label, mask)
            for label, mask in strata
            if (
                bench.name,
                task,
                m_label,
                model_name,
                label,
                _normalize_cell_deg(0.0),
                RANDOM_FOLD_ALGORITHM,
            )
            not in completed
        ]
        if not needed:
            continue

        if method in {'linear', 'chunked'}:
            assert features is not None
            valid = _valid_row_mask(labels, bench.task_type, features)
            n_samples = int(valid.sum())
            if method == 'linear':
                _, _, oof = linear_probe_score(
                    features,
                    labels,
                    task_type=bench.task_type,
                    folds=cfg.evaluation.folds,
                    seed=cfg.runtime.seed,
                    device=cfg.runtime.device,
                    return_oof=True,
                )
            else:
                _, _, oof = chunked_probe_score(
                    features,
                    labels,
                    task_type=bench.task_type,
                    folds=cfg.evaluation.folds,
                    seed=cfg.runtime.seed,
                    device=cfg.runtime.device,
                    return_oof=True,
                )
            y_cls: NDArray[np.int64] | None = None
            n_classes = 0
            if bench.task_type == 'classification':
                y_cls, n_classes = _encode_labels_to_int(labels, valid)

            for split_label, stratum_mask in needed:
                sub = valid & stratum_mask
                n_test = int(sub.sum())
                if bench.task_type == 'regression':
                    score = _r2_score_np(
                        labels[sub].astype(np.float64),
                        np.asarray(oof, dtype=np.float64)[sub],
                    )
                    macro_f1 = bal_acc = kappa = None
                else:
                    assert y_cls is not None
                    cm = confusion_matrix(
                        y_cls[sub], np.asarray(oof, dtype=np.int64)[sub], n_classes
                    )
                    stats = classification_metrics_from_cm(cm)
                    score = stats['accuracy']
                    macro_f1 = stats['macro_f1']
                    bal_acc = stats['balanced_accuracy']
                    kappa = stats['cohen_kappa']
                results.append(
                    CoordResult(
                        dataset=bench.name,
                        task=task,
                        task_type=bench.task_type,
                        method=m_label,
                        split=split_label,
                        metric_name=metric_name,
                        metric_value=score,
                        ci_lower=score,
                        ci_upper=score,
                        n_folds=cfg.evaluation.folds,
                        cell_deg=0.0,
                        feature_dim=feat_dim,
                        n_samples=n_samples,
                        n_test=n_test,
                        seed=cfg.runtime.seed,
                        model_name=model_name,
                        model_target=model_target,
                        fold_algorithm=RANDOM_FOLD_ALGORITHM,
                        macro_f1=macro_f1,
                        balanced_accuracy=bal_acc,
                        cohen_kappa=kappa,
                    )
                )
            continue

        probe_feats = None if method in _SPATIAL_METHODS else features
        for split_label, stratum_mask in needed:
            sub_feats = None if probe_feats is None else probe_feats[stratum_mask]
            sub_labels = labels[stratum_mask]
            valid_sub = _valid_row_mask(sub_labels, bench.task_type, sub_feats)
            n_sub = int(valid_sub.sum())
            confusions: list[NDArray[np.int64]] | None = (
                [] if bench.task_type == 'classification' else None
            )
            score, fold_scores = _run_single_probe(
                method=method,
                features=sub_feats,
                lat=bench.lat[stratum_mask],
                lon=bench.lon[stratum_mask],
                labels=sub_labels,
                task_type=bench.task_type,
                cfg=cfg,
                confusions=confusions,
            )
            ci_lo, ci_hi = _score_bounds(score, fold_scores)
            macro_f1, bal_acc, kappa = _mean_clf_metrics(confusions or [])
            results.append(
                CoordResult(
                    dataset=bench.name,
                    task=task,
                    task_type=bench.task_type,
                    method=m_label,
                    split=split_label,
                    metric_name=metric_name,
                    metric_value=score,
                    ci_lower=ci_lo,
                    ci_upper=ci_hi,
                    n_folds=len(fold_scores),
                    cell_deg=0.0,
                    feature_dim=feat_dim,
                    n_samples=n_sub,
                    n_test=n_sub,
                    seed=cfg.runtime.seed,
                    model_name=model_name,
                    model_target=model_target,
                    fold_algorithm=RANDOM_FOLD_ALGORITHM,
                    macro_f1=macro_f1,
                    balanced_accuracy=bal_acc,
                    cohen_kappa=kappa,
                )
            )
    return results


def _eval_urban_rural_holdout(
    *,
    bench: CoordBenchmark,
    features: NDArray[np.float32] | None,
    tags: NDArray[np.int64],
    method: str,
    cfg: CoordConfig,
    model_name: str,
    model_target: str,
    completed: set[tuple[str, ...]],
) -> list[CoordResult]:
    """Evaluate 2-fold urban/rural holdout dropping buffer-zone points."""
    m_label = _method_label(method, cfg.evaluation.knn_k)
    metric_name = 'r2' if bench.task_type == 'regression' else 'accuracy'
    feat_dim = (
        0 if method in _SPATIAL_METHODS or features is None else features.shape[1]
    )
    keep = tags >= 0
    fa = tags[keep]
    sub_lat = bench.lat[keep]
    sub_lon = bench.lon[keep]
    probe_feats = (
        None if method in _SPATIAL_METHODS or features is None else features[keep]
    )

    holdout_splits: tuple[tuple[int, str], ...] = (
        (0, 'train_rural_test_urban'),
        (1, 'train_urban_test_rural'),
    )
    results: list[CoordResult] = []
    for task, labels in bench.tasks.items():
        needed = [
            (fold_idx, split_label)
            for fold_idx, split_label in holdout_splits
            if (
                bench.name,
                task,
                m_label,
                model_name,
                split_label,
                _normalize_cell_deg(0.0),
                URBAN_RURAL_HOLDOUT_ALGORITHM,
            )
            not in completed
        ]
        if not needed:
            continue

        sub_labels = labels[keep]
        valid = _valid_row_mask(sub_labels, bench.task_type, probe_feats)
        n_samples = int(valid.sum())
        confusions: list[NDArray[np.int64]] | None = (
            [] if bench.task_type == 'classification' else None
        )
        _, fold_scores = _run_single_probe(
            method=method,
            features=probe_feats,
            lat=sub_lat,
            lon=sub_lon,
            labels=sub_labels,
            task_type=bench.task_type,
            cfg=cfg,
            fold_assign=fa,
            confusions=confusions,
        )
        for fold_idx, split_label in needed:
            score = float(fold_scores[fold_idx])
            n_test = int((valid & (fa == fold_idx)).sum())
            if confusions:
                stats = classification_metrics_from_cm(confusions[fold_idx])
                macro_f1: float | None = stats['macro_f1']
                bal_acc: float | None = stats['balanced_accuracy']
                kappa: float | None = stats['cohen_kappa']
            else:
                macro_f1 = bal_acc = kappa = None
            results.append(
                CoordResult(
                    dataset=bench.name,
                    task=task,
                    task_type=bench.task_type,
                    method=m_label,
                    split=split_label,
                    metric_name=metric_name,
                    metric_value=score,
                    ci_lower=score,
                    ci_upper=score,
                    n_folds=1,
                    cell_deg=0.0,
                    feature_dim=feat_dim,
                    n_samples=n_samples,
                    n_test=n_test,
                    seed=cfg.runtime.seed,
                    model_name=model_name,
                    model_target=model_target,
                    fold_algorithm=URBAN_RURAL_HOLDOUT_ALGORITHM,
                    macro_f1=macro_f1,
                    balanced_accuracy=bal_acc,
                    cohen_kappa=kappa,
                )
            )
    return results


def _resolve_encoder(
    cfg: CoordConfig, encoder: LocationEncoder | None
) -> tuple[LocationEncoder | None, str, str]:
    """Resolve model name, target string, and optional LocationEncoder instance."""
    needs_encoder = any(m not in _SPATIAL_METHODS for m in cfg.evaluation.methods)
    model_name = cfg.model.name

    if model_name in _SPATIAL_METHODS and encoder is None:
        if needs_encoder:
            raise ConfigError(
                f'Spatial baseline {model_name!r} only supports methods '
                f'{sorted(_SPATIAL_METHODS)}.'
            )
        return None, model_name, f'neuralftw.probes.spatial.{model_name}_probe_score'

    if encoder is not None:
        target = (
            cfg.model.target
            or f'{type(encoder).__module__}.{type(encoder).__qualname__}'
        )
        return encoder, model_name, target

    if not needs_encoder:
        return None, model_name, cfg.model.target or ''

    preset = resolve_model_preset(
        cfg.model.name, target=cfg.model.target, kwargs=cfg.model.kwargs
    )
    built = build_model(preset, device=cfg.runtime.device)
    return built, model_name, preset.target


def run_coordbench(
    cfg: CoordConfig,
    *,
    root: Path | None = None,
    repo_id: str | None = None,
    encoder: LocationEncoder | None = None,
) -> list[CoordResult]:
    """Run a configured CoordBench evaluation suite and persist CSV results.

    Args:
        cfg: Strictly validated :class:`~neuralftw.coordbench.config.CoordConfig`.
        root: Optional local directory containing CoordBench parquet files.
        repo_id: Optional Hugging Face dataset repo id overriding ``COORDBENCH_REPO``.
        encoder: Optional pre-built :class:`~neuralftw.models.LocationEncoder` instance
            bypassing preset resolution and model construction.

    Returns:
        List of newly computed :class:`CoordResult` rows written to ``cfg.output.file``.

    Raises:
        ConfigError: If an encoder-free spatial baseline is paired with feature-based
            probe methods.
    """
    torch.manual_seed(cfg.runtime.seed)
    resolved_encoder, model_name, model_target = _resolve_encoder(cfg, encoder)
    completed = _load_completed_keys(cfg.output.file) if cfg.output.resume else set()

    benchmarks = load_benchmarks(cfg.datasets, root=root, repo_id=repo_id)

    cities_lat: NDArray[np.float64] | None = None
    cities_lon: NDArray[np.float64] | None = None
    if cfg.evaluation.split in {'urban_rural_stratified', 'urban_rural_holdout'}:
        cities_lat, cities_lon = load_geonames_cities(cfg.evaluation.cities_path)

    all_results: list[CoordResult] = []
    for bench in benchmarks:
        if _benchmark_fully_completed(bench, cfg, model_name, completed):
            continue

        methods = _applicable_methods(bench, cfg)
        if not methods:
            continue

        use_official = (
            bench.test_mask is not None and cfg.evaluation.respect_official_splits
        )
        tags: NDArray[np.int64] | None = None
        if not use_official and cfg.evaluation.split in {
            'urban_rural_stratified',
            'urban_rural_holdout',
        }:
            assert cities_lat is not None
            assert cities_lon is not None
            tags = urban_rural_tags(
                bench.lat,
                bench.lon,
                cities_lat,
                cities_lon,
                urban_km=cfg.evaluation.urban_km,
                rural_km=cfg.evaluation.rural_km,
            )
            if (
                int(np.sum(tags == 0)) < MIN_URBAN_RURAL_SAMPLES
                or int(np.sum(tags == 1)) < MIN_URBAN_RURAL_SAMPLES
            ):
                continue

        features: NDArray[np.float32] | None = None
        if resolved_encoder is not None and any(
            m not in _SPATIAL_METHODS for m in methods
        ):
            features = resolved_encoder.encode(bench.lon, bench.lat, year=bench.year)

        bench_results: list[CoordResult] = []
        if not use_official and cfg.evaluation.split == 'urban_rural_stratified':
            assert tags is not None
            for method in methods:
                bench_results.extend(
                    _eval_urban_rural_stratified(
                        bench=bench,
                        features=features,
                        tags=tags,
                        method=method,
                        cfg=cfg,
                        model_name=model_name,
                        model_target=model_target,
                        completed=completed,
                    )
                )
        elif not use_official and cfg.evaluation.split == 'urban_rural_holdout':
            assert tags is not None
            for method in methods:
                bench_results.extend(
                    _eval_urban_rural_holdout(
                        bench=bench,
                        features=features,
                        tags=tags,
                        method=method,
                        cfg=cfg,
                        model_name=model_name,
                        model_target=model_target,
                        completed=completed,
                    )
                )
        else:
            specs = _planned_split_specs(bench, cfg)
            for split_label, cell_deg, fold_algo in specs:
                for method in methods:
                    bench_results.extend(
                        _eval_standard_split(
                            bench=bench,
                            features=features,
                            method=method,
                            split_label=split_label,
                            cell_deg=cell_deg,
                            fold_algo=fold_algo,
                            cfg=cfg,
                            model_name=model_name,
                            model_target=model_target,
                            completed=completed,
                        )
                    )

        rows = [r.to_row() for r in bench_results]
        append_rows_atomic(cfg.output.file, rows)
        for row in rows:
            completed.add(_resume_key_from_mapping(row))
        all_results.extend(bench_results)

    return all_results


__all__ = (
    'CSV_COLUMNS',
    'MIN_URBAN_RURAL_SAMPLES',
    'RESUME_KEY_COLS',
    'SPATIAL_FOLD_ALGORITHM',
    'URBAN_RURAL_HOLDOUT_ALGORITHM',
    'CoordResult',
    'append_rows_atomic',
    'run_coordbench',
)
