# Copyright (c) Taylor Geospatial. All rights reserved.
# Licensed under the MIT License.

"""Classification confusion matrices, metrics, and compact sparse storage."""

import os
from collections.abc import Sequence
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor

from .._types import Path


def confusion_matrix(
    y_true: NDArray[np.integer[Any]] | Tensor,
    y_pred: NDArray[np.integer[Any]] | Tensor,
    n_classes: int,
) -> NDArray[np.int64]:
    """Compute a ``[n_classes, n_classes]`` confusion matrix of counts.

    Rows index the true class and columns index the predicted class.

    Args:
        y_true: Ground-truth integer class indices in ``[0, n_classes - 1]``.
        y_pred: Predicted integer class indices in ``[0, n_classes - 1]``.
        n_classes: Total number of classes (must be positive).

    Returns:
        Two-dimensional ``int64`` array of shape ``(n_classes, n_classes)``.

    Raises:
        ValueError: If *n_classes* < 1 or *y_true*/*y_pred* shapes mismatch.
    """
    if n_classes < 1:
        raise ValueError(f'n_classes must be positive, got {n_classes}.')
    t = (
        y_true.detach().cpu().numpy().astype(np.int64)
        if isinstance(y_true, torch.Tensor)
        else np.asarray(y_true, dtype=np.int64)
    )
    p = (
        y_pred.detach().cpu().numpy().astype(np.int64)
        if isinstance(y_pred, torch.Tensor)
        else np.asarray(y_pred, dtype=np.int64)
    )
    if t.shape != p.shape:
        raise ValueError(
            f'y_true shape {t.shape} does not match y_pred shape {p.shape}.'
        )
    flat = t.ravel() * n_classes + p.ravel()
    return (
        np.bincount(flat, minlength=n_classes * n_classes)
        .reshape(n_classes, n_classes)
        .astype(np.int64)
    )


def classification_metrics_from_cm(cm: NDArray[np.integer[Any]]) -> dict[str, float]:
    """Compute standard classification metrics from a confusion matrix.

    Follows scikit-learn's convention for class inclusion:

    - ``accuracy``: overall fraction of correct predictions.
    - ``macro_f1``: unweighted mean F1 over classes present in true or predicted labels.
    - ``balanced_accuracy``: mean recall over classes with positive true support.
    - ``cohen_kappa``: chance-corrected agreement between true and predicted marginals.

    Args:
        cm: Square integer confusion matrix with shape ``(C, C)``.

    Returns:
        Dictionary mapping metric names to scalar float values.

    Raises:
        ValueError: If *cm* is not a non-empty square 2D matrix.
    """
    mat = np.asarray(cm, dtype=np.int64)
    if mat.ndim != 2 or mat.shape[0] != mat.shape[1] or mat.shape[0] == 0:
        raise ValueError('cm must be a non-empty square 2D array.')
    total = int(mat.sum())
    if total == 0:
        return {
            'accuracy': 0.0,
            'macro_f1': 0.0,
            'balanced_accuracy': 0.0,
            'cohen_kappa': 0.0,
        }
    tp = np.diag(mat).astype(np.float64)
    support = mat.sum(axis=1)
    predicted = mat.sum(axis=0)

    acc = float(tp.sum() / total)
    present = (support + predicted) > 0
    macro_f1 = float(np.mean(2.0 * tp[present] / (support + predicted)[present]))
    seen = support > 0
    bal_acc = float(np.mean(tp[seen] / support[seen]))
    pe = float((predicted * support).sum()) / float(total * total)
    kappa = float((acc - pe) / (1.0 - pe)) if pe < 1.0 else 0.0
    return {
        'accuracy': acc,
        'macro_f1': macro_f1,
        'balanced_accuracy': bal_acc,
        'cohen_kappa': kappa,
    }


def save_sparse_cm(path: Path, matrices: Sequence[NDArray[np.integer[Any]]]) -> None:
    """Save a sequence of square confusion matrices as COO triplets in one ``.npz``.

    Args:
        path: Output ``.npz`` file path.
        matrices: Sequence of square count matrices.
    """
    ids: list[NDArray[np.int32]] = []
    rows: list[NDArray[np.int32]] = []
    cols: list[NDArray[np.int32]] = []
    vals: list[NDArray[np.int64]] = []
    sizes: list[int] = []
    for i, cm in enumerate(matrices):
        mat = np.asarray(cm, dtype=np.int64)
        r, c = np.nonzero(mat)
        ids.append(np.full(len(r), i, dtype=np.int32))
        rows.append(r.astype(np.int32))
        cols.append(c.astype(np.int32))
        vals.append(mat[r, c].astype(np.int64))
        sizes.append(int(mat.shape[0]))
    empty32 = np.zeros(0, dtype=np.int32)
    empty64 = np.zeros(0, dtype=np.int64)
    np.savez_compressed(
        os.fspath(path),
        matrix=np.concatenate(ids) if ids else empty32,
        true=np.concatenate(rows) if rows else empty32,
        pred=np.concatenate(cols) if cols else empty32,
        count=np.concatenate(vals) if vals else empty64,
        n_classes=np.asarray(sizes, dtype=np.int32),
    )


def load_sparse_cm(path: Path) -> list[NDArray[np.int64]]:
    """Load confusion matrices previously stored with :func:`save_sparse_cm`.

    Args:
        path: Input ``.npz`` file path.

    Returns:
        List of dense ``[C, C]`` ``int64`` confusion matrices.
    """
    with np.load(os.fspath(path)) as z:
        ids = z['matrix']
        true_idx = z['true']
        pred_idx = z['pred']
        counts = z['count']
        sizes = z['n_classes']
    bounds = np.searchsorted(ids, np.arange(len(sizes) + 1))
    out: list[NDArray[np.int64]] = []
    for i, k in enumerate(sizes):
        sl = slice(int(bounds[i]), int(bounds[i + 1]))
        cm = np.zeros((int(k), int(k)), dtype=np.int64)
        cm[true_idx[sl], pred_idx[sl]] = counts[sl]
        out.append(cm)
    return out


__all__ = (
    'classification_metrics_from_cm',
    'confusion_matrix',
    'load_sparse_cm',
    'save_sparse_cm',
)
