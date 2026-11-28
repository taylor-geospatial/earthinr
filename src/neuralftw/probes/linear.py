# Copyright (c) Taylor Geospatial. All rights reserved.
# Licensed under the MIT License.

"""Closed-form ridge and Chunked Penalty linear probes with unpenalized intercepts."""

import warnings
from collections.abc import Callable
from typing import Any, Literal, overload

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor

from .._utils import float64_device, resolve_device
from .metrics import confusion_matrix
from .splits import random_fold_ids

RIDGE_ALPHAS: tuple[float, ...] = tuple(
    float(a) for a in 10.0 ** np.arange(-4.0, 6.5, 0.5)
)
MRL_BOUNDARIES: tuple[int, ...] = (64, 128, 256, 512, 1024, 2048)
CHUNK_GAMMAS: tuple[float, ...] = (1.0, 3.0, 10.0, 100.0, 1000.0)


def chunked_penalty_weights(
    dim: int, gamma: float, boundaries: tuple[int, ...] = MRL_BOUNDARIES
) -> NDArray[np.float64]:
    """Compute geometric Chunked Penalty multipliers per embedding dimension.

    Chunk ``j`` starts at ``boundaries[j - 1]`` (with chunk 0 starting at 0) and
    receives penalty multiplier ``gamma ** j``. Setting ``gamma=1.0`` produces
    uniform ridge weights.

    Args:
        dim: Embedding dimensionality (must be positive).
        gamma: Per-chunk geometric growth factor (must be finite and >= 1.0).
        boundaries: Strictly increasing positive integer chunk boundaries.

    Returns:
        Float64 vector of shape ``(dim,)`` with penalty multipliers.

    Raises:
        ValueError: If *dim*, *gamma*, or *boundaries* are invalid.
    """
    if dim < 1:
        raise ValueError(f'dim must be positive, got {dim}.')
    if not np.isfinite(gamma) or gamma < 1.0:
        raise ValueError(f'gamma must be finite and at least 1.0, got {gamma}.')
    stops = np.asarray(boundaries)
    if stops.ndim != 1 or (
        stops.size > 0
        and (
            not np.issubdtype(stops.dtype, np.integer)
            or bool(np.any(stops <= 0))
            or bool(np.any(np.diff(stops) <= 0))
        )
    ):
        raise ValueError('boundaries must be increasing positive integers.')
    chunk = np.searchsorted(stops, np.arange(dim), side='right')
    return np.asarray(float(gamma) ** chunk.astype(np.float64), dtype=np.float64)


def power_penalty_weights(dim: int, power: float) -> NDArray[np.float64]:
    """Compute smooth power-law Tikhonov penalty weights ``(i + 1) ** power``.

    Args:
        dim: Embedding dimensionality (must be positive).
        power: Non-negative finite exponent.

    Returns:
        Float64 vector of shape ``(dim,)`` with penalty multipliers.

    Raises:
        ValueError: If *dim* < 1 or *power* is negative or non-finite.
    """
    if dim < 1:
        raise ValueError(f'dim must be positive, got {dim}.')
    if not np.isfinite(power) or power < 0.0:
        raise ValueError(f'power must be finite and non-negative, got {power}.')
    return np.asarray(
        (np.arange(dim, dtype=np.float64) + 1.0) ** float(power), dtype=np.float64
    )


class RidgeCP:
    """Fitted Chunked Penalty ridge probe returned by :func:`fit_ridge_cp`."""

    def __init__(
        self,
        mean: NDArray[np.float64],
        scale: NDArray[np.float64],
        coef: NDArray[np.float64],
        intercept: float | NDArray[np.float64],
        standardize: bool = True,
    ) -> None:
        """Initialize a fitted Chunked Penalty ridge probe.

        Args:
            mean: Training-set per-channel mean of shape ``(D,)``.
            scale: Training-set per-channel sample std of shape ``(D,)``.
            coef: Learned linear weights on standardized (or raw) features.
            intercept: Unpenalized intercept scalar or vector.
            standardize: Whether input features are standardized prior to projection.
        """
        self.mean_ = mean
        self.scale_ = scale
        self.coef_ = coef
        self.intercept_ = intercept
        self.standardize = standardize

    def predict(self, features: NDArray[np.floating[Any]]) -> NDArray[np.float64]:
        """Predict targets from embeddings using training-set normalization.

        Args:
            features: Two-dimensional array of embeddings with shape ``(N, D)``.

        Returns:
            Float64 predictions with shape ``(N,)`` or ``(N, T)``.

        Raises:
            ValueError: If feature dimensionality does not match the fitted probe.
        """
        x = np.asarray(features, dtype=np.float64)
        if x.ndim != 2 or x.shape[0] == 0 or not np.isfinite(x).all():
            raise ValueError('features must be a non-empty finite 2D array.')
        if x.shape[1] != self.mean_.size:
            raise ValueError(
                f'Expected {self.mean_.size} embedding dimensions, got {x.shape[1]}.'
            )
        if self.standardize:
            x = (x - self.mean_) / self.scale_
        return np.asarray(x @ self.coef_.T + self.intercept_, dtype=np.float64)

    def __repr__(self) -> str:
        """Return a concise string representation of the fitted probe."""
        return (
            f'RidgeCP(dim={self.mean_.size}, standardize={self.standardize}, '
            f'coef_shape={self.coef_.shape})'
        )


def fit_ridge_cp(
    x: NDArray[np.floating[Any]],
    y: NDArray[np.floating[Any]],
    alpha: float = 1.0,
    gamma: float = 1.0,
    boundaries: tuple[int, ...] = MRL_BOUNDARIES,
    standardize: bool = True,
) -> RidgeCP:
    """Fit a Chunked Penalty generalized ridge probe with an unpenalized intercept.

    Solves ``min_w ||X w + b - y||_2^2 + alpha * w^T diag(g) w`` where ``g_i =
    gamma ** chunk(i)`` by rescaling features with ``g ** (-1/2)`` and fitting
    standard ridge regression. Matches ``sklearn.linear_model.Ridge(fit_intercept=True)``.

    Args:
        x: Training embeddings of shape ``(N, D)``.
        y: Regression targets of shape ``(N,)`` or ``(N, T)``.
        alpha: Base L2 penalty (must be finite and positive).
        gamma: Per-chunk geometric penalty multiplier (must be finite and >= 1.0).
        boundaries: Strictly increasing positive chunk start indices.
        standardize: Whether to z-score features with training mean and sample std
            floored at ``1e-6``.

    Returns:
        Fitted :class:`RidgeCP` predictor.

    Raises:
        ValueError: If *alpha*, *gamma*, or *boundaries* are invalid.
    """
    features = np.asarray(x, dtype=np.float64)
    if features.ndim != 2 or features.shape[0] < 2 or not np.isfinite(features).all():
        raise ValueError('x must be a finite 2D array with at least 2 samples.')
    targets = np.asarray(y, dtype=np.float64)
    if (
        targets.ndim not in (1, 2)
        or targets.shape[0] != features.shape[0]
        or not np.isfinite(targets).all()
    ):
        raise ValueError(
            'y must be a finite 1D or 2D array with the same number of rows as x.'
        )
    if not np.isfinite(alpha) or alpha <= 0.0:
        raise ValueError(f'alpha must be finite and positive, got {alpha}.')
    weights = chunked_penalty_weights(features.shape[1], gamma, boundaries)
    if standardize:
        mean = features.mean(axis=0)
        scale = features.std(axis=0, ddof=1).clip(min=1e-6)
        z = (features - mean) / scale
    else:
        mean = np.zeros(features.shape[1], dtype=np.float64)
        scale = np.ones(features.shape[1], dtype=np.float64)
        z = features
    rescale = weights**-0.5
    z_scaled = z * rescale
    x_mean = z_scaled.mean(axis=0)
    y_mean = targets.mean(axis=0)
    xc = z_scaled - x_mean
    yc = targets - y_mean
    u, s, vt = np.linalg.svd(xc, full_matrices=False)
    shrink = s / (s * s + alpha)
    if targets.ndim == 1:
        w = vt.T @ (shrink * (u.T @ yc))
        intercept: float | NDArray[np.float64] = float(y_mean - x_mean @ w)
        coef = np.asarray(w * rescale, dtype=np.float64)
    else:
        w = vt.T @ (shrink[:, None] * (u.T @ yc))
        intercept = np.asarray(y_mean - x_mean @ w, dtype=np.float64)
        coef = np.asarray(w.T * rescale, dtype=np.float64)
    return RidgeCP(
        mean=mean, scale=scale, coef=coef, intercept=intercept, standardize=standardize
    )


def _valid_feature_label_mask(
    features: NDArray[np.floating[Any]], labels: NDArray[Any], task_type: str
) -> NDArray[np.bool_]:
    """Return boolean mask of rows with finite features and valid labels."""
    if task_type not in ('regression', 'classification'):
        raise ValueError(
            f"task_type must be 'regression' or 'classification', got {task_type!r}."
        )
    feat_arr = np.asarray(features)
    if feat_arr.ndim != 2:
        raise ValueError(f'features must be 2D, got shape {feat_arr.shape}.')
    lab_arr = np.asarray(labels)
    if lab_arr.shape[0] != feat_arr.shape[0]:
        raise ValueError(
            f'features ({feat_arr.shape[0]}) and labels ({lab_arr.shape[0]}) '
            'must have the same number of rows.'
        )
    if task_type == 'regression':
        lab_f = lab_arr.astype(np.float64)
        valid_y = (
            np.isfinite(lab_f).all(axis=1) if lab_f.ndim == 2 else np.isfinite(lab_f)
        )
    else:
        valid_y = np.array(
            [v is not None and (isinstance(v, str) or np.isfinite(v)) for v in lab_arr],
            dtype=np.bool_,
        )
    valid_x = np.isfinite(feat_arr).all(axis=1)
    return np.asarray(valid_y & valid_x, dtype=np.bool_)


def _standardize_train_test(
    x_tr: Tensor, x_te: Tensor, thresh: float = 1e-4
) -> tuple[Tensor, Tensor]:
    """Z-score using training-fold statistics and zero near-constant channels."""
    mean = x_tr.mean(0, keepdim=True)
    std = x_tr.std(0, keepdim=True)
    keep = std > thresh
    sd = torch.where(keep, std, torch.ones_like(std))
    return ((x_tr - mean) / sd) * keep, ((x_te - mean) / sd) * keep


def _resolve_outer_splits(
    n_valid: int,
    valid_mask: NDArray[np.bool_],
    folds: int,
    seed: int,
    test_mask: NDArray[np.bool_] | None,
    fold_assign: NDArray[np.integer[Any]] | None,
) -> tuple[
    list[tuple[NDArray[np.int64], NDArray[np.int64]]], list[NDArray[np.int64]] | None
]:
    """Build outer ``(train_idx, test_idx)`` splits and optional spatial groups."""
    all_idx = np.arange(n_valid, dtype=np.int64)
    if test_mask is not None:
        is_test = np.asarray(test_mask, dtype=np.bool_)[valid_mask]
        tr = all_idx[~is_test]
        te = all_idx[is_test]
        if tr.size == 0 or te.size == 0:
            raise ValueError('test_mask must have both training and test samples.')
        return [(tr, te)], None
    if fold_assign is not None:
        fa = np.asarray(fold_assign, dtype=np.int64)[valid_mask]
        groups = [all_idx[fa == f] for f in np.unique(fa)]
        if len(groups) < 2:
            raise ValueError('fold_assign must contain at least 2 nonempty folds.')
        splits = [
            (
                np.sort(
                    np.concatenate([groups[j] for j in range(len(groups)) if j != f])
                ),
                te,
            )
            for f, te in enumerate(groups)
        ]
        return splits, groups
    fa = random_fold_ids(n_valid, folds=folds, seed=seed)
    groups = [all_idx[fa == f] for f in range(folds) if np.any(fa == f)]
    splits = [
        (np.sort(np.concatenate([groups[j] for j in range(len(groups)) if j != f])), te)
        for f, te in enumerate(groups)
    ]
    return splits, None


def _inner_splits(
    train_idx: NDArray[np.int64],
    outer_groups: list[NDArray[np.int64]] | None,
    outer_fold_pos: int,
    folds: int,
    seed: int,
) -> list[tuple[NDArray[np.int64], NDArray[np.int64]]]:
    """Build nested inner CV ``(in_tr, in_val)`` splits strictly inside *train_idx*."""
    if outer_groups is not None and len(outer_groups) >= 3:
        inner_groups = [
            outer_groups[j] for j in range(len(outer_groups)) if j != outer_fold_pos
        ]
        return [
            (
                np.sort(
                    np.concatenate(
                        [inner_groups[k] for k in range(len(inner_groups)) if k != i]
                    )
                ),
                val,
            )
            for i, val in enumerate(inner_groups)
        ]
    n_inner = min(folds, int(train_idx.size))
    fa = random_fold_ids(int(train_idx.size), folds=n_inner, seed=seed)
    inner_groups = [train_idx[fa == f] for f in range(n_inner) if np.any(fa == f)]
    return [
        (
            np.sort(
                np.concatenate(
                    [inner_groups[k] for k in range(len(inner_groups)) if k != i]
                )
            ),
            val,
        )
        for i, val in enumerate(inner_groups)
    ]


def _score_predictions(
    pred: Tensor, targets_te: Tensor, class_idx_te: Tensor | None, task_type: str
) -> NDArray[np.float64]:
    """Score predictions as R^2 (shape ``(T,)``) or accuracy (shape ``(1,)``)."""
    if task_type == 'regression':
        y_te = targets_te.double()
        ss_res = ((y_te - pred) ** 2).sum(dim=0)
        ss_tot = (
            ((y_te - y_te.mean(dim=0, keepdim=True)) ** 2).sum(dim=0).clamp_min(1e-12)
        )
        return (1.0 - ss_res / ss_tot).cpu().numpy().astype(np.float64)
    assert class_idx_te is not None
    acc = float((pred.argmax(dim=1) == class_idx_te).float().mean())
    return np.array([acc], dtype=np.float64)


def _centered_gram(
    feats: Tensor,
    targets: Tensor,
    train_idx: Tensor,
    test_idx: Tensor,
    standardize: bool,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Standardize and center on ``train_idx`` in float64; return Gram, RHS, x_te, y_mean."""
    x_tr, x_te = feats[train_idx], feats[test_idx]
    if standardize:
        x_tr, x_te = _standardize_train_test(x_tr, x_te)
    x_tr_d = x_tr.double()
    x_te_d = x_te.double()
    y_tr_d = targets[train_idx].double()
    x_mean = x_tr_d.mean(0, keepdim=True)
    y_mean = y_tr_d.mean(0, keepdim=True)
    xc = x_tr_d - x_mean
    xte_c = x_te_d - x_mean
    return xc.T @ xc, xc.T @ (y_tr_d - y_mean), xte_c, y_mean


def _eval_profiles_alpha_path(
    feats: Tensor,
    targets: Tensor,
    class_idx: Tensor | None,
    train_idx: Tensor,
    test_idx: Tensor,
    alphas: tuple[float, ...],
    scales: list[Tensor | None],
    task_type: str,
    standardize: bool,
) -> NDArray[np.float64]:
    """Score all ``(profile, alpha)`` pairs from one centered float64 Gram matrix."""
    gram, rhs, xte_c, y_mean = _centered_gram(
        feats, targets, train_idx, test_idx, standardize
    )
    n_targets = targets.shape[1] if task_type == 'regression' else 1
    out = np.empty((len(scales), len(alphas), n_targets), dtype=np.float64)
    y_te = targets[test_idx]
    c_te = class_idx[test_idx] if class_idx is not None else None
    for pi, scale in enumerate(scales):
        if scale is None:
            g_p, rhs_p, xte_p = gram, rhs, xte_c
        else:
            g_p = scale[:, None] * gram * scale[None, :]
            rhs_p = scale[:, None] * rhs
            xte_p = xte_c * scale[None, :]
        evals, evecs = torch.linalg.eigh(g_p)
        evals = evals.clamp_min(0.0)
        proj = evecs.T @ rhs_p
        xte_v = xte_p @ evecs
        for ai, alpha in enumerate(alphas):
            pred = xte_v @ (proj / (evals[:, None] + alpha)) + y_mean
            out[pi, ai] = _score_predictions(pred, y_te, c_te, task_type)
    return out


def _predict_selected(
    feats: Tensor,
    targets: Tensor,
    train_idx: Tensor,
    test_idx: Tensor,
    alphas: tuple[float, ...],
    scales: list[Tensor | None],
    chosen_pi: NDArray[np.int64],
    chosen_ai: NDArray[np.int64],
    task_type: str,
    standardize: bool,
) -> Tensor:
    """Refit on ``train_idx`` for each selected ``(profile, alpha)`` and predict ``test_idx``."""
    gram, rhs, xte_c, y_mean = _centered_gram(
        feats, targets, train_idx, test_idx, standardize
    )
    n_test = int(test_idx.shape[0])
    if task_type == 'classification':
        pi = int(chosen_pi[0])
        ai = int(chosen_ai[0])
        scale = scales[pi]
        if scale is None:
            g_p, rhs_p, xte_p = gram, rhs, xte_c
        else:
            g_p = scale[:, None] * gram * scale[None, :]
            rhs_p = scale[:, None] * rhs
            xte_p = xte_c * scale[None, :]
        evals, evecs = torch.linalg.eigh(g_p)
        evals = evals.clamp_min(0.0)
        proj = evecs.T @ rhs_p
        return (xte_p @ evecs) @ (proj / (evals[:, None] + alphas[ai])) + y_mean

    pred = torch.empty(
        (n_test, targets.shape[1]), dtype=torch.float64, device=feats.device
    )
    for pi in np.unique(chosen_pi):
        cols = np.flatnonzero(chosen_pi == pi)
        scale = scales[int(pi)]
        if scale is None:
            g_p, rhs_p, xte_p = gram, rhs[:, cols], xte_c
        else:
            g_p = scale[:, None] * gram * scale[None, :]
            rhs_p = scale[:, None] * rhs[:, cols]
            xte_p = xte_c * scale[None, :]
        evals, evecs = torch.linalg.eigh(g_p)
        evals = evals.clamp_min(0.0)
        proj = evecs.T @ rhs_p
        xte_v = xte_p @ evecs
        for ai in np.unique(chosen_ai[cols]):
            local_mask = chosen_ai[cols] == ai
            local_idx = np.flatnonzero(local_mask)
            orig_cols = cols[local_idx]
            pred[:, orig_cols] = (
                xte_v @ (proj[:, local_idx] / (evals[:, None] + alphas[int(ai)]))
                + y_mean[:, orig_cols]
            )
    return pred


def _run_ridge_cv(
    features: NDArray[np.floating[Any]],
    labels: NDArray[Any],
    *,
    task_type: str,
    folds: int,
    seed: int,
    device: str | torch.device,
    test_mask: NDArray[np.bool_] | None,
    fold_assign: NDArray[np.integer[Any]] | None,
    alphas: tuple[float, ...],
    scale_vectors: list[NDArray[np.float64] | None],
    standardize: bool,
    confusions: list[NDArray[np.int64]] | None,
    fold_features: Callable[[NDArray[np.int64]], NDArray[np.floating[Any]]] | None,
    return_oof: bool,
) -> (
    tuple[float | NDArray[np.float64], list[float | NDArray[np.float64]]]
    | tuple[
        float | NDArray[np.float64],
        list[float | NDArray[np.float64]],
        NDArray[np.float64] | NDArray[np.int64],
    ]
):
    """Shared nested-CV closed-form ridge engine for linear and chunked probes."""
    if len(alphas) == 0:
        raise ValueError('alphas must be non-empty.')
    if any(not np.isfinite(a) or a <= 0.0 for a in alphas):
        raise ValueError('All alphas must be finite and positive.')
    if test_mask is not None and fold_features is not None:
        raise ValueError('fold_features applies to k-fold splits only, not test_mask.')

    dev = float64_device(resolve_device(device))
    valid = _valid_feature_label_mask(features, labels, task_type)
    valid_rows = np.flatnonzero(valid)
    n_valid = int(valid_rows.size)
    if n_valid < 2:
        raise ValueError(f'Need at least 2 valid samples, got {n_valid}.')

    feats = torch.as_tensor(features[valid], dtype=torch.float32, device=dev)
    lab_arr = np.asarray(labels)
    multi = task_type == 'regression' and lab_arr.ndim == 2
    class_idx: Tensor | None = None
    n_classes = 0
    if task_type == 'regression':
        y_valid = lab_arr[valid].astype(np.float64)
        targets = torch.as_tensor(y_valid, dtype=torch.float32, device=dev)
        if not multi:
            targets = targets[:, None]
    else:
        classes, inverse = np.unique(lab_arr[valid], return_inverse=True)
        n_classes = len(classes)
        class_idx = torch.as_tensor(inverse, dtype=torch.long, device=dev)
        targets = torch.nn.functional.one_hot(class_idx, num_classes=n_classes).float()

    scales_t: list[Tensor | None] = [
        None if sv is None else torch.as_tensor(sv, dtype=torch.float64, device=dev)
        for sv in scale_vectors
    ]

    outer_splits, outer_groups = _resolve_outer_splits(
        n_valid, valid, folds, seed, test_mask, fold_assign
    )

    oof: NDArray[np.float64] | NDArray[np.int64] | None = None
    if return_oof:
        if task_type == 'regression':
            oof_shape = (len(features), targets.shape[1]) if multi else (len(features),)
            oof = np.full(oof_shape, np.nan, dtype=np.float64)
        else:
            oof = np.full(len(features), -1, dtype=np.int64)

    fold_scores_raw: list[NDArray[np.float64]] = []
    for f, (tr_np, te_np) in enumerate(outer_splits):
        fx = feats
        if fold_features is not None:
            orig_tr = valid_rows[tr_np]
            fx = torch.as_tensor(
                fold_features(orig_tr)[valid], dtype=torch.float32, device=dev
            )

        n_targets = targets.shape[1] if task_type == 'regression' else 1
        if len(scales_t) == 1 and len(alphas) == 1:
            chosen_pi = np.zeros(n_targets, dtype=np.int64)
            chosen_ai = np.zeros(n_targets, dtype=np.int64)
        else:
            in_splits = _inner_splits(tr_np, outer_groups, f, folds, seed)
            in_scores = np.mean(
                [
                    _eval_profiles_alpha_path(
                        fx,
                        targets,
                        class_idx,
                        torch.as_tensor(in_tr, dtype=torch.long, device=dev),
                        torch.as_tensor(in_val, dtype=torch.long, device=dev),
                        alphas,
                        scales_t,
                        task_type,
                        standardize,
                    )
                    for in_tr, in_val in in_splits
                ],
                axis=0,
            )  # [P, A, T]
            flat_best = in_scores.reshape(-1, n_targets).argmax(axis=0)
            chosen_pi, chosen_ai = np.divmod(flat_best, len(alphas))
            if len(alphas) > 1 and any(
                int(ai) in (0, len(alphas) - 1) for ai in chosen_ai
            ):
                warnings.warn(
                    'Ridge alpha selected at grid edge; consider widening alphas.',
                    UserWarning,
                    stacklevel=3,
                )

        tr_t = torch.as_tensor(tr_np, dtype=torch.long, device=dev)
        te_t = torch.as_tensor(te_np, dtype=torch.long, device=dev)
        pred = _predict_selected(
            fx,
            targets,
            tr_t,
            te_t,
            alphas,
            scales_t,
            chosen_pi,
            chosen_ai,
            task_type,
            standardize,
        )
        c_te = class_idx[te_t] if class_idx is not None else None
        fold_scores_raw.append(_score_predictions(pred, targets[te_t], c_te, task_type))

        if task_type == 'classification' and confusions is not None:
            assert c_te is not None
            confusions.append(confusion_matrix(c_te, pred.argmax(dim=1), n_classes))

        if oof is not None:
            orig_te = valid_rows[te_np]
            if task_type == 'regression':
                pred_np = pred.cpu().numpy()
                oof[orig_te] = pred_np if multi else pred_np[:, 0]
            else:
                oof[orig_te] = pred.argmax(dim=1).cpu().numpy().astype(np.int64)

    mean_arr = np.mean(np.stack(fold_scores_raw, axis=0), axis=0)
    fold_out: list[float | NDArray[np.float64]]
    if multi:
        score_out: float | NDArray[np.float64] = mean_arr
        fold_out = list(fold_scores_raw)
    else:
        score_out = float(mean_arr[0])
        fold_out = [float(fs[0]) for fs in fold_scores_raw]

    if return_oof:
        assert oof is not None
        return score_out, fold_out, oof
    return score_out, fold_out


@overload
def linear_probe_score(
    features: NDArray[np.floating[Any]],
    labels: NDArray[Any],
    *,
    task_type: str = 'regression',
    folds: int = 5,
    seed: int = 0,
    device: str | torch.device = 'cpu',
    test_mask: NDArray[np.bool_] | None = None,
    fold_assign: NDArray[np.integer[Any]] | None = None,
    alphas: tuple[float, ...] = RIDGE_ALPHAS,
    penalty_weights: NDArray[np.floating[Any]] | None = None,
    standardize: bool = True,
    confusions: list[NDArray[np.int64]] | None = None,
    fold_features: Callable[[NDArray[np.int64]], NDArray[np.floating[Any]]]
    | None = None,
    return_oof: Literal[False] = False,
) -> tuple[float | NDArray[np.float64], list[float | NDArray[np.float64]]]: ...


@overload
def linear_probe_score(
    features: NDArray[np.floating[Any]],
    labels: NDArray[Any],
    *,
    task_type: str = 'regression',
    folds: int = 5,
    seed: int = 0,
    device: str | torch.device = 'cpu',
    test_mask: NDArray[np.bool_] | None = None,
    fold_assign: NDArray[np.integer[Any]] | None = None,
    alphas: tuple[float, ...] = RIDGE_ALPHAS,
    penalty_weights: NDArray[np.floating[Any]] | None = None,
    standardize: bool = True,
    confusions: list[NDArray[np.int64]] | None = None,
    fold_features: Callable[[NDArray[np.int64]], NDArray[np.floating[Any]]]
    | None = None,
    return_oof: Literal[True],
) -> tuple[
    float | NDArray[np.float64],
    list[float | NDArray[np.float64]],
    NDArray[np.float64] | NDArray[np.int64],
]: ...


def linear_probe_score(
    features: NDArray[np.floating[Any]],
    labels: NDArray[Any],
    *,
    task_type: str = 'regression',
    folds: int = 5,
    seed: int = 0,
    device: str | torch.device = 'cpu',
    test_mask: NDArray[np.bool_] | None = None,
    fold_assign: NDArray[np.integer[Any]] | None = None,
    alphas: tuple[float, ...] = RIDGE_ALPHAS,
    penalty_weights: NDArray[np.floating[Any]] | None = None,
    standardize: bool = True,
    confusions: list[NDArray[np.int64]] | None = None,
    fold_features: Callable[[NDArray[np.int64]], NDArray[np.floating[Any]]]
    | None = None,
    return_oof: bool = False,
) -> (
    tuple[float | NDArray[np.float64], list[float | NDArray[np.float64]]]
    | tuple[
        float | NDArray[np.float64],
        list[float | NDArray[np.float64]],
        NDArray[np.float64] | NDArray[np.int64],
    ]
):
    """Evaluate frozen features with a nested-CV closed-form ridge linear probe.

    Solves ridge regression (for regression targets) or one-hot ridge (for
    classification) in ``float64`` with an unpenalized intercept via a single
    Gram eigendecomposition per fold, matching
    ``sklearn.linear_model.Ridge(fit_intercept=True)``. Supports single-target
    ``(N,)`` and batched multi-target ``(N, T)`` regression labels.

    Split precedence: ``test_mask`` > ``fold_assign`` > random folds. When
    ``fold_assign`` has at least 3 nonempty groups, inner penalty selection uses
    leave-one-group-out CV over the remaining outer groups.

    Args:
        features: Two-dimensional feature matrix of shape ``(N, D)``.
        labels: Target array of shape ``(N,)`` or ``(N, T)`` (multi-target regression).
        task_type: Either ``'regression'`` (R^2) or ``'classification'`` (accuracy).
        folds: Number of CV folds.
        seed: Random seed for fold partitioning.
        device: PyTorch device string or object.
        test_mask: Optional boolean mask marking held-out test rows.
        fold_assign: Optional integer fold-group assignment per row.
        alphas: Tuple of candidate positive L2 penalties.
        penalty_weights: Optional positive per-channel penalty weights ``g`` of shape
            ``(D,)`` (solving generalized Tikhonov with ``alpha * diag(g)``).
        standardize: Whether to z-score features per training fold, zeroing channels
            with ``std <= 1e-4``.
        confusions: Optional list populated with one ``[C, C]`` confusion matrix per
            scored outer fold (classification only).
        fold_features: Optional callable mapping outer training row indices to a
            replacement ``(N, D')`` feature matrix (e.g. inductive fold-fit PCA).
        return_oof: Whether to also return out-of-fold predictions aligned to *features*.

    Returns:
        ``(mean_score, fold_scores)`` or ``(mean_score, fold_scores, oof_preds)``
        when ``return_oof=True``.

    Raises:
        ValueError: If inputs, penalty weights, or split specifications are invalid.
    """
    scale_vec: NDArray[np.float64] | None = None
    if penalty_weights is not None:
        pw = np.asarray(penalty_weights, dtype=np.float64)
        if pw.ndim != 1 or pw.shape[0] != np.asarray(features).shape[1]:
            raise ValueError(
                'penalty_weights must be 1D with length matching feature dimensionality.'
            )
        if not np.all(np.isfinite(pw)) or bool(np.any(pw <= 0.0)):
            raise ValueError('penalty_weights must be finite and strictly positive.')
        scale_vec = pw**-0.5

    return _run_ridge_cv(
        features,
        labels,
        task_type=task_type,
        folds=folds,
        seed=seed,
        device=device,
        test_mask=test_mask,
        fold_assign=fold_assign,
        alphas=alphas,
        scale_vectors=[scale_vec],
        standardize=standardize,
        confusions=confusions,
        fold_features=fold_features,
        return_oof=return_oof,
    )


@overload
def chunked_probe_score(
    features: NDArray[np.floating[Any]],
    labels: NDArray[Any],
    *,
    task_type: str = 'regression',
    folds: int = 5,
    seed: int = 0,
    device: str | torch.device = 'cpu',
    test_mask: NDArray[np.bool_] | None = None,
    fold_assign: NDArray[np.integer[Any]] | None = None,
    alphas: tuple[float, ...] = RIDGE_ALPHAS,
    gammas: tuple[float, ...] = CHUNK_GAMMAS,
    boundaries: tuple[int, ...] = MRL_BOUNDARIES,
    standardize: bool = True,
    confusions: list[NDArray[np.int64]] | None = None,
    return_oof: Literal[False] = False,
) -> tuple[float | NDArray[np.float64], list[float | NDArray[np.float64]]]: ...


@overload
def chunked_probe_score(
    features: NDArray[np.floating[Any]],
    labels: NDArray[Any],
    *,
    task_type: str = 'regression',
    folds: int = 5,
    seed: int = 0,
    device: str | torch.device = 'cpu',
    test_mask: NDArray[np.bool_] | None = None,
    fold_assign: NDArray[np.integer[Any]] | None = None,
    alphas: tuple[float, ...] = RIDGE_ALPHAS,
    gammas: tuple[float, ...] = CHUNK_GAMMAS,
    boundaries: tuple[int, ...] = MRL_BOUNDARIES,
    standardize: bool = True,
    confusions: list[NDArray[np.int64]] | None = None,
    return_oof: Literal[True],
) -> tuple[
    float | NDArray[np.float64],
    list[float | NDArray[np.float64]],
    NDArray[np.float64] | NDArray[np.int64],
]: ...


def chunked_probe_score(
    features: NDArray[np.floating[Any]],
    labels: NDArray[Any],
    *,
    task_type: str = 'regression',
    folds: int = 5,
    seed: int = 0,
    device: str | torch.device = 'cpu',
    test_mask: NDArray[np.bool_] | None = None,
    fold_assign: NDArray[np.integer[Any]] | None = None,
    alphas: tuple[float, ...] = RIDGE_ALPHAS,
    gammas: tuple[float, ...] = CHUNK_GAMMAS,
    boundaries: tuple[int, ...] = MRL_BOUNDARIES,
    standardize: bool = True,
    confusions: list[NDArray[np.int64]] | None = None,
    return_oof: bool = False,
) -> (
    tuple[float | NDArray[np.float64], list[float | NDArray[np.float64]]]
    | tuple[
        float | NDArray[np.float64],
        list[float | NDArray[np.float64]],
        NDArray[np.float64] | NDArray[np.int64],
    ]
):
    """Evaluate features with MIND's nested-CV Chunked Penalty ridge probe.

    Selects ``(gamma, alpha)`` jointly inside each outer training pool using inner
    CV (standardizing strictly inside each inner fold with zero leakage), where
    chunk ``j`` receives penalty ``alpha * gamma ** j``.

    Args:
        features: Two-dimensional feature matrix of shape ``(N, D)``.
        labels: Target array of shape ``(N,)`` or ``(N, T)`` (multi-target regression).
        task_type: Either ``'regression'`` (R^2) or ``'classification'`` (accuracy).
        folds: Number of CV folds.
        seed: Random seed for fold partitioning.
        device: PyTorch device string or object.
        test_mask: Optional boolean mask marking held-out test rows.
        fold_assign: Optional integer fold-group assignment per row.
        alphas: Tuple of candidate positive base L2 penalties.
        gammas: Tuple of candidate chunk penalty multipliers (>= 1.0).
        boundaries: Strictly increasing positive integer chunk boundaries.
        standardize: Whether to z-score features per training fold.
        confusions: Optional list populated with per-fold confusion matrices.
        return_oof: Whether to also return out-of-fold predictions.

    Returns:
        ``(mean_score, fold_scores)`` or ``(mean_score, fold_scores, oof_preds)``
        when ``return_oof=True``.

    Raises:
        ValueError: If *gammas* is empty or any parameter is invalid.
    """
    if len(gammas) == 0:
        raise ValueError('gammas must be non-empty.')
    feat_arr = np.asarray(features)
    if feat_arr.ndim != 2:
        raise ValueError(f'features must be 2D, got shape {feat_arr.shape}.')
    dim = feat_arr.shape[1]
    scale_vectors: list[NDArray[np.float64] | None] = [
        None
        if gamma == 1.0
        else chunked_penalty_weights(dim, gamma, boundaries) ** -0.5
        for gamma in gammas
    ]
    return _run_ridge_cv(
        features,
        labels,
        task_type=task_type,
        folds=folds,
        seed=seed,
        device=device,
        test_mask=test_mask,
        fold_assign=fold_assign,
        alphas=alphas,
        scale_vectors=scale_vectors,
        standardize=standardize,
        confusions=confusions,
        fold_features=None,
        return_oof=return_oof,
    )


__all__ = (
    'CHUNK_GAMMAS',
    'MRL_BOUNDARIES',
    'RIDGE_ALPHAS',
    'RidgeCP',
    'chunked_penalty_weights',
    'chunked_probe_score',
    'fit_ridge_cp',
    'linear_probe_score',
    'power_penalty_weights',
)
