# Copyright (c) Taylor Geospatial. All rights reserved.
# Licensed under the MIT License.

"""Encoder-free spatial interpolation baselines: Coordinate IDW and Local Kriging."""

from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import least_squares
from scipy.spatial import KDTree
from sklearn.neighbors import BallTree

from .linear import _resolve_outer_splits
from .metrics import confusion_matrix
from .splits import EARTH_RADIUS_KM, _unit_xyz, spatial_fold_ids

_VARIO_POINTS = 3000
_VARIO_RANDOM_PAIRS = 200_000
_VARIO_NN = 10
_VARIO_BINS = 40
_KRIGE_CHUNK = 4096
_VARIOGRAM_MODELS: tuple[str, ...] = ('exponential', 'spherical', 'gaussian')


def _valid_coord_label_mask(
    lat: NDArray[np.floating[Any]],
    lon: NDArray[np.floating[Any]],
    labels: NDArray[Any],
    task_type: str,
) -> NDArray[np.bool_]:
    """Validate shapes and return mask of rows with finite coordinates and labels."""
    if task_type not in ('regression', 'classification'):
        raise ValueError(
            f"task_type must be 'regression' or 'classification', got {task_type!r}."
        )
    lat_arr = np.asarray(lat, dtype=np.float64)
    lon_arr = np.asarray(lon, dtype=np.float64)
    lab_arr = np.asarray(labels)
    if lat_arr.ndim != 1 or lat_arr.shape != lon_arr.shape:
        raise ValueError('lat and lon must be 1D arrays of equal length.')
    if lab_arr.ndim != 1 or lab_arr.shape[0] != lat_arr.shape[0]:
        raise ValueError('labels must be a 1D array matching lat and lon length.')
    valid_xy = np.isfinite(lat_arr) & np.isfinite(lon_arr)
    if task_type == 'regression':
        valid_y = np.isfinite(lab_arr.astype(np.float64))
    else:
        valid_y = np.array(
            [v is not None and (isinstance(v, str) or np.isfinite(v)) for v in lab_arr],
            dtype=np.bool_,
        )
    return np.asarray(valid_xy & valid_y, dtype=np.bool_)


def idw_probe_score(
    lat: NDArray[np.floating[Any]],
    lon: NDArray[np.floating[Any]],
    labels: NDArray[Any],
    *,
    task_type: str = 'regression',
    k: int = 50,
    folds: int = 5,
    seed: int = 0,
    test_mask: NDArray[np.bool_] | None = None,
    fold_assign: NDArray[np.integer[Any]] | None = None,
    confusions: list[NDArray[np.int64]] | None = None,
) -> tuple[float, list[float]]:
    """Evaluate inverse-distance-weighted great-circle k-NN interpolation.

    Queries a Haversine :class:`sklearn.neighbors.BallTree` on ``(lat, lon)`` with
    inverse-distance weights floored at 10 m (``0.01 km``) for coincident points.
    Uses deterministic :func:`neuralftw.probes.splits.random_fold_ids` for random folds.

    Args:
        lat: One-dimensional array of latitudes in degrees.
        lon: One-dimensional array of longitudes in degrees.
        labels: One-dimensional target array.
        task_type: Either ``'regression'`` (R^2) or ``'classification'`` (accuracy).
        k: Number of nearest training neighbors (must be at least 1).
        folds: Number of CV folds.
        seed: Random seed for fold partitioning.
        test_mask: Optional boolean mask marking held-out test rows.
        fold_assign: Optional integer fold-group assignment per row.
        confusions: Optional list populated with per-fold confusion matrices
            (classification only).

    Returns:
        Tuple ``(mean_score, fold_scores)``.

    Raises:
        ValueError: If *k* < 1, fewer than 2 valid samples exist, or inputs are invalid.
    """
    if k < 1:
        raise ValueError(f'k must be at least 1, got {k}.')
    valid = _valid_coord_label_mask(lat, lon, labels, task_type)
    n_valid = int(valid.sum())
    if n_valid < 2:
        raise ValueError(f'Need at least 2 valid samples, got {n_valid}.')

    la = np.asarray(lat, dtype=np.float64)[valid]
    lo = np.asarray(lon, dtype=np.float64)[valid]
    lab = np.asarray(labels)[valid]
    is_reg = task_type == 'regression'
    if is_reg:
        y_reg = lab.astype(np.float64)
        y_cls = np.empty(0, dtype=np.int64)
        n_classes = 0
    else:
        classes, y_cls = np.unique(lab, return_inverse=True)
        y_reg = np.empty(0, dtype=np.float64)
        n_classes = len(classes)

    outer_splits, _ = _resolve_outer_splits(
        n_valid, valid, folds, seed, test_mask, fold_assign
    )

    rad = np.radians(np.stack([la, lo], axis=1))
    fold_scores: list[float] = []
    for tr_idx, te_idx in outer_splits:
        tree = BallTree(rad[tr_idx], metric='haversine')
        k_eff = min(k, len(tr_idx))
        dist, nn_idx = tree.query(rad[te_idx], k=k_eff)
        dist_km = dist * EARTH_RADIUS_KM
        inv = 1.0 / np.maximum(dist_km, 0.01)
        weights = inv / inv.sum(axis=1, keepdims=True)
        if is_reg:
            pred = (weights * y_reg[tr_idx][nn_idx]).sum(axis=1)
            y_te = y_reg[te_idx]
            ss_res = float(np.sum((y_te - pred) ** 2))
            ss_tot = max(float(np.sum((y_te - y_te.mean()) ** 2)), 1e-12)
            fold_scores.append(float(1.0 - ss_res / ss_tot))
        else:
            neigh = y_cls[tr_idx][nn_idx]
            votes = np.zeros((len(te_idx), n_classes), dtype=np.float64)
            rows = np.repeat(np.arange(len(te_idx)), k_eff)
            np.add.at(votes, (rows, neigh.ravel()), weights.ravel())
            pred_cls = votes.argmax(axis=1).astype(np.int64)
            if confusions is not None:
                confusions.append(confusion_matrix(y_cls[te_idx], pred_cls, n_classes))
            fold_scores.append(float(np.mean(pred_cls == y_cls[te_idx])))

    return float(np.mean(fold_scores)), fold_scores


def _variogram_corr(
    model: str, h: NDArray[np.float64], rng: float
) -> NDArray[np.float64]:
    """Evaluate isotropic correlation function ``rho(h)`` for a variogram model."""
    if model == 'exponential':
        return np.exp(-3.0 * h / rng)
    if model == 'gaussian':
        return np.exp(-3.0 * (h / rng) ** 2)
    r = np.minimum(h / rng, 1.0)
    return np.asarray(1.0 - 1.5 * r + 0.5 * r**3, dtype=np.float64)


def _fit_variograms(
    x: NDArray[np.float64],
    y: NDArray[np.float64] | NDArray[np.int64],
    seed: int,
    categorical: bool = False,
) -> dict[str, dict[str, Any]]:
    """Fit exponential, spherical, and Gaussian variograms by Cressie-weighted LS."""
    rs = np.random.default_rng(seed)
    n = len(x)
    s = rs.choice(n, min(n, _VARIO_POINTS), replace=False)
    i = rs.choice(n, _VARIO_RANDOM_PAIRS)
    j = rs.choice(n, _VARIO_RANDOM_PAIRS)
    k_nn = min(_VARIO_NN + 1, n)
    _, nn = KDTree(x).query(x[s], k=k_nn)
    nn_arr = np.asarray(nn).reshape(len(s), k_nn)
    i = np.concatenate([i, np.repeat(s, k_nn - 1)])
    j = np.concatenate([j, nn_arr[:, 1:].ravel()])
    keep = i != j
    h = np.linalg.norm(x[i[keep]] - x[j[keep]], axis=1)
    if categorical:
        g = (y[i[keep]] != y[j[keep]]).astype(np.float64)
        p = np.bincount(y.astype(np.int64)) / float(len(y))
        var = float(1.0 - np.sum(p**2))
    else:
        g = 0.5 * (y[i[keep]] - y[j[keep]]) ** 2
        var = float(np.var(y))

    pos = h > 0.0
    h, g = h[pos], g[pos]
    if h.size == 0:
        return {
            m: {'model': m, 'nugget': max(var, 1e-6), 'psill': 0.0, 'range': 1.0}
            for m in _VARIOGRAM_MODELS
        }

    edges = np.unique(np.quantile(h, np.linspace(0.0, 1.0, _VARIO_BINS + 1)))
    if len(edges) < 2:
        hb = np.array([float(np.mean(h))], dtype=np.float64)
        gb = np.array([float(np.mean(g))], dtype=np.float64)
        nb = np.array([float(len(h))], dtype=np.float64)
    else:
        b = np.clip(np.searchsorted(edges, h, side='right') - 1, 0, len(edges) - 2)
        cnt = np.bincount(b, minlength=len(edges) - 1)
        ok = cnt > 0
        hb = np.bincount(b, weights=h, minlength=len(cnt))[ok] / cnt[ok]
        gb = np.bincount(b, weights=g, minlength=len(cnt))[ok] / cnt[ok]
        nb = cnt[ok].astype(np.float64)

    fits: dict[str, dict[str, Any]] = {}
    upper_sill = 4.0 * var + 1e-12
    upper_range = max(10.0 * float(hb.max()), 1e-2)
    p0 = (0.1 * var, 0.9 * var, max(float(np.median(hb)), 1e-3))
    for model in _VARIOGRAM_MODELS:

        def resid(params: NDArray[np.float64], m: str = model) -> NDArray[np.float64]:
            nug, ps, rg = float(params[0]), float(params[1]), float(params[2])
            gm = nug + ps * (1.0 - _variogram_corr(m, hb, rg))
            return np.asarray(
                np.sqrt(nb) * (gb - gm) / np.maximum(gm, 1e-12 * var + 1e-30),
                dtype=np.float64,
            )

        res = least_squares(
            resid, p0, bounds=([0.0, 0.0, 1e-3], [upper_sill, upper_sill, upper_range])
        )
        nug, ps, rg = res.x
        fits[model] = {
            'model': model,
            'nugget': float(nug),
            'psill': float(ps),
            'range': float(rg),
        }
    return fits


def _krige_weights(
    x_tr: NDArray[np.float64], x_te: NDArray[np.float64], vario: dict[str, Any], k: int
) -> tuple[NDArray[np.int64], NDArray[np.float64]]:
    """Compute local ordinary kriging weights for the *k* nearest training points."""
    k_eff = min(k, len(x_tr))
    _, raw_idx = KDTree(x_tr).query(x_te, k=k_eff)
    idx = np.asarray(raw_idx, dtype=np.int64).reshape(len(x_te), k_eff)
    sill = float(vario['nugget']) + float(vario['psill'])
    jitter = 1e-8 * max(sill, 1e-12)
    out = np.empty((len(x_te), k_eff), dtype=np.float64)
    for start in range(0, len(x_te), _KRIGE_CHUNK):
        stop = min(start + _KRIGE_CHUNK, len(x_te))
        ii = idx[start:stop]
        xn = x_tr[ii]  # (m, k, 3)
        dnn = np.linalg.norm(xn[:, :, None, :] - xn[:, None, :, :], axis=-1)
        cov = float(vario['psill']) * _variogram_corr(
            vario['model'], dnn, float(vario['range'])
        )
        diag = np.arange(k_eff)
        cov[:, diag, diag] = sill + jitter
        d0 = np.linalg.norm(xn - x_te[start:stop, None, :], axis=-1)
        c0 = float(vario['psill']) * _variogram_corr(
            vario['model'], d0, float(vario['range'])
        )
        c0 = np.where(d0 == 0.0, sill, c0)
        m = len(ii)
        lhs = np.ones((m, k_eff + 1, k_eff + 1), dtype=np.float64)
        lhs[:, :k_eff, :k_eff] = cov
        lhs[:, k_eff, k_eff] = 0.0
        rhs = np.ones((m, k_eff + 1), dtype=np.float64)
        rhs[:, :k_eff] = c0
        try:
            w = np.linalg.solve(lhs, rhs[..., None])[..., 0]
        except np.linalg.LinAlgError:
            w = np.stack(
                [
                    np.linalg.lstsq(lhs_i, rhs_i, rcond=None)[0]
                    for lhs_i, rhs_i in zip(lhs, rhs, strict=True)
                ]
            )
        out[start:stop] = w[:, :k_eff]
    return idx, out


def _krige_regress(
    x_tr: NDArray[np.float64],
    y_tr: NDArray[np.float64],
    x_te: NDArray[np.float64],
    vario: dict[str, Any],
    k: int,
) -> NDArray[np.float64]:
    """Predict continuous values via local ordinary kriging."""
    idx, w = _krige_weights(x_tr, x_te, vario, k)
    return np.asarray((w * y_tr[idx]).sum(axis=1), dtype=np.float64)


def _krige_classify(
    x_tr: NDArray[np.float64],
    y_tr: NDArray[np.int64],
    x_te: NDArray[np.float64],
    vario: dict[str, Any],
    n_classes: int,
    k: int,
) -> NDArray[np.int64]:
    """Predict classes via median indicator kriging."""
    idx, w = _krige_weights(x_tr, x_te, vario, k)
    scores = np.zeros((len(x_te), n_classes), dtype=np.float64)
    rows = np.repeat(np.arange(len(x_te)), idx.shape[1])
    np.add.at(scores, (rows, y_tr[idx].ravel()), w.ravel())
    return scores.argmax(axis=1).astype(np.int64)


def _select_kriging_model(
    x_tr: NDArray[np.float64],
    y_tr: NDArray[np.float64] | NDArray[np.int64],
    cell_deg: float,
    seed: int,
    k: int,
    n_classes: int = 0,
) -> str:
    """Select variogram model on an inner validation split of the training fold."""
    if cell_deg == 0.0:
        val = np.random.default_rng(seed).random(len(x_tr)) < 0.2
    else:
        lat_tr = np.degrees(np.arcsin(np.clip(x_tr[:, 2] / EARTH_RADIUS_KM, -1.0, 1.0)))
        lon_tr = np.degrees(np.arctan2(x_tr[:, 1], x_tr[:, 0]))
        val = (
            spatial_fold_ids(lat_tr, lon_tr, folds=5, cell_deg=cell_deg, seed=seed) == 0
        )
    if bool(val.all()) or not bool(val.any()):
        return 'exponential'
    fits = _fit_variograms(x_tr[~val], y_tr[~val], seed, categorical=n_classes > 0)
    if n_classes > 0:
        y_int = y_tr.astype(np.int64)
        err = {
            m: float(
                np.mean(
                    _krige_classify(x_tr[~val], y_int[~val], x_tr[val], v, n_classes, k)
                    != y_int[val]
                )
            )
            for m, v in fits.items()
        }
    else:
        y_flt = y_tr.astype(np.float64)
        err = {
            m: float(
                np.mean(
                    (
                        _krige_regress(x_tr[~val], y_flt[~val], x_tr[val], v, k)
                        - y_flt[val]
                    )
                    ** 2
                )
            )
            for m, v in fits.items()
        }
    return min(err, key=err.__getitem__)


def kriging_probe_score(
    lat: NDArray[np.floating[Any]],
    lon: NDArray[np.floating[Any]],
    labels: NDArray[Any],
    *,
    task_type: str = 'regression',
    k: int = 50,
    folds: int = 5,
    seed: int = 0,
    cell_deg: float = 0.0,
    test_mask: NDArray[np.bool_] | None = None,
    fold_assign: NDArray[np.integer[Any]] | None = None,
    confusions: list[NDArray[np.int64]] | None = None,
) -> tuple[float, list[float]]:
    """Evaluate local ordinary kriging (regression) or median indicator kriging (classification).

    Fits isotropic exponential, spherical, and Gaussian variograms with nugget by
    Cressie-weighted least squares on 3D Earth-sphere chord distances, selects the
    best variogram model via a nested inner validation split (random 20% when
    ``cell_deg == 0`` or one spatial block group when ``cell_deg > 0``), refits on
    the full outer training fold, and predicts from the *k* nearest neighbors.

    Args:
        lat: One-dimensional array of latitudes in degrees.
        lon: One-dimensional array of longitudes in degrees.
        labels: One-dimensional target array.
        task_type: Either ``'regression'`` (R^2) or ``'classification'`` (accuracy).
        k: Number of nearest training neighbors (must be at least 1).
        folds: Number of CV folds.
        seed: Random seed for outer folds and variogram subsampling.
        cell_deg: Spatial block size in degrees used for inner variogram model
            selection (``0.0`` uses a random 20% inner validation split).
        test_mask: Optional boolean mask marking held-out test rows.
        fold_assign: Optional integer fold-group assignment per row.
        confusions: Optional list populated with per-fold confusion matrices
            (classification only).

    Returns:
        Tuple ``(mean_score, fold_scores)``.

    Raises:
        ValueError: If *k* < 1, *cell_deg* < 0, fewer than 2 valid samples exist, or
            inputs are invalid.
    """
    if k < 1:
        raise ValueError(f'k must be at least 1, got {k}.')
    if cell_deg < 0.0:
        raise ValueError(f'cell_deg must be non-negative, got {cell_deg}.')
    valid = _valid_coord_label_mask(lat, lon, labels, task_type)
    n_valid = int(valid.sum())
    if n_valid < 2:
        raise ValueError(f'Need at least 2 valid samples, got {n_valid}.')

    la = np.asarray(lat, dtype=np.float64)[valid]
    lo = np.asarray(lon, dtype=np.float64)[valid]
    lab = np.asarray(labels)[valid]
    is_reg = task_type == 'regression'
    if is_reg:
        y_reg = lab.astype(np.float64)
        y_cls = np.empty(0, dtype=np.int64)
        n_classes = 0
    else:
        classes, y_cls = np.unique(lab, return_inverse=True)
        y_reg = np.empty(0, dtype=np.float64)
        n_classes = len(classes)

    outer_splits, _ = _resolve_outer_splits(
        n_valid, valid, folds, seed, test_mask, fold_assign
    )

    xyz = _unit_xyz(la, lo) * EARTH_RADIUS_KM
    fold_scores: list[float] = []
    for f, (tr_idx, te_idx) in enumerate(outer_splits):
        fseed = 1000 * seed + f
        x_tr, x_te = xyz[tr_idx], xyz[te_idx]
        if is_reg:
            y_tr_raw = y_reg[tr_idx]
            mu = float(y_tr_raw.mean())
            sd = float(y_tr_raw.std()) or 1.0
            y_tr = (y_tr_raw - mu) / sd
            model = _select_kriging_model(x_tr, y_tr, cell_deg, fseed, k)
            vario = _fit_variograms(x_tr, y_tr, seed=fseed)[model]
            pred = _krige_regress(x_tr, y_tr, x_te, vario, k) * sd + mu
            y_te = y_reg[te_idx]
            ss_res = float(np.sum((y_te - pred) ** 2))
            ss_tot = max(float(np.sum((y_te - y_te.mean()) ** 2)), 1e-12)
            fold_scores.append(float(1.0 - ss_res / ss_tot))
        else:
            cls, y_tr_compact = np.unique(y_cls[tr_idx], return_inverse=True)
            model = _select_kriging_model(
                x_tr, y_tr_compact, cell_deg, fseed, k, n_classes=len(cls)
            )
            vario = _fit_variograms(x_tr, y_tr_compact, seed=fseed, categorical=True)[
                model
            ]
            pred_cls = cls[
                _krige_classify(x_tr, y_tr_compact, x_te, vario, len(cls), k)
            ].astype(np.int64)
            if confusions is not None:
                confusions.append(confusion_matrix(y_cls[te_idx], pred_cls, n_classes))
            fold_scores.append(float(np.mean(pred_cls == y_cls[te_idx])))

    return float(np.mean(fold_scores)), fold_scores


__all__ = ('idw_probe_score', 'kriging_probe_score')
