# Copyright (c) Taylor Geospatial. All rights reserved.
# Licensed under the MIT License.

"""Downstream probing, splitting, metrics, and spatial interpolation utilities."""

from .classifiers import (
    DEFAULT_LOGREG_CS,
    LBFGSLogReg,
    knn_probe_score,
    logreg_probe_score,
    mlp_probe_score,
)
from .linear import (
    CHUNK_GAMMAS,
    MRL_BOUNDARIES,
    RIDGE_ALPHAS,
    RidgeCP,
    chunked_penalty_weights,
    chunked_probe_score,
    fit_ridge_cp,
    linear_probe_score,
    power_penalty_weights,
)
from .metrics import (
    classification_metrics_from_cm,
    confusion_matrix,
    load_sparse_cm,
    save_sparse_cm,
)
from .spatial import idw_probe_score, kriging_probe_score
from .splits import (
    EARTH_RADIUS_KM,
    GEONAMES_CITIES15000_URL,
    RANDOM_FOLD_ALGORITHM,
    buffered_exclusion_mask,
    load_geonames_cities,
    random_fold_ids,
    spatial_fold_ids,
    urban_rural_tags,
)

__all__ = (
    'CHUNK_GAMMAS',
    'DEFAULT_LOGREG_CS',
    'EARTH_RADIUS_KM',
    'GEONAMES_CITIES15000_URL',
    'MRL_BOUNDARIES',
    'RANDOM_FOLD_ALGORITHM',
    'RIDGE_ALPHAS',
    'LBFGSLogReg',
    'RidgeCP',
    'buffered_exclusion_mask',
    'chunked_penalty_weights',
    'chunked_probe_score',
    'classification_metrics_from_cm',
    'confusion_matrix',
    'fit_ridge_cp',
    'idw_probe_score',
    'knn_probe_score',
    'kriging_probe_score',
    'linear_probe_score',
    'load_geonames_cities',
    'load_sparse_cm',
    'logreg_probe_score',
    'mlp_probe_score',
    'power_penalty_weights',
    'random_fold_ids',
    'save_sparse_cm',
    'spatial_fold_ids',
    'urban_rural_tags',
)
