# Copyright (c) Taylor Geospatial. All rights reserved.
# Licensed under the MIT License.

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from sklearn.linear_model import LogisticRegression

from neuralftw._errors import DependencyNotFoundError
from neuralftw.probes import classifiers as clf_mod
from neuralftw.probes.classifiers import (
    LBFGSLogReg,
    knn_probe_score,
    logreg_probe_score,
    mlp_probe_score,
)


def _make_clf_data(
    seed: int = 3, n: int = 90, d: int = 6
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, d)).astype(np.float32)
    x[:, -1] = 3.0  # constant channel to exercise near-constant handling
    y_reg = (2.0 * x[:, 0] - 1.5 * x[:, 1] + 0.2 * rng.normal(size=n)).astype(
        np.float32
    )
    y_cls = np.digitize(y_reg, np.quantile(y_reg, [0.33, 0.67]))
    return x, y_reg, y_cls


class TestKNNProbeScore:
    def test_sklearn_fallback_and_confusions(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        x, _, y_cls = _make_clf_data()

        def _missing(_name: str, extra: str | None = None) -> object:
            raise DependencyNotFoundError(f'missing {extra}')

        monkeypatch.setattr(clf_mod, 'lazy_import', _missing)

        cms: list[np.ndarray] = []
        score, fold_scores = knn_probe_score(
            x, y_cls, k=5, folds=3, seed=0, confusions=cms
        )
        assert 0.5 < score <= 1.0
        assert len(fold_scores) == 3
        assert len(cms) == 3

        # test_mask path without confusions
        tm = np.arange(len(x)) >= 60
        s_tm, f_tm = knn_probe_score(x, y_cls, k=3, test_mask=tm)
        assert 0.5 < s_tm <= 1.0
        assert len(f_tm) == 1

    def test_faissknn_branch_via_monkeypatch(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        x, _, y_cls = _make_clf_data(n=60)
        seen_devices: list[str] = []

        class _FakeFaissKNN:
            def __init__(
                self, n_neighbors: int, n_classes: int, device: str, metric: str
            ) -> None:
                assert n_neighbors >= 1
                assert n_classes == 3
                assert metric == 'l2'
                seen_devices.append(device)
                self._x: np.ndarray | None = None
                self._y: np.ndarray | None = None

            def fit(self, x_tr: np.ndarray, y_tr: np.ndarray) -> '_FakeFaissKNN':
                self._x = x_tr
                self._y = y_tr
                return self

            def predict(self, x_te: np.ndarray) -> np.ndarray:
                assert self._x is not None
                assert self._y is not None
                d2 = ((x_te[:, None, :] - self._x[None, :, :]) ** 2).sum(axis=2)
                return self._y[d2.argmin(axis=1)]

        fake_mod = SimpleNamespace(FaissKNNClassifier=_FakeFaissKNN)
        monkeypatch.setattr(clf_mod, 'lazy_import', lambda _n, extra=None: fake_mod)
        score, folds = knn_probe_score(x, y_cls, k=3, folds=3, device='cpu')
        assert 0.5 < score <= 1.0
        assert len(folds) == 3
        assert seen_devices == ['cpu', 'cpu', 'cpu']

    def test_validation_errors(self) -> None:
        x, _, y_cls = _make_clf_data(n=10)
        with pytest.raises(ValueError, match='k must be at least 1'):
            knn_probe_score(x, y_cls, k=0)
        with pytest.raises(ValueError, match='at least 2 valid samples'):
            knn_probe_score(np.full((4, 2), np.nan), np.array([0, 1, 0, 1]))


class TestLBFGSLogRegAndProbe:
    def test_lbfgs_logreg_matches_sklearn_and_errors(self) -> None:
        x, _, y_cls = _make_clf_data(n=120, d=4)
        x_tr, x_te = x[:90, :3], x[90:, :3]
        y_tr, y_te = y_cls[:90], y_cls[90:]

        model = LBFGSLogReg(c=1.0, max_iter=500, device='cpu', seed=0)
        with pytest.raises(RuntimeError, match='must be fitted'):
            model.predict(torch.as_tensor(x_te))

        model.fit(torch.as_tensor(x_tr), torch.as_tensor(y_tr, dtype=torch.long))
        pred = model.predict(torch.as_tensor(x_te))
        ref = LogisticRegression(C=1.0, max_iter=500).fit(x_tr, y_tr).predict(x_te)
        assert float(np.mean(pred == ref)) >= 0.9
        assert float(np.mean(pred == y_te)) >= 0.7

        with pytest.raises(ValueError, match='c must be finite'):
            LBFGSLogReg(c=0.0)
        with pytest.raises(ValueError, match='max_iter'):
            LBFGSLogReg(max_iter=0)
        with pytest.raises(ValueError, match='tol must be finite'):
            LBFGSLogReg(tol=0.0)

    def test_logreg_probe_nested_cv_single_c_and_confusions(self) -> None:
        x, _, y_cls = _make_clf_data(n=90, d=4)
        cms: list[np.ndarray] = []
        s_multi, f_multi = logreg_probe_score(
            x,
            y_cls,
            folds=3,
            seed=0,
            c_values=(0.1, 1.0, 10.0),
            max_iter=200,
            confusions=cms,
        )
        assert s_multi > 0.7
        assert len(f_multi) == 3
        assert len(cms) == 3

        # Single C skips inner CV; spatial fold_assign
        fa = np.arange(len(x)) % 3
        s_single, f_single = logreg_probe_score(
            x, y_cls, c_values=(1.0,), max_iter=200, fold_assign=fa
        )
        assert s_single > 0.7
        assert len(f_single) == 3

    def test_logreg_validation_errors(self) -> None:
        x, _, y_cls = _make_clf_data(n=10, d=3)
        with pytest.raises(ValueError, match='c_values must be non-empty'):
            logreg_probe_score(x, y_cls, c_values=())
        with pytest.raises(ValueError, match='at least 2 valid samples'):
            logreg_probe_score(np.full((4, 2), np.nan), np.array([0, 1, 0, 1]))


class TestMLPProbeScore:
    def test_regression_and_classification_with_early_stopping_and_nan_filter(
        self,
    ) -> None:
        x, y_reg, y_cls = _make_clf_data(n=100, d=5)
        x_nan = x.copy()
        x_nan[0, 0] = np.nan

        s_reg, f_reg = mlp_probe_score(
            x_nan,
            y_reg,
            task_type='regression',
            folds=2,
            seed=0,
            hidden=16,
            dropout=0.1,
            epochs=30,
            batch_size=32,
            lr=1e-2,
            patience=3,
        )
        assert s_reg > 0.5
        assert len(f_reg) == 2

        cms: list[np.ndarray] = []
        s_cls, f_cls = mlp_probe_score(
            x_nan,
            y_cls,
            task_type='classification',
            folds=2,
            seed=0,
            hidden=16,
            dropout=0.0,
            epochs=25,
            batch_size=32,
            lr=1e-2,
            patience=2,
            confusions=cms,
        )
        assert s_cls > 0.6
        assert len(f_cls) == 2
        assert len(cms) == 2

        # Classification without confusions + 1 epoch (no early stop trigger)
        tm = np.arange(len(x)) >= 80
        s_one, f_one = mlp_probe_score(
            x,
            y_cls,
            task_type='classification',
            test_mask=tm,
            hidden=8,
            epochs=1,
            patience=5,
        )
        assert 0.0 <= s_one <= 1.0
        assert len(f_one) == 1

    def test_validation_errors(self) -> None:
        x, y_reg, _ = _make_clf_data(n=10, d=3)
        with pytest.raises(ValueError, match='must be positive'):
            mlp_probe_score(x, y_reg, hidden=0)
        with pytest.raises(ValueError, match='dropout must be in'):
            mlp_probe_score(x, y_reg, dropout=1.0)
        with pytest.raises(ValueError, match='at least 2 valid samples'):
            mlp_probe_score(np.full((4, 2), np.nan), np.ones(4))
