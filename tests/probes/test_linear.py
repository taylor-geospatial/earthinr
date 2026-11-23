# Copyright (c) Taylor Geospatial. All rights reserved.
# Licensed under the MIT License.

import warnings

import numpy as np
import pytest
from sklearn.linear_model import Ridge
from sklearn.metrics import accuracy_score, r2_score

from earthinr.probes.linear import (
    RIDGE_ALPHAS,
    chunked_penalty_weights,
    chunked_probe_score,
    fit_ridge_cp,
    linear_probe_score,
    power_penalty_weights,
)


def _make_regression_data(
    seed: int = 7, n: int = 120, d: int = 8
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, d)) * np.arange(1, d + 1) + 5.0
    signal = x[:, 0] - 2.0 * x[:, 1] + 0.5 * rng.normal(size=n)
    y_reg = signal + 300.0
    y_cls = np.digitize(signal, np.quantile(signal, [0.33, 0.67]))
    return x, y_reg, y_cls


class TestPenaltyWeights:
    def test_chunked_and_power_weights(self) -> None:
        w = chunked_penalty_weights(9, gamma=3.0, boundaries=(3, 6))
        np.testing.assert_allclose(w, [1, 1, 1, 3, 3, 3, 9, 9, 9])
        w_empty = chunked_penalty_weights(4, gamma=10.0, boundaries=())
        np.testing.assert_allclose(w_empty, [1, 1, 1, 1])

        p = power_penalty_weights(4, power=2.0)
        np.testing.assert_allclose(p, [1.0, 4.0, 9.0, 16.0])

    def test_invalid_weight_args(self) -> None:
        with pytest.raises(ValueError, match='dim must be positive'):
            chunked_penalty_weights(0, gamma=1.0)
        with pytest.raises(ValueError, match='gamma must be finite'):
            chunked_penalty_weights(4, gamma=0.5)
        with pytest.raises(ValueError, match='gamma must be finite'):
            chunked_penalty_weights(4, gamma=float('inf'))
        with pytest.raises(ValueError, match='increasing positive integers'):
            chunked_penalty_weights(4, gamma=2.0, boundaries=(2, 1))
        with pytest.raises(ValueError, match='increasing positive integers'):
            chunked_penalty_weights(4, gamma=2.0, boundaries=(0, 2))
        bad_bounds: tuple[object, ...] = (1.5,)
        with pytest.raises(ValueError, match='increasing positive integers'):
            chunked_penalty_weights(4, gamma=2.0, boundaries=bad_bounds)  # ty: ignore[invalid-argument-type]
        with pytest.raises(ValueError, match='dim must be positive'):
            power_penalty_weights(0, power=1.0)
        with pytest.raises(ValueError, match='non-negative'):
            power_penalty_weights(4, power=-0.5)


class TestFitRidgeCP:
    def test_generalized_ridge_matches_closed_form_and_sklearn(self) -> None:
        rng = np.random.default_rng(7)
        x_tr = rng.normal(size=(40, 9)) * np.arange(1, 10) + 4.0
        x_tr[:, 0] = 2.0  # constant column floored at 1e-6
        y_tr = rng.normal(size=(40, 2)) + 100.0
        x_te = rng.normal(size=(10, 9)) + 12.0

        model = fit_ridge_cp(
            x_tr, y_tr, alpha=2.0, gamma=3.0, boundaries=(3, 6), standardize=True
        )
        assert 'RidgeCP(dim=9' in repr(model)

        mean = x_tr.mean(axis=0)
        scale = x_tr.std(axis=0, ddof=1).clip(min=1e-6)
        design = np.column_stack(((x_tr - mean) / scale, np.ones(40)))
        penalty = np.diag([1, 1, 1, 3, 3, 3, 9, 9, 9, 0])
        weights = np.linalg.solve(design.T @ design + 2.0 * penalty, design.T @ y_tr)
        expected = np.column_stack(((x_te - mean) / scale, np.ones(10))) @ weights
        np.testing.assert_allclose(model.predict(x_te), expected, rtol=1e-9, atol=1e-9)

        # Target shift invariance (unpenalized intercept)
        shifted = fit_ridge_cp(
            x_tr, y_tr + 1000.0, alpha=2.0, gamma=3.0, boundaries=(3, 6)
        )
        np.testing.assert_allclose(shifted.predict(x_te), model.predict(x_te) + 1000.0)

        # Unstandardized matches sklearn Ridge(alpha=alpha, fit_intercept=True)
        raw_model = fit_ridge_cp(
            x_tr, y_tr[:, 0], alpha=5.0, gamma=1.0, standardize=False
        )
        ref = Ridge(alpha=5.0, fit_intercept=True).fit(x_tr, y_tr[:, 0])
        np.testing.assert_allclose(
            raw_model.predict(x_te), ref.predict(x_te), atol=1e-9
        )

    def test_invalid_inputs(self) -> None:
        x = np.ones((6, 3))
        y = np.ones(6)
        with pytest.raises(ValueError, match='alpha must be finite'):
            fit_ridge_cp(x, y, alpha=0.0)
        model = fit_ridge_cp(x, y, alpha=1.0)
        with pytest.raises(ValueError, match='Expected 3 embedding dimensions'):
            model.predict(np.ones((2, 4)))


class TestLinearProbeScore:
    def test_exact_parity_with_sklearn_ridge_regression_and_classification(
        self,
    ) -> None:
        x, y_reg, y_cls = _make_regression_data()
        test_mask = np.zeros(len(x), dtype=np.bool_)
        test_mask[90:] = True

        for standardize in (False, True):
            for alpha in (0.01, 1.0, 100.0, 10000.0):
                score_r, folds_r = linear_probe_score(
                    x,
                    y_reg,
                    task_type='regression',
                    test_mask=test_mask,
                    alphas=(alpha,),
                    standardize=standardize,
                )
                x_tr = x[~test_mask].astype(np.float32)
                x_te = x[test_mask].astype(np.float32)
                if standardize:
                    mu = x_tr.mean(0, keepdims=True)
                    sd = x_tr.std(0, ddof=1, keepdims=True)
                    keep = sd > 1e-4
                    s = np.where(keep, sd, 1.0)
                    x_tr = ((x_tr - mu) / s) * keep
                    x_te = ((x_te - mu) / s) * keep
                ref_r = Ridge(alpha=alpha, fit_intercept=True).fit(
                    x_tr.astype(np.float64), y_reg[~test_mask].astype(np.float32)
                )
                expected_r = r2_score(
                    y_reg[test_mask].astype(np.float32),
                    ref_r.predict(x_te.astype(np.float64)),
                )
                assert isinstance(score_r, float)
                assert len(folds_r) == 1
                np.testing.assert_allclose(score_r, expected_r, rtol=0, atol=1e-6)

                score_c, _ = linear_probe_score(
                    x,
                    y_cls,
                    task_type='classification',
                    test_mask=test_mask,
                    alphas=(alpha,),
                    standardize=standardize,
                )
                onehot = np.eye(3, dtype=np.float32)[y_cls[~test_mask]]
                ref_c = Ridge(alpha=alpha, fit_intercept=True).fit(
                    x_tr.astype(np.float64), onehot
                )
                pred_c = ref_c.predict(x_te.astype(np.float64)).argmax(axis=1)
                expected_c = accuracy_score(y_cls[test_mask], pred_c)
                np.testing.assert_allclose(score_c, expected_c, rtol=0, atol=1e-6)

    def test_multi_target_regression_penalty_weights_and_oof(self) -> None:
        x, y_reg, y_cls = _make_regression_data()
        y_multi = np.stack([y_reg, -0.5 * y_reg + 20.0], axis=1)
        weights = chunked_penalty_weights(x.shape[1], gamma=5.0, boundaries=(4,))

        with warnings.catch_warnings():
            warnings.simplefilter('ignore', UserWarning)
            score_m, folds_m, oof_m = linear_probe_score(
                x,
                y_multi,
                task_type='regression',
                folds=4,
                seed=0,
                alphas=(0.1, 10.0, 1000.0),
                penalty_weights=weights,
                return_oof=True,
            )
        assert isinstance(score_m, np.ndarray)
        assert score_m.shape == (2,)
        assert len(folds_m) == 4
        assert oof_m.shape == y_multi.shape
        assert np.all(np.isfinite(oof_m))

        # Verify classification OOF + confusions + string labels + NaN/None filtering
        x_nan = x.copy()
        x_nan[0, 0] = np.nan
        labels_str = np.array([f'c{c}' for c in y_cls], dtype=object)
        labels_str[1] = None
        cms: list[np.ndarray] = []
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', UserWarning)
            score_c, folds_c, oof_c = linear_probe_score(
                x_nan,
                labels_str,
                task_type='classification',
                folds=4,
                seed=1,
                alphas=(0.1, 10.0),
                penalty_weights=weights,
                confusions=cms,
                return_oof=True,
            )
        assert isinstance(score_c, float)
        assert len(folds_c) == 4
        assert len(cms) == 4
        assert oof_c[0] == -1
        assert oof_c[1] == -1
        assert np.all(oof_c[2:] >= 0)
        # Recompute accuracy from stored confusion matrices
        cm_accs = [float(np.trace(cm) / cm.sum()) for cm in cms]
        np.testing.assert_allclose(cm_accs, [float(v) for v in folds_c], atol=1e-6)

    def test_spatial_fold_assign_two_groups_and_three_groups_and_fold_features(
        self,
    ) -> None:
        x, y_reg, _ = _make_regression_data()
        # 2-group fold_assign -> falls back to random inner CV
        fa2 = np.arange(len(x)) % 2
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', UserWarning)
            s2, f2, oof2 = linear_probe_score(
                x,
                y_reg,
                task_type='regression',
                fold_assign=fa2,
                alphas=(1.0, 10.0),
                return_oof=True,
            )
        assert isinstance(s2, float)
        assert len(f2) == 2
        assert oof2.shape == (len(x),)

        # 4-group fold_assign -> leave-one-group-out inner CV + fold_features callback
        fa4 = np.arange(len(x)) % 4
        seen_tr_lens: list[int] = []

        def _fold_proj(tr_rows: np.ndarray) -> np.ndarray:
            seen_tr_lens.append(len(tr_rows))
            return x[:, :4]

        with warnings.catch_warnings():
            warnings.simplefilter('ignore', UserWarning)
            s4, f4 = linear_probe_score(
                x,
                y_reg,
                task_type='regression',
                fold_assign=fa4,
                alphas=(1.0, 10.0),
                fold_features=_fold_proj,
            )
        assert isinstance(s4, float)
        assert len(f4) == 4
        assert seen_tr_lens == [90, 90, 90, 90]

    def test_alpha_grid_edge_warning_and_single_alpha_classification(self) -> None:
        x, y_reg, y_cls = _make_regression_data()
        with pytest.warns(UserWarning, match='grid edge'):
            linear_probe_score(
                x, y_reg, task_type='regression', folds=3, alphas=(1e-4, 1e6)
            )
        # Interior alpha selection emits no grid-edge warning
        rng = np.random.default_rng(11)
        x_noisy = rng.normal(size=(80, 20))
        y_noisy = x_noisy[:, 0] + 2.0 * rng.normal(size=80)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            s_int, _ = linear_probe_score(
                x_noisy, y_noisy, task_type='regression', folds=3, alphas=RIDGE_ALPHAS
            )
        assert isinstance(s_int, float)
        assert not [w for w in caught if issubclass(w.category, UserWarning)]
        # Single alpha classification with test_mask & confusions
        tm = np.arange(len(x)) >= 90
        cms: list[np.ndarray] = []
        s, f = linear_probe_score(
            x,
            y_cls,
            task_type='classification',
            test_mask=tm,
            alphas=(1.0,),
            confusions=cms,
        )
        assert isinstance(s, float)
        assert len(f) == 1
        assert len(cms) == 1
        assert len(RIDGE_ALPHAS) == 21

    def test_validation_errors(self) -> None:
        x, y_reg, _ = _make_regression_data(n=10, d=4)
        with pytest.raises(ValueError, match='alphas must be non-empty'):
            linear_probe_score(x, y_reg, alphas=())
        with pytest.raises(ValueError, match='finite and positive'):
            linear_probe_score(x, y_reg, alphas=(0.0,))
        with pytest.raises(ValueError, match='fold_features applies to k-fold'):
            linear_probe_score(
                x, y_reg, test_mask=np.arange(10) >= 8, fold_features=lambda _r: x
            )
        with pytest.raises(ValueError, match='task_type'):
            linear_probe_score(x, y_reg, task_type='other')
        with pytest.raises(ValueError, match='features must be 2D'):
            linear_probe_score(np.ones(10), y_reg)
        with pytest.raises(ValueError, match='same number of rows'):
            linear_probe_score(x, y_reg[:5])
        with pytest.raises(ValueError, match='at least 2 valid samples'):
            linear_probe_score(np.full((4, 2), np.nan), np.ones(4))
        with pytest.raises(ValueError, match='penalty_weights must be 1D'):
            linear_probe_score(x, y_reg, penalty_weights=np.ones(2))
        with pytest.raises(ValueError, match='strictly positive'):
            linear_probe_score(x, y_reg, penalty_weights=np.array([1.0, 0.0, 1.0, 1.0]))
        with pytest.raises(ValueError, match='both training and test'):
            linear_probe_score(x, y_reg, test_mask=np.ones(10, dtype=np.bool_))
        with pytest.raises(ValueError, match='at least 2 nonempty folds'):
            linear_probe_score(x, y_reg, fold_assign=np.zeros(10, dtype=np.int64))


class TestChunkedProbeScore:
    def test_regression_and_classification_nested_selection(self) -> None:
        x, y_reg, y_cls = _make_regression_data()
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', UserWarning)
            s_r, f_r, oof_r = chunked_probe_score(
                x,
                y_reg,
                task_type='regression',
                folds=3,
                alphas=(0.1, 10.0),
                gammas=(1.0, 10.0),
                boundaries=(4,),
                return_oof=True,
            )
            cms: list[np.ndarray] = []
            s_c, f_c = chunked_probe_score(
                x,
                y_cls,
                task_type='classification',
                folds=3,
                alphas=(0.1, 10.0),
                gammas=(1.0, 10.0),
                boundaries=(4,),
                confusions=cms,
            )
        assert isinstance(s_r, float)
        assert s_r > 0.7
        assert len(f_r) == 3
        assert oof_r.shape == (len(x),)
        assert isinstance(s_c, float)
        assert s_c > 0.6
        assert len(f_c) == 3
        assert len(cms) == 3

        # Multi-target regression where different targets can pick different gammas/alphas
        y_two = np.stack([y_reg, x[:, 5] * 3.0 + 1.0], axis=1)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', UserWarning)
            s_m, f_m = chunked_probe_score(
                x,
                y_two,
                task_type='regression',
                folds=3,
                alphas=(0.01, 100.0),
                gammas=(1.0, 1000.0),
                boundaries=(4,),
            )
        assert isinstance(s_m, np.ndarray)
        assert s_m.shape == (2,)
        assert len(f_m) == 3

    def test_validation_errors(self) -> None:
        x, y_reg, _ = _make_regression_data(n=10, d=4)
        with pytest.raises(ValueError, match='gammas must be non-empty'):
            chunked_probe_score(x, y_reg, gammas=())
        with pytest.raises(ValueError, match='features must be 2D'):
            chunked_probe_score(np.ones(10), y_reg, gammas=(1.0,))
