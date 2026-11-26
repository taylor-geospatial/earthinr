# Copyright (c) Taylor Geospatial. All rights reserved.
# Licensed under the MIT License.

import pathlib

import numpy as np
import pytest
import torch
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    f1_score,
)

from neuralftw.probes.metrics import (
    classification_metrics_from_cm,
    confusion_matrix,
    load_sparse_cm,
    save_sparse_cm,
)


class TestConfusionMatrix:
    def test_numpy_and_torch_inputs(self) -> None:
        y_true = np.array([0, 0, 1, 2, 2, 2])
        y_pred = np.array([0, 1, 1, 2, 0, 2])
        cm_np = confusion_matrix(y_true, y_pred, n_classes=4)
        cm_pt = confusion_matrix(
            torch.as_tensor(y_true), torch.as_tensor(y_pred), n_classes=4
        )
        expected = np.array(
            [[1, 1, 0, 0], [0, 1, 0, 0], [1, 0, 2, 0], [0, 0, 0, 0]], dtype=np.int64
        )
        np.testing.assert_array_equal(cm_np, expected)
        np.testing.assert_array_equal(cm_pt, expected)

    def test_invalid_inputs(self) -> None:
        with pytest.raises(ValueError, match='positive'):
            confusion_matrix(np.array([0]), np.array([0]), n_classes=0)
        with pytest.raises(ValueError, match='does not match'):
            confusion_matrix(np.array([0, 1]), np.array([0]), n_classes=2)


class TestClassificationMetricsFromCM:
    def test_matches_sklearn_including_unseen_and_predicted_only_classes(self) -> None:
        # Class 3 is unseen in truth & pred; class 2 is predicted only
        y_true = np.array([0, 0, 0, 1, 1, 1, 1, 0])
        y_pred = np.array([0, 0, 1, 1, 1, 2, 0, 0])
        cm = confusion_matrix(y_true, y_pred, n_classes=4)
        got = classification_metrics_from_cm(cm)

        assert got['accuracy'] == pytest.approx(accuracy_score(y_true, y_pred))
        assert got['macro_f1'] == pytest.approx(
            f1_score(y_true, y_pred, average='macro')
        )
        assert got['balanced_accuracy'] == pytest.approx(
            balanced_accuracy_score(y_true, y_pred)
        )
        assert got['cohen_kappa'] == pytest.approx(cohen_kappa_score(y_true, y_pred))

    def test_degenerate_single_class_and_empty_cm(self) -> None:
        cm_single = np.array([[5, 0], [0, 0]], dtype=np.int64)
        got_single = classification_metrics_from_cm(cm_single)
        assert got_single['accuracy'] == 1.0
        assert got_single['cohen_kappa'] == 0.0

        cm_zero = np.zeros((3, 3), dtype=np.int64)
        got_zero = classification_metrics_from_cm(cm_zero)
        assert got_zero == {
            'accuracy': 0.0,
            'macro_f1': 0.0,
            'balanced_accuracy': 0.0,
            'cohen_kappa': 0.0,
        }

    def test_invalid_cm_raises(self) -> None:
        with pytest.raises(ValueError, match='square 2D'):
            classification_metrics_from_cm(np.zeros((2, 3), dtype=np.int64))
        with pytest.raises(ValueError, match='square 2D'):
            classification_metrics_from_cm(np.zeros((0, 0), dtype=np.int64))
        with pytest.raises(ValueError, match='square 2D'):
            classification_metrics_from_cm(np.zeros(4, dtype=np.int64))


class TestSparseCMIO:
    def test_roundtrip_multiple_and_empty_matrices(
        self, tmp_path: pathlib.Path
    ) -> None:
        m0 = np.array([[4, 1], [0, 3]], dtype=np.int64)
        m1 = np.zeros((3, 3), dtype=np.int64)
        m2 = np.array([[0, 0, 2], [1, 0, 0], [0, 0, 5]], dtype=np.int64)
        path = tmp_path / 'cms.npz'
        save_sparse_cm(path, [m0, m1, m2])
        loaded = load_sparse_cm(path)
        assert len(loaded) == 3
        np.testing.assert_array_equal(loaded[0], m0)
        np.testing.assert_array_equal(loaded[1], m1)
        np.testing.assert_array_equal(loaded[2], m2)

        empty_path = tmp_path / 'empty.npz'
        save_sparse_cm(empty_path, [])
        assert load_sparse_cm(empty_path) == []
