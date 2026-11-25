# Copyright (c) Taylor Geospatial. All rights reserved.
# Licensed under the MIT License.

import csv
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from neuralftw import ConfigError
from neuralftw.coordbench import run as run_mod
from neuralftw.coordbench.config import (
    CoordConfig,
    CoordEvaluationConfig,
    CoordModelConfig,
    CoordOutputConfig,
    CoordRuntimeConfig,
)
from neuralftw.coordbench.datasets import CoordBenchmark
from neuralftw.coordbench.run import (
    CSV_COLUMNS,
    CoordResult,
    append_rows_atomic,
    run_coordbench,
)
from neuralftw.models.encoders import SinCosLocationEncoder, XYZLocationEncoder


def _make_reg_bench(
    name: str = 'worldclim-bio1',
    n: int = 80,
    *,
    multi_task: bool = False,
    with_official_mask: bool = False,
    seed: int = 0,
) -> CoordBenchmark:
    rng = np.random.default_rng(seed)
    lat = rng.uniform(-50.0, 50.0, size=n)
    lon = rng.uniform(-120.0, 120.0, size=n)
    y1 = np.sin(np.deg2rad(lat)) + 0.5 * np.cos(np.deg2rad(lon))
    tasks: dict[str, np.ndarray] = {'t1': y1}
    if multi_task:
        y2 = np.cos(np.deg2rad(lat)) - 0.3 * np.sin(np.deg2rad(lon))
        y3 = y1.copy()
        y3[0] = np.nan
        tasks['t2'] = y2
        tasks['t3_nan'] = y3
    test_mask = None
    if with_official_mask:
        test_mask = np.zeros(n, dtype=np.bool_)
        test_mask[:20] = True
    return CoordBenchmark(
        name=name,
        lat=lat,
        lon=lon,
        tasks=tasks,
        task_type='regression',
        test_mask=test_mask,
    )


def _make_clf_bench(
    name: str = 'satclip-country',
    n: int = 80,
    *,
    with_official_mask: bool = False,
    seed: int = 1,
) -> CoordBenchmark:
    rng = np.random.default_rng(seed)
    lat = rng.uniform(-50.0, 50.0, size=n)
    lon = rng.uniform(-120.0, 120.0, size=n)
    labels = np.where(lat >= 0.0, 1, 0).astype(np.int64)
    test_mask = None
    if with_official_mask:
        test_mask = np.zeros(n, dtype=np.bool_)
        test_mask[:20] = True
    return CoordBenchmark(
        name=name,
        lat=lat,
        lon=lon,
        tasks={'cls': labels},
        task_type='classification',
        test_mask=test_mask,
    )


def _make_urban_rural_bench(
    name: str, *, task_type: str = 'regression', n_urban: int = 30, n_rural: int = 30
) -> CoordBenchmark:
    rng = np.random.default_rng(7)
    # City at (lat=0, lon=0): 0.02 deg ~= 2.2 km (urban <= 10 km),
    # 0.25 deg ~= 27.8 km (buffer), 2.0 deg ~= 222 km (rural > 50 km).
    u_lat = rng.uniform(-0.02, 0.02, size=n_urban)
    u_lon = rng.uniform(-0.02, 0.02, size=n_urban)
    b_lat = np.full(8, 0.25)
    b_lon = np.zeros(8)
    r_lat = rng.uniform(1.5, 4.0, size=n_rural)
    r_lon = rng.uniform(1.5, 4.0, size=n_rural)
    lat = np.concatenate([u_lat, b_lat, r_lat])
    lon = np.concatenate([u_lon, b_lon, r_lon])
    if task_type == 'regression':
        y = 0.5 * lat - 0.3 * lon + rng.normal(0.0, 0.05, size=len(lat))
        tasks = {'val': y}
    else:
        y = np.arange(len(lat), dtype=np.int64) % 2
        tasks = {'cls': y}
    return CoordBenchmark(name=name, lat=lat, lon=lon, tasks=tasks, task_type=task_type)


def _write_cities_file(path: Path) -> Path:
    fields = ['1', 'ZeroCity', 'ZeroCity', '', '0.0', '0.0'] + [''] * 13
    path.write_text('\t'.join(fields) + '\n', encoding='utf-8')
    return path


class TestCoordResultAndAtomicCSV:
    def test_coord_result_and_atomic_append(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        res = CoordResult(
            dataset='worldclim-bio1',
            task='bio1',
            task_type='regression',
            method='linear',
            split='random',
            metric_name='r2',
            metric_value=0.85,
            ci_lower=0.80,
            ci_upper=0.90,
            n_folds=5,
            cell_deg=0.0,
            feature_dim=4,
            n_samples=100,
            n_test=100,
            seed=0,
            model_name='sincos',
            model_target='neuralftw.models.encoders.SinCosLocationEncoder',
            fold_algorithm='torch-randperm-v1',
        )
        row = res.to_row()
        assert tuple(row.keys()) == CSV_COLUMNS
        assert row['macro_f1'] is None
        assert 'worldclim-bio1' in repr(res)
        assert run_mod._normalize_cell_deg(None) == '0'
        assert run_mod._normalize_cell_deg('') == '0'
        assert run_mod._normalize_cell_deg(10.0) == '10'

        csv_path = tmp_path / 'sub' / 'out.csv'
        # Empty rows is a no-op
        append_rows_atomic(csv_path, [])
        assert not csv_path.exists()

        # First write creates parent dirs + file
        append_rows_atomic(csv_path, [{'a': 1, 'b': 2}])
        # Second write with identical schema appends
        append_rows_atomic(csv_path, [{'a': 3, 'b': 4}])
        # Third write with new column triggers atomic schema-evolution rewrite
        append_rows_atomic(csv_path, [{'a': 5, 'b': 6, 'c': 7}])

        with open(csv_path, encoding='utf-8', newline='') as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 3
        assert rows[0] == {'a': '1', 'b': '2', 'c': ''}
        assert rows[2] == {'a': '5', 'b': '6', 'c': '7'}

        # Also test writing directly in cwd (empty dirname)
        monkeypatch.chdir(tmp_path)
        append_rows_atomic('flat.csv', [{'x': 1}])
        assert (tmp_path / 'flat.csv').exists()

        # Error during atomic replace cleans up temp file
        def _boom(src: str, dst: str) -> None:
            raise OSError('disk full')

        monkeypatch.setattr(run_mod.os, 'replace', _boom)
        with pytest.raises(OSError, match='disk full'):
            run_mod._write_csv_atomic(str(tmp_path / 'fail.csv'), ['x'], [{'x': 1}])
        assert not list(tmp_path.glob('.neuralftw_*.tmp'))

        # If temp file was already removed before exception, unlink branch is skipped
        def _unlink_and_boom(src: str, dst: str) -> None:
            Path(src).unlink()
            raise OSError('already gone')

        monkeypatch.setattr(run_mod.os, 'replace', _unlink_and_boom)
        with pytest.raises(OSError, match='already gone'):
            run_mod._write_csv_atomic(str(tmp_path / 'fail2.csv'), ['x'], [{'x': 1}])


class TestRunCoordBench:
    def test_standard_splits_batching_and_resume(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        reg_multi = _make_reg_bench('pdfm-conus27', multi_task=True)
        reg_official = _make_reg_bench(
            'sustainbench-asset', with_official_mask=True, seed=2
        )
        clf_bench = _make_clf_bench('satclip-country')
        benches = [reg_multi, reg_official, clf_bench]
        monkeypatch.setattr(
            run_mod, 'load_benchmarks', lambda *args, **kwargs: list(benches)
        )

        out_csv = tmp_path / 'results.csv'
        cfg = CoordConfig(
            model=CoordModelConfig(name='sincos'),
            datasets=('worldclim',),
            evaluation=CoordEvaluationConfig(
                methods=('linear', 'knn'),
                split='both',
                folds=3,
                cell_deg=(10.0, 30.0),
                knn_k=3,
                respect_official_splits=True,
            ),
            runtime=CoordRuntimeConfig(device='cpu', seed=0),
            output=CoordOutputConfig(file=str(out_csv), resume=True),
        )
        results = run_coordbench(cfg)

        # reg_multi (3 tasks) x linear x 3 splits (random + 2 cells) = 9
        # reg_official (1 task) x linear x 1 split (official) = 1
        # clf_bench (1 task) x (linear, knn3) x 3 splits = 6
        assert len(results) == 16
        official_rows = [r for r in results if r.split == 'official']
        assert len(official_rows) == 1
        assert official_rows[0].n_test == 20
        clf_rows = [r for r in results if r.task_type == 'classification']
        assert all(r.macro_f1 is not None for r in clf_rows)
        assert all(r.balanced_accuracy is not None for r in clf_rows)
        assert all(r.cohen_kappa is not None for r in clf_rows)
        reg_rows = [r for r in results if r.task_type == 'regression']
        assert all(r.macro_f1 is None for r in reg_rows)

        # Full resume skips encoding completely
        encode_calls = 0

        class _CountingEncoder(SinCosLocationEncoder):
            def encode(self, *args: Any, **kwargs: Any) -> np.ndarray:
                nonlocal encode_calls
                encode_calls += 1
                return super().encode(*args, **kwargs)

        resumed = run_coordbench(cfg, encoder=_CountingEncoder())
        assert resumed == []
        assert encode_calls == 0

        # Partial resume: add a new cell_deg=50.0 -> only new spatial rows run
        cfg_more = CoordConfig(
            model=CoordModelConfig(name='sincos'),
            datasets=('worldclim',),
            evaluation=CoordEvaluationConfig(
                methods=('linear', 'knn'),
                split='spatial',
                folds=3,
                cell_deg=(10.0, 50.0),
                knn_k=3,
                respect_official_splits=True,
            ),
            runtime=CoordRuntimeConfig(device='cpu', seed=0),
            output=CoordOutputConfig(file=str(out_csv), resume=True),
        )
        more = run_coordbench(cfg_more)
        # reg_multi (3 tasks x 1 new cell) + clf_bench (1 task x 2 methods x 1 new cell)
        assert len(more) == 5
        assert all(r.cell_deg == 50.0 for r in more)

        # Test official mask ignored when respect_official_splits=False
        out_no_off = tmp_path / 'no_official.csv'
        monkeypatch.setattr(
            run_mod, 'load_benchmarks', lambda *args, **kwargs: [reg_official]
        )
        cfg_no_off = CoordConfig(
            model=CoordModelConfig(name='xyz'),
            datasets=('sustainbench',),
            evaluation=CoordEvaluationConfig(
                methods=('linear',),
                split='random',
                folds=3,
                respect_official_splits=False,
            ),
            output=CoordOutputConfig(file=str(out_no_off), resume=False),
        )
        no_off_res = run_coordbench(cfg_no_off)
        assert len(no_off_res) == 1
        assert no_off_res[0].split == 'random'

    def test_all_methods_and_spatial_baselines(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        reg_bench = _make_reg_bench('satclip-air-temp', n=60)
        clf_bench = _make_clf_bench('satclip-country', n=60)
        monkeypatch.setattr(
            run_mod, 'load_benchmarks', lambda *args, **kwargs: [reg_bench, clf_bench]
        )
        # Keep MLP fast in unit tests
        orig_mlp = run_mod.mlp_probe_score
        monkeypatch.setattr(
            run_mod,
            'mlp_probe_score',
            lambda *args, **kwargs: orig_mlp(*args, epochs=2, hidden=16, **kwargs),
        )

        # All 7 methods on xyz encoder
        out_all = tmp_path / 'all_methods.csv'
        cfg_all = CoordConfig(
            model=CoordModelConfig(name='xyz'),
            datasets=('air_temp',),
            evaluation=CoordEvaluationConfig(
                methods=('linear', 'chunked', 'knn', 'logreg', 'mlp', 'idw', 'kriging'),
                split='random',
                folds=2,
                knn_k=3,
            ),
            output=CoordOutputConfig(file=str(out_all), resume=False),
        )
        res_all = run_coordbench(cfg_all, encoder=XYZLocationEncoder())
        # reg skips knn & logreg -> 5 methods; clf runs all 7 -> 12 total
        assert len(res_all) == 12
        by_method = {(r.task_type, r.method): r for r in res_all}
        assert by_method[('regression', 'idw')].feature_dim == 0
        assert by_method[('classification', 'kriging')].feature_dim == 0
        assert by_method[('classification', 'knn3')].feature_dim == 3

        # Encoder-free spatial baseline mode (idw & kriging)
        out_spatial = tmp_path / 'spatial_only.csv'
        cfg_idw = CoordConfig(
            model=CoordModelConfig(name='idw'),
            datasets=('air_temp',),
            evaluation=CoordEvaluationConfig(
                methods=('idw',), split='spatial', folds=2, cell_deg=20.0
            ),
            output=CoordOutputConfig(file=str(out_spatial), resume=False),
        )
        res_idw = run_coordbench(cfg_idw)
        assert len(res_idw) == 2
        assert res_idw[0].model_target == 'neuralftw.probes.spatial.idw_probe_score'

        # Spatial baseline paired with feature method raises ConfigError
        cfg_bad = CoordConfig(
            model=CoordModelConfig(name='kriging'),
            datasets=('air_temp',),
            evaluation=CoordEvaluationConfig(methods=('linear',), folds=2),
            output=CoordOutputConfig(file=str(out_spatial), resume=False),
        )
        with pytest.raises(ConfigError, match='Spatial baseline'):
            run_coordbench(cfg_bad)

        # Non-spatial model name with spatial-only methods skips encoder build
        out_mixed = tmp_path / 'mixed.csv'
        cfg_no_enc = CoordConfig(
            model=CoordModelConfig(name='custom-spatial'),
            datasets=('air_temp',),
            evaluation=CoordEvaluationConfig(methods=('idw',), folds=2),
            output=CoordOutputConfig(file=str(out_mixed), resume=False),
        )
        res_no_enc = run_coordbench(cfg_no_enc)
        assert len(res_no_enc) == 2

        # Regression benchmark skipped cleanly when only clf-only methods requested
        monkeypatch.setattr(
            run_mod, 'load_benchmarks', lambda *args, **kwargs: [reg_bench]
        )
        cfg_clf_only = CoordConfig(
            model=CoordModelConfig(name='xyz'),
            datasets=('air_temp',),
            evaluation=CoordEvaluationConfig(methods=('knn', 'logreg'), folds=2),
            output=CoordOutputConfig(file=str(tmp_path / 'empty.csv'), resume=False),
        )
        assert run_coordbench(cfg_clf_only) == []

    def test_urban_rural_stratified_and_holdout(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        cities_file = _write_cities_file(tmp_path / 'cities15000.txt')
        ur_reg = _make_urban_rural_bench('worldclim-bio1', task_type='regression')
        ur_clf = _make_urban_rural_bench('satclip-country', task_type='classification')
        ur_small = _make_urban_rural_bench(
            'worldclim-bio12', task_type='regression', n_urban=5, n_rural=30
        )
        monkeypatch.setattr(
            run_mod,
            'load_benchmarks',
            lambda *args, **kwargs: [ur_reg, ur_clf, ur_small],
        )

        # Stratified split with linear, chunked, knn, idw (both OOF and non-OOF paths)
        out_strat = tmp_path / 'strat.csv'
        cfg_strat = CoordConfig(
            model=CoordModelConfig(name='sincos'),
            datasets=('worldclim',),
            evaluation=CoordEvaluationConfig(
                methods=('linear', 'chunked', 'knn', 'idw'),
                split='urban_rural_stratified',
                folds=3,
                knn_k=3,
                cities_path=str(cities_file),
            ),
            output=CoordOutputConfig(file=str(out_strat), resume=True),
        )
        res_strat = run_coordbench(cfg_strat)
        # ur_small skipped (<20 urban).
        # ur_reg: 3 methods (linear, chunked, idw) x 2 strata = 6
        # ur_clf: 4 methods (linear, chunked, knn3, idw) x 2 strata = 8
        assert len(res_strat) == 14
        assert {r.split for r in res_strat} == {'urban_stratified', 'rural_stratified'}
        assert 'worldclim-bio12' not in {r.dataset for r in res_strat}

        # Partial resume on stratified: add a second task to ur_reg so bench is not
        # fully completed while first task's strata are already in CSV.
        ur_reg_ext = CoordBenchmark(
            name='worldclim-bio1',
            lat=ur_reg.lat,
            lon=ur_reg.lon,
            tasks={'val': ur_reg.tasks['val'], 'val2': ur_reg.tasks['val'] * 2.0},
            task_type='regression',
        )
        monkeypatch.setattr(
            run_mod, 'load_benchmarks', lambda *args, **kwargs: [ur_reg_ext]
        )
        res_strat_ext = run_coordbench(cfg_strat)
        assert len(res_strat_ext) == 6
        assert {r.task for r in res_strat_ext} == {'val2'}

        # Holdout split with linear and idw (regression + classification)
        monkeypatch.setattr(
            run_mod,
            'load_benchmarks',
            lambda *args, **kwargs: [ur_reg, ur_clf, ur_small],
        )
        out_hold = tmp_path / 'holdout.csv'
        cfg_hold = CoordConfig(
            model=CoordModelConfig(name='sincos'),
            datasets=('worldclim',),
            evaluation=CoordEvaluationConfig(
                methods=('linear', 'idw'),
                split='urban_rural_holdout',
                folds=2,
                cities_path=str(cities_file),
            ),
            output=CoordOutputConfig(file=str(out_hold), resume=True),
        )
        res_hold = run_coordbench(cfg_hold)
        # ur_reg (2 methods x 2 splits) + ur_clf (2 methods x 2 splits) = 8
        assert len(res_hold) == 8
        assert {r.split for r in res_hold} == {
            'train_rural_test_urban',
            'train_urban_test_rural',
        }
        assert all(r.n_folds == 1 for r in res_hold)
        assert all(
            r.macro_f1 is not None for r in res_hold if r.task_type == 'classification'
        )

        # Partial resume on holdout
        monkeypatch.setattr(
            run_mod, 'load_benchmarks', lambda *args, **kwargs: [ur_reg_ext]
        )
        res_hold_ext = run_coordbench(cfg_hold)
        assert len(res_hold_ext) == 4
        assert {r.task for r in res_hold_ext} == {'val2'}

    def test_partial_resume_skips_completed_standard_split_tasks(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        reg_bench = _make_reg_bench('worldclim-bio1', multi_task=False)
        out_csv = tmp_path / 'partial.csv'
        monkeypatch.setattr(
            run_mod, 'load_benchmarks', lambda *args, **kwargs: [reg_bench]
        )
        cfg1 = CoordConfig(
            model=CoordModelConfig(name='xyz'),
            datasets=('worldclim',),
            evaluation=CoordEvaluationConfig(
                methods=('linear',), split='random', folds=2
            ),
            output=CoordOutputConfig(file=str(out_csv), resume=True),
        )
        assert len(run_coordbench(cfg1)) == 1

        # Add a second method so benchmark is not fully completed, while 'linear' is
        cfg2 = CoordConfig(
            model=CoordModelConfig(name='xyz'),
            datasets=('worldclim',),
            evaluation=CoordEvaluationConfig(
                methods=('linear', 'idw'), split='random', folds=2
            ),
            output=CoordOutputConfig(file=str(out_csv), resume=True),
        )
        more = run_coordbench(cfg2)
        assert len(more) == 1
        assert more[0].method == 'idw'

        # Also test official split with multi-target regression batching
        reg_off_multi = _make_reg_bench(
            'pdfm-conus27', multi_task=True, with_official_mask=True
        )
        monkeypatch.setattr(
            run_mod, 'load_benchmarks', lambda *args, **kwargs: [reg_off_multi]
        )
        cfg_off = CoordConfig(
            model=CoordModelConfig(name='xyz'),
            datasets=('pdfm',),
            evaluation=CoordEvaluationConfig(
                methods=('linear',),
                split='random',
                folds=2,
                respect_official_splits=True,
            ),
            output=CoordOutputConfig(
                file=str(tmp_path / 'off_multi.csv'), resume=False
            ),
        )
        off_res = run_coordbench(cfg_off)
        assert len(off_res) == 3
        assert all(r.split == 'official' for r in off_res)
        assert off_res[0].n_test == 20
