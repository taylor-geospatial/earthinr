# Copyright (c) Taylor Geospatial. All rights reserved.
# Licensed under the MIT License.

from pathlib import Path

import huggingface_hub
import numpy as np
import pandas as pd
import pytest

from earthinr._errors import DatasetNotFoundError
from earthinr.coordbench.catalog import (
    CDC_PLACES_MEASURES,
    COORDBENCH_REPO,
    DEEPMIND_EVAL_CONFIGS,
    SUSTAINBENCH_TASKS,
    USAVARS_LABELS,
    USAVARS_LOG_LABELS,
)
from earthinr.coordbench.datasets import (
    CoordBenchmark,
    load_air_temp,
    load_benchmarks,
    load_better_together,
    load_california_housing,
    load_cdc_places,
    load_config,
    load_country,
    load_deepmind,
    load_ecoregions,
    load_pdfm,
    load_satclip,
    load_soilgrids,
    load_sustainbench,
    load_usavars,
    load_worldclim,
)


def _write_table(
    root: Path, config: str, df: pd.DataFrame, *, nested: bool = True
) -> Path:
    base = root / 'data' / config if nested else root / config
    base.mkdir(parents=True, exist_ok=True)
    path = base / 'data.parquet'
    df.to_parquet(path, index=False)
    return path


def _populate_full_coordbench(root: Path) -> None:
    _write_table(
        root,
        'pdfm_conus27',
        pd.DataFrame(
            {
                'lon': [-122.4, -74.0, -87.6],
                'lat': [37.7, 40.7, 41.8],
                'place': ['a', 'b', 'c'],
                'county': ['c1', 'c2', 'c3'],
                'state': ['CA', 'NY', 'IL'],
                'Count_Person': [100.0, 200.0, 300.0],
                'imputation_split': ['train', 'test', 'train'],
                'superresolution_split': ['test', 'train', 'train'],
                'extrapolation_split': ['train', 'train', 'test'],
                'timestamp': [1_600_000_000_000, 1_600_000_000_000, 1_600_000_000_000],
                'task_a': [1.0, 2.0, 3.0],
                'task_b': [4.0, 5.0, 6.0],
            }
        ),
    )

    _write_table(
        root,
        'air_temp',
        pd.DataFrame(
            {
                'lon': [0.0, 10.0],
                'lat': [10.0, 20.0],
                'meanP': [101.0, 100.0],
                'meanT': [15.5, 22.0],
            }
        ),
    )

    _write_table(
        root,
        'california_housing',
        pd.DataFrame(
            {
                'longitude': [-121.0, -118.0],
                'latitude': [37.5, 34.0],
                'MedHouseVal': [3.5, 4.2],
            }
        ),
    )

    _write_table(
        root,
        'satclip_country',
        pd.DataFrame(
            {'lon': [0.0, 1.0, 2.0], 'lat': [0.0, 1.0, 2.0], 'country': [0, 1, 2]}
        ),
    )
    _write_table(
        root,
        'satclip_ecoregion',
        pd.DataFrame(
            {
                'lon': [0.0, 1.0, 2.0],
                'lat': [0.0, 1.0, 2.0],
                'ecoregion_class_index': [-1, 10, 20],
                'biome_class_index': [-1, 2, 3],
            }
        ),
    )
    _write_table(
        root,
        'satclip_population',
        pd.DataFrame(
            {'lon': [0.0, 1.0], 'lat': [0.0, 1.0], 'log_population': [1.2, 3.4]}
        ),
    )
    _write_table(
        root,
        'satclip_elevation',
        pd.DataFrame(
            {'lon': [0.0, 1.0], 'lat': [0.0, 1.0], 'elevation': [100.0, 250.0]}
        ),
    )

    sb_data: dict[str, object] = {
        'lon': [10.0, 20.0, np.nan, 30.0],
        'lat': [1.0, 2.0, 3.0, 4.0],
        'split': ['trainval', 'test', 'trainval', 'test'],
    }
    for idx, col in enumerate(SUSTAINBENCH_TASKS.values()):
        sb_data[col] = [float(idx), np.nan, 1.0, float(idx + 2)]
    _write_table(root, 'sustainbench', pd.DataFrame(sb_data))

    _write_table(
        root,
        'bt_cropharvest',
        pd.DataFrame(
            {'lon': [1.0, 2.0], 'lat': [3.0, 4.0], 'label_name': ['maize', 'wheat']}
        ),
    )
    _write_table(
        root,
        'bt_biomass',
        pd.DataFrame(
            {
                'lon': [1.0, 2.0, 3.0],
                'lat': [3.0, 4.0, 5.0],
                'biomass_mean': [12.0, np.nan, 34.0],
            }
        ),
    )
    dw_data: dict[str, object] = {
        'id': [1, 2, 3],
        'lon': [1.0, 2.0, 3.0],
        'lat': [4.0, 5.0, 6.0],
        'random_sample': [1, 0, 1],
    }
    for col in (
        'water',
        'trees',
        'grass',
        'flooded_vegetation',
        'crops',
        'shrub_and_scrub',
        'built',
        'bare',
        'snow_and_ice',
    ):
        dw_data[col] = [0.1, 0.2, 0.3]
    _write_table(root, 'bt_landcover', pd.DataFrame(dw_data))

    bio_data: dict[str, object] = {
        'id': [1, 2, 3],
        'lon': [1.0, 2.0, 3.0],
        'lat': [4.0, 5.0, 6.0],
    }
    for i in range(1, 20):
        bio_data[f'bioclim_{i:02d}'] = [float(i), float(i + 1), float(i + 2)]
    _write_table(root, 'bt_bioclim', pd.DataFrame(bio_data))

    _write_table(
        root,
        'bt_human_footprint',
        pd.DataFrame(
            {
                'id': [1, 2, 3],
                'lon': [1.0, 2.0, 3.0],
                'lat': [4.0, 5.0, 6.0],
                'pop_density': [10.0, 20.0, 30.0],
                'maxdist_road': [5.0, 15.0, 25.0],
            }
        ),
    )

    cdc_data: dict[str, object] = {
        'lon': [-90.0, -80.0, np.nan],
        'lat': [35.0, 40.0, 42.0],
    }
    for idx, pref in enumerate(CDC_PLACES_MEASURES.values()):
        cdc_data[f'{pref.lower()}_crudeprev'] = [10.0 + idx, np.nan, 20.0]
    _write_table(root, 'cdc_places', pd.DataFrame(cdc_data))

    for label in USAVARS_LABELS:
        _write_table(
            root,
            f'usavars_{label}',
            pd.DataFrame(
                {
                    'id': [1, 2, 3],
                    'lon': [-100.0, -95.0, -90.0],
                    'lat': [35.0, 36.0, 37.0],
                    label: [10.0, -999.0, 25.0],
                }
            ),
        )

    _write_table(
        root,
        'country',
        pd.DataFrame(
            {'lon': [0.0, 10.0], 'lat': [50.0, 45.0], 'country': ['UK', 'IT']}
        ),
    )
    _write_table(
        root,
        'ecoregions',
        pd.DataFrame(
            {
                'lon': [0.0, 10.0],
                'lat': [50.0, 45.0],
                'biome': ['forest', 'grass'],
                'ecoregion': ['e1', 'e2'],
            }
        ),
    )
    _write_table(
        root,
        'worldclim_bio',
        pd.DataFrame(
            {
                'lon': [0.0, 1.0, 2.0],
                'lat': [0.0, 1.0, 2.0],
                'bio1': [10.0, 15.0, 20.0],
                'bio12': [500.0, 600.0, 700.0],
            }
        ),
    )
    _write_table(
        root,
        'soilgrids',
        pd.DataFrame(
            {
                'lon': [0.0, 1.0, 2.0],
                'lat': [0.0, 1.0, 2.0],
                'soc': [1.0, 2.0, 3.0],
                'phh2o': [6.0, 6.5, 7.0],
            }
        ),
    )

    for stem in DEEPMIND_EVAL_CONFIGS:
        if stem in ('aster_ged', 'openet_ensemble'):
            df = pd.DataFrame(
                {
                    'lon': [0.0, 1.0],
                    'lat': [10.0, 11.0],
                    'label': [1.25, 2.75],
                    'timestamp': [1_577_836_800_000, 1_609_459_200_000],
                }
            )
        else:
            df = pd.DataFrame(
                {
                    'lon': [0.0, 1.0],
                    'lat': [10.0, 11.0],
                    'label': [0, 1],
                    'timestamp': [1_577_836_800_000, 1_609_459_200_000],
                }
            )
        _write_table(root, f'dm_{stem}', df)


class TestCoordBenchmark:
    def test_init_repr_and_validation(self) -> None:
        bench = CoordBenchmark(
            name='demo', lat=np.array([1.0, 2.0]), lon=np.array([3.0, 4.0])
        )
        assert bench.tasks == {}
        assert bench.task_type == 'regression'
        assert (
            "CoordBenchmark(name='demo', n_samples=2, tasks=0, task_type='regression')"
            in repr(bench)
        )

        with pytest.raises(ValueError, match='Invalid task_type'):
            CoordBenchmark(
                name='bad',
                lat=np.array([1.0]),
                lon=np.array([2.0]),
                task_type='segmentation',
            )


class TestLoadConfig:
    def test_direct_and_nested_local_root(self, tmp_path: Path) -> None:
        df = pd.DataFrame({'lon': [1.0], 'lat': [2.0], 'val': [3.0]})
        _write_table(tmp_path / 'direct', 'sample', df, nested=False)
        _write_table(tmp_path / 'nested', 'sample', df, nested=True)

        pd.testing.assert_frame_equal(
            load_config('sample', root=tmp_path / 'direct'), df
        )
        pd.testing.assert_frame_equal(
            load_config('sample', root=tmp_path / 'nested'), df
        )

    def test_env_var_and_missing_local_table(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        df = pd.DataFrame({'lon': [1.0], 'lat': [2.0]})
        _write_table(tmp_path, 'country', df)
        monkeypatch.setenv('COORDBENCH_ROOT', str(tmp_path))

        pd.testing.assert_frame_equal(load_config('country'), df)
        with pytest.raises(DatasetNotFoundError, match='not found under'):
            load_config('missing_table')

    def test_hub_fallback(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv('COORDBENCH_ROOT', raising=False)
        monkeypatch.delenv('COORDBENCH_REPO', raising=False)
        df = pd.DataFrame({'lon': [1.0], 'lat': [2.0]})
        parquet = tmp_path / 'hub.parquet'
        df.to_parquet(parquet, index=False)

        calls: list[tuple[str, str, str]] = []

        def fake_download(repo_id: str, filename: str, *, repo_type: str) -> str:
            calls.append((repo_id, filename, repo_type))
            return str(parquet)

        monkeypatch.setattr(huggingface_hub, 'hf_hub_download', fake_download)
        pd.testing.assert_frame_equal(load_config('country'), df)
        pd.testing.assert_frame_equal(load_config('country', repo_id='custom/repo'), df)

        assert calls == [
            (COORDBENCH_REPO, 'data/country/data.parquet', 'dataset'),
            ('custom/repo', 'data/country/data.parquet', 'dataset'),
        ]


class TestFamilyLoaders:
    def test_pdfm_and_official_splits(self, tmp_path: Path) -> None:
        _populate_full_coordbench(tmp_path)
        benches = load_pdfm(root=tmp_path)
        assert len(benches) == 1
        assert benches[0].name == 'pdfm-conus27'
        assert set(benches[0].tasks) == {'task_a', 'task_b'}
        assert benches[0].test_mask is None

        imp = load_pdfm(root=tmp_path, split='imputation')[0]
        assert imp.name == 'pdfm-conus27-imputation'
        assert imp.test_mask is not None
        np.testing.assert_array_equal(imp.test_mask, [False, True, False])

        with pytest.raises(ValueError, match='Invalid PDFM split'):
            load_pdfm(root=tmp_path, split='bad')

        _write_table(
            tmp_path / 'nosplit',
            'pdfm_conus27',
            pd.DataFrame({'lon': [0.0], 'lat': [1.0], 'task_a': [2.0]}),
        )
        with pytest.raises(ValueError, match='not found in table'):
            load_pdfm(root=tmp_path / 'nosplit', split='imputation')

    def test_air_temp_and_fallback_column(self, tmp_path: Path) -> None:
        _populate_full_coordbench(tmp_path)
        bench = load_air_temp(root=tmp_path)[0]
        assert bench.name == 'satclip-air-temp'
        np.testing.assert_array_equal(bench.tasks['air_temperature'], [15.5, 22.0])

        _write_table(
            tmp_path / 'fallback',
            'air_temp',
            pd.DataFrame({'lon': [0.0], 'lat': [1.0], 'mean_val': [12.3]}),
        )
        fb = load_air_temp(root=tmp_path / 'fallback')[0]
        np.testing.assert_array_equal(fb.tasks['air_temperature'], [12.3])

    def test_california_housing_and_satclip_sentinels(self, tmp_path: Path) -> None:
        _populate_full_coordbench(tmp_path)
        cal = load_california_housing(root=tmp_path)[0]
        assert cal.name == 'california-housing'
        np.testing.assert_array_equal(cal.tasks['median_house_value'], [3.5, 4.2])

        sat = {b.name: b for b in load_satclip(root=tmp_path)}
        assert len(sat) == 5
        assert len(sat['satclip-country'].lat) == 2
        assert sat['satclip-country'].task_type == 'classification'
        assert len(sat['satclip-ecoregion'].lat) == 2
        assert len(sat['satclip-biome'].lat) == 2
        assert sat['satclip-population'].task_type == 'regression'
        assert sat['satclip-elevation'].task_type == 'regression'

    def test_sustainbench_and_better_together(self, tmp_path: Path) -> None:
        _populate_full_coordbench(tmp_path)
        sb = load_sustainbench(root=tmp_path)
        assert len(sb) == 6
        assert sb[0].test_mask is not None
        np.testing.assert_array_equal(sb[0].test_mask, [False, True])

        bt = {b.name: b for b in load_better_together(root=tmp_path)}
        assert len(bt) == 6
        assert bt['bt-cropharvest'].task_type == 'classification'
        assert len(bt['bt-biomass'].lat) == 2
        assert len(bt['bt-landcover'].lat) == 2
        assert len(bt['bt-landcover'].tasks) == 9
        assert len(bt['bt-bioclim'].lat) == 2
        assert len(bt['bt-bioclim'].tasks) == 19
        assert len(bt['bt-population'].lat) == 2
        assert len(bt['bt-distroad'].lat) == 2

    def test_cdc_places_and_missing_column_warning(self, tmp_path: Path) -> None:
        _populate_full_coordbench(tmp_path)
        places = load_cdc_places(root=tmp_path)
        assert len(places) == 12
        assert len(places[0].lat) == 1

        _write_table(
            tmp_path / 'partial',
            'cdc_places',
            pd.DataFrame({'lon': [-90.0], 'lat': [35.0], 'phlth_crudeprev': [11.0]}),
        )
        with pytest.warns(UserWarning, match='Missing CDC PLACES column'):
            partial = load_cdc_places(root=tmp_path / 'partial')
        assert [b.name for b in partial] == ['places-phys_health']

    def test_usavars_sentinels_log1p_and_joins(self, tmp_path: Path) -> None:
        _populate_full_coordbench(tmp_path)
        usavars = {b.name: b for b in load_usavars(root=tmp_path)}
        assert len(usavars) == 7
        for label in USAVARS_LABELS:
            vals = usavars[f'mosaiks-{label}'].tasks[label]
            expected = np.array([10.0, 25.0])
            if label in USAVARS_LOG_LABELS:
                expected = np.log1p(expected)
            np.testing.assert_allclose(vals, expected)

        # Edge cases: id join when lat/lon missing, price fallback, no numeric, no coords.
        alt = tmp_path / 'alt_usavars'
        _write_table(
            alt,
            'usavars_treecover',
            pd.DataFrame(
                {
                    'ID': [1, 2],
                    'lat': [35.0, 36.0],
                    'lon': [-100.0, -99.0],
                    'val': [5.0, 6.0],
                }
            ),
        )
        _write_table(
            alt,
            'usavars_housing',
            pd.DataFrame({'id': [1, 2], 'price': [100.0, -999.0]}),
        )
        _write_table(
            alt, 'usavars_roads', pd.DataFrame({'id': [1, 2], 'note': ['a', 'b']})
        )
        _write_table(alt, 'usavars_income', pd.DataFrame({'income': [50.0, 60.0]}))
        for label in ('elevation', 'population', 'nightlights'):
            _write_table(
                alt,
                f'usavars_{label}',
                pd.DataFrame({'lat': [35.0], 'lon': [-100.0], label: [2.0]}),
            )
        alt_benches = {b.name: b for b in load_usavars(root=alt)}
        assert 'mosaiks-housing' in alt_benches
        np.testing.assert_allclose(
            alt_benches['mosaiks-housing'].tasks['housing'], [np.log1p(100.0)]
        )
        assert 'mosaiks-roads' not in alt_benches
        assert 'mosaiks-income' not in alt_benches

        # Edge case: no table has id/lat/lon so id2ll is None.
        no_id = tmp_path / 'no_id'
        for label in USAVARS_LABELS:
            _write_table(
                no_id, f'usavars_{label}', pd.DataFrame({'id': [1], label: [1.0]})
            )
        assert load_usavars(root=no_id) == []

    def test_country_ecoregions_worldclim_soilgrids(self, tmp_path: Path) -> None:
        _populate_full_coordbench(tmp_path)
        assert load_country(root=tmp_path)[0].task_type == 'classification'
        eco = load_ecoregions(root=tmp_path)[0]
        assert set(eco.tasks) == {'biome', 'ecoregion'}

        wc_full = load_worldclim(root=tmp_path, max_points=10)
        assert [b.name for b in wc_full] == ['worldclim-bio1', 'worldclim-bio12']
        wc_sub1 = load_worldclim(root=tmp_path, max_points=2, seed=0)
        wc_sub2 = load_worldclim(root=tmp_path, max_points=2, seed=0)
        assert len(wc_sub1[0].lat) == 2
        np.testing.assert_array_equal(wc_sub1[0].lat, wc_sub2[0].lat)

        sg_full = load_soilgrids(root=tmp_path, max_points=10)
        assert [b.name for b in sg_full] == ['soilgrids-soc', 'soilgrids-phh2o']
        sg_sub1 = load_soilgrids(root=tmp_path, max_points=2, seed=0)
        sg_sub2 = load_soilgrids(root=tmp_path, max_points=2, seed=0)
        assert len(sg_sub1[0].lat) == 2
        np.testing.assert_array_equal(sg_sub1[0].lat, sg_sub2[0].lat)

        # Partial columns
        _write_table(
            tmp_path / 'partial_env',
            'worldclim_bio',
            pd.DataFrame({'lon': [0.0], 'lat': [0.0], 'bio1': [1.0]}),
        )
        _write_table(
            tmp_path / 'partial_env',
            'soilgrids',
            pd.DataFrame({'lon': [0.0], 'lat': [0.0], 'soc': [1.0]}),
        )
        assert [b.name for b in load_worldclim(root=tmp_path / 'partial_env')] == [
            'worldclim-bio1'
        ]
        assert [b.name for b in load_soilgrids(root=tmp_path / 'partial_env')] == [
            'soilgrids-soc'
        ]

    def test_deepmind_task_inference_and_timestamps(self, tmp_path: Path) -> None:
        _populate_full_coordbench(tmp_path)
        # Override a few configs to cover missing label, null timestamp, no timestamp, valid_time_start_ms
        _write_table(
            tmp_path,
            'dm_africa_crop_mask',
            pd.DataFrame({'lon': [0.0, 1.0], 'lat': [1.0, 2.0], 'other': [0, 1]}),
        )
        _write_table(
            tmp_path,
            'dm_descals',
            pd.DataFrame(
                {
                    'lon': [0.0, 1.0],
                    'lat': [1.0, 2.0],
                    'label': [0, 1],
                    'timestamp': [None, None],
                }
            ),
        )
        _write_table(
            tmp_path,
            'dm_glance',
            pd.DataFrame({'lon': [0.0, 1.0], 'lat': [1.0, 2.0], 'label': [0, 1]}),
        )
        _write_table(
            tmp_path,
            'dm_us_trees',
            pd.DataFrame(
                {
                    'lon': [0.0, 1.0],
                    'lat': [1.0, 2.0],
                    'label': [0, 1],
                    'valid_time_start_ms': [1_609_459_200_000, 1_640_995_200_000],
                }
            ),
        )
        dm = {b.name: b for b in load_deepmind(root=tmp_path)}
        assert 'dm-africa_crop_mask' not in dm
        assert dm['dm-aster_ged'].task_type == 'regression'
        assert dm['dm-aster_ged'].tasks['aster_ged'].dtype == np.float64
        assert dm['dm-descals'].task_type == 'classification'
        assert dm['dm-descals'].year is None
        assert dm['dm-glance'].year is None
        assert dm['dm-us_trees'].year is not None
        np.testing.assert_array_equal(dm['dm-us_trees'].year, [2021, 2022])


class TestLoadBenchmarks:
    def test_selection_modes_and_deduplication(self, tmp_path: Path) -> None:
        _populate_full_coordbench(tmp_path)

        all_benches = load_benchmarks('all', root=tmp_path)
        assert len(all_benches) == 60

        official_benches = load_benchmarks('official', root=tmp_path)
        assert len(official_benches) == 52
        assert 'country' not in {b.name for b in official_benches}

        official_flag = load_benchmarks('all', root=tmp_path, official_only=True)
        assert len(official_flag) == 52

        single = load_benchmarks('worldclim-bio1', root=tmp_path)
        assert [b.name for b in single] == ['worldclim-bio1']

        # Family before single + single before family + duplicate entries deduplicated
        combo = load_benchmarks(
            [
                'worldclim',
                'worldclim-bio1',
                'soilgrids-soc',
                'soilgrids',
                'worldclim-bio1',
            ],
            root=tmp_path,
        )
        assert [b.name for b in combo] == [
            'worldclim-bio1',
            'worldclim-bio12',
            'soilgrids-soc',
            'soilgrids-phh2o',
        ]

        # Comma-separated string with whitespace
        csv_sel = load_benchmarks('country, worldclim-bio12', root=tmp_path)
        assert [b.name for b in csv_sel] == ['country', 'worldclim-bio12']

        # Duplicate single benchmark under same family
        dup_single = load_benchmarks(
            ['worldclim-bio1', 'worldclim-bio1'], root=tmp_path
        )
        assert [b.name for b in dup_single] == ['worldclim-bio1']

        # Duplicate family name should not emit duplicates
        dup_fam = load_benchmarks(['country', 'country'], root=tmp_path)
        assert [b.name for b in dup_fam] == ['country']

    def test_unknown_and_official_only_rejection(self, tmp_path: Path) -> None:
        with pytest.raises(KeyError, match='Unknown coordbench selection'):
            load_benchmarks('not-a-benchmark', root=tmp_path)

        with pytest.raises(KeyError, match='Unknown coordbench selection'):
            load_benchmarks('country', root=tmp_path, official_only=True)

        with pytest.raises(KeyError, match='Unknown coordbench selection'):
            load_benchmarks('bt-biomass', root=tmp_path, official_only=True)
