# Copyright (c) Taylor Geospatial. All rights reserved.
# Licensed under the MIT License.

from pathlib import Path
from typing import Any, cast

import pytest

from neuralftw._errors import ConfigError
from neuralftw.coordbench.config import (
    CoordConfig,
    CoordEvaluationConfig,
    CoordModelConfig,
    CoordOutputConfig,
    CoordRuntimeConfig,
    dump_coord_config_yaml,
    load_coord_config,
    parse_coord_config,
)


class TestConfigClasses:
    def test_defaults_repr_and_equality(self) -> None:
        model = CoordModelConfig(name='mind')
        ev = CoordEvaluationConfig()
        rt = CoordRuntimeConfig()
        out = CoordOutputConfig()
        cfg = CoordConfig(model=model)

        assert cfg.schema_version == 1
        assert cfg.datasets == ('all',)
        assert cfg.evaluation == ev
        assert cfg.runtime == rt
        assert cfg.output == out
        assert cfg == CoordConfig(model=CoordModelConfig(name='mind'))
        assert model == CoordModelConfig(name='mind')
        assert cfg != 'not-a-config'
        assert model != 'not-a-model'
        assert ev != 'not-an-eval'
        assert rt != 'not-a-runtime'
        assert out != 'not-an-output'

        assert 'mind' in repr(model)
        assert 'random' in repr(ev)
        assert 'cpu' in repr(rt)
        assert 'coordbench_results.csv' in repr(out)
        assert 'CoordConfig' in repr(cfg)

    def test_model_config_validation(self) -> None:
        custom = CoordModelConfig(
            name='custom', target='pkg.mod.Encoder', kwargs={'dim': 64}
        )
        assert custom.target == 'pkg.mod.Encoder'
        assert custom.kwargs == {'dim': 64}

        with pytest.raises(ConfigError, match=r'model\.name'):
            CoordModelConfig(name='   ')
        with pytest.raises(ConfigError, match=r'model\.target'):
            CoordModelConfig(name='x', target='SingleName')
        with pytest.raises(ConfigError, match=r'model\.target'):
            CoordModelConfig(name='x', target='bad-pkg.Cls')
        with pytest.raises(ConfigError, match=r'model\.kwargs'):
            CoordModelConfig(name='x', kwargs=cast(Any, ['not', 'mapping']))
        with pytest.raises(ConfigError, match=r'model\.kwargs'):
            CoordModelConfig(name='x', kwargs=cast(Any, {1: 'bad'}))

    def test_evaluation_config_validation(self) -> None:
        ev = CoordEvaluationConfig(
            methods=['linear', 'chunked', 'knn', 'logreg', 'mlp', 'idw', 'kriging'],
            split='urban_rural_holdout',
            folds=3,
            cell_deg=[2.0, 10, 20.5],
            knn_k=10,
            knn_device='cuda:0',
            respect_official_splits=False,
            cities_path='/tmp/cities15000.txt',
            urban_km=5.0,
            rural_km=25.0,
        )
        assert ev.cell_deg == (2.0, 10.0, 20.5)
        assert ev.to_dict()['cell_deg'] == [2.0, 10.0, 20.5]
        assert ev.cities_path == '/tmp/cities15000.txt'

        assert CoordEvaluationConfig(knn_device='cuda').knn_device == 'cuda'

        bad_cases: list[dict[str, Any]] = [
            {'methods': 'linear'},
            {'methods': []},
            {'methods': ['unknown']},
            {'methods': ['linear', 'linear']},
            {'split': 'official'},
            {'folds': 1},
            {'folds': True},
            {'folds': '5'},
            {'cell_deg': 0},
            {'cell_deg': -1.0},
            {'cell_deg': float('nan')},
            {'cell_deg': float('inf')},
            {'cell_deg': '10.0'},
            {'cell_deg': True},
            {'cell_deg': []},
            {'cell_deg': [5.0, 0.0]},
            {'knn_k': 0},
            {'knn_k': True},
            {'knn_device': 'auto'},
            {'knn_device': 'mps'},
            {'knn_device': 'cuda:-1'},
            {'respect_official_splits': 'true'},
            {'cities_path': '   '},
            {'urban_km': 0.0},
            {'rural_km': 10.0, 'urban_km': 10.0},
            {'rural_km': 5.0, 'urban_km': 10.0},
        ]
        for kwargs in bad_cases:
            with pytest.raises(ConfigError):
                CoordEvaluationConfig(**kwargs)

    def test_runtime_and_output_validation(self) -> None:
        for dev in ('auto', 'cpu', 'mps', 'cuda', 'cuda:0', 'cuda:12'):
            assert CoordRuntimeConfig(device=dev).device == dev
        assert CoordRuntimeConfig(seed=2**64 - 1).seed == 2**64 - 1

        for bad_dev in ('', 'cuda:-1', 'cuda:bad', 'tpu'):
            with pytest.raises(ConfigError):
                CoordRuntimeConfig(device=bad_dev)
        bad_seeds: list[Any] = [-1, 2**64, True, 1.5, '0']
        for bad_seed in bad_seeds:
            with pytest.raises(ConfigError):
                CoordRuntimeConfig(seed=bad_seed)

        with pytest.raises(ConfigError):
            CoordOutputConfig(file='  ')
        with pytest.raises(ConfigError):
            CoordOutputConfig(resume=cast(Any, 'false'))

    def test_coord_config_validation(self) -> None:
        model = CoordModelConfig(name='sincos')
        assert CoordConfig(model=model, datasets=('official',)).datasets == (
            'official',
        )
        assert CoordConfig(
            model=model, datasets=('country', 'worldclim-bio1')
        ).datasets == ('country', 'worldclim-bio1')

        with pytest.raises(ConfigError, match='schema_version'):
            CoordConfig(model=model, schema_version=cast(Any, True))
        with pytest.raises(ConfigError, match='schema_version'):
            CoordConfig(model=model, schema_version=2)
        with pytest.raises(ConfigError, match='CoordModelConfig'):
            CoordConfig(model=cast(Any, {'name': 'sincos'}))
        with pytest.raises(ConfigError, match='sequence'):
            CoordConfig(model=model, datasets='country')
        with pytest.raises(ConfigError, match='empty'):
            CoordConfig(model=model, datasets=[])
        with pytest.raises(ConfigError, match='distinct'):
            CoordConfig(model=model, datasets=['country', 'country'])
        with pytest.raises(ConfigError, match='cannot be combined'):
            CoordConfig(model=model, datasets=['all', 'country'])
        with pytest.raises(ConfigError, match='cannot be combined'):
            CoordConfig(model=model, datasets=['official', 'country'])
        with pytest.raises(ConfigError, match='Unknown coordinate datasets'):
            CoordConfig(model=model, datasets=['does-not-exist'])


class TestParseAndLoadYaml:
    def test_parse_minimal_and_errors(self) -> None:
        minimal = parse_coord_config({'model': {'name': 'sincos'}})
        assert minimal == CoordConfig(model=CoordModelConfig(name='sincos'))

        with pytest.raises(ConfigError, match='mapping'):
            parse_coord_config(cast(Any, ['not', 'mapping']))
        with pytest.raises(ConfigError, match='Unknown keys in config'):
            parse_coord_config({'model': {'name': 'sincos'}, 'extra': 1})
        with pytest.raises(ConfigError, match="Missing required 'model'"):
            parse_coord_config({})
        with pytest.raises(ConfigError, match='model must be a mapping'):
            parse_coord_config({'model': 'sincos'})
        with pytest.raises(ConfigError, match='Unknown keys in model'):
            parse_coord_config({'model': {'name': 'sincos', 'bad': 1}})
        with pytest.raises(ConfigError, match=r"Missing required 'model\.name'"):
            parse_coord_config({'model': {}})
        with pytest.raises(ConfigError, match='Unknown keys in evaluation'):
            parse_coord_config({'model': {'name': 'sincos'}, 'evaluation': {'bad': 1}})
        with pytest.raises(ConfigError, match='Unknown keys in runtime'):
            parse_coord_config({'model': {'name': 'sincos'}, 'runtime': {'bad': 1}})
        with pytest.raises(ConfigError, match='Unknown keys in output'):
            parse_coord_config({'model': {'name': 'sincos'}, 'output': {'bad': 1}})

    def test_yaml_roundtrip_and_scientific_float(self, tmp_path: Path) -> None:
        path = tmp_path / 'coord.yaml'
        path.write_text(
            'schema_version: 1\n'
            'model:\n'
            '  name: mind\n'
            '  kwargs:\n'
            '    dim: 64\n'
            'datasets: [official]\n'
            'evaluation:\n'
            '  methods: [linear, chunked]\n'
            '  split: both\n'
            '  folds: 5\n'
            '  cell_deg: 1e1\n'
            '  respect_official_splits: false\n'
            'runtime:\n'
            '  device: auto\n'
            '  seed: 7\n'
            'output:\n'
            '  file: results/out.csv\n'
            '  resume: true\n',
            encoding='utf-8',
        )
        cfg = load_coord_config(path)
        assert cfg.evaluation.cell_deg == 10.0
        assert cfg.evaluation.respect_official_splits is False
        assert cfg.runtime.seed == 7
        assert cfg.output.resume is True

        dumped = dump_coord_config_yaml(cfg)
        roundtrip_path = tmp_path / 'roundtrip.yaml'
        roundtrip_path.write_text(dumped, encoding='utf-8')
        assert load_coord_config(roundtrip_path) == cfg

    @pytest.mark.parametrize(
        'content',
        [
            '[]',
            'model: [',
            'model: {name: sincos, name: mind}',
            '{[1, 2]: value}',
            'model: !!python/object:builtins.dict {}',
        ],
    )
    def test_bad_yaml_raises_config_error(self, tmp_path: Path, content: str) -> None:
        path = tmp_path / 'bad.yaml'
        path.write_text(content, encoding='utf-8')
        with pytest.raises(ConfigError):
            load_coord_config(path)

    def test_missing_yaml_raises_file_not_found(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_coord_config(tmp_path / 'absent.yaml')
