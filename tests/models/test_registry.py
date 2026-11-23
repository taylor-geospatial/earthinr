# Copyright (c) Taylor Geospatial. All rights reserved.
# Licensed under the MIT License.

import importlib
import inspect
from pathlib import Path
from typing import Any

import pytest

import earthinr
from earthinr import ConfigError
from earthinr.models import LocationEncoder, ReSIRENLocationEncoder, registry
from earthinr.models.registry import (
    ModelPreset,
    build_model,
    get_model,
    get_model_preset,
    list_models,
    resolve_model_preset,
)

EXPECTED_PRESETS: tuple[str, ...] = (
    'aef',
    'climplicit',
    'csp-fmow',
    'csp-inat',
    'gair',
    'geoclip',
    'mind',
    'mind-3072',
    'mind-head',
    'nerf',
    'osmgraphclip',
    'osmgraphclip-a-l10',
    'osmgraphclip-a-l40',
    'osmgraphclip-ms-l10',
    'rff3072',
    'satclip',
    'satclip-resnet18-l10',
    'satclip-resnet18-l40',
    'satclip-resnet50-l10',
    'satclip-resnet50-l40',
    'satclip-vit16-l10',
    'sh40',
    'sincos',
    'sinr',
    'sled',
    'sled-s2',
    'spherical-harmonics',
    'taxabind',
    'terranova',
    'terranova-st',
    'tessera',
    'tte',
    'xyz',
)


def _not_an_encoder(**kwargs: Any) -> dict[str, Any]:
    return kwargs


class TestModelPreset:
    def test_valid_and_dunders(self) -> None:
        preset = ModelPreset(
            name='demo',
            target='earthinr.models.encoders.XYZLocationEncoder',
            kwargs={'batch_size': 64},
            description='Demo preset',
        )
        other = ModelPreset(
            name='demo',
            target='earthinr.models.encoders.XYZLocationEncoder',
            kwargs={'batch_size': 64},
            description='Demo preset',
        )
        assert preset == other
        assert (preset == 'not-a-preset') is False
        assert 'ModelPreset' in repr(preset)
        assert 'earthinr.models.encoders.XYZLocationEncoder' in preset.to_yaml()

        no_kwargs = ModelPreset(
            name='plain', target='earthinr.models.encoders.XYZLocationEncoder'
        )
        assert no_kwargs.kwargs == {}
        assert no_kwargs.description == ''

    def test_validation_errors(self) -> None:
        with pytest.raises(ConfigError, match='name'):
            ModelPreset(name='  ', target='earthinr.models.encoders.XYZLocationEncoder')
        with pytest.raises(ConfigError, match='target must be a non-empty'):
            ModelPreset(name='x', target='')
        with pytest.raises(ConfigError, match='dotted Python symbol'):
            ModelPreset(name='x', target='singleident')
        with pytest.raises(ConfigError, match='dotted Python symbol'):
            ModelPreset(name='x', target='earthinr.not-valid')
        bad_kwargs: Any = ['bad']
        bad_key_kwargs: Any = {1: 'a'}
        bad_desc: Any = 123
        with pytest.raises(ConfigError, match='kwargs must be a mapping'):
            ModelPreset(
                name='x',
                target='earthinr.models.encoders.XYZLocationEncoder',
                kwargs=bad_kwargs,
            )
        with pytest.raises(ConfigError, match='kwargs keys must be strings'):
            ModelPreset(
                name='x',
                target='earthinr.models.encoders.XYZLocationEncoder',
                kwargs=bad_key_kwargs,
            )
        with pytest.raises(ConfigError, match='description must be a string'):
            ModelPreset(
                name='x',
                target='earthinr.models.encoders.XYZLocationEncoder',
                description=bad_desc,
            )


class TestPackagedPresets:
    def test_all_packaged_presets_parse_and_resolve(self) -> None:
        names = list_models()
        assert names == EXPECTED_PRESETS

        for name in names:
            preset = get_model_preset(name)
            assert preset.name == name
            assert preset.description
            mod_path, attr = preset.target.rsplit('.', 1)
            module = importlib.import_module(mod_path)
            target_obj = getattr(module, attr)
            assert callable(target_obj)
            sig = inspect.signature(target_obj)
            for key in preset.kwargs:
                assert key in sig.parameters, f'{name}: unexpected kwarg {key!r}'

    @pytest.mark.parametrize(
        ('name', 'expected_dim'),
        [
            ('sincos', 4),
            ('xyz', 3),
            ('nerf', 96),
            ('spherical-harmonics', 16),
            ('sh40', 1681),
            ('rff3072', 3072),
            ('aef', 64),
            ('tessera', 128),
        ],
    )
    def test_analytic_and_product_get_model(self, name: str, expected_dim: int) -> None:
        enc = get_model(name, device='cpu')
        assert isinstance(enc, LocationEncoder)
        assert enc.embed_dim == expected_dim

    @pytest.mark.parametrize(
        'name',
        [
            'satclip',
            'satclip-vit16-l10',
            'satclip-resnet18-l40',
            'satclip-resnet18-l10',
            'satclip-resnet50-l40',
            'satclip-resnet50-l10',
            'geoclip',
            'sinr',
            'climplicit',
            'gair',
            'taxabind',
            'sled',
            'sled-s2',
            'tte',
            'terranova',
            'terranova-st',
            'csp-inat',
            'csp-fmow',
            'osmgraphclip',
            'osmgraphclip-ms-l10',
            'osmgraphclip-a-l40',
            'osmgraphclip-a-l10',
        ],
    )
    def test_neural_presets_build_with_uninitialized_weights(self, name: str) -> None:
        extra: dict[str, Any] = {'weights': None, 'device': 'cpu'}
        if name == 'terranova-st':
            extra.update(
                {
                    'location_dim': 16,
                    'wire_widths': (16,),
                    'hash_n_levels': 2,
                    'hash_n_features': 2,
                    'hash_log2_size': 6,
                    'hash_hidden': (16,),
                    'time_dim': 8,
                    'time_width': 16,
                    'time_depth': 1,
                    'time_fourier': 4,
                    'st_width': 16,
                    'st_heads': 2,
                    'st_depth': 1,
                }
            )
        elif name == 'terranova':
            extra.update(
                {
                    'location_dim': 16,
                    'wire_widths': (16,),
                    'hash_n_levels': 2,
                    'hash_n_features': 2,
                    'hash_log2_size': 6,
                    'hash_hidden': (16,),
                }
            )
        elif name == 'climplicit':
            extra.update({'dim_hidden': 16, 'dim_out': 8, 'num_layers': 2})
        elif name == 'tte':
            extra.update(
                {
                    'n_sites': 16,
                    'site_embed_dim': 16,
                    'hidden_dim': 16,
                    'output_dim': 16,
                    'n_reslayers': 1,
                    'n_registers': 4,
                }
            )
        enc = get_model(name, **extra)
        assert isinstance(enc, LocationEncoder)

    def test_mind_presets_build_with_injected_model(self) -> None:
        tiny = ReSIRENLocationEncoder(embed_dim=16, out_dim=8, depth=1)
        enc_pooled = get_model('mind', weights=None, model=tiny, dim=8)
        assert enc_pooled.embed_dim == 8
        enc_head = get_model('mind-head', weights=None, model=tiny)
        assert enc_head.embed_dim == 8

    def test_resolve_and_build_edge_cases(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        custom = resolve_model_preset(
            'custom-xyz',
            target='earthinr.models.encoders.XYZLocationEncoder',
            kwargs={'batch_size': 32},
        )
        enc = build_model(custom, device='cpu')
        assert enc.batch_size == 32

        same = resolve_model_preset('xyz')
        assert same == get_model_preset('xyz')

        merged = resolve_model_preset('nerf', kwargs={'include_xyz': True})
        assert merged.kwargs == {'num_frequencies': 16, 'include_xyz': True}

        bad_preset = ModelPreset(
            name='bad', target='tests.models.test_registry._not_an_encoder'
        )
        with pytest.raises(TypeError, match='expected LocationEncoder'):
            build_model(bad_preset)

        with pytest.raises(ConfigError, match='non-empty string'):
            get_model_preset('')
        with pytest.raises(ConfigError, match='Invalid model preset name'):
            get_model_preset('../xyz')
        with pytest.raises(ConfigError, match='Unknown model preset'):
            get_model_preset('does-not-exist')

        # Missing preset directory -> empty list
        monkeypatch.setattr(registry, '_PRESET_DIR', str(tmp_path / 'missing'))
        assert list_models() == ()

        # Malformed YAML files in custom preset directory
        bad_dir = tmp_path / 'conf'
        bad_dir.mkdir()
        monkeypatch.setattr(registry, '_PRESET_DIR', str(bad_dir))
        (bad_dir / 'syntax.yaml').write_text('a: [1, 2\n', encoding='utf-8')
        (bad_dir / 'list.yaml').write_text('- one\n- two\n', encoding='utf-8')
        (bad_dir / 'unknown.yaml').write_text(
            'target: earthinr.models.encoders.XYZLocationEncoder\nextra: 1\n',
            encoding='utf-8',
        )
        (bad_dir / 'notarget.yaml').write_text('name: notarget\n', encoding='utf-8')

        with pytest.raises(ConfigError, match=r'syntax\.yaml'):
            get_model_preset('syntax')
        with pytest.raises(ConfigError, match='must be a YAML mapping'):
            get_model_preset('list')
        with pytest.raises(ConfigError, match='unknown preset keys'):
            get_model_preset('unknown')
        with pytest.raises(ConfigError, match="missing required 'target'"):
            get_model_preset('notarget')

    def test_lazy_top_level_exports(self) -> None:
        assert earthinr.__version__ == '0.1.0'
        assert earthinr.ConfigError is ConfigError
        assert earthinr.models.XYZLocationEncoder is not None
        assert earthinr.coordbench.CoordConfig is not None
