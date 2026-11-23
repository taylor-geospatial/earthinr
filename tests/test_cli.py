# Copyright (c) Taylor Geospatial. All rights reserved.
# Licensed under the MIT License.

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from earthinr.cli import main


class TestCLIBasicsAndModels:
    def test_version_and_no_subcommand(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        with pytest.raises(SystemExit) as exc_info:
            main(['--version'])
        assert exc_info.value.code == 0
        assert 'earthinr 0.1.0' in capsys.readouterr().out

        rc = main([])
        assert rc == 0
        assert 'usage: earthinr' in capsys.readouterr().out

    def test_models_subcommand(self, capsys: pytest.CaptureFixture[str]) -> None:
        assert main(['models']) == 0
        out = capsys.readouterr().out
        assert 'mind' in out
        assert 'satclip' in out
        assert 'tessera' in out

        assert main(['models', 'mind']) == 0
        parsed = yaml.safe_load(capsys.readouterr().out)
        assert parsed['name'] == 'mind'
        assert parsed['target'] == 'earthinr.models.mind.MINDLocationEncoder'
        assert parsed['kwargs']['dim'] == 64

        assert main(['models', 'not-a-model']) == 2
        assert 'error:' in capsys.readouterr().err


class TestCLICoordAndDryRun:
    def test_dry_run_imports_no_heavy_dependencies_in_subprocess(self) -> None:
        code = (
            'import sys; '
            'from earthinr.cli import main; '
            "rc1 = main(['models']); "
            "rc2 = main(['coord', '--model', 'mind', '--datasets', 'worldclim', "
            "'--cell-deg', '2', '10', '--dry-run']); "
            'assert rc1 == 0 and rc2 == 0; '
            "heavy = [m for m in ('torch', 'numpy', 'pandas') if m in sys.modules]; "
            'assert not heavy, f"Unexpected heavy imports: {heavy}"'
        )
        proc = subprocess.run(
            [sys.executable, '-c', code], capture_output=True, text=True, check=False
        )
        assert proc.returncode == 0, proc.stderr

    def test_coord_dry_run_flags_and_yaml_merge(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        cfg_file = tmp_path / 'base.yaml'
        cfg_file.write_text(
            yaml.safe_dump(
                {
                    'schema_version': 1,
                    'model': {
                        'name': 'custom',
                        'target': 'earthinr.models.encoders.XYZLocationEncoder',
                        'kwargs': {'batch_size': 128},
                    },
                    'datasets': ['air_temp'],
                }
            ),
            encoding='utf-8',
        )

        # Config without --model keeps custom target/kwargs
        rc = main(['coord', '--config', str(cfg_file), '--dry-run'])
        assert rc == 0
        out_base = yaml.safe_load(capsys.readouterr().out)
        assert (
            out_base['model']['target'] == 'earthinr.models.encoders.XYZLocationEncoder'
        )

        # CLI overrides replace model root and set all evaluation/runtime/output flags
        rc = main(
            [
                'coord',
                '--config',
                str(cfg_file),
                '--model',
                'sincos',
                '--datasets',
                'worldclim',
                'air_temp',
                '--methods',
                'linear',
                'chunked',
                '--split',
                'both',
                '--folds',
                '3',
                '--cell-deg',
                '2',
                '10',
                '--knn-k',
                '7',
                '--knn-device',
                'cpu',
                '--device',
                'cpu',
                '--seed',
                '42',
                '--output',
                str(tmp_path / 'out.csv'),
                '--resume',
                '--no-respect-official-splits',
                '--cities-path',
                '/tmp/cities15000.txt',
                '--dry-run',
            ]
        )
        assert rc == 0
        merged = yaml.safe_load(capsys.readouterr().out)
        assert merged['model'] == {'name': 'sincos', 'target': None, 'kwargs': {}}
        assert merged['datasets'] == ['worldclim', 'air_temp']
        assert merged['evaluation']['methods'] == ['linear', 'chunked']
        assert merged['evaluation']['split'] == 'both'
        assert merged['evaluation']['folds'] == 3
        assert merged['evaluation']['cell_deg'] == [2.0, 10.0]
        assert merged['evaluation']['knn_k'] == 7
        assert merged['evaluation']['respect_official_splits'] is False
        assert merged['evaluation']['cities_path'] == '/tmp/cities15000.txt'
        assert merged['runtime'] == {'device': 'cpu', 'seed': 42}
        assert merged['output']['resume'] is True

        # Single --cell-deg collapses to scalar float, and spatial baseline validates
        rc = main(
            [
                'coord',
                '--model',
                'kriging',
                '--methods',
                'kriging',
                '--cell-deg',
                '5.0',
                '--dry-run',
            ]
        )
        assert rc == 0
        scalar = yaml.safe_load(capsys.readouterr().out)
        assert scalar['evaluation']['cell_deg'] == 5.0

    def test_coord_end_to_end_and_leaderboard(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        # Create synthetic local air_temp parquet under tmp_path
        rng = np.random.default_rng(0)
        n = 50
        lat = rng.uniform(-40.0, 40.0, size=n)
        lon = rng.uniform(-100.0, 100.0, size=n)
        df = pd.DataFrame(
            {'lat': lat, 'lon': lon, 'meanT': np.sin(np.deg2rad(lat)) * 20.0}
        )
        data_dir = tmp_path / 'air_temp'
        data_dir.mkdir()
        df.to_parquet(data_dir / 'data.parquet', index=False)

        out_csv = tmp_path / 'results.csv'
        rc = main(
            [
                'coord',
                '--model',
                'sincos',
                '--datasets',
                'air_temp',
                '--methods',
                'linear',
                '--split',
                'random',
                '--folds',
                '3',
                '--root',
                str(tmp_path),
                '--output',
                str(out_csv),
            ]
        )
        assert rc == 0
        assert 'Wrote 1 result row(s)' in capsys.readouterr().out
        assert out_csv.exists()

        # Run leaderboard subcommand on the produced CSV
        rc_lb = main(
            ['leaderboard', str(out_csv), '--method', 'linear', '--split', 'random']
        )
        assert rc_lb == 0
        lb_out = capsys.readouterr().out
        assert 'Mean Rank' in lb_out
        assert 'Mean Score' in lb_out
        assert 'sincos' in lb_out

    def test_cli_errors_and_rejected_legacy_overrides(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        # Legacy Hydra-style overrides rejected with exit code 2
        assert main(['coord', '--model', 'sincos', 'evaluation.folds=3']) == 2
        assert 'Legacy key=value override' in capsys.readouterr().err

        assert main(['coord', '--model', 'sincos', '+model.name=xyz']) == 2
        assert 'Legacy key=value override' in capsys.readouterr().err

        # Unrecognized non-override flag triggers argparse SystemExit(2)
        with pytest.raises(SystemExit) as exc_info:
            main(['coord', '--not-a-real-flag'])
        assert exc_info.value.code == 2

        # Missing config file
        assert (
            main(['coord', '--config', str(tmp_path / 'nope.yaml'), '--dry-run']) == 2
        )
        assert 'error: Config file not found' in capsys.readouterr().err

        # Missing model section
        assert main(['coord', '--dry-run']) == 2
        assert 'error:' in capsys.readouterr().err

        # Unknown model preset
        assert main(['coord', '--model', 'unknown-model', '--dry-run']) == 2
        assert 'error: Unknown model preset' in capsys.readouterr().err

        # Leaderboard errors
        assert main(['leaderboard', str(tmp_path / 'missing.csv')]) == 2
        assert 'error:' in capsys.readouterr().err
