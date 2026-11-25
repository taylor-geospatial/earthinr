# Copyright (c) Taylor Geospatial. All rights reserved.
# Licensed under the MIT License.

import json
import math
from pathlib import Path
from typing import Any, cast

import huggingface_hub
import pytest
import torch

from neuralftw._errors import ConfigError
from neuralftw.models._utils import (
    equal_earth_projection,
    hf_download,
    load_json_config,
    load_safetensors,
    lonlat_to_xyz,
)


class TestEqualEarthProjection:
    def test_origin_and_symmetry(self) -> None:
        coords = torch.tensor([[0.0, 0.0], [90.0, 45.0], [-90.0, -45.0]])
        out = equal_earth_projection(coords, coord_order='lonlat')
        assert out.shape == (3, 2)
        assert out.dtype == torch.float32
        assert torch.allclose(out[0], torch.zeros(2), atol=1e-7)
        assert torch.allclose(out[1], -out[2], atol=1e-6)

    def test_latlon_order_matches_lonlat(self) -> None:
        lonlat = torch.tensor([[-122.4, 37.8], [139.7, 35.7]], dtype=torch.float64)
        latlon = lonlat.flip(-1)
        out_lonlat = equal_earth_projection(lonlat, coord_order='lonlat')
        out_latlon = equal_earth_projection(latlon, coord_order='latlon')
        assert torch.allclose(out_lonlat, out_latlon, atol=1e-12)

    def test_exact_reference_values(self) -> None:
        lon_deg, lat_deg = 45.0, 30.0
        lon = math.radians(lon_deg)
        lat = math.radians(lat_deg)
        s3 = math.sqrt(3.0)
        a1, a2, a3, a4, sf = 1.340264, -0.081106, 0.000893, 0.003796, 66.50336
        theta = math.asin((s3 / 2.0) * math.sin(lat))
        expected_x = (
            (2.0 * s3 * lon * math.cos(theta))
            / (3.0 * (9 * a4 * theta**8 + 7 * a3 * theta**6 + 3 * a2 * theta**2 + a1))
            * (sf / 180.0)
        )
        expected_y = (a4 * theta**9 + a3 * theta**7 + a2 * theta**3 + a1 * theta) * (
            sf / 180.0
        )
        out = equal_earth_projection(
            torch.tensor([[lon_deg, lat_deg]], dtype=torch.float64)
        )
        assert torch.allclose(
            out[0],
            torch.tensor([expected_x, expected_y], dtype=torch.float64),
            atol=1e-12,
        )

    def test_3d_batch_shape(self) -> None:
        coords = torch.zeros((2, 4, 2))
        out = equal_earth_projection(coords)
        assert out.shape == (2, 4, 2)

    def test_invalid_inputs(self) -> None:
        with pytest.raises(ValueError, match=r'\(\.\.\., 2\)'):
            equal_earth_projection(torch.tensor(1.0))
        with pytest.raises(ValueError, match=r'\(\.\.\., 2\)'):
            equal_earth_projection(torch.zeros((4, 3)))
        with pytest.raises(ValueError, match='coord_order'):
            equal_earth_projection(torch.zeros((2, 2)), coord_order=cast(Any, 'xy'))


class TestLonLatToXYZ:
    def test_cardinal_points(self) -> None:
        lon = torch.tensor([0.0, 90.0, 0.0, 0.0, 180.0], dtype=torch.float64)
        lat = torch.tensor([0.0, 0.0, 90.0, -90.0, 0.0], dtype=torch.float64)
        xyz = lonlat_to_xyz(lon, lat)
        expected = torch.tensor(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.0, 0.0, -1.0],
                [-1.0, 0.0, 0.0],
            ],
            dtype=torch.float64,
        )
        assert xyz.shape == (5, 3)
        assert torch.allclose(xyz, expected, atol=1e-12)
        norms = torch.linalg.norm(xyz, dim=-1)
        assert torch.allclose(norms, torch.ones(5, dtype=torch.float64), atol=1e-12)

    def test_non_finite_passthrough(self) -> None:
        lon = torch.tensor([float('nan'), 0.0])
        lat = torch.tensor([float('nan'), float('inf')])
        xyz = lonlat_to_xyz(lon, lat)
        assert xyz.shape == (2, 3)
        assert torch.isnan(xyz[0]).all()

    def test_invalid_inputs(self) -> None:
        with pytest.raises(ValueError, match='same shape'):
            lonlat_to_xyz(torch.zeros(2), torch.zeros(3))
        with pytest.raises(ValueError, match=r'\[-90, 90\]'):
            lonlat_to_xyz(torch.tensor([0.0]), torch.tensor([90.1]))
        with pytest.raises(ValueError, match=r'\[-90, 90\]'):
            lonlat_to_xyz(torch.tensor([0.0]), torch.tensor([-91.0]))


class TestCheckpointUtils:
    def test_hf_download(self, monkeypatch: pytest.MonkeyPatch) -> None:
        captured: dict[str, Any] = {}

        def _fake_hf_hub_download(**kwargs: Any) -> str:
            captured.update(kwargs)
            return '/cache/model.safetensors'

        monkeypatch.setattr(huggingface_hub, 'hf_hub_download', _fake_hf_hub_download)
        path = hf_download(
            'org/repo', 'model.safetensors', repo_type='dataset', revision='v1'
        )
        assert path == '/cache/model.safetensors'
        assert captured == {
            'repo_id': 'org/repo',
            'filename': 'model.safetensors',
            'repo_type': 'dataset',
            'revision': 'v1',
        }

    def test_load_safetensors(self, fake_safetensors: Path) -> None:
        state = load_safetensors(fake_safetensors, device='cpu')
        assert list(state.keys()) == ['weight']
        assert torch.equal(state['weight'], torch.arange(6, dtype=torch.float32))

    def test_load_json_config(self, tmp_path: Path) -> None:
        cfg_path = tmp_path / 'config.json'
        cfg_path.write_text(json.dumps({'depth': 4, 'dim': 256}), encoding='utf-8')
        assert load_json_config(cfg_path) == {'depth': 4, 'dim': 256}

        bad_path = tmp_path / 'list.json'
        bad_path.write_text('[1, 2, 3]', encoding='utf-8')
        with pytest.raises(ConfigError, match='Expected JSON object'):
            load_json_config(bad_path)
