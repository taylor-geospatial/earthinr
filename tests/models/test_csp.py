# Copyright (c) Taylor Geospatial. All rights reserved.
# Licensed under the MIT License.

from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest
import safetensors.torch
import torch

from neuralftw.models.csp import (
    CSP_Weights,
    CSPLocationEncoder,
    GridCellSpatialRelationEncoder,
    MultiLayerFeedForwardNN,
    SingleFeedForwardNN,
    csp,
)


class TestCSPComponents:
    @pytest.mark.parametrize('act', ['leakyrelu', 'gelu', 'relu', 'sigmoid', 'tanh'])
    def test_single_ffn_activations_and_options(self, act: str) -> None:
        layer = SingleFeedForwardNN(
            8,
            8,
            activation=act,
            dropout_rate=0.1,
            use_layernorm=True,
            skip_connection=True,
        ).eval()
        out = layer(torch.randn(3, 8))
        assert out.shape == (3, 8)

    def test_invalid_activation_raises(self) -> None:
        with pytest.raises(ValueError, match='Unsupported CSP activation'):
            SingleFeedForwardNN(4, 4, activation='swish')

    def test_multilayer_ffn_zero_and_multiple_hidden(self) -> None:
        ffn0 = MultiLayerFeedForwardNN(8, 4, num_hidden_layers=0)
        assert len(ffn0.layers) == 1
        assert ffn0(torch.randn(2, 8)).shape == (2, 4)

        ffn2 = MultiLayerFeedForwardNN(8, 4, num_hidden_layers=2, hidden_dim=6)
        assert len(ffn2.layers) == 3
        assert ffn2(torch.randn(2, 8)).shape == (2, 4)

    def test_gridcell_validation_and_single_freq(self) -> None:
        with pytest.raises(ValueError, match='frequency_num'):
            GridCellSpatialRelationEncoder(frequency_num=0)
        with pytest.raises(ValueError, match='radius bounds'):
            GridCellSpatialRelationEncoder(min_radius=0.0, max_radius=10.0)
        with pytest.raises(ValueError, match='radius bounds'):
            GridCellSpatialRelationEncoder(min_radius=10.0, max_radius=5.0)

        enc1 = GridCellSpatialRelationEncoder(
            frequency_num=1, hidden_dim=8, output_dim=6, normalize_coords=False
        )
        assert enc1.freqs.shape == (1,)
        assert enc1(torch.randn(2, 2)).shape == (2, 6)
        assert 'freqs' not in enc1.state_dict()


class TestCSPLocationEncoder:
    @pytest.fixture
    def tiny_csp_ckpt(self, tmp_path: Path) -> Path:
        torch.manual_seed(0)
        model = GridCellSpatialRelationEncoder(
            frequency_num=4, hidden_dim=12, output_dim=8, activation='leakyrelu'
        )
        path = tmp_path / 'csp.safetensors'
        safetensors.torch.save_file(model.state_dict(), path)
        return path

    def test_dynamic_batch_sizes_and_encode(self, tiny_csp_ckpt: Path) -> None:
        enc = csp(
            weights=None,
            checkpoint_path=tiny_csp_ckpt,
            frequency_num=4,
            hidden_dim=12,
            output_dim=8,
        )
        assert enc.name == 'csp'
        assert enc.embed_dim == 8

        for batch_size in (1, 7):
            coords = torch.randn(batch_size, 2) * 45.0
            out = enc(coords)
            assert out.shape == (batch_size, 8)
            assert out.dtype == torch.float32

        lon = np.array([-120.0, 0.0, 120.0], dtype=np.float64)
        lat = np.array([35.0, 0.0, -35.0], dtype=np.float64)
        emb = enc.encode(lon, lat)
        assert emb.shape == (3, 8)
        assert emb.dtype == np.float32

    @pytest.mark.parametrize(
        ('weight_arg', 'expected_act'),
        [(CSP_Weights.INAT, torch.nn.LeakyReLU), ('FMOW', torch.nn.GELU)],
    )
    def test_weights_presets_and_activation_inference(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tiny_csp_ckpt: Path,
        weight_arg: CSP_Weights | str,
        expected_act: type[torch.nn.Module],
    ) -> None:
        monkeypatch.setattr(CSP_Weights, 'download', lambda self: str(tiny_csp_ckpt))
        enc = CSPLocationEncoder(
            weights=weight_arg, frequency_num=4, hidden_dim=12, output_dim=8
        )
        first_layer = cast(Any, enc.model.ffn.layers[0])
        assert isinstance(first_layer.act, expected_act)
        assert enc(torch.zeros(2, 2)).shape == (2, 8)

    def test_random_init_explicit_activation(self) -> None:
        enc = CSPLocationEncoder(
            weights=None, activation='gelu', frequency_num=4, hidden_dim=8, output_dim=6
        )
        assert enc(torch.zeros(3, 2)).shape == (3, 6)
