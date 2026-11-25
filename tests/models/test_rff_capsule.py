# Copyright (c) Taylor Geospatial. All rights reserved.
# Licensed under the MIT License.

from pathlib import Path

import numpy as np
import pytest
import safetensors.torch
import torch

from neuralftw.models.rff_capsule import (
    EqualEarthRFFEncoder,
    GAIR_Weights,
    GAIRLocationEncoder,
    GaussianEncoding,
    GeoCLIP_Weights,
    GeoCLIPLocationEncoder,
    RFFCapsule,
    SLED_Weights,
    SLEDLocationEncoder,
    TaxaBind_Weights,
    TaxaBindLocationEncoder,
    gair,
    geoclip,
    sled,
    taxabind,
)


class TestGaussianEncodingAndCapsule:
    def test_explicit_and_sampled_b(self) -> None:
        b = torch.ones((4, 2))
        enc = GaussianEncoding(b=b)
        assert not enc.b.requires_grad
        out = enc(torch.zeros((3, 2)))
        assert out.shape == (3, 8)
        assert torch.allclose(out[:, :4], torch.ones((3, 4)))
        assert torch.allclose(out[:, 4:], torch.zeros((3, 4)), atol=1e-6)

        sampled = GaussianEncoding(sigma=1.0, input_size=2, encoded_size=4)
        assert sampled.b.shape == (4, 2)

    def test_validation(self) -> None:
        with pytest.raises(ValueError, match='required when b is None'):
            GaussianEncoding(sigma=1.0)
        with pytest.raises(ValueError, match='not both'):
            GaussianEncoding(sigma=1.0, b=torch.zeros((2, 2)))
        with pytest.raises(ValueError, match='non-empty'):
            EqualEarthRFFEncoder(sigma=())

    def test_state_dict_key_schema(self) -> None:
        enc = EqualEarthRFFEncoder(
            sigma=(1.0, 16.0), encoded_size=8, hidden_dim=16, out_dim=12
        )
        keys = set(enc.state_dict().keys())
        expected = {
            f'LocEnc{i}.{suffix}'
            for i in range(2)
            for suffix in (
                'capsule.0.b',
                'capsule.1.weight',
                'capsule.1.bias',
                'capsule.3.weight',
                'capsule.3.bias',
                'capsule.5.weight',
                'capsule.5.bias',
                'head.0.weight',
                'head.0.bias',
            )
        }
        assert keys == expected
        assert isinstance(enc._modules['LocEnc0'], RFFCapsule)


class TestCoordOrderParityAndEncoders:
    @pytest.mark.parametrize('batch_size', [1, 7])
    def test_geoclip_gair_coord_order_parity(
        self, tmp_path: Path, batch_size: int
    ) -> None:
        # Shared weights loaded into both a latlon-native model (GeoCLIP)
        # and a lonlat-native model (GAIR) must produce identical outputs
        # when both receive universal (lon, lat) WGS84 degrees.
        raw_latlon = EqualEarthRFFEncoder(
            sigma=(1.0, 16.0),
            encoded_size=8,
            hidden_dim=16,
            out_dim=12,
            coord_order='latlon',
        )
        ckpt = tmp_path / 'shared.safetensors'
        safetensors.torch.save_file(raw_latlon.state_dict(), ckpt)

        geo_enc = GeoCLIPLocationEncoder(
            checkpoint=ckpt,
            sigma=(1.0, 16.0),
            encoded_size=8,
            hidden_dim=16,
            out_dim=12,
        )
        gair_enc = GAIRLocationEncoder(
            checkpoint=ckpt,
            sigma=(1.0, 16.0),
            encoded_size=8,
            hidden_dim=16,
            out_dim=12,
        )

        lon = torch.linspace(-150.0, 150.0, batch_size)
        lat = torch.linspace(-70.0, 70.0, batch_size)
        lonlat = torch.stack((lon, lat), dim=-1)
        latlon = torch.stack((lat, lon), dim=-1)

        out_geo = geo_enc(lonlat)
        out_gair = gair_enc(lonlat)
        out_raw = raw_latlon(latlon)

        assert out_geo.shape == (batch_size, 12)
        assert out_gair.shape == (batch_size, 12)
        assert torch.allclose(out_geo, out_raw, atol=1e-6)
        assert torch.allclose(out_gair, out_raw, atol=1e-6)

        # Verify numpy batched encode matches forward
        np_out = geo_enc.encode(lon.numpy(), lat.numpy())
        assert np.allclose(np_out, out_geo.detach().numpy(), atol=1e-6)

    def test_gair_legacy_pth_loading(self, tmp_path: Path) -> None:
        ref = EqualEarthRFFEncoder(
            sigma=(1.0, 16.0),
            encoded_size=8,
            hidden_dim=16,
            out_dim=12,
            coord_order='lonlat',
        )
        wrapped = {
            'state_dict': {
                **{f'module.loc_enc.{k}': v for k, v in ref.state_dict().items()},
                'module.vision.weight': torch.zeros((4, 4)),
            }
        }
        pth_path = tmp_path / 'gair.pth'
        torch.save(wrapped, pth_path)

        loaded = GAIRLocationEncoder(
            checkpoint=pth_path,
            sigma=(1.0, 16.0),
            encoded_size=8,
            hidden_dim=16,
            out_dim=12,
        )
        coords = torch.tensor([[-122.4, 37.8], [2.35, 48.85]])
        assert torch.allclose(loaded(coords), ref(coords), atol=1e-6)

        bad_pth = tmp_path / 'bad.pth'
        torch.save({'other.weight': torch.zeros((2, 2))}, bad_pth)
        with pytest.raises(KeyError, match='LocEnc0'):
            GAIRLocationEncoder(
                checkpoint=bad_pth,
                sigma=(1.0,),
                encoded_size=4,
                hidden_dim=8,
                out_dim=4,
            )

    @pytest.mark.parametrize('batch_size', [1, 7])
    def test_factories_and_weights(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, batch_size: int
    ) -> None:
        tiny_meta = {
            'sigma': (1.0, 4.0),
            'encoded_size': 4,
            'hidden_dim': 8,
            'output_dim': 6,
        }
        ref = EqualEarthRFFEncoder(
            sigma=(1.0, 4.0), encoded_size=4, hidden_dim=8, out_dim=6
        )
        ckpt = tmp_path / 'tiny.safetensors'
        safetensors.torch.save_file(ref.state_dict(), ckpt)

        coords = torch.linspace(-40.0, 40.0, batch_size * 2).reshape(batch_size, 2)

        for weights_cls, member, factory, expected_name in (
            (GeoCLIP_Weights, GeoCLIP_Weights.DEFAULT, geoclip, 'geoclip'),
            (TaxaBind_Weights, TaxaBind_Weights.ECOGEO, taxabind, 'taxabind'),
            (SLED_Weights, SLED_Weights.S1_S2_LS, sled, 'sled'),
            (SLED_Weights, SLED_Weights.S2_LS, sled, 'sled'),
            (GAIR_Weights, GAIR_Weights.DEFAULT, gair, 'gair'),
        ):
            monkeypatch.setattr(member.value, 'meta', tiny_meta)
            monkeypatch.setattr(weights_cls, 'download', lambda self: str(ckpt))
            model = factory(weights=member.name)
            assert model.name == expected_name
            assert model.embed_dim == 6
            out = model(coords)
            assert out.shape == (batch_size, 6)
            assert out.dtype == torch.float32

    def test_untrained_and_errors(self, tmp_path: Path) -> None:
        enc = TaxaBindLocationEncoder(
            sigma=(1.0,), encoded_size=4, hidden_dim=8, out_dim=6
        )
        assert enc(torch.zeros((2, 2))).shape == (2, 6)

        enc_sled = SLEDLocationEncoder(
            sigma=(2.0,), encoded_size=4, hidden_dim=8, out_dim=6
        )
        assert enc_sled(torch.zeros((2, 2))).shape == (2, 6)

        bad_file = tmp_path / 'model.pt'
        bad_file.write_bytes(b'')
        with pytest.raises(ValueError, match=r'\.safetensors'):
            GeoCLIPLocationEncoder(checkpoint=bad_file)
        with pytest.raises(ValueError, match='at most one'):
            GeoCLIPLocationEncoder(weights=GeoCLIP_Weights.DEFAULT, checkpoint=bad_file)
