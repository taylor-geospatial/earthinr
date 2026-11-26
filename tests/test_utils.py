# Copyright (c) Taylor Geospatial. All rights reserved.
# Licensed under the MIT License.

import math

import pytest
import torch

import neuralftw
from neuralftw._errors import (
    ConfigError,
    DatasetNotFoundError,
    DependencyNotFoundError,
    NeuralFTWError,
)
from neuralftw._utils import float64_device, float64_dtype, lazy_import, resolve_device
from neuralftw.cli import main


class TestPackage:
    def test_version_and_exports(self) -> None:
        assert isinstance(neuralftw.__version__, str)
        assert neuralftw.__version__ == '0.1.0'
        assert hasattr(neuralftw, 'models')
        assert hasattr(neuralftw, 'probes')
        assert hasattr(neuralftw, 'coordbench')

    def test_error_hierarchy(self) -> None:
        assert issubclass(DependencyNotFoundError, NeuralFTWError)
        assert issubclass(DatasetNotFoundError, NeuralFTWError)
        assert issubclass(DatasetNotFoundError, FileNotFoundError)
        assert issubclass(ConfigError, NeuralFTWError)
        assert issubclass(ConfigError, ValueError)


class TestCLI:
    def test_main_no_args(self) -> None:
        assert main([]) == 0

    def test_main_version(self, capsys: pytest.CaptureFixture[str]) -> None:
        with pytest.raises(SystemExit) as exc_info:
            main(['--version'])
        assert exc_info.value.code == 0
        captured = capsys.readouterr()
        assert neuralftw.__version__ in captured.out


class TestLazyImport:
    def test_import_existing_module(self) -> None:
        mod = lazy_import('math')
        assert mod is math

    def test_import_missing_without_extra(self) -> None:
        with pytest.raises(
            DependencyNotFoundError, match='pip install _nonexistent_neuralftw_pkg_xyz'
        ):
            lazy_import('_nonexistent_neuralftw_pkg_xyz')

    def test_import_missing_with_extra(self) -> None:
        with pytest.raises(
            DependencyNotFoundError, match=r'pip install neuralftw\[products\]'
        ):
            lazy_import('_nonexistent_neuralftw_pkg_xyz', extra='products')


class TestResolveDevice:
    def test_cpu_string_never_queries_cuda(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def _fail_cuda() -> bool:
            raise AssertionError('CUDA must not be queried for cpu')

        monkeypatch.setattr(torch.cuda, 'is_available', _fail_cuda)
        assert resolve_device('cpu') == torch.device('cpu')
        assert resolve_device('  cpu  ') == torch.device('cpu')

    def test_cpu_device_object_never_queries_cuda(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def _fail_cuda() -> bool:
            raise AssertionError('CUDA must not be queried for cpu device')

        monkeypatch.setattr(torch.cuda, 'is_available', _fail_cuda)
        assert resolve_device(torch.device('cpu')) == torch.device('cpu')
        assert resolve_device('cpu:0') == torch.device('cpu', 0)

    def test_auto_picks_cpu_when_cuda_unavailable(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
        assert resolve_device('auto') == torch.device('cpu')

    def test_auto_picks_current_cuda_when_available(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
        monkeypatch.setattr(torch.cuda, 'current_device', lambda: 1)
        assert resolve_device('auto') == torch.device('cuda', 1)

    def test_cuda_unavailable_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
        with pytest.raises(ValueError, match='CUDA is not available'):
            resolve_device('cuda')
        with pytest.raises(ValueError, match='CUDA is not available'):
            resolve_device(torch.device('cuda', 0))

    def test_cuda_available_and_index_validation(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
        monkeypatch.setattr(torch.cuda, 'device_count', lambda: 2)
        assert resolve_device('cuda') == torch.device('cuda')
        assert resolve_device('cuda:0') == torch.device('cuda', 0)
        assert resolve_device('cuda:1') == torch.device('cuda', 1)
        with pytest.raises(ValueError, match='out of range'):
            resolve_device('cuda:2')

    def test_mps_validation(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(torch.backends.mps, 'is_available', lambda: False)
        with pytest.raises(ValueError, match='MPS is not available'):
            resolve_device('mps')

        monkeypatch.setattr(torch.backends.mps, 'is_available', lambda: True)
        assert resolve_device('mps') == torch.device('mps')

    def test_other_torch_device_passthrough(self) -> None:
        assert resolve_device('meta') == torch.device('meta')

    @pytest.mark.parametrize('bad_device', ['', '   ', 'invalid_dev', 'cuda:-1'])
    def test_invalid_device_strings(self, bad_device: str) -> None:
        with pytest.raises(ValueError, match=r'Device string|Invalid device'):
            resolve_device(bad_device)

    def test_float64_mps_fallbacks(self) -> None:
        assert float64_dtype(torch.device('cpu')) == torch.float64
        assert float64_dtype(torch.device('mps')) == torch.float32
        assert float64_device(torch.device('cpu')) == torch.device('cpu')
        assert float64_device(torch.device('mps')) == torch.device('cpu')
