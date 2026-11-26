# Copyright (c) Taylor Geospatial. All rights reserved.
# Licensed under the MIT License.

"""Strict pure-Python YAML configuration schema for CoordBench runs."""

import math
import os
import re
from collections.abc import Mapping, Sequence
from typing import Any

import yaml
from yaml.constructor import ConstructorError

from .._errors import ConfigError
from .._types import Path
from .catalog import ALL_FAMILIES, list_benchmarks

VALID_METHODS: frozenset[str] = frozenset(
    {'linear', 'chunked', 'knn', 'logreg', 'mlp', 'idw', 'kriging'}
)
VALID_SPLITS: frozenset[str] = frozenset(
    {'random', 'spatial', 'both', 'urban_rural_stratified', 'urban_rural_holdout'}
)

_DEVICE_RE = re.compile(r'^(cpu|mps|cuda(?::[0-9]+)?)$')
_MAX_SEED = 2**64 - 1


class _UniqueKeyLoader(yaml.SafeLoader):
    """YAML SafeLoader that rejects duplicate or unhashable mapping keys."""

    def construct_mapping(
        self, node: yaml.MappingNode, deep: bool = False
    ) -> dict[Any, Any]:
        """Construct a dictionary while rejecting duplicate or unhashable keys."""
        mapping: dict[Any, Any] = {}
        for key_node, value_node in node.value:
            key = self.construct_object(key_node, deep=deep)
            try:
                duplicate = key in mapping
            except TypeError as err:
                raise ConstructorError(
                    'while constructing a mapping',
                    node.start_mark,
                    'found an unhashable key',
                    key_node.start_mark,
                ) from err
            if duplicate:
                raise ConstructorError(
                    'while constructing a mapping',
                    node.start_mark,
                    f'found duplicate key {key!r}',
                    key_node.start_mark,
                )
            mapping[key] = self.construct_object(value_node, deep=deep)
        return mapping


_UniqueKeyLoader.add_implicit_resolver(
    'tag:yaml.org,2002:float',
    re.compile(
        r"""^(?:[-+]?(?:[0-9][0-9_]*)\.[0-9_]*(?:[eE][-+]?[0-9]+)?
        |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)|\.[0-9_]+(?:[eE][-+]?[0-9]+)?
        |[-+]?\.(?:inf|Inf|INF)|\.(?:nan|NaN|NAN))$""",
        re.VERBOSE,
    ),
    list('-+0123456789.'),
)


def _require_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ConfigError(f'{name} must be a mapping.')
    return value


def _reject_unknown_keys(
    mapping: Mapping[str, Any], allowed: frozenset[str], section: str
) -> None:
    unknown = sorted(set(mapping) - allowed)
    if unknown:
        raise ConfigError(f'Unknown keys in {section}: {unknown}.')


def _require_strict_int(
    value: Any, name: str, *, min_value: int | None = None, max_value: int | None = None
) -> int:
    if type(value) is not int:
        raise ConfigError(f'{name} must be an integer.')
    if min_value is not None and value < min_value:
        raise ConfigError(f'{name} must be >= {min_value}.')
    if max_value is not None and value > max_value:
        raise ConfigError(f'{name} must be <= {max_value}.')
    return value


def _require_strict_bool(value: Any, name: str) -> bool:
    if type(value) is not bool:
        raise ConfigError(f'{name} must be a boolean.')
    return value


def _require_positive_float(value: Any, name: str) -> float:
    if type(value) not in (int, float):
        raise ConfigError(f'{name} must be a number.')
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ConfigError(f'{name} must be a finite positive number.')
    return result


def _require_non_blank_str(value: Any, name: str) -> str:
    if type(value) is not str or not value.strip():
        raise ConfigError(f'{name} must be a non-empty string.')
    return value


def _validate_device(value: Any, name: str, *, allow_auto: bool) -> str:
    device = _require_non_blank_str(value, name)
    if _DEVICE_RE.fullmatch(device) or (allow_auto and device == 'auto'):
        return device
    if allow_auto:
        raise ConfigError(
            f"{name} must be 'auto', 'cpu', 'mps', 'cuda', or 'cuda:<index>'."
        )
    raise ConfigError(f"{name} must be 'cpu', 'mps', 'cuda', or 'cuda:<index>'.")


class CoordModelConfig:
    """Model selection and constructor keyword arguments for CoordBench."""

    def __init__(
        self,
        name: str,
        target: str | None = None,
        kwargs: Mapping[str, Any] | None = None,
    ) -> None:
        """Initialize a new CoordModelConfig instance.

        Args:
            name: Model preset or display name.
            target: Optional dotted Python import path to a custom encoder class.
            kwargs: Optional mapping of keyword arguments passed to the constructor.

        Raises:
            ConfigError: If any field fails strict validation.
        """
        self.name = _require_non_blank_str(name, 'model.name')
        if target is not None:
            checked_target = _require_non_blank_str(target, 'model.target')
            parts = checked_target.split('.')
            if len(parts) < 2 or any(not p.isidentifier() for p in parts):
                raise ConfigError('model.target must be a dotted Python symbol.')
            self.target: str | None = checked_target
        else:
            self.target = None

        if kwargs is None:
            self.kwargs: dict[str, Any] = {}
        else:
            raw_kwargs = _require_mapping(kwargs, 'model.kwargs')
            if any(type(k) is not str for k in raw_kwargs):
                raise ConfigError('model.kwargs keys must be strings.')
            self.kwargs = dict(raw_kwargs)

    def to_dict(self) -> dict[str, Any]:
        """Return a YAML-serializable dictionary representation."""
        return {'name': self.name, 'target': self.target, 'kwargs': dict(self.kwargs)}

    def __eq__(self, other: object) -> bool:
        """Compare two model configurations for value equality."""
        if not isinstance(other, CoordModelConfig):
            return NotImplemented
        return self.to_dict() == other.to_dict()

    def __repr__(self) -> str:
        """Return a readable string representation."""
        return (
            f'CoordModelConfig(name={self.name!r}, target={self.target!r}, '
            f'kwargs={self.kwargs!r})'
        )


class CoordEvaluationConfig:
    """Cross-validation, splitting, and probing configuration for CoordBench."""

    def __init__(
        self,
        methods: Sequence[str] = ('linear', 'knn'),
        split: str = 'random',
        folds: int = 5,
        cell_deg: float | Sequence[float] = 10.0,
        knn_k: int = 5,
        knn_device: str = 'cpu',
        respect_official_splits: bool = True,
        cities_path: str | None = None,
        urban_km: float = 10.0,
        rural_km: float = 50.0,
    ) -> None:
        """Initialize a new CoordEvaluationConfig instance.

        Args:
            methods: Non-empty distinct probe methods to run.
            split: Split protocol name.
            folds: Number of cross-validation folds (>= 2).
            cell_deg: Positive grid-cell size in degrees or sequence of sizes for
                sweeps.
            knn_k: Number of nearest neighbors for k-NN probing (>= 1).
            knn_device: Execution device for k-NN probing (``'cpu'``, ``'mps'``,
                ``'cuda'``, or ``'cuda:<index>'``).
            respect_official_splits: Whether official benchmark test masks override CV.
            cities_path: Optional custom GeoNames ``.parquet`` or ``cities15000.txt``
                path for urban/rural splits (defaults to bundled
                ``cities15000.parquet``).
            urban_km: Maximum distance to nearest city in km to classify as urban.
            rural_km: Minimum distance to nearest city in km to classify as rural
                (must be strictly greater than *urban_km*).

        Raises:
            ConfigError: If any field fails strict validation.
        """
        if isinstance(methods, (str, bytes)) or not isinstance(methods, Sequence):
            raise ConfigError('evaluation.methods must be a sequence of method names.')
        method_list = list(methods)
        if not method_list:
            raise ConfigError('evaluation.methods must not be empty.')
        for item in method_list:
            if type(item) is not str or item not in VALID_METHODS:
                raise ConfigError(
                    f'Invalid method {item!r}; expected one of {sorted(VALID_METHODS)}.'
                )
        if len(set(method_list)) != len(method_list):
            raise ConfigError('evaluation.methods must not contain duplicates.')
        self.methods: tuple[str, ...] = tuple(method_list)

        if type(split) is not str or split not in VALID_SPLITS:
            raise ConfigError(
                f'Invalid evaluation.split {split!r}; expected one of '
                f'{sorted(VALID_SPLITS)}.'
            )
        self.split = split
        self.folds = _require_strict_int(folds, 'evaluation.folds', min_value=2)

        if isinstance(cell_deg, Sequence) and not isinstance(cell_deg, (str, bytes)):
            if not cell_deg:
                raise ConfigError('evaluation.cell_deg sequence must not be empty.')
            self.cell_deg: float | tuple[float, ...] = tuple(
                _require_positive_float(v, 'evaluation.cell_deg') for v in cell_deg
            )
        else:
            self.cell_deg = _require_positive_float(cell_deg, 'evaluation.cell_deg')

        self.knn_k = _require_strict_int(knn_k, 'evaluation.knn_k', min_value=1)
        self.knn_device = _validate_device(
            knn_device, 'evaluation.knn_device', allow_auto=False
        )
        self.respect_official_splits = _require_strict_bool(
            respect_official_splits, 'evaluation.respect_official_splits'
        )
        if cities_path is None:
            self.cities_path: str | None = None
        else:
            self.cities_path = _require_non_blank_str(
                cities_path, 'evaluation.cities_path'
            )
        self.urban_km = _require_positive_float(urban_km, 'evaluation.urban_km')
        self.rural_km = _require_positive_float(rural_km, 'evaluation.rural_km')
        if self.rural_km <= self.urban_km:
            raise ConfigError(
                'evaluation.rural_km must be strictly greater than evaluation.urban_km.'
            )

    def to_dict(self) -> dict[str, Any]:
        """Return a YAML-serializable dictionary representation."""
        cell_val: float | list[float] = (
            list(self.cell_deg) if isinstance(self.cell_deg, tuple) else self.cell_deg
        )
        return {
            'methods': list(self.methods),
            'split': self.split,
            'folds': self.folds,
            'cell_deg': cell_val,
            'knn_k': self.knn_k,
            'knn_device': self.knn_device,
            'respect_official_splits': self.respect_official_splits,
            'cities_path': self.cities_path,
            'urban_km': self.urban_km,
            'rural_km': self.rural_km,
        }

    def __eq__(self, other: object) -> bool:
        """Compare two evaluation configurations for value equality."""
        if not isinstance(other, CoordEvaluationConfig):
            return NotImplemented
        return self.to_dict() == other.to_dict()

    def __repr__(self) -> str:
        """Return a readable string representation."""
        return (
            f'CoordEvaluationConfig(methods={self.methods!r}, split={self.split!r}, '
            f'folds={self.folds!r}, cell_deg={self.cell_deg!r})'
        )


class CoordRuntimeConfig:
    """Runtime execution device and RNG seed configuration."""

    def __init__(self, device: str = 'cpu', seed: int = 0) -> None:
        """Initialize a new CoordRuntimeConfig instance.

        Args:
            device: Execution device string (``'auto'``, ``'cpu'``, ``'mps'``,
                ``'cuda'``, or ``'cuda:<index>'``).
            seed: Deterministic integer seed in ``[0, 2**64 - 1]``.

        Raises:
            ConfigError: If any field fails strict validation.
        """
        self.device = _validate_device(device, 'runtime.device', allow_auto=True)
        self.seed = _require_strict_int(
            seed, 'runtime.seed', min_value=0, max_value=_MAX_SEED
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a YAML-serializable dictionary representation."""
        return {'device': self.device, 'seed': self.seed}

    def __eq__(self, other: object) -> bool:
        """Compare two runtime configurations for value equality."""
        if not isinstance(other, CoordRuntimeConfig):
            return NotImplemented
        return self.to_dict() == other.to_dict()

    def __repr__(self) -> str:
        """Return a readable string representation."""
        return f'CoordRuntimeConfig(device={self.device!r}, seed={self.seed!r})'


class CoordOutputConfig:
    """Result CSV output path and resume configuration."""

    def __init__(
        self, file: str = 'results/coordbench_results.csv', resume: bool = False
    ) -> None:
        """Initialize a new CoordOutputConfig instance.

        Args:
            file: Non-blank output CSV path.
            resume: Whether to skip rows already present in *file*.

        Raises:
            ConfigError: If any field fails strict validation.
        """
        self.file = _require_non_blank_str(file, 'output.file')
        self.resume = _require_strict_bool(resume, 'output.resume')

    def to_dict(self) -> dict[str, Any]:
        """Return a YAML-serializable dictionary representation."""
        return {'file': self.file, 'resume': self.resume}

    def __eq__(self, other: object) -> bool:
        """Compare two output configurations for value equality."""
        if not isinstance(other, CoordOutputConfig):
            return NotImplemented
        return self.to_dict() == other.to_dict()

    def __repr__(self) -> str:
        """Return a readable string representation."""
        return f'CoordOutputConfig(file={self.file!r}, resume={self.resume!r})'


class CoordConfig:
    """Complete strictly validated CoordBench evaluation configuration."""

    def __init__(
        self,
        model: CoordModelConfig,
        datasets: Sequence[str] = ('all',),
        evaluation: CoordEvaluationConfig | None = None,
        runtime: CoordRuntimeConfig | None = None,
        output: CoordOutputConfig | None = None,
        schema_version: int = 1,
    ) -> None:
        """Initialize a new CoordConfig instance.

        Args:
            model: Model selection and constructor options.
            datasets: Non-empty sequence of distinct dataset or family names (or
                ``('all',)`` / ``('official',)``).
            evaluation: Probe and split configuration.
            runtime: Execution device and seed configuration.
            output: Result CSV path and resume configuration.
            schema_version: Schema version integer (must be exact integer ``1``).

        Raises:
            ConfigError: If any field fails strict validation.
        """
        if type(schema_version) is not int or schema_version != 1:
            raise ConfigError('schema_version must be the integer 1.')
        self.schema_version = schema_version

        if not isinstance(model, CoordModelConfig):
            raise ConfigError('model must be a CoordModelConfig instance.')
        self.model = model

        if isinstance(datasets, (str, bytes)) or not isinstance(datasets, Sequence):
            raise ConfigError('datasets must be a sequence of dataset names.')
        ds_list = list(datasets)
        if not ds_list:
            raise ConfigError('datasets must not be empty.')
        for item in ds_list:
            _require_non_blank_str(item, 'datasets')
        if len(set(ds_list)) != len(ds_list):
            raise ConfigError('datasets must contain distinct names.')
        if ('all' in ds_list or 'official' in ds_list) and len(ds_list) != 1:
            raise ConfigError(
                "'all' and 'official' cannot be combined with other datasets."
            )
        known = {'all', 'official', *ALL_FAMILIES, *list_benchmarks()}
        unknown = sorted(set(ds_list) - known)
        if unknown:
            raise ConfigError(f'Unknown coordinate datasets: {unknown}.')
        self.datasets: tuple[str, ...] = tuple(ds_list)

        self.evaluation = CoordEvaluationConfig() if evaluation is None else evaluation
        self.runtime = CoordRuntimeConfig() if runtime is None else runtime
        self.output = CoordOutputConfig() if output is None else output

    def to_dict(self) -> dict[str, Any]:
        """Return a YAML-serializable dictionary with explicit defaults."""
        return {
            'schema_version': self.schema_version,
            'model': self.model.to_dict(),
            'datasets': list(self.datasets),
            'evaluation': self.evaluation.to_dict(),
            'runtime': self.runtime.to_dict(),
            'output': self.output.to_dict(),
        }

    def __eq__(self, other: object) -> bool:
        """Compare two CoordConfig instances for value equality."""
        if not isinstance(other, CoordConfig):
            return NotImplemented
        return self.to_dict() == other.to_dict()

    def __repr__(self) -> str:
        """Return a readable string representation."""
        return (
            f'CoordConfig(model={self.model!r}, datasets={self.datasets!r}, '
            f'evaluation={self.evaluation!r})'
        )


_TOP_KEYS: frozenset[str] = frozenset(
    {'schema_version', 'model', 'datasets', 'evaluation', 'runtime', 'output'}
)
_MODEL_KEYS: frozenset[str] = frozenset({'name', 'target', 'kwargs'})
_EVAL_KEYS: frozenset[str] = frozenset(
    {
        'methods',
        'split',
        'folds',
        'cell_deg',
        'knn_k',
        'knn_device',
        'respect_official_splits',
        'cities_path',
        'urban_km',
        'rural_km',
    }
)
_RUNTIME_KEYS: frozenset[str] = frozenset({'device', 'seed'})
_OUTPUT_KEYS: frozenset[str] = frozenset({'file', 'resume'})


def parse_coord_config(data: Mapping[str, Any]) -> CoordConfig:
    """Strictly validate a plain mapping into a :class:`CoordConfig`.

    Args:
        data: Top-level configuration mapping.

    Returns:
        Validated :class:`CoordConfig` instance.

    Raises:
        ConfigError: If any key is unknown, missing, or fails validation.
    """
    raw = _require_mapping(data, 'config')
    _reject_unknown_keys(raw, _TOP_KEYS, 'config')

    if 'model' not in raw:
        raise ConfigError("Missing required 'model' section.")
    model_raw = _require_mapping(raw['model'], 'model')
    _reject_unknown_keys(model_raw, _MODEL_KEYS, 'model')
    if 'name' not in model_raw:
        raise ConfigError("Missing required 'model.name'.")
    model = CoordModelConfig(**model_raw)

    evaluation = CoordEvaluationConfig()
    if 'evaluation' in raw:
        eval_raw = _require_mapping(raw['evaluation'], 'evaluation')
        _reject_unknown_keys(eval_raw, _EVAL_KEYS, 'evaluation')
        evaluation = CoordEvaluationConfig(**eval_raw)

    runtime = CoordRuntimeConfig()
    if 'runtime' in raw:
        runtime_raw = _require_mapping(raw['runtime'], 'runtime')
        _reject_unknown_keys(runtime_raw, _RUNTIME_KEYS, 'runtime')
        runtime = CoordRuntimeConfig(**runtime_raw)

    output = CoordOutputConfig()
    if 'output' in raw:
        output_raw = _require_mapping(raw['output'], 'output')
        _reject_unknown_keys(output_raw, _OUTPUT_KEYS, 'output')
        output = CoordOutputConfig(**output_raw)

    return CoordConfig(
        model=model,
        datasets=raw.get('datasets', ('all',)),
        evaluation=evaluation,
        runtime=runtime,
        output=output,
        schema_version=raw.get('schema_version', 1),
    )


def load_coord_config(path: Path) -> CoordConfig:
    """Load and strictly validate a CoordBench YAML configuration file.

    Args:
        path: Path to a YAML file.

    Returns:
        Validated :class:`CoordConfig` instance.

    Raises:
        FileNotFoundError: If *path* does not exist.
        ConfigError: If the YAML syntax, tags, duplicate keys, or schema are invalid.
    """
    filepath = os.fspath(path)
    with open(filepath, encoding='utf-8') as f:
        try:
            loaded = yaml.load(f, Loader=_UniqueKeyLoader)
        except yaml.YAMLError as err:
            raise ConfigError(f'{filepath}: {err}') from err
    if not isinstance(loaded, dict):
        raise ConfigError(f'{filepath}: top level must be a YAML mapping.')
    return parse_coord_config(loaded)


def dump_coord_config_yaml(cfg: CoordConfig) -> str:
    """Serialize a :class:`CoordConfig` to canonical YAML text.

    Args:
        cfg: Configuration instance to serialize.

    Returns:
        YAML string preserving field ordering and explicit defaults.
    """
    return yaml.safe_dump(cfg.to_dict(), sort_keys=False)


__all__ = (
    'VALID_METHODS',
    'VALID_SPLITS',
    'CoordConfig',
    'CoordEvaluationConfig',
    'CoordModelConfig',
    'CoordOutputConfig',
    'CoordRuntimeConfig',
    'dump_coord_config_yaml',
    'load_coord_config',
    'parse_coord_config',
)
