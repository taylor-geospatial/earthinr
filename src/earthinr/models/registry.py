# Copyright (c) Taylor Geospatial. All rights reserved.
# Licensed under the MIT License.

"""YAML-backed model preset registry and lazy encoder builder."""

import importlib
import os
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Literal

import yaml

from .._errors import ConfigError

if TYPE_CHECKING:
    from ._base import LocationEncoder

_PRESET_DIR = os.path.normpath(
    os.path.join(os.path.dirname(__file__), '..', 'conf', 'model')
)
_ALLOWED_PRESET_KEYS: frozenset[str] = frozenset(
    {'name', 'target', 'kwargs', 'description'}
)


class ModelPreset:
    """Typed specification for a named location encoder preset."""

    def __init__(
        self,
        name: str,
        target: str,
        kwargs: Mapping[str, Any] | None = None,
        description: str = '',
    ) -> None:
        """Initialize a new ModelPreset instance.

        Args:
            name: Unique preset identifier (e.g. ``'mind'`` or ``'satclip'``).
            target: Dotted Python import path to a :class:`LocationEncoder` class or
                factory callable.
            kwargs: Optional keyword arguments forwarded when building the encoder.
            description: Short human-readable summary of the model preset.

        Raises:
            ConfigError: If *name*, *target*, *kwargs*, or *description* is invalid.
        """
        if type(name) is not str or not name.strip():
            raise ConfigError('preset name must be a non-empty string.')
        if type(target) is not str or not target.strip():
            raise ConfigError('preset target must be a non-empty string.')
        parts = target.split('.')
        if len(parts) < 2 or any(not part.isidentifier() for part in parts):
            raise ConfigError(
                f'preset target must be a dotted Python symbol, got {target!r}.'
            )
        if kwargs is None:
            resolved_kwargs: dict[str, Any] = {}
        elif not isinstance(kwargs, Mapping):
            raise ConfigError('preset kwargs must be a mapping.')
        else:
            if any(type(k) is not str for k in kwargs):
                raise ConfigError('preset kwargs keys must be strings.')
            resolved_kwargs = dict(kwargs)
        if type(description) is not str:
            raise ConfigError('preset description must be a string.')

        self.name = name
        self.target = target
        self.kwargs = resolved_kwargs
        self.description = description

    def to_dict(self) -> dict[str, Any]:
        """Return a YAML-serializable dictionary representation."""
        return {
            'name': self.name,
            'target': self.target,
            'kwargs': dict(self.kwargs),
            'description': self.description,
        }

    def to_yaml(self) -> str:
        """Serialize this preset to canonical YAML text."""
        return yaml.safe_dump(self.to_dict(), sort_keys=False)

    def __eq__(self, other: object) -> bool:
        """Compare two ModelPreset instances for value equality."""
        if not isinstance(other, ModelPreset):
            return NotImplemented
        return self.to_dict() == other.to_dict()

    def __repr__(self) -> str:
        """Return a readable string representation."""
        return (
            f'ModelPreset(name={self.name!r}, target={self.target!r}, '
            f'kwargs={self.kwargs!r}, description={self.description!r})'
        )


def _preset_dir() -> str:
    """Return the filesystem directory containing packaged model preset YAMLs."""
    return _PRESET_DIR


def list_models() -> tuple[str, ...]:
    """Return all registered model preset names in sorted order.

    Returns:
        Sorted tuple of model preset identifiers.
    """
    directory = _preset_dir()
    if not os.path.isdir(directory):
        return ()
    names = [
        entry[:-5]
        for entry in os.listdir(directory)
        if entry.endswith('.yaml') and os.path.isfile(os.path.join(directory, entry))
    ]
    return tuple(sorted(names))


def get_model_preset(name: str) -> ModelPreset:
    """Load and validate a packaged model preset by name.

    Args:
        name: Registered model preset identifier (matching ``<name>.yaml``).

    Returns:
        Parsed :class:`ModelPreset` instance.

    Raises:
        ConfigError: If *name* is invalid, unknown, or its YAML definition is malformed.
    """
    if type(name) is not str or not name.strip():
        raise ConfigError('Model preset name must be a non-empty string.')
    if os.path.basename(name) != name or '/' in name or '\\' in name or '..' in name:
        raise ConfigError(f'Invalid model preset name {name!r}.')

    yaml_path = os.path.join(_preset_dir(), f'{name}.yaml')
    if not os.path.isfile(yaml_path):
        available = ', '.join(list_models())
        raise ConfigError(
            f'Unknown model preset {name!r}. Available presets: {available}.'
        )

    with open(yaml_path, encoding='utf-8') as f:
        try:
            raw = yaml.safe_load(f)
        except yaml.YAMLError as err:
            raise ConfigError(f'{yaml_path}: {err}') from err

    if not isinstance(raw, Mapping):
        raise ConfigError(f'{yaml_path}: preset must be a YAML mapping.')
    unknown = sorted(set(raw) - _ALLOWED_PRESET_KEYS)
    if unknown:
        raise ConfigError(f'{yaml_path}: unknown preset keys {unknown}.')
    if 'target' not in raw:
        raise ConfigError(f"{yaml_path}: missing required 'target' field.")

    return ModelPreset(
        name=raw.get('name', name),
        target=raw['target'],
        kwargs=raw.get('kwargs'),
        description=raw.get('description', ''),
    )


def resolve_model_preset(
    name: str, *, target: str | None = None, kwargs: Mapping[str, Any] | None = None
) -> ModelPreset:
    """Resolve a model selection into a concrete :class:`ModelPreset`.

    When *target* is provided, constructs a custom preset directly without looking up
    a YAML file. Otherwise loads the named YAML preset and merges *kwargs* on top of
    its defaults.

    Args:
        name: Preset or custom display name.
        target: Optional dotted Python target overriding preset lookup.
        kwargs: Optional keyword overrides merged into preset kwargs.

    Returns:
        Resolved :class:`ModelPreset` instance.
    """
    if target is not None:
        return ModelPreset(name=name, target=target, kwargs=kwargs)
    preset = get_model_preset(name)
    if not kwargs:
        return preset
    merged = dict(preset.kwargs)
    merged.update(kwargs)
    return ModelPreset(
        name=preset.name,
        target=preset.target,
        kwargs=merged,
        description=preset.description,
    )


def build_model(preset: ModelPreset, **runtime_kwargs: Any) -> 'LocationEncoder':
    """Instantiate a :class:`LocationEncoder` from a :class:`ModelPreset`.

    Args:
        preset: Resolved model preset containing target import path and default kwargs.
        **runtime_kwargs: Runtime keyword overrides (e.g. ``device='cpu'`` or
            ``weights=None``) merged over ``preset.kwargs``.

    Returns:
        Instantiated :class:`LocationEncoder`.

    Raises:
        TypeError: If the resolved target does not return a :class:`LocationEncoder`.
    """
    from ._base import LocationEncoder

    module_path, attr_name = preset.target.rsplit('.', 1)
    module = importlib.import_module(module_path)
    factory = getattr(module, attr_name)
    merged_kwargs = dict(preset.kwargs)
    merged_kwargs.update(runtime_kwargs)
    encoder = factory(**merged_kwargs)
    if not isinstance(encoder, LocationEncoder):
        raise TypeError(
            f'Target {preset.target!r} returned {type(encoder).__name__}, '
            'expected LocationEncoder.'
        )
    return encoder


def get_model(
    name: str, *, backend: Literal['torch', 'onnx'] = 'torch', **kwargs: Any
) -> 'LocationEncoder':
    """Load a named model preset and instantiate its :class:`LocationEncoder`.

    Args:
        name: Registered model preset identifier (see :func:`list_models`).
        backend: Inference backend, either ``'torch'`` (default) or ``'onnx'``.
        **kwargs: Optional keyword overrides forwarded to the model constructor.

    Returns:
        Instantiated :class:`LocationEncoder`.

    Raises:
        ValueError: If *backend* is invalid or *name* does not have an ONNX
            export when ``backend='onnx'``.
    """
    if backend == 'torch':
        return build_model(get_model_preset(name), **kwargs)
    if backend == 'onnx':
        from .onnx import ONNX_PRESET_SPECS, onnx_encoder

        if name not in ONNX_PRESET_SPECS:
            supported = ', '.join(sorted(ONNX_PRESET_SPECS))
            raise ValueError(
                f'Preset {name!r} does not have an ONNX export. '
                f'Supported ONNX presets: {supported}.'
            )
        return onnx_encoder(name, **kwargs)
    raise ValueError(f"Unsupported backend {backend!r}. Expected 'torch' or 'onnx'.")


__all__ = (
    'ModelPreset',
    'build_model',
    'get_model',
    'get_model_preset',
    'list_models',
    'resolve_model_preset',
)
