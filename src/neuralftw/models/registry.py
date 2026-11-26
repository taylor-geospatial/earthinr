# Copyright (c) Taylor Geospatial. All rights reserved.
# Licensed under the MIT License.

"""YAML-backed and programmatic model preset registry and lazy encoder builder."""

import importlib
import os
from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Any, Literal, TypeVar, overload

import yaml

from .._errors import ConfigError

if TYPE_CHECKING:
    from ._base import LocationEncoder

_F = TypeVar('_F', bound=Callable[..., Any])

_PRESET_DIR = os.path.normpath(
    os.path.join(os.path.dirname(__file__), '..', 'conf', 'model')
)
_ALLOWED_PRESET_KEYS: frozenset[str] = frozenset(
    {'name', 'target', 'kwargs', 'description'}
)
_CUSTOM_PRESETS: dict[str, 'ModelPreset'] = {}
_CUSTOM_FACTORIES: dict[tuple[str, str], Callable[..., Any]] = {}


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


def _validate_preset_name(name: str) -> str:
    """Validate a model preset name against empty strings and path traversal."""
    if type(name) is not str or not name.strip():
        raise ConfigError('Model preset name must be a non-empty string.')
    if os.path.basename(name) != name or '/' in name or '\\' in name or '..' in name:
        raise ConfigError(f'Invalid model preset name {name!r}.')
    return name


def _packaged_model_names() -> set[str]:
    """Return the set of model preset names shipped as packaged YAML files."""
    directory = _preset_dir()
    if not os.path.isdir(directory):
        return set()
    return {
        entry[:-5]
        for entry in os.listdir(directory)
        if entry.endswith('.yaml') and os.path.isfile(os.path.join(directory, entry))
    }


def list_models() -> tuple[str, ...]:
    """Return all registered model preset names in sorted order.

    Includes both packaged YAML presets and models registered programmatically
    via :func:`register_model`.

    Returns:
        Sorted tuple of model preset identifiers.
    """
    return tuple(sorted(_packaged_model_names() | set(_CUSTOM_PRESETS)))


@overload
def register_model(
    name: str,
    target: None = None,
    *,
    kwargs: Mapping[str, Any] | None = None,
    description: str = '',
    overwrite: bool = False,
) -> Callable[[_F], _F]: ...


@overload
def register_model(
    name: str,
    target: str | Callable[..., Any],
    *,
    kwargs: Mapping[str, Any] | None = None,
    description: str = '',
    overwrite: bool = False,
) -> ModelPreset: ...


def register_model(
    name: str,
    target: str | Callable[..., Any] | None = None,
    *,
    kwargs: Mapping[str, Any] | None = None,
    description: str = '',
    overwrite: bool = False,
) -> ModelPreset | Callable[[_F], _F]:
    """Register a custom model preset in the in-memory registry.

    Can be called directly with a :class:`LocationEncoder` class, factory
    callable, or dotted import path string, or used as a class/function
    decorator when *target* is omitted::

        @register_model('my-encoder', description='Custom MLP location encoder.')
        class MyEncoder(LocationEncoder): ...

    Args:
        name: Unique preset identifier used with :func:`get_model` and
            :func:`list_models`.
        target: :class:`LocationEncoder` subclass, factory callable, dotted
            Python import path (e.g. ``'my_pkg.MyEncoder'``), or ``None`` to
            return a decorator.
        kwargs: Optional default keyword arguments forwarded when instantiating
            the encoder.
        description: Short human-readable summary shown by ``neuralftw models``.
        overwrite: Whether to allow replacing an existing preset with the same
            *name*.

    Returns:
        The registered :class:`ModelPreset` when *target* is provided, or a
        decorator that registers and returns the decorated class/callable.

    Raises:
        ConfigError: If *name* or *target* is invalid, or if *name* is already
            registered and *overwrite* is ``False``.
    """
    _validate_preset_name(name)
    if target is None:

        def _decorator(obj: _F) -> _F:
            register_model(
                name, obj, kwargs=kwargs, description=description, overwrite=overwrite
            )
            return obj

        return _decorator

    if not overwrite and name in (_packaged_model_names() | set(_CUSTOM_PRESETS)):
        raise ConfigError(
            f'Model preset {name!r} is already registered. '
            'Pass overwrite=True to replace it.'
        )

    factory: Callable[..., Any] | None = None
    if isinstance(target, str):
        target_path = target
    elif callable(target):
        module = getattr(target, '__module__', '')
        obj_name = getattr(target, '__name__', '')
        if not module or not obj_name or not obj_name.isidentifier():
            raise ConfigError(
                'Callable target must define valid __module__ and __name__ attributes.'
            )
        target_path = f'{module}.{obj_name}'
        factory = target
    else:
        raise ConfigError(
            f'target must be a callable or dotted Python path string, got {target!r}.'
        )

    preset = ModelPreset(
        name=name, target=target_path, kwargs=kwargs, description=description
    )
    prev = _CUSTOM_PRESETS.get(name)
    if prev is not None:
        _CUSTOM_FACTORIES.pop((name, prev.target), None)
    _CUSTOM_PRESETS[name] = preset
    if factory is not None:
        _CUSTOM_FACTORIES[(name, target_path)] = factory
    return preset


def unregister_model(name: str) -> None:
    """Remove a programmatically registered model preset.

    Args:
        name: Preset identifier previously passed to :func:`register_model`.

    Raises:
        ConfigError: If *name* is not a registered custom preset.
    """
    preset = _CUSTOM_PRESETS.pop(name, None)
    if preset is None:
        raise ConfigError(f'No custom model preset registered under {name!r}.')
    _CUSTOM_FACTORIES.pop((name, preset.target), None)


def get_model_preset(name: str) -> ModelPreset:
    """Load and validate a registered model preset by name.

    Checks custom presets registered via :func:`register_model` first, then
    packaged YAML presets.

    Args:
        name: Registered model preset identifier.

    Returns:
        Parsed :class:`ModelPreset` instance.

    Raises:
        ConfigError: If *name* is invalid, unknown, or its YAML definition is malformed.
    """
    _validate_preset_name(name)
    if name in _CUSTOM_PRESETS:
        return _CUSTOM_PRESETS[name]

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
    a YAML file. Otherwise loads the named preset and merges *kwargs* on top of
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

    factory = _CUSTOM_FACTORIES.get((preset.name, preset.target))
    if factory is None:
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
        **kwargs: Optional keyword overrides forwarded to the model constructor
            (including ``precision='fp32'|'fp16'|'int8'|'int4'`` when
            ``backend='onnx'``).

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
    'register_model',
    'resolve_model_preset',
    'unregister_model',
)
