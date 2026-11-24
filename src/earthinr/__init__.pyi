# Copyright (c) Taylor Geospatial. All rights reserved.
# Licensed under the MIT License.

"""Type stubs for top-level EarthINR lazy exports."""

from . import analysis, cli, coordbench, models, probes
from ._errors import (
    ConfigError,
    DatasetNotFoundError,
    DependencyNotFoundError,
    EarthINRError,
)

__version__: str

__all__: tuple[str, ...] = (
    'ConfigError',
    'DatasetNotFoundError',
    'DependencyNotFoundError',
    'EarthINRError',
    '__version__',
    'analysis',
    'cli',
    'coordbench',
    'models',
    'probes',
)
