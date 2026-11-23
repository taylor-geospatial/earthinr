# Copyright (c) Taylor Geospatial. All rights reserved.
# Licensed under the MIT License.

"""Custom exception hierarchy for EarthINR."""


class EarthINRError(Exception):
    """Base exception for all EarthINR errors."""


class DependencyNotFoundError(EarthINRError):
    """Raised when an optional dependency is not installed."""


class DatasetNotFoundError(EarthINRError, FileNotFoundError):
    """Raised when a required dataset or file cannot be found."""


class ConfigError(EarthINRError, ValueError):
    """Raised when a configuration file or parameter is invalid."""


__all__ = (
    'ConfigError',
    'DatasetNotFoundError',
    'DependencyNotFoundError',
    'EarthINRError',
)
