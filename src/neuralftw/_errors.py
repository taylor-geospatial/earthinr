# Copyright (c) Taylor Geospatial. All rights reserved.
# Licensed under the MIT License.

"""Custom exception hierarchy for NeuralFTW."""


class NeuralFTWError(Exception):
    """Base exception for all NeuralFTW errors."""


class DependencyNotFoundError(NeuralFTWError):
    """Raised when an optional dependency is not installed."""


class DatasetNotFoundError(NeuralFTWError, FileNotFoundError):
    """Raised when a required dataset or file cannot be found."""


class ConfigError(NeuralFTWError, ValueError):
    """Raised when a configuration file or parameter is invalid."""


__all__ = (
    'ConfigError',
    'DatasetNotFoundError',
    'DependencyNotFoundError',
    'NeuralFTWError',
)
