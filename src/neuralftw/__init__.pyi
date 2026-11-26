# Copyright (c) Taylor Geospatial. All rights reserved.
# Licensed under the MIT License.

"""Type stubs for top-level NeuralFTW lazy exports."""

from . import analysis, cli, coordbench, models, probes, viz
from ._errors import (
    ConfigError,
    DatasetNotFoundError,
    DependencyNotFoundError,
    NeuralFTWError,
)
from .models._base import LocationEncoder
from .models.registry import get_model, list_models, register_model, unregister_model
from .viz import embeddings_to_rgb, encode_grid, make_coord_grid, plot_embedding_map

__version__: str

__all__: tuple[str, ...] = (
    'ConfigError',
    'DatasetNotFoundError',
    'DependencyNotFoundError',
    'LocationEncoder',
    'NeuralFTWError',
    '__version__',
    'analysis',
    'cli',
    'coordbench',
    'embeddings_to_rgb',
    'encode_grid',
    'get_model',
    'list_models',
    'make_coord_grid',
    'models',
    'plot_embedding_map',
    'probes',
    'register_model',
    'unregister_model',
    'viz',
)
