# Copyright (c) Taylor Geospatial. All rights reserved.
# Licensed under the MIT License.

"""Shared type aliases for NeuralFTW."""

import os
from typing import Literal

type Path = str | os.PathLike[str]
type CoordOrder = Literal['lonlat', 'latlon']

__all__ = ('CoordOrder', 'Path')
