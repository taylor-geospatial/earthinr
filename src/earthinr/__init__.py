# Copyright (c) Taylor Geospatial. All rights reserved.
# Licensed under the MIT License.

"""EarthINR: Implicit Neural Representations for Earth Observation."""

import lazy_loader as lazy

__version__ = '0.1.0'

__getattr__, __dir__, _lazy_all = lazy.attach_stub(__name__, __file__)

__all__ = tuple(sorted({'__version__', *_lazy_all}))
