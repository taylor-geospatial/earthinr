# Copyright (c) Taylor Geospatial. All rights reserved.
# Licensed under the MIT License.

"""CoordBench benchmark loading, evaluation, and leaderboard tools."""

import lazy_loader as lazy

__getattr__, __dir__, _lazy_all = lazy.attach_stub(__name__, __file__)

__all__ = tuple(sorted(_lazy_all))
