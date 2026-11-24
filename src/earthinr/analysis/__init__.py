# Copyright (c) Taylor Geospatial. All rights reserved.
# Licensed under the MIT License.

"""Spatial statistics and implicit resolution analysis utilities."""

from .variogram import (
    EARTH_RADIUS_KM,
    KM_PER_DEGREE,
    analyze_implicit_resolution,
    empirical_semivariogram,
    geodesic_offset,
    half_sill_distance,
    sample_sphere_points,
)

__all__ = (
    'EARTH_RADIUS_KM',
    'KM_PER_DEGREE',
    'analyze_implicit_resolution',
    'empirical_semivariogram',
    'geodesic_offset',
    'half_sill_distance',
    'sample_sphere_points',
)
