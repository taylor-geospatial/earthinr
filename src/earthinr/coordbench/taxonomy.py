# Copyright (c) Taylor Geospatial. All rights reserved.
# Licensed under the MIT License.

"""Task taxonomy for grouping and aggregating CoordBench results."""

SOCIO: frozenset[str] = frozenset(
    {
        'pdfm-conus27',
        'california-housing',
        'mosaiks-population',
        'mosaiks-nightlights',
        'mosaiks-income',
        'mosaiks-housing',
        'mosaiks-roads',
        'satclip-population',
        'bt-population',
        'bt-distroad',
        *(
            f'sustainbench-{k}'
            for k in (
                'asset',
                'water',
                'sanitation',
                'child_mortality',
                'women_edu',
                'women_bmi',
            )
        ),
        *(
            f'places-{k}'
            for k in (
                'asthma',
                'cancer',
                'chd',
                'checkup',
                'copd',
                'diabetes',
                'high_chol',
                'mental_health',
                'obesity',
                'phys_health',
                'sleep_lt7',
                'smoking',
            )
        ),
    }
)

ENV: frozenset[str] = frozenset(
    {
        'worldclim-bio1',
        'worldclim-bio12',
        'soilgrids-soc',
        'soilgrids-phh2o',
        'dm-aster_ged',
        'dm-openet_ensemble',
        'mosaiks-elevation',
        'mosaiks-treecover',
        'satclip-elevation',
        'satclip-air-temp',
        'bt-biomass',
        'bt-bioclim',
    }
)

LAND: frozenset[str] = frozenset(
    {
        *(
            f'dm-{k}'
            for k in (
                'africa_crop_mask',
                'canada_crops_coarse',
                'canada_crops_fine',
                'descals',
                'ethiopia_crops',
                'glance',
                'lcmap_lc',
                'lcmap_lcc',
                'lcmap_lu',
                'lcmap_luc',
                'lucas_lc',
                'lucas_lu',
                'us_trees',
            )
        ),
        'satclip-biome',
        'satclip-country',
        'satclip-ecoregion',
        'country',
        'ecoregions',
        'bt-cropharvest',
        'bt-landcover',
    }
)

FAMILIES: tuple[str, ...] = ('socio', 'env', 'land')

R2_FLOOR: float = -1.0


def family_of(name: str) -> str:
    """Return the task family for a CoordBench benchmark name.

    Args:
        name: Benchmark identifier (e.g. ``'worldclim-bio1'``).

    Returns:
        One of ``'socio'``, ``'env'``, ``'land'``, or ``'other'``.
    """
    if name in SOCIO:
        return 'socio'
    if name in ENV:
        return 'env'
    if name in LAND:
        return 'land'
    return 'other'


__all__ = ('ENV', 'FAMILIES', 'LAND', 'R2_FLOOR', 'SOCIO', 'family_of')
