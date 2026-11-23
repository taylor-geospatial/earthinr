# Copyright (c) Taylor Geospatial. All rights reserved.
# Licensed under the MIT License.

"""Static CoordBench benchmark catalog with zero heavy imports."""

COORDBENCH_REPO = 'taylor-geospatial/CoordBench'

DEEPMIND_EVAL_CONFIGS: tuple[str, ...] = (
    'africa_crop_mask',
    'aster_ged',
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
    'openet_ensemble',
    'us_trees',
)

USAVARS_LABELS: tuple[str, ...] = (
    'treecover',
    'elevation',
    'population',
    'nightlights',
    'income',
    'roads',
    'housing',
)

USAVARS_LOG_LABELS: frozenset[str] = frozenset(
    {'population', 'income', 'nightlights', 'housing'}
)

SUSTAINBENCH_TASKS: dict[str, str] = {
    'asset': 'asset_index',
    'water': 'water_index',
    'sanitation': 'sanitation_index',
    'child_mortality': 'under5_mort',
    'women_edu': 'women_edu',
    'women_bmi': 'women_bmi',
}

CDC_PLACES_MEASURES: dict[str, str] = {
    'phys_health': 'PHLTH',
    'diabetes': 'DIABETES',
    'copd': 'COPD',
    'cancer': 'CANCER',
    'chd': 'CHD',
    'mental_health': 'MHLTH',
    'checkup': 'CHECKUP',
    'sleep_lt7': 'SLEEP',
    'asthma': 'CASTHMA',
    'obesity': 'OBESITY',
    'smoking': 'CSMOKING',
    'high_chol': 'HIGHCHOL',
}

FAMILY_BENCHMARKS: dict[str, tuple[str, ...]] = {
    'pdfm': ('pdfm-conus27',),
    'air_temp': ('satclip-air-temp',),
    'california_housing': ('california-housing',),
    'satclip': (
        'satclip-country',
        'satclip-ecoregion',
        'satclip-biome',
        'satclip-population',
        'satclip-elevation',
    ),
    'sustainbench': tuple(f'sustainbench-{k}' for k in SUSTAINBENCH_TASKS),
    'better_together': (
        'bt-cropharvest',
        'bt-biomass',
        'bt-landcover',
        'bt-bioclim',
        'bt-population',
        'bt-distroad',
    ),
    'cdc_places': tuple(f'places-{k}' for k in CDC_PLACES_MEASURES),
    'usavars': tuple(f'mosaiks-{label}' for label in USAVARS_LABELS),
    'country': ('country',),
    'ecoregions': ('ecoregions',),
    'worldclim': ('worldclim-bio1', 'worldclim-bio12'),
    'soilgrids': ('soilgrids-soc', 'soilgrids-phh2o'),
    'deepmind': tuple(f'dm-{stem}' for stem in DEEPMIND_EVAL_CONFIGS),
}

OFFICIAL_FAMILIES: tuple[str, ...] = (
    'pdfm',
    'air_temp',
    'california_housing',
    'satclip',
    'sustainbench',
    'cdc_places',
    'usavars',
    'worldclim',
    'soilgrids',
    'deepmind',
)

ALL_FAMILIES: tuple[str, ...] = tuple(FAMILY_BENCHMARKS)

URBAN_RURAL_BENCHMARKS: tuple[str, ...] = (
    'worldclim-bio1',
    'worldclim-bio12',
    'satclip-country',
    'satclip-ecoregion',
    'satclip-biome',
    'satclip-population',
    'satclip-elevation',
    'satclip-air-temp',
)


def list_families(*, official_only: bool = False) -> list[str]:
    """Return registered CoordBench family names.

    Args:
        official_only: If True, return only the 10 official paper families.

    Returns:
        List of family names in registration order.
    """
    return list(OFFICIAL_FAMILIES if official_only else ALL_FAMILIES)


def list_benchmarks(*, official_only: bool = False) -> list[str]:
    """Return individual CoordBench benchmark names.

    Args:
        official_only: If True, return only benchmarks from the 10 official families.

    Returns:
        List of benchmark names in family registration order.
    """
    return [
        name
        for family in list_families(official_only=official_only)
        for name in FAMILY_BENCHMARKS[family]
    ]


__all__ = (
    'ALL_FAMILIES',
    'CDC_PLACES_MEASURES',
    'COORDBENCH_REPO',
    'DEEPMIND_EVAL_CONFIGS',
    'FAMILY_BENCHMARKS',
    'OFFICIAL_FAMILIES',
    'SUSTAINBENCH_TASKS',
    'URBAN_RURAL_BENCHMARKS',
    'USAVARS_LABELS',
    'USAVARS_LOG_LABELS',
    'list_benchmarks',
    'list_families',
)
