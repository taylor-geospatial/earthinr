# Copyright (c) Taylor Geospatial. All rights reserved.
# Licensed under the MIT License.

"""CoordBench dataset container and parquet benchmark loaders."""

import os
import warnings
from collections.abc import Callable, Sequence

import numpy as np
import pandas as pd

from .._errors import DatasetNotFoundError
from .._types import Path
from .catalog import (
    ALL_FAMILIES,
    CDC_PLACES_MEASURES,
    COORDBENCH_REPO,
    DEEPMIND_EVAL_CONFIGS,
    FAMILY_BENCHMARKS,
    OFFICIAL_FAMILIES,
    SUSTAINBENCH_TASKS,
    USAVARS_LABELS,
    USAVARS_LOG_LABELS,
)

_CANONICAL_EXTRA: frozenset[str] = frozenset(
    {'timestamp', 'timestamp_end', 'split', 'id'}
)

USAVARS_NODATA: float = -999.0

PDFM_NON_TASK: frozenset[str] = frozenset(
    {
        'place',
        'county',
        'state',
        'latitude',
        'longitude',
        'Count_Person',
        'imputation_split',
        'superresolution_split',
        'extrapolation_split',
    }
)

_PDFM_SPLITS: frozenset[str] = frozenset(
    {'imputation', 'superresolution', 'extrapolation'}
)

_VALID_TASK_TYPES: frozenset[str] = frozenset({'regression', 'classification'})


class CoordBenchmark:
    """Single coordinate-to-label benchmark table."""

    def __init__(
        self,
        name: str,
        lat: np.ndarray,
        lon: np.ndarray,
        tasks: dict[str, np.ndarray] | None = None,
        task_type: str = 'regression',
        year: np.ndarray | None = None,
        test_mask: np.ndarray | None = None,
    ) -> None:
        """Initialize a new CoordBenchmark instance.

        Args:
            name: Unique benchmark identifier (e.g. ``'sustainbench-asset'``).
            lat: Latitudes in degrees, shape ``(N,)``.
            lon: Longitudes in degrees, shape ``(N,)``.
            tasks: Mapping of task name to label array, each shape ``(N,)``.
            task_type: Either ``'regression'`` or ``'classification'``.
            year: Optional per-point calendar year array, shape ``(N,)``.
            test_mask: Optional boolean held-out test mask for official splits.

        Raises:
            ValueError: If *task_type* is not ``'regression'`` or ``'classification'``.
        """
        if task_type not in _VALID_TASK_TYPES:
            raise ValueError(
                f"Invalid task_type {task_type!r}; expected 'regression' or 'classification'."
            )
        self.name = name
        self.lat = lat
        self.lon = lon
        self.tasks: dict[str, np.ndarray] = {} if tasks is None else dict(tasks)
        self.task_type = task_type
        self.year = year
        self.test_mask = test_mask

    def __repr__(self) -> str:
        """Return a concise summary string for debugging."""
        return (
            f'CoordBenchmark(name={self.name!r}, n_samples={len(self.lat)}, '
            f'tasks={len(self.tasks)}, task_type={self.task_type!r})'
        )


def load_config(
    config: str, *, root: Path | None = None, repo_id: str | None = None
) -> pd.DataFrame:
    """Read one CoordBench config parquet table from disk or Hugging Face Hub.

    Checks *root* first, then the ``COORDBENCH_ROOT`` environment variable, looking for
    ``<root>/<config>/data.parquet`` or ``<root>/data/<config>/data.parquet``. Falls
    back to downloading ``data/<config>/data.parquet`` from Hugging Face Hub.

    Args:
        config: Parquet configuration stem (e.g. ``'california_housing'``).
        root: Optional local directory containing CoordBench parquet files.
        repo_id: Optional Hugging Face dataset repo id overriding ``COORDBENCH_REPO``.

    Returns:
        Loaded :class:`pandas.DataFrame`.

    Raises:
        DatasetNotFoundError: If a local root is set and the parquet file is missing.
    """
    local_root = (
        os.fspath(root) if root is not None else os.environ.get('COORDBENCH_ROOT')
    )
    if local_root is not None:
        base = os.path.expanduser(local_root)
        direct = os.path.join(base, config, 'data.parquet')
        nested = os.path.join(base, 'data', config, 'data.parquet')
        if os.path.isfile(direct):
            return pd.read_parquet(direct)
        if os.path.isfile(nested):
            return pd.read_parquet(nested)
        raise DatasetNotFoundError(
            f'CoordBench table {config!r} not found under {base!r} '
            f'(checked {direct!r} and {nested!r}).'
        )

    from huggingface_hub import hf_hub_download

    repo = repo_id or os.environ.get('COORDBENCH_REPO', COORDBENCH_REPO)
    path = hf_hub_download(repo, f'data/{config}/data.parquet', repo_type='dataset')
    return pd.read_parquet(path)


def _lonlat_cols(df: pd.DataFrame) -> tuple[str, str]:
    """Return the longitude and latitude column names from *df*."""
    lon_col = next(c for c in ('lon', 'longitude', 'Lon', 'x') if c in df.columns)
    lat_col = next(c for c in ('lat', 'latitude', 'Lat', 'y') if c in df.columns)
    return lon_col, lat_col


def _first_col(df: pd.DataFrame, *names: str) -> str | None:
    """Return the first column name in *names* present in *df*, or None."""
    return next((name for name in names if name in df.columns), None)


def load_pdfm(
    *, root: Path | None = None, repo_id: str | None = None, split: str | None = None
) -> list[CoordBenchmark]:
    """Load PDFM CONUS-27 US ZIP-level regression tasks.

    Args:
        root: Optional local CoordBench directory.
        repo_id: Optional Hugging Face dataset repo id.
        split: Optional official PDFM split (``'imputation'``, ``'superresolution'``,
            or ``'extrapolation'``).

    Returns:
        Single-element list containing the PDFM benchmark.

    Raises:
        ValueError: If *split* is invalid or the split column is missing.
    """
    df = load_config('pdfm_conus27', root=root, repo_id=repo_id)
    lon_col, lat_col = _lonlat_cols(df)
    non_task = PDFM_NON_TASK | {lon_col, lat_col} | _CANONICAL_EXTRA
    task_cols = [c for c in df.columns if c not in non_task]
    test_mask = None
    name = 'pdfm-conus27'
    if split is not None:
        if split not in _PDFM_SPLITS:
            raise ValueError(
                f'Invalid PDFM split {split!r}; expected one of {sorted(_PDFM_SPLITS)}.'
            )
        split_col = f'{split}_split'
        if split_col not in df.columns:
            raise ValueError(f'PDFM split column {split_col!r} not found in table.')
        test_mask = (df[split_col].astype(str) == 'test').to_numpy()
        name = f'pdfm-conus27-{split}'
    return [
        CoordBenchmark(
            name=name,
            lat=df[lat_col].to_numpy(np.float64),
            lon=df[lon_col].to_numpy(np.float64),
            tasks={c: df[c].to_numpy(np.float64) for c in task_cols},
            test_mask=test_mask,
        )
    ]


def load_air_temp(
    *, root: Path | None = None, repo_id: str | None = None
) -> list[CoordBenchmark]:
    """Load SatCLIP global air temperature station regression benchmark.

    Args:
        root: Optional local CoordBench directory.
        repo_id: Optional Hugging Face dataset repo id.

    Returns:
        Single-element list containing ``'satclip-air-temp'``.
    """
    df = load_config('air_temp', root=root, repo_id=repo_id)
    lon_col, lat_col = _lonlat_cols(df)
    cols = {c.lower(): c for c in df.columns}
    temp_keys = [k for k in cols if 'temp' in k or k == 'meant'] or [
        k for k in cols if k.startswith('mean')
    ]
    temp_col = cols[temp_keys[0]]
    return [
        CoordBenchmark(
            name='satclip-air-temp',
            lat=df[lat_col].to_numpy(np.float64),
            lon=df[lon_col].to_numpy(np.float64),
            tasks={'air_temperature': df[temp_col].to_numpy(np.float64)},
        )
    ]


def load_california_housing(
    *, root: Path | None = None, repo_id: str | None = None
) -> list[CoordBenchmark]:
    """Load California housing median house value regression benchmark.

    Args:
        root: Optional local CoordBench directory.
        repo_id: Optional Hugging Face dataset repo id.

    Returns:
        Single-element list containing ``'california-housing'``.
    """
    df = load_config('california_housing', root=root, repo_id=repo_id)
    lon_col, lat_col = _lonlat_cols(df)
    return [
        CoordBenchmark(
            name='california-housing',
            lat=df[lat_col].to_numpy(np.float64),
            lon=df[lon_col].to_numpy(np.float64),
            tasks={'median_house_value': df['MedHouseVal'].to_numpy(np.float64)},
        )
    ]


def load_satclip(
    *, root: Path | None = None, repo_id: str | None = None
) -> list[CoordBenchmark]:
    """Load SatCLIP's five downstream classification and regression benchmarks.

    Drops ocean nodata sentinels (country index ``0`` and ecoregion/biome index ``-1``).

    Args:
        root: Optional local CoordBench directory.
        repo_id: Optional Hugging Face dataset repo id.

    Returns:
        List of five SatCLIP benchmarks.
    """
    out: list[CoordBenchmark] = []

    country = load_config('satclip_country', root=root, repo_id=repo_id)
    lon_c, lat_c = _lonlat_cols(country)
    country = country[country['country'] != 0]
    out.append(
        CoordBenchmark(
            name='satclip-country',
            lat=country[lat_c].to_numpy(np.float64),
            lon=country[lon_c].to_numpy(np.float64),
            tasks={'country': country['country'].to_numpy()},
            task_type='classification',
        )
    )

    eco = load_config('satclip_ecoregion', root=root, repo_id=repo_id)
    lon_e, lat_e = _lonlat_cols(eco)
    eco = eco[eco['ecoregion_class_index'] != -1]
    for col, name in (
        ('ecoregion_class_index', 'satclip-ecoregion'),
        ('biome_class_index', 'satclip-biome'),
    ):
        out.append(
            CoordBenchmark(
                name=name,
                lat=eco[lat_e].to_numpy(np.float64),
                lon=eco[lon_e].to_numpy(np.float64),
                tasks={col: eco[col].to_numpy()},
                task_type='classification',
            )
        )

    pop = load_config('satclip_population', root=root, repo_id=repo_id)
    lon_p, lat_p = _lonlat_cols(pop)
    out.append(
        CoordBenchmark(
            name='satclip-population',
            lat=pop[lat_p].to_numpy(np.float64),
            lon=pop[lon_p].to_numpy(np.float64),
            tasks={'log_population': pop['log_population'].to_numpy(np.float64)},
        )
    )

    elev = load_config('satclip_elevation', root=root, repo_id=repo_id)
    lon_el, lat_el = _lonlat_cols(elev)
    out.append(
        CoordBenchmark(
            name='satclip-elevation',
            lat=elev[lat_el].to_numpy(np.float64),
            lon=elev[lon_el].to_numpy(np.float64),
            tasks={'elevation': elev['elevation'].to_numpy(np.float64)},
        )
    )
    return out


def load_sustainbench(
    *, root: Path | None = None, repo_id: str | None = None
) -> list[CoordBenchmark]:
    """Load six SustainBench DHS socioeconomic regressions with official test masks.

    Args:
        root: Optional local CoordBench directory.
        repo_id: Optional Hugging Face dataset repo id.

    Returns:
        List of six SustainBench benchmarks.
    """
    df = load_config('sustainbench', root=root, repo_id=repo_id)
    is_test = (df['split'] == 'test').to_numpy()
    out: list[CoordBenchmark] = []
    for key, col in SUSTAINBENCH_TASKS.items():
        mask = df[col].notna() & df['lat'].notna() & df['lon'].notna()
        sub = df[mask]
        out.append(
            CoordBenchmark(
                name=f'sustainbench-{key}',
                lat=sub['lat'].to_numpy(np.float64),
                lon=sub['lon'].to_numpy(np.float64),
                tasks={col: sub[col].to_numpy(np.float64)},
                test_mask=is_test[mask.to_numpy()],
            )
        )
    return out


def load_better_together(
    *, root: Path | None = None, repo_id: str | None = None
) -> list[CoordBenchmark]:
    """Load six Better Together coordinate benchmarks.

    Restricts ``bt-landcover``, ``bt-bioclim``, ``bt-population``, and ``bt-distroad``
    to the ``random_sample == 1`` shared point pool.

    Args:
        root: Optional local CoordBench directory.
        repo_id: Optional Hugging Face dataset repo id.

    Returns:
        List of six Better Together benchmarks.
    """
    out: list[CoordBenchmark] = []

    ch = load_config('bt_cropharvest', root=root, repo_id=repo_id)
    out.append(
        CoordBenchmark(
            name='bt-cropharvest',
            lat=ch['lat'].to_numpy(np.float64),
            lon=ch['lon'].to_numpy(np.float64),
            tasks={'crop_type': ch['label_name'].to_numpy()},
            task_type='classification',
        )
    )

    bm = load_config('bt_biomass', root=root, repo_id=repo_id)
    mask = bm['biomass_mean'].notna()
    out.append(
        CoordBenchmark(
            name='bt-biomass',
            lat=bm.loc[mask, 'lat'].to_numpy(np.float64),
            lon=bm.loc[mask, 'lon'].to_numpy(np.float64),
            tasks={'biomass': bm.loc[mask, 'biomass_mean'].to_numpy(np.float64)},
        )
    )

    dw = load_config('bt_landcover', root=root, repo_id=repo_id)
    dw = dw[dw['random_sample'] == 1]
    lc_cols = (
        'water',
        'trees',
        'grass',
        'flooded_vegetation',
        'crops',
        'shrub_and_scrub',
        'built',
        'bare',
        'snow_and_ice',
    )
    out.append(
        CoordBenchmark(
            name='bt-landcover',
            lat=dw['lat'].to_numpy(np.float64),
            lon=dw['lon'].to_numpy(np.float64),
            tasks={c: dw[c].to_numpy(np.float64) for c in lc_cols},
        )
    )

    pool_ids = set(dw['id'])
    bio = load_config('bt_bioclim', root=root, repo_id=repo_id)
    bio = bio[bio['id'].isin(pool_ids)]
    bio_cols = tuple(f'bioclim_{i:02d}' for i in range(1, 20))
    out.append(
        CoordBenchmark(
            name='bt-bioclim',
            lat=bio['lat'].to_numpy(np.float64),
            lon=bio['lon'].to_numpy(np.float64),
            tasks={c: bio[c].to_numpy(np.float64) for c in bio_cols},
        )
    )

    hf = load_config('bt_human_footprint', root=root, repo_id=repo_id)
    hf = hf[hf['id'].isin(pool_ids)]
    out.append(
        CoordBenchmark(
            name='bt-population',
            lat=hf['lat'].to_numpy(np.float64),
            lon=hf['lon'].to_numpy(np.float64),
            tasks={'pop_density': hf['pop_density'].to_numpy(np.float64)},
        )
    )
    out.append(
        CoordBenchmark(
            name='bt-distroad',
            lat=hf['lat'].to_numpy(np.float64),
            lon=hf['lon'].to_numpy(np.float64),
            tasks={'maxdist_road': hf['maxdist_road'].to_numpy(np.float64)},
        )
    )
    return out


def load_cdc_places(
    *, root: Path | None = None, repo_id: str | None = None
) -> list[CoordBenchmark]:
    """Load CDC PLACES ZCTA health and behavior prevalence regressions.

    Args:
        root: Optional local CoordBench directory.
        repo_id: Optional Hugging Face dataset repo id.

    Returns:
        List of up to twelve CDC PLACES benchmarks.
    """
    df = load_config('cdc_places', root=root, repo_id=repo_id)
    out: list[CoordBenchmark] = []
    for key, pref in CDC_PLACES_MEASURES.items():
        col = f'{pref.lower()}_crudeprev'
        if col not in df.columns:
            warnings.warn(
                f'Missing CDC PLACES column {col!r}; skipping {key!r}.',
                UserWarning,
                stacklevel=2,
            )
            continue
        mask = df[col].notna() & df['lat'].notna() & df['lon'].notna()
        sub = df[mask]
        out.append(
            CoordBenchmark(
                name=f'places-{key}',
                lat=sub['lat'].to_numpy(np.float64),
                lon=sub['lon'].to_numpy(np.float64),
                tasks={col: sub[col].to_numpy(np.float64)},
            )
        )
    return out


def load_usavars(
    *, root: Path | None = None, repo_id: str | None = None
) -> list[CoordBenchmark]:
    """Load seven MOSAIKS/USAVars regression benchmarks.

    Drops ``-999.0`` nodata sentinels and applies ``log1p`` to ``population``,
    ``income``, ``nightlights``, and ``housing``.

    Args:
        root: Optional local CoordBench directory.
        repo_id: Optional Hugging Face dataset repo id.

    Returns:
        List of USAVars benchmarks.
    """
    frames: dict[str, pd.DataFrame] = {}
    for label in USAVARS_LABELS:
        df = load_config(f'usavars_{label}', root=root, repo_id=repo_id)
        df.columns = [str(c).strip() for c in df.columns]
        frames[label] = df

    id_maps: list[pd.DataFrame] = []
    for df in frames.values():
        cl = {c.lower(): c for c in df.columns}
        if {'id', 'lat', 'lon'} <= set(cl):
            mapped = df[[cl['id'], cl['lat'], cl['lon']]].copy()
            mapped.columns = ['ID', 'lat', 'lon']
            id_maps.append(mapped)
    id2ll = (
        pd.concat(id_maps).dropna().drop_duplicates('ID').set_index('ID')
        if id_maps
        else None
    )

    out: list[CoordBenchmark] = []
    skip = {'id', 'lat', 'lon'} | _CANONICAL_EXTRA
    for label, df in frames.items():
        cl = {c.lower(): c for c in df.columns}
        cands = [
            c
            for c in df.columns
            if c.lower() not in skip
            and not c.startswith('Unnamed')
            and pd.api.types.is_numeric_dtype(df[c])
        ]
        if not cands:
            continue
        value_col = next((c for c in cands if label in c.lower()), None)
        value_col = (
            value_col
            or next((c for c in cands if c.lower() == 'price'), None)
            or cands[0]
        )
        if {'lat', 'lon'} <= set(cl):
            lat_s, lon_s, val_s = df[cl['lat']], df[cl['lon']], df[value_col]
        elif id2ll is not None and 'id' in cl:
            joined = (
                df[[cl['id'], value_col]]
                .rename(columns={cl['id']: 'ID'})
                .merge(id2ll, left_on='ID', right_index=True)
            )
            lat_s, lon_s, val_s = joined['lat'], joined['lon'], joined[value_col]
        else:
            continue

        lat_a = lat_s.to_numpy(np.float64)
        lon_a = lon_s.to_numpy(np.float64)
        val_a = val_s.to_numpy(np.float64)
        keep = val_a != USAVARS_NODATA
        lat_a, lon_a, val_a = lat_a[keep], lon_a[keep], val_a[keep]
        if label in USAVARS_LOG_LABELS:
            val_a = np.log1p(val_a)
        out.append(
            CoordBenchmark(
                name=f'mosaiks-{label}', lat=lat_a, lon=lon_a, tasks={label: val_a}
            )
        )
    return out


def load_country(
    *, root: Path | None = None, repo_id: str | None = None
) -> list[CoordBenchmark]:
    """Load global Natural Earth admin-0 country classification benchmark.

    Args:
        root: Optional local CoordBench directory.
        repo_id: Optional Hugging Face dataset repo id.

    Returns:
        Single-element list containing ``'country'``.
    """
    df = load_config('country', root=root, repo_id=repo_id)
    return [
        CoordBenchmark(
            name='country',
            lat=df['lat'].to_numpy(np.float64),
            lon=df['lon'].to_numpy(np.float64),
            tasks={'country': df['country'].to_numpy()},
            task_type='classification',
        )
    ]


def load_ecoregions(
    *, root: Path | None = None, repo_id: str | None = None
) -> list[CoordBenchmark]:
    """Load global RESOLVE Ecoregions2017 biome and ecoregion classification benchmark.

    Args:
        root: Optional local CoordBench directory.
        repo_id: Optional Hugging Face dataset repo id.

    Returns:
        Single-element list containing ``'ecoregions'``.
    """
    df = load_config('ecoregions', root=root, repo_id=repo_id)
    return [
        CoordBenchmark(
            name='ecoregions',
            lat=df['lat'].to_numpy(np.float64),
            lon=df['lon'].to_numpy(np.float64),
            tasks={
                'biome': df['biome'].to_numpy(),
                'ecoregion': df['ecoregion'].to_numpy(),
            },
            task_type='classification',
        )
    ]


def load_worldclim(
    *,
    root: Path | None = None,
    repo_id: str | None = None,
    max_points: int = 20000,
    seed: int = 0,
) -> list[CoordBenchmark]:
    """Load WorldClim v2.1 bio1 (temperature) and bio12 (precipitation) regressions.

    Deterministically subsamples to *max_points* with *seed* if the table exceeds
    *max_points*, matching MIND's evaluation protocol.

    Args:
        root: Optional local CoordBench directory.
        repo_id: Optional Hugging Face dataset repo id.
        max_points: Maximum number of rows to retain.
        seed: Random state used for deterministic subsampling.

    Returns:
        List of WorldClim benchmarks present in the table.
    """
    df = load_config('worldclim_bio', root=root, repo_id=repo_id)
    if len(df) > max_points:
        df = df.sample(n=max_points, random_state=seed)
    return [
        CoordBenchmark(
            name=f'worldclim-bio{v}',
            lat=df['lat'].to_numpy(np.float64),
            lon=df['lon'].to_numpy(np.float64),
            tasks={f'bio{v}': df[f'bio{v}'].to_numpy(np.float64)},
        )
        for v in (1, 12)
        if f'bio{v}' in df.columns
    ]


def load_soilgrids(
    *,
    root: Path | None = None,
    repo_id: str | None = None,
    max_points: int = 6000,
    seed: int = 0,
) -> list[CoordBenchmark]:
    """Load ISRIC SoilGrids v2 soil organic carbon and pH regressions.

    Deterministically subsamples to *max_points* with *seed* if the table exceeds
    *max_points*, matching MIND's evaluation protocol.

    Args:
        root: Optional local CoordBench directory.
        repo_id: Optional Hugging Face dataset repo id.
        max_points: Maximum number of rows to retain.
        seed: Random state used for deterministic subsampling.

    Returns:
        List of SoilGrids benchmarks present in the table.
    """
    df = load_config('soilgrids', root=root, repo_id=repo_id)
    if len(df) > max_points:
        df = df.sample(n=max_points, random_state=seed)
    return [
        CoordBenchmark(
            name=f'soilgrids-{prop}',
            lat=df['lat'].to_numpy(np.float64),
            lon=df['lon'].to_numpy(np.float64),
            tasks={prop: df[prop].to_numpy(np.float64)},
        )
        for prop in ('soc', 'phh2o')
        if prop in df.columns
    ]


def load_deepmind(
    *, root: Path | None = None, repo_id: str | None = None
) -> list[CoordBenchmark]:
    """Load DeepMind/AlphaEarth geospatial evaluation benchmarks.

    Infers classification vs. regression from label integrality and cardinality
    (<= 100 unique integer values) and extracts per-point year from Unix-millisecond
    timestamps when available.

    Args:
        root: Optional local CoordBench directory.
        repo_id: Optional Hugging Face dataset repo id.

    Returns:
        List of DeepMind geospatial evaluation benchmarks.
    """
    out: list[CoordBenchmark] = []
    for stem in DEEPMIND_EVAL_CONFIGS:
        df = load_config(f'dm_{stem}', root=root, repo_id=repo_id)
        if 'label' not in df.columns:
            continue
        lon_col, lat_col = _lonlat_cols(df)
        ts_col = _first_col(df, 'timestamp', 'valid_time_start_ms')
        label = df['label'].to_numpy()
        integral = np.all(np.isfinite(label)) and np.allclose(label, np.round(label))
        is_clf = bool(integral and np.unique(label).size <= 100)
        year = None
        if ts_col is not None:
            yr = pd.to_datetime(df[ts_col], unit='ms').dt.year
            if yr.notna().any():
                year = yr.to_numpy()
        out.append(
            CoordBenchmark(
                name=f'dm-{stem}',
                lat=df[lat_col].to_numpy(np.float64),
                lon=df[lon_col].to_numpy(np.float64),
                tasks={
                    stem: label.astype(np.int64) if is_clf else label.astype(np.float64)
                },
                task_type='classification' if is_clf else 'regression',
                year=year,
            )
        )
    return out


type _FamilyLoader = Callable[..., list[CoordBenchmark]]

FAMILY_LOADERS: dict[str, _FamilyLoader] = {
    'pdfm': load_pdfm,
    'air_temp': load_air_temp,
    'california_housing': load_california_housing,
    'satclip': load_satclip,
    'sustainbench': load_sustainbench,
    'better_together': load_better_together,
    'cdc_places': load_cdc_places,
    'usavars': load_usavars,
    'country': load_country,
    'ecoregions': load_ecoregions,
    'worldclim': load_worldclim,
    'soilgrids': load_soilgrids,
    'deepmind': load_deepmind,
}

_BENCHMARK_TO_FAMILY: dict[str, str] = {
    name: family for family, names in FAMILY_BENCHMARKS.items() for name in names
}


def load_benchmarks(
    names: str | Sequence[str] = 'all',
    *,
    root: Path | None = None,
    repo_id: str | None = None,
    official_only: bool = False,
) -> list[CoordBenchmark]:
    """Load a selection of CoordBench benchmarks by family or benchmark name.

    Only the families needed to satisfy *names* are read from disk or Hub.

    Args:
        names: ``'all'``, ``'official'``, a comma-separated string, or a sequence of
            family and/or individual benchmark names.
        root: Optional local directory containing CoordBench parquet tables.
        repo_id: Optional Hugging Face dataset repo id.
        official_only: If True, restrict selection and ``'all'`` expansion to the 10
            official paper families.

    Returns:
        Deduplicated list of :class:`CoordBenchmark` instances in family order.

    Raises:
        KeyError: If any entry in *names* is not a recognized family or benchmark name
            under the active *official_only* scope.
    """
    allowed_families = set(OFFICIAL_FAMILIES if official_only else ALL_FAMILIES)
    if isinstance(names, str):
        raw_selection = [n.strip() for n in names.split(',') if n.strip()]
    else:
        raw_selection = list(names)

    expanded: list[str] = []
    for entry in raw_selection:
        if entry == 'all':
            expanded.extend(OFFICIAL_FAMILIES if official_only else ALL_FAMILIES)
        elif entry == 'official':
            expanded.extend(OFFICIAL_FAMILIES)
        else:
            expanded.append(entry)

    families_to_load: list[str] = []
    name_filter: dict[str, set[str]] = {}
    for entry in expanded:
        if entry in allowed_families:
            family = entry
            name_filter[family] = set()
        elif (
            entry in _BENCHMARK_TO_FAMILY
            and _BENCHMARK_TO_FAMILY[entry] in allowed_families
        ):
            family = _BENCHMARK_TO_FAMILY[entry]
            if family not in name_filter or name_filter[family]:
                name_filter.setdefault(family, set()).add(entry)
        else:
            raise KeyError(
                f'Unknown coordbench selection {entry!r}; expected a family '
                f'({sorted(allowed_families)}) or benchmark name.'
            )
        if family not in families_to_load:
            families_to_load.append(family)

    out: list[CoordBenchmark] = []
    seen: set[str] = set()
    for family in families_to_load:
        wanted = name_filter.get(family, set())
        for bench in FAMILY_LOADERS[family](root=root, repo_id=repo_id):
            if bench.name in seen or (wanted and bench.name not in wanted):
                continue
            seen.add(bench.name)
            out.append(bench)
    return out


__all__ = (
    'FAMILY_LOADERS',
    'PDFM_NON_TASK',
    'USAVARS_NODATA',
    'CoordBenchmark',
    'load_air_temp',
    'load_benchmarks',
    'load_better_together',
    'load_california_housing',
    'load_cdc_places',
    'load_config',
    'load_country',
    'load_deepmind',
    'load_ecoregions',
    'load_pdfm',
    'load_satclip',
    'load_soilgrids',
    'load_sustainbench',
    'load_usavars',
    'load_worldclim',
)
