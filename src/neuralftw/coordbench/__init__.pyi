# Copyright (c) Taylor Geospatial. All rights reserved.
# Licensed under the MIT License.

"""Type stubs for lazy CoordBench exports."""

from .catalog import (
    ALL_FAMILIES,
    CDC_PLACES_MEASURES,
    COORDBENCH_REPO,
    DEEPMIND_EVAL_CONFIGS,
    FAMILY_BENCHMARKS,
    OFFICIAL_FAMILIES,
    SUSTAINBENCH_TASKS,
    URBAN_RURAL_BENCHMARKS,
    USAVARS_LABELS,
    USAVARS_LOG_LABELS,
    list_benchmarks,
    list_families,
)
from .config import (
    VALID_METHODS,
    VALID_SPLITS,
    CoordConfig,
    CoordEvaluationConfig,
    CoordModelConfig,
    CoordOutputConfig,
    CoordRuntimeConfig,
    dump_coord_config_yaml,
    load_coord_config,
    parse_coord_config,
)
from .datasets import (
    FAMILY_LOADERS,
    PDFM_NON_TASK,
    USAVARS_NODATA,
    CoordBenchmark,
    load_benchmarks,
    load_config,
)
from .leaderboard import bench_scores, leaderboard, summarize_results
from .run import (
    CSV_COLUMNS,
    RESUME_KEY_COLS,
    CoordResult,
    append_rows_atomic,
    run_coordbench,
)
from .taxonomy import ENV, FAMILIES, LAND, R2_FLOOR, SOCIO, family_of

__all__: tuple[str, ...] = (
    'ALL_FAMILIES',
    'CDC_PLACES_MEASURES',
    'COORDBENCH_REPO',
    'CSV_COLUMNS',
    'DEEPMIND_EVAL_CONFIGS',
    'ENV',
    'FAMILIES',
    'FAMILY_BENCHMARKS',
    'FAMILY_LOADERS',
    'LAND',
    'OFFICIAL_FAMILIES',
    'PDFM_NON_TASK',
    'R2_FLOOR',
    'RESUME_KEY_COLS',
    'SOCIO',
    'SUSTAINBENCH_TASKS',
    'URBAN_RURAL_BENCHMARKS',
    'USAVARS_LABELS',
    'USAVARS_LOG_LABELS',
    'USAVARS_NODATA',
    'VALID_METHODS',
    'VALID_SPLITS',
    'CoordBenchmark',
    'CoordConfig',
    'CoordEvaluationConfig',
    'CoordModelConfig',
    'CoordOutputConfig',
    'CoordResult',
    'CoordRuntimeConfig',
    'append_rows_atomic',
    'bench_scores',
    'dump_coord_config_yaml',
    'family_of',
    'leaderboard',
    'list_benchmarks',
    'list_families',
    'load_benchmarks',
    'load_config',
    'load_coord_config',
    'parse_coord_config',
    'run_coordbench',
    'summarize_results',
)
