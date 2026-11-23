# Copyright (c) Taylor Geospatial. All rights reserved.
# Licensed under the MIT License.

"""Command-line interface for EarthINR."""

import argparse
import sys
from collections.abc import Sequence
from typing import Any

from . import __version__
from ._errors import ConfigError
from .coordbench.config import (
    CoordConfig,
    dump_coord_config_yaml,
    load_coord_config,
    parse_coord_config,
)
from .models.registry import get_model_preset, list_models

_SPATIAL_BASELINES: frozenset[str] = frozenset({'idw', 'kriging'})


def _build_parser() -> argparse.ArgumentParser:
    """Build the top-level ``earthinr`` argument parser."""
    parser = argparse.ArgumentParser(
        prog='earthinr',
        description='EarthINR: Implicit Neural Representations for Earth Observation.',
    )
    parser.add_argument(
        '--version', action='version', version=f'%(prog)s {__version__}'
    )
    subparsers = parser.add_subparsers(dest='command')

    models_parser = subparsers.add_parser(
        'models', help='List registered model presets or print a preset YAML.'
    )
    models_parser.add_argument(
        'name', nargs='?', default=None, help='Optional preset name to inspect.'
    )

    coord_parser = subparsers.add_parser(
        'coord', help='Run CoordBench coordinate evaluation.'
    )
    coord_parser.add_argument(
        '--config', default=None, help='Path to a YAML CoordBench configuration file.'
    )
    coord_parser.add_argument(
        '--model', default=None, help='Model preset name (replaces YAML model section).'
    )
    coord_parser.add_argument(
        '--dataset',
        '--datasets',
        '--name',
        '--names',
        dest='datasets',
        nargs='+',
        default=None,
        help='Benchmark or family names to evaluate.',
    )
    coord_parser.add_argument(
        '--methods',
        nargs='+',
        default=None,
        help='Probe methods to run (e.g. linear chunked knn logreg mlp idw kriging).',
    )
    coord_parser.add_argument(
        '--split',
        default=None,
        help=(
            'Split protocol (random, spatial, both, urban_rural_stratified, '
            'urban_rural_holdout).'
        ),
    )
    coord_parser.add_argument(
        '--folds', type=int, default=None, help='Number of cross-validation folds.'
    )
    coord_parser.add_argument(
        '--cell-deg',
        nargs='+',
        type=float,
        default=None,
        help='One or more spatial block grid sizes in degrees.',
    )
    coord_parser.add_argument(
        '--knn-k', type=int, default=None, help='Number of neighbors for k-NN probe.'
    )
    coord_parser.add_argument(
        '--knn-device', default=None, help='Device for k-NN probe (cpu or cuda[:idx]).'
    )
    coord_parser.add_argument(
        '--device',
        default=None,
        help='Execution device (auto, cpu, mps, cuda, cuda:<idx>).',
    )
    coord_parser.add_argument(
        '--seed', type=int, default=None, help='Deterministic integer random seed.'
    )
    coord_parser.add_argument(
        '--output', default=None, help='Output CSV path for evaluation rows.'
    )
    coord_parser.add_argument(
        '--resume',
        action=argparse.BooleanOptionalAction,
        default=None,
        help='Skip rows already recorded in the output CSV.',
    )
    coord_parser.add_argument(
        '--respect-official-splits',
        action=argparse.BooleanOptionalAction,
        default=None,
        help='Use official benchmark test masks when present.',
    )
    coord_parser.add_argument(
        '--cities-path',
        default=None,
        help='Local path to GeoNames cities15000.txt for urban/rural splits.',
    )
    coord_parser.add_argument(
        '--root',
        default=None,
        help='Optional local directory containing CoordBench parquet files.',
    )
    coord_parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Validate configuration and print effective YAML without executing.',
    )

    lb_parser = subparsers.add_parser(
        'leaderboard', help='Summarize a CoordBench results CSV into rankings.'
    )
    lb_parser.add_argument('csv', help='Path to a CoordBench results CSV file.')
    lb_parser.add_argument(
        '--method',
        default='linear',
        help="Probe method filter (default: 'linear'; supports prefixes like 'knn').",
    )
    lb_parser.add_argument(
        '--split', default=None, help='Optional split filter (e.g. random or spatial).'
    )

    return parser


def _merge_coord_args(args: argparse.Namespace) -> CoordConfig:
    """Merge optional YAML config with CLI overrides and validate strictly."""
    if args.config is not None:
        try:
            base = load_coord_config(args.config).to_dict()
        except FileNotFoundError as err:
            raise ConfigError(f'Config file not found: {args.config!r}.') from err
    else:
        base: dict[str, Any] = {}

    if args.model is not None:
        base['model'] = {'name': args.model}
    if args.datasets is not None:
        base['datasets'] = list(args.datasets)

    eval_cfg: dict[str, Any] = dict(base.get('evaluation', {}))
    if args.methods is not None:
        eval_cfg['methods'] = list(args.methods)
    if args.split is not None:
        eval_cfg['split'] = args.split
    if args.folds is not None:
        eval_cfg['folds'] = args.folds
    if args.cell_deg is not None:
        eval_cfg['cell_deg'] = (
            args.cell_deg[0] if len(args.cell_deg) == 1 else list(args.cell_deg)
        )
    if args.knn_k is not None:
        eval_cfg['knn_k'] = args.knn_k
    if args.knn_device is not None:
        eval_cfg['knn_device'] = args.knn_device
    if args.respect_official_splits is not None:
        eval_cfg['respect_official_splits'] = args.respect_official_splits
    if args.cities_path is not None:
        eval_cfg['cities_path'] = args.cities_path
    if eval_cfg:
        base['evaluation'] = eval_cfg

    runtime_cfg: dict[str, Any] = dict(base.get('runtime', {}))
    if args.device is not None:
        runtime_cfg['device'] = args.device
    if args.seed is not None:
        runtime_cfg['seed'] = args.seed
    if runtime_cfg:
        base['runtime'] = runtime_cfg

    output_cfg: dict[str, Any] = dict(base.get('output', {}))
    if args.output is not None:
        output_cfg['file'] = args.output
    if args.resume is not None:
        output_cfg['resume'] = args.resume
    if output_cfg:
        base['output'] = output_cfg

    cfg = parse_coord_config(base)
    if cfg.model.target is None and cfg.model.name not in _SPATIAL_BASELINES:
        get_model_preset(cfg.model.name)
    return cfg


def _cmd_models(name: str | None) -> int:
    """Handle ``earthinr models [name]``."""
    if name is not None:
        preset = get_model_preset(name)
        sys.stdout.write(preset.to_yaml())
        return 0
    for preset_name in list_models():
        preset = get_model_preset(preset_name)
        sys.stdout.write(f'{preset_name:<24} {preset.description}\n')
    return 0


def _cmd_coord(args: argparse.Namespace) -> int:
    """Handle ``earthinr coord ...``."""
    cfg = _merge_coord_args(args)
    if args.dry_run:
        sys.stdout.write(dump_coord_config_yaml(cfg))
        return 0

    from .coordbench.run import run_coordbench

    results = run_coordbench(cfg, root=args.root)
    sys.stdout.write(f'Wrote {len(results)} result row(s) to {cfg.output.file}\n')
    return 0


def _cmd_leaderboard(args: argparse.Namespace) -> int:
    """Handle ``earthinr leaderboard <csv> ...``."""
    from .coordbench.leaderboard import summarize_results

    try:
        ranks, metrics = summarize_results(
            args.csv, method=args.method, split=args.split
        )
    except (FileNotFoundError, ValueError) as err:
        raise ConfigError(str(err)) from err

    sys.stdout.write('Mean Rank (lower is better)\n')
    sys.stdout.write(ranks.to_string(float_format=lambda v: f'{v:.2f}'))
    sys.stdout.write('\n\nMean Score (higher is better)\n')
    sys.stdout.write(metrics.to_string(float_format=lambda v: f'{v:.4f}'))
    sys.stdout.write('\n')
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Run the EarthINR command-line interface.

    Args:
        argv: Optional sequence of command-line arguments (defaults to ``sys.argv[1:]``).

    Returns:
        Process exit code (``0`` on success, ``2`` on configuration or argument error).
    """
    parser = _build_parser()
    args, extras = parser.parse_known_args(argv)
    if extras:
        for tok in extras:
            if not tok.startswith('--') and '=' in tok:
                sys.stderr.write(
                    f'error: Legacy key=value override {tok!r} is not supported; '
                    'use explicit CLI flags.\n'
                )
                return 2
        parser.error(f'unrecognized arguments: {" ".join(extras)}')

    if args.command is None:
        parser.print_help()
        return 0

    try:
        if args.command == 'models':
            return _cmd_models(args.name)
        if args.command == 'coord':
            return _cmd_coord(args)
        return _cmd_leaderboard(args)
    except ConfigError as err:
        sys.stderr.write(f'error: {err}\n')
        return 2


__all__ = ('main',)
