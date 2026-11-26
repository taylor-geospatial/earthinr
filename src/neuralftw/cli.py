# Copyright (c) Taylor Geospatial. All rights reserved.
# Licensed under the MIT License.

"""Command-line interface for NeuralFTW."""

import argparse
import sys
from collections.abc import Sequence
from typing import Any

import yaml

from . import __version__
from ._errors import ConfigError
from .coordbench.catalog import FAMILY_BENCHMARKS, list_families
from .coordbench.config import (
    CoordConfig,
    dump_coord_config_yaml,
    load_coord_config,
    parse_coord_config,
)
from .models.registry import get_model_preset, list_models

_SPATIAL_BASELINES: frozenset[str] = frozenset({'idw', 'kriging'})

_MAIN_EPILOG = (
    'Examples:\n'
    '  neuralftw models\n'
    '  neuralftw datasets --official\n'
    '  neuralftw coord --model sinr --dataset air_temp --split both\n'
    '  neuralftw coord --model my-enc --target my_pkg.MyEncoder '
    '--dataset air_temp --split both\n'
    '  neuralftw leaderboard results/coordbench_results.csv --split spatial\n'
)

_MODELS_EPILOG = (
    'Examples:\n'
    '  neuralftw models          # list all presets\n'
    '  neuralftw models mind     # inspect preset YAML\n'
)

_DATASETS_EPILOG = (
    'Examples:\n'
    '  neuralftw datasets             # list all 13 families (60 benchmarks)\n'
    '  neuralftw datasets --official  # list the 10 official paper families\n'
)

_COORD_EPILOG = (
    'Examples:\n'
    '  # Run a built-in preset on random + spatial splits\n'
    '  neuralftw coord --model satclip --datasets worldclim air_temp '
    '--split both --cell-deg 10 20\n'
    '\n'
    '  # Evaluate a custom LocationEncoder subclass without writing a YAML file\n'
    '  neuralftw coord --model my-enc --target my_pkg.MyEncoder '
    '--model-kwarg hidden_dim=256 --dataset air_temp --split both\n'
)

_LEADERBOARD_EPILOG = (
    'Examples:\n'
    '  neuralftw leaderboard results/coordbench_results.csv --method linear '
    '--split spatial\n'
)


def _build_parser() -> argparse.ArgumentParser:
    """Build the top-level ``neuralftw`` argument parser."""
    parser = argparse.ArgumentParser(
        prog='neuralftw',
        description='NeuralFTW: Implicit Neural Representations for Earth Observation.',
        epilog=_MAIN_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        '--version', action='version', version=f'%(prog)s {__version__}'
    )
    subparsers = parser.add_subparsers(dest='command')

    models_parser = subparsers.add_parser(
        'models',
        help='List registered model presets or print a preset YAML.',
        epilog=_MODELS_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    models_parser.add_argument(
        'name', nargs='?', default=None, help='Optional preset name to inspect.'
    )

    datasets_parser = subparsers.add_parser(
        'datasets',
        help='List CoordBench dataset families and benchmark names.',
        epilog=_DATASETS_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    datasets_parser.add_argument(
        '--official',
        action='store_true',
        help='Show only the 10 official paper benchmark families.',
    )

    coord_parser = subparsers.add_parser(
        'coord',
        help='Run CoordBench coordinate evaluation.',
        epilog=_COORD_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    coord_parser.add_argument(
        '--config', default=None, help='Path to a YAML CoordBench configuration file.'
    )
    coord_parser.add_argument(
        '--model',
        default=None,
        help='Model preset name or custom display name when paired with --target.',
    )
    coord_parser.add_argument(
        '--target',
        default=None,
        help=(
            'Dotted Python path to a custom LocationEncoder class or factory '
            '(e.g. my_pkg.MyEncoder).'
        ),
    )
    coord_parser.add_argument(
        '--model-kwarg',
        dest='model_kwargs',
        action='append',
        default=None,
        metavar='KEY=VALUE',
        help=(
            'Model constructor keyword override parsed as YAML '
            '(repeatable, e.g. --model-kwarg dim=256).'
        ),
    )
    coord_parser.add_argument(
        '--dataset',
        '--datasets',
        '--name',
        '--names',
        dest='datasets',
        nargs='+',
        default=None,
        help='Benchmark or family names to evaluate (see `neuralftw datasets`).',
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
        help=(
            'Optional custom GeoNames cities .parquet or .txt path '
            '(defaults to bundled cities15000.parquet).'
        ),
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
        'leaderboard',
        help='Summarize a CoordBench results CSV into rankings.',
        epilog=_LEADERBOARD_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
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


def _parse_model_kwargs(items: Sequence[str]) -> dict[str, Any]:
    """Parse repeatable ``--model-kwarg KEY=VALUE`` items using safe YAML scalars."""
    parsed: dict[str, Any] = {}
    for item in items:
        if '=' not in item:
            raise ConfigError(
                f'Invalid --model-kwarg {item!r}; expected KEY=VALUE format.'
            )
        key, raw_val = item.split('=', 1)
        key = key.strip()
        if not key or not key.isidentifier():
            raise ConfigError(
                f'Invalid --model-kwarg key {key!r}; expected a Python identifier.'
            )
        try:
            parsed[key] = yaml.safe_load(raw_val)
        except yaml.YAMLError as err:
            raise ConfigError(
                f'Invalid YAML value in --model-kwarg {item!r}: {err}'
            ) from err
    return parsed


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
    if args.target is not None:
        model_cfg = dict(base.get('model', {}))
        model_cfg['target'] = args.target
        if 'name' not in model_cfg:
            model_cfg['name'] = args.target.rsplit('.', 1)[-1]
        base['model'] = model_cfg
    if args.model_kwargs is not None:
        extra_kwargs = _parse_model_kwargs(args.model_kwargs)
        model_cfg = dict(base.get('model', {}))
        merged_kwargs = dict(model_cfg.get('kwargs') or {})
        merged_kwargs.update(extra_kwargs)
        model_cfg['kwargs'] = merged_kwargs
        base['model'] = model_cfg
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
    """Handle ``neuralftw models [name]``."""
    if name is not None:
        preset = get_model_preset(name)
        sys.stdout.write(preset.to_yaml())
        return 0
    names = list_models()
    for preset_name in names:
        preset = get_model_preset(preset_name)
        sys.stdout.write(f'{preset_name:<24} {preset.description}\n')
    sys.stdout.write(
        f'\n{len(names)} registered presets. '
        "Inspect one with 'neuralftw models <name>' or pass "
        "'--target pkg.Class' to 'neuralftw coord' for a custom model.\n"
    )
    return 0


def _cmd_datasets(*, official_only: bool) -> int:
    """Handle ``neuralftw datasets [--official]``."""
    families = list_families(official_only=official_only)
    total = 0
    for family in families:
        benches = FAMILY_BENCHMARKS[family]
        total += len(benches)
        sys.stdout.write(f'{family:<20} ({len(benches):>2}) {", ".join(benches)}\n')
    scope = 'official' if official_only else 'all'
    sys.stdout.write(f'\n{len(families)} {scope} families ({total} benchmarks).\n')
    return 0


def _cmd_coord(args: argparse.Namespace) -> int:
    """Handle ``neuralftw coord ...``."""
    cfg = _merge_coord_args(args)
    if args.dry_run:
        sys.stdout.write(dump_coord_config_yaml(cfg))
        return 0

    from .coordbench.run import run_coordbench

    results = run_coordbench(cfg, root=args.root)
    sys.stdout.write(f'Wrote {len(results)} result row(s) to {cfg.output.file}\n')
    return 0


def _cmd_leaderboard(args: argparse.Namespace) -> int:
    """Handle ``neuralftw leaderboard <csv> ...``."""
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
    """Run the NeuralFTW command-line interface.

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
        if args.command == 'datasets':
            return _cmd_datasets(official_only=args.official)
        if args.command == 'coord':
            return _cmd_coord(args)
        return _cmd_leaderboard(args)
    except ConfigError as err:
        sys.stderr.write(f'error: {err}\n')
        return 2


__all__ = ('main',)
