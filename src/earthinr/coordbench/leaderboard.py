# Copyright (c) Taylor Geospatial. All rights reserved.
# Licensed under the MIT License.

"""Aggregate CoordBench result CSVs into per-family rank and metric leaderboards."""

import os

import pandas as pd

from .._types import Path
from .taxonomy import FAMILIES, R2_FLOOR, family_of

_REQUIRED_SCORE_COLS: frozenset[str] = frozenset(
    {'model_name', 'dataset', 'metric_name', 'metric_value'}
)


def bench_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse per-task results to one score per ``(model_name, dataset)``.

    Floors regression :math:`R^2` metrics (``metric_name == 'r2'``) at
    :data:`~earthinr.coordbench.taxonomy.R2_FLOOR` before averaging across tasks
    within each dataset.

    Args:
        df: Result table with columns ``model_name``, ``dataset``, ``metric_name``,
            and ``metric_value``.

    Returns:
        Aggregated DataFrame with columns ``model_name``, ``dataset``, ``score``, and
        ``family``.

    Raises:
        ValueError: If required columns are missing from *df*.
    """
    missing = sorted(_REQUIRED_SCORE_COLS - set(df.columns))
    if missing:
        raise ValueError(f'Missing required columns: {missing}.')
    copied = df.copy()
    is_r2 = copied['metric_name'] == 'r2'
    copied.loc[is_r2, 'metric_value'] = copied.loc[is_r2, 'metric_value'].clip(
        lower=R2_FLOOR
    )
    scores = copied.groupby(['model_name', 'dataset'], as_index=False).agg(
        score=('metric_value', 'mean')
    )
    scores['family'] = scores['dataset'].map(family_of)
    return scores


def leaderboard(scores: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute per-family and macro mean-rank and mean-metric leaderboard tables.

    Ranks models within each dataset (1 = best; ties receive their average rank) and
    sorts rows by ascending macro mean rank.

    Args:
        scores: DataFrame returned by :func:`bench_scores`.

    Returns:
        Tuple ``(ranks, metrics)`` of DataFrames indexed by ``model_name`` with
        columns for each family in ``('socio', 'env', 'land')`` plus ``'macro'``.

    Raises:
        ValueError: If *scores* is empty.
    """
    if scores.empty:
        raise ValueError('No benchmark scores to aggregate.')
    ranked = scores.copy()
    ranked['rank'] = ranked.groupby('dataset')['score'].rank(
        ascending=False, method='average'
    )
    models = sorted(ranked['model_name'].unique())

    def _pivot(col: str) -> pd.DataFrame:
        out: dict[str, pd.Series] = {}
        for fam in FAMILIES:
            sub = ranked[ranked['family'] == fam]
            out[f'{fam} (n={sub["dataset"].nunique()})'] = (
                sub.groupby('model_name')[col].mean().reindex(models)
            )
        out['macro'] = ranked.groupby('model_name')[col].mean().reindex(models)
        return pd.DataFrame(out)

    ranks = _pivot('rank')
    metrics = _pivot('score')
    order = ranks['macro'].sort_values(kind='mergesort').index
    return ranks.reindex(order), metrics.reindex(order)


def summarize_results(
    csv_path: Path, *, method: str = 'linear', split: str | None = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read a CoordBench results CSV and build rank and metric leaderboards.

    Args:
        csv_path: Path to a CoordBench results CSV file.
        method: Probe method selector (exact match if present, otherwise prefix match
            such as ``'knn'`` matching ``'knn5'``).
        split: Optional split view (e.g. ``'random'`` or ``'spatial'``). When set,
            rows with ``split == 'official'`` are also included.

    Returns:
        Tuple ``(ranks, metrics)`` DataFrames from :func:`leaderboard`.

    Raises:
        ValueError: If required columns are missing or no rows match *method* / *split*.
    """
    filepath = os.fspath(csv_path)
    df = pd.read_csv(filepath)
    if 'method' not in df.columns:
        raise ValueError(f"Missing required column 'method' in {filepath!r}.")
    method_col = df['method'].astype(str)
    exact = method_col == method
    mask = exact if exact.any() else method_col.str.startswith(method)
    filtered = df[mask]
    if filtered.empty:
        raise ValueError(f'No rows for method={method!r} in {filepath!r}.')

    if split is not None:
        if 'split' not in filtered.columns:
            raise ValueError(f"Missing required column 'split' in {filepath!r}.")
        filtered = filtered[filtered['split'].isin([split, 'official'])]
        if filtered.empty:
            raise ValueError(
                f'No rows for method={method!r}, split={split!r} in {filepath!r}.'
            )

    return leaderboard(bench_scores(filtered))


__all__ = ('bench_scores', 'leaderboard', 'summarize_results')
