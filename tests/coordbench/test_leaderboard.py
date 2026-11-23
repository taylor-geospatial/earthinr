# Copyright (c) Taylor Geospatial. All rights reserved.
# Licensed under the MIT License.

from pathlib import Path

import pandas as pd
import pytest

from earthinr.coordbench.leaderboard import bench_scores, leaderboard, summarize_results


class TestLeaderboard:
    def test_bench_scores_and_leaderboard_ranks(self) -> None:
        results = pd.DataFrame(
            [
                ('alpha', 'california-housing', 'r2', -100.0),
                ('alpha', 'california-housing', 'r2', 0.6),
                ('beta', 'california-housing', 'r2', 0.2),
                ('beta', 'california-housing', 'r2', 0.4),
                ('alpha', 'worldclim-bio1', 'r2', 0.5),
                ('beta', 'worldclim-bio1', 'r2', 0.5),
                ('alpha', 'country', 'accuracy', 0.8),
                ('beta', 'country', 'accuracy', 0.6),
            ],
            columns=['model_name', 'dataset', 'metric_name', 'metric_value'],
        )
        original = results.copy(deep=True)
        scores = bench_scores(results)
        pd.testing.assert_frame_equal(results, original)

        ranks, metrics = leaderboard(scores)
        assert list(ranks.index) == ['alpha', 'beta']
        assert ranks['socio (n=1)'].tolist() == [2.0, 1.0]
        assert ranks['env (n=1)'].tolist() == [1.5, 1.5]
        assert ranks['land (n=1)'].tolist() == [1.0, 2.0]
        assert ranks['macro'].tolist() == [1.5, 1.5]
        assert metrics['socio (n=1)'].tolist() == pytest.approx([-0.2, 0.3])

    def test_validation_errors(self) -> None:
        with pytest.raises(ValueError, match='Missing required columns'):
            bench_scores(pd.DataFrame({'model_name': ['a']}))

        empty = pd.DataFrame(columns=['model_name', 'dataset', 'score', 'family'])
        with pytest.raises(ValueError, match='No benchmark scores'):
            leaderboard(empty)

    def test_summarize_results_filters_and_ordering(self, tmp_path: Path) -> None:
        csv_path = tmp_path / 'results.csv'
        pd.DataFrame(
            [
                ('alpha', 'california-housing', 'random', 'linear', 'r2', 0.1),
                ('beta', 'california-housing', 'random', 'linear', 'r2', 0.9),
                ('alpha', 'worldclim-bio1', 'official', 'linear', 'r2', 0.2),
                ('beta', 'worldclim-bio1', 'official', 'linear', 'r2', 0.8),
                ('alpha', 'unregistered', 'spatial', 'linear', 'r2', 0.4),
                ('nearest', 'country', 'official', 'knn5', 'accuracy', 0.9),
                ('other_knn', 'country', 'official', 'knn10', 'accuracy', 0.7),
            ],
            columns=[
                'model_name',
                'dataset',
                'split',
                'method',
                'metric_name',
                'metric_value',
            ],
        ).to_csv(csv_path, index=False)

        # Best macro rank ('beta') comes first
        ranks, metrics = summarize_results(csv_path, method='linear', split='random')
        assert list(ranks.index) == ['beta', 'alpha']
        assert ranks.loc['beta', 'macro'] == 1.0
        assert metrics.loc['beta', 'socio (n=1)'] == pytest.approx(0.9)

        # Split=None includes all splits
        ranks_all, _ = summarize_results(csv_path, method='linear')
        assert set(ranks_all.index) == {'alpha', 'beta'}

        # Prefix match ('knn' matches both knn5 and knn10) vs exact match ('knn5')
        ranks_knn, _ = summarize_results(csv_path, method='knn', split='spatial')
        assert list(ranks_knn.index) == ['nearest', 'other_knn']

        ranks_knn5, _ = summarize_results(csv_path, method='knn5', split='official')
        assert list(ranks_knn5.index) == ['nearest']

        # Missing method match
        with pytest.raises(ValueError, match="No rows for method='absent'"):
            summarize_results(csv_path, method='absent')

        # Missing split match
        no_spatial = tmp_path / 'no_spatial.csv'
        pd.DataFrame(
            [('alpha', 'country', 'random', 'linear', 'accuracy', 0.8)],
            columns=[
                'model_name',
                'dataset',
                'split',
                'method',
                'metric_name',
                'metric_value',
            ],
        ).to_csv(no_spatial, index=False)
        with pytest.raises(ValueError, match="split='spatial'"):
            summarize_results(no_spatial, method='linear', split='spatial')

        # Missing 'method' column
        no_method = tmp_path / 'no_method.csv'
        pd.DataFrame({'model_name': ['a']}).to_csv(no_method, index=False)
        with pytest.raises(ValueError, match="Missing required column 'method'"):
            summarize_results(no_method)

        # Missing 'split' column when split filter requested
        no_split = tmp_path / 'no_split.csv'
        pd.DataFrame(
            [('alpha', 'country', 'linear', 'accuracy', 0.8)],
            columns=['model_name', 'dataset', 'method', 'metric_name', 'metric_value'],
        ).to_csv(no_split, index=False)
        with pytest.raises(ValueError, match="Missing required column 'split'"):
            summarize_results(no_split, method='linear', split='random')
