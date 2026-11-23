# Copyright (c) Taylor Geospatial. All rights reserved.
# Licensed under the MIT License.

import subprocess
import sys

from earthinr.coordbench.catalog import (
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


class TestCatalog:
    def test_repo_and_counts(self) -> None:
        assert COORDBENCH_REPO == 'taylor-geospatial/CoordBench'
        assert len(OFFICIAL_FAMILIES) == 10
        assert len(ALL_FAMILIES) == 13
        assert set(ALL_FAMILIES) - set(OFFICIAL_FAMILIES) == {
            'better_together',
            'country',
            'ecoregions',
        }
        assert len(DEEPMIND_EVAL_CONFIGS) == 15
        assert len(USAVARS_LABELS) == 7
        expected_log_labels = frozenset(
            {'population', 'income', 'nightlights', 'housing'}
        )
        assert expected_log_labels == USAVARS_LOG_LABELS
        assert len(SUSTAINBENCH_TASKS) == 6
        assert len(CDC_PLACES_MEASURES) == 12
        assert len(URBAN_RURAL_BENCHMARKS) == 8

    def test_list_families_and_benchmarks(self) -> None:
        assert list_families() == list(ALL_FAMILIES)
        assert list_families(official_only=True) == list(OFFICIAL_FAMILIES)

        all_benches = list_benchmarks()
        official_benches = list_benchmarks(official_only=True)
        assert len(all_benches) == 60
        assert len(set(all_benches)) == 60
        assert len(official_benches) == 52
        assert len(set(official_benches)) == 52
        assert set(URBAN_RURAL_BENCHMARKS) <= set(official_benches)
        assert set(FAMILY_BENCHMARKS) == set(ALL_FAMILIES)

    def test_catalog_and_config_have_zero_heavy_imports(self) -> None:
        script = """
import importlib.abc
import importlib.util
import os
import sys
import types

BLOCKED = {
    'torch',
    'numpy',
    'pandas',
    'pyarrow',
    'scipy',
    'sklearn',
    'huggingface_hub',
}

class BlockHeavy(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in BLOCKED:
            raise AssertionError(f'heavy import triggered: {fullname}')

sys.meta_path.insert(0, BlockHeavy())

# Stub parent packages so we isolate catalog/taxonomy/config from earthinr.__init__.
src_root = os.path.abspath('src/earthinr')
for pkg_name, pkg_path in (
    ('earthinr', src_root),
    ('earthinr.coordbench', os.path.join(src_root, 'coordbench')),
):
    pkg = types.ModuleType(pkg_name)
    pkg.__path__ = [pkg_path]
    sys.modules[pkg_name] = pkg

import earthinr.coordbench.catalog as cat
import earthinr.coordbench.config as cfg
import earthinr.coordbench.taxonomy as tax

assert len(cat.list_benchmarks()) == 60
assert tax.family_of('pdfm-conus27') == 'socio'
parsed = cfg.parse_coord_config({'model': {'name': 'sincos'}})
assert parsed.datasets == ('all',)
"""
        result = subprocess.run(
            [sys.executable, '-c', script], capture_output=True, text=True, check=False
        )
        assert result.returncode == 0, result.stderr
