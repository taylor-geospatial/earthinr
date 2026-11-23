# Copyright (c) Taylor Geospatial. All rights reserved.
# Licensed under the MIT License.

from earthinr.coordbench.catalog import list_benchmarks
from earthinr.coordbench.taxonomy import ENV, FAMILIES, LAND, R2_FLOOR, SOCIO, family_of


class TestTaxonomy:
    def test_families_cover_all_benchmarks_disjointly(self) -> None:
        assert FAMILIES == ('socio', 'env', 'land')
        assert R2_FLOOR == -1.0
        assert SOCIO.isdisjoint(ENV)
        assert SOCIO.isdisjoint(LAND)
        assert ENV.isdisjoint(LAND)

        all_benches = set(list_benchmarks())
        assert all_benches == SOCIO | ENV | LAND

    def test_family_of(self) -> None:
        assert family_of('pdfm-conus27') == 'socio'
        assert family_of('sustainbench-asset') == 'socio'
        assert family_of('places-diabetes') == 'socio'
        assert family_of('worldclim-bio1') == 'env'
        assert family_of('dm-openet_ensemble') == 'env'
        assert family_of('country') == 'land'
        assert family_of('dm-descals') == 'land'
        assert family_of('unknown-benchmark') == 'other'
