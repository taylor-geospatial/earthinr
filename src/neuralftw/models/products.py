# Copyright (c) Taylor Geospatial. All rights reserved.
# Licensed under the MIT License.

"""Remote-sensing embedding product point-lookup encoders (AlphaEarth, TESSERA)."""

import hashlib
import os
import threading
from collections import defaultdict
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor

from .._types import Path
from .._utils import lazy_import
from ._base import LocationEncoder

AEF_MOSAIC_URL = 's3://us-west-2.opendata.source.coop/tge-labs/aef-mosaic'
AEF_YEARS: tuple[int, ...] = (2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025)
AEF_DIM = 64
AEF_NODATA = -128
AEF_LON_ORIGIN = -180.0
AEF_LAT_ORIGIN = 83.68570533713473
AEF_PIXEL_DEG = 0.00008983111749910169
AEF_CHUNK_PX = 256

TESSERA_ZARR_URL = (
    's3://us-west-2.opendata.source.coop/tessera/tessera/zarr/v1.1-dclimate'
)
TESSERA_YEARS: tuple[int, ...] = (2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025)
TESSERA_DIM = 128
TESSERA_SHARD_PX = 4096
TESSERA_CHUNK_PX = 32
_TESSERA_CHUNKS_PER_SIDE = TESSERA_SHARD_PX // TESSERA_CHUNK_PX
_TESSERA_INDEX_BYTES = _TESSERA_CHUNKS_PER_SIDE * _TESSERA_CHUNKS_PER_SIDE * 16 + 4
_TESSERA_MISSING = np.uint64(2**64 - 1)

_S3_STORAGE_OPTIONS: dict[str, Any] = {
    'anon': True,
    'client_kwargs': {'region_name': 'us-west-2'},
    'config_kwargs': {'s3': {'addressing_style': 'path'}},
}
_TLS = threading.local()


def impute_nan_rows(features: NDArray[np.float32]) -> NDArray[np.float32]:
    """Impute NaN entries with per-channel means computed across valid rows.

    Channels that are entirely NaN are filled with ``0.0``.

    Args:
        features: 2D array of shape ``(N, D)`` potentially containing NaNs.

    Returns:
        Float32 array of shape ``(N, D)`` with all NaN entries replaced.
    """
    arr = np.asarray(features, dtype=np.float32)
    nan_mask = np.isnan(arr)
    if not nan_mask.any():
        return arr
    valid_counts = (~nan_mask).sum(axis=0)
    col_sums = np.where(nan_mask, 0.0, arr).sum(axis=0, dtype=np.float64)
    col_mean = np.divide(
        col_sums,
        valid_counts,
        out=np.zeros_like(col_sums, dtype=np.float64),
        where=valid_counts > 0,
    ).astype(np.float32)
    return np.where(nan_mask, col_mean[None, :], arr).astype(np.float32)


def dequantize_aef(raw: NDArray[Any]) -> NDArray[np.float32]:
    """Apply AlphaEarth signed-square dequantization followed by L2 normalization.

    Args:
        raw: Array of shape ``(..., 64)`` containing int8/float32 quantized values
            (with optional NaN rows).

    Returns:
        Float32 L2-normalized dequantized embeddings of the same shape.
    """
    f = np.asarray(raw, dtype=np.float32)
    deq = np.sign(f) * (np.abs(f) / np.float32(127.5)) ** 2
    norms = np.linalg.norm(deq, axis=-1, keepdims=True)
    safe_norms = np.where(norms > 0.0, norms, np.float32(1.0))
    return (deq / safe_norms).astype(np.float32)


def aef_lonlat_to_pixel(
    lon: NDArray[np.floating[Any]], lat: NDArray[np.floating[Any]]
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Map WGS84 ``(lon, lat)`` degrees to fractional ``(row, col)`` AEF pixel indices.

    Args:
        lon: 1D longitude array in degrees.
        lat: 1D latitude array in degrees.

    Returns:
        Tuple of ``(row, col)`` float64 arrays.
    """
    lon_f = np.asarray(lon, dtype=np.float64)
    lat_f = np.asarray(lat, dtype=np.float64)
    col = (lon_f - AEF_LON_ORIGIN) / AEF_PIXEL_DEG - 0.5
    row = (AEF_LAT_ORIGIN - lat_f) / AEF_PIXEL_DEG - 0.5
    return row, col


def utm_zone(lon: NDArray[np.floating[Any]]) -> NDArray[np.intp]:
    """Compute UTM zone numbers ``1..60`` for WGS84 longitudes.

    Args:
        lon: 1D longitude array in degrees.

    Returns:
        Integer array of UTM zone indices in ``[1, 60]``.
    """
    zones = (
        np.floor((np.asarray(lon, dtype=np.float64) + 180.0) / 6.0).astype(np.intp) + 1
    )
    return np.clip(zones, 1, 60)


def resolve_years(
    n: int,
    year: NDArray[np.floating[Any]] | float | None,
    *,
    default_year: int = 2021,
    fixed_year: int | None = None,
    valid_years: tuple[int, ...] = AEF_YEARS,
) -> NDArray[np.intp]:
    """Resolve per-point integer years clipped to a product's valid year range.

    Args:
        n: Number of coordinate rows.
        year: Optional per-point year array or scalar.
        default_year: Fallback year when *year* is ``None`` or contains NaNs.
        fixed_year: Optional fixed year overriding *year* for all rows.
        valid_years: Ordered tuple of supported product years.

    Returns:
        1D integer array of length *n* within ``[valid_years[0], valid_years[-1]]``.
    """
    if fixed_year is not None:
        clipped = int(np.clip(fixed_year, valid_years[0], valid_years[-1]))
        return np.full(n, clipped, dtype=np.intp)
    if year is None:
        return np.full(
            n,
            int(np.clip(default_year, valid_years[0], valid_years[-1])),
            dtype=np.intp,
        )
    y = np.broadcast_to(np.asarray(year, dtype=np.float64), (n,))
    y = np.where(np.isnan(y), float(default_year), y)
    return np.clip(y.astype(np.intp), valid_years[0], valid_years[-1])


def _cache_digest(
    lon: NDArray[np.float64], lat: NDArray[np.float64], years: NDArray[np.intp]
) -> str:
    """Compute a deterministic SHA256 digest for caching point queries."""
    stacked = np.concatenate((lon, lat, years.astype(np.float64))).astype(np.float64)
    return hashlib.sha256(stacked.tobytes()).hexdigest()[:32]


def _open_aef_mosaic(
    url: str = AEF_MOSAIC_URL, *, anon: bool = True, concurrency: int = 16
) -> Any:
    """Open the remote AlphaEarth Zarr mosaic ``embeddings`` array."""
    lazy_import('s3fs', extra='products')
    zarr = lazy_import('zarr', extra='products')
    zarr.config.set({'async.concurrency': concurrency})
    options = {**_S3_STORAGE_OPTIONS, 'anon': anon}
    store = zarr.storage.FsspecStore.from_url(url, storage_options=options)
    return zarr.open_group(store, mode='r')['embeddings']


def _open_tessera_zone(zone: int, *, url: str = TESSERA_ZARR_URL) -> dict[str, Any]:
    """Open one TESSERA v1.1 UTM zone group and return its grid metadata."""
    lazy_import('s3fs', extra='products')
    zarr = lazy_import('zarr', extra='products')
    zone_url = f'{url}/utm{zone:02d}'
    store = zarr.storage.FsspecStore.from_url(
        zone_url, storage_options=_S3_STORAGE_OPTIONS
    )
    group = zarr.open_group(store, mode='r')
    a, _, c, _, e, f = [float(v) for v in group.attrs['spatial:transform']]
    emb = group['embeddings']
    https = zone_url.replace('s3://', 'https://s3.us-west-2.amazonaws.com/')
    return {
        'epsg': int(str(group.attrs['proj:code']).split(':')[1]),
        'transform': (a, c, e, f),
        'ny': int(emb.shape[2]),
        'nx': int(emb.shape[3]),
        'emb_url': f'{https}/embeddings/c',
        'scales_url': f'{https}/scales/c',
    }


def _project_to_utm(
    epsg: int, lon: NDArray[np.float64], lat: NDArray[np.float64]
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Project WGS84 ``(lon, lat)`` degrees into projected ``(east, north)`` meters."""
    pyproj = lazy_import('pyproj', extra='products')
    transformer = pyproj.Transformer.from_crs(
        'EPSG:4326', f'EPSG:{epsg}', always_xy=True
    )
    east, north = transformer.transform(lon, lat)
    return np.asarray(east, dtype=np.float64), np.asarray(north, dtype=np.float64)


def _http_session() -> Any:
    """Return a thread-local keep-alive ``requests.Session`` with retries."""
    requests = lazy_import('requests', extra='products')
    session = getattr(_TLS, 'session', None)
    if session is None:
        session = requests.Session()
        _TLS.session = session
    return session


def _range_read(url: str, byte_range: str, *, timeout: float = 120.0) -> bytes | None:
    """Perform an HTTP Range read, returning ``None`` on HTTP 404."""
    response = _http_session().get(
        url, headers={'Range': f'bytes={byte_range}'}, timeout=timeout
    )
    if response.status_code == 404:
        return None
    response.raise_for_status()
    return bytes(response.content)


def _parse_shard_index(raw: bytes | None) -> NDArray[np.uint64] | None:
    """Parse a Zarr v3 shard index footer into ``(N_chunks, 2)`` ``(offset, nbytes)``."""
    if raw is None or len(raw) != _TESSERA_INDEX_BYTES:
        return None
    return np.frombuffer(raw[:-4], dtype='<u8').reshape(-1, 2)


def _decode_blosc_chunk(raw: bytes, shape: tuple[int, ...], dtype: str) -> NDArray[Any]:
    """Decode a Blosc-compressed Zarr v3 inner chunk."""
    numcodecs = lazy_import('numcodecs', extra='products')
    decoded = numcodecs.Blosc().decode(raw)
    return np.frombuffer(decoded, dtype=dtype).reshape(shape)


def _read_inner_chunk(
    url: str,
    index: NDArray[np.uint64],
    iy: int,
    ix: int,
    shape: tuple[int, ...],
    dtype: str,
) -> NDArray[Any] | None:
    """Read and decode a single inner chunk from a Zarr v3 shard via HTTP Range."""
    offset, nbytes = index[iy * _TESSERA_CHUNKS_PER_SIDE + ix]
    if offset == _TESSERA_MISSING:
        return None
    raw = _range_read(url, f'{int(offset)}-{int(offset) + int(nbytes) - 1}')
    if raw is None:
        return None
    return _decode_blosc_chunk(raw, shape, dtype)


class AlphaEarthLocationEncoder(LocationEncoder):
    """AlphaEarth Foundations (AEF) 64-d Zarr mosaic point-lookup encoder.

    Reads 10 m pixel embeddings from the global EPSG:4326 Zarr mosaic, applies
    signed-square dequantization and L2 normalization, and optionally imputes
    off-coverage NaN rows with the per-channel mean.

    .. note::

       Requires the ``products`` extra (``pip install neuralftw[products]``).
    """

    name = 'aef'
    embed_dim = AEF_DIM

    def __init__(
        self,
        *,
        url: str = AEF_MOSAIC_URL,
        default_year: int = 2021,
        fixed_year: int | None = None,
        impute_nan: bool = True,
        workers: int = 32,
        cache_dir: Path | None = None,
        anon: bool = True,
        device: str | torch.device = 'cpu',
        batch_size: int = 1_000_000,
    ) -> None:
        """Initialize a new AlphaEarthLocationEncoder instance.

        Args:
            url: S3 or filesystem URL to the AlphaEarth Zarr mosaic store.
            default_year: Default year used when *year* is omitted or NaN.
            fixed_year: Optional fixed year overriding per-point years.
            impute_nan: Whether to impute NaN rows with per-channel means.
            workers: Number of worker threads for parallel chunk fetches.
            cache_dir: Optional directory for caching point-query ``.npy`` files.
            anon: Whether to access S3 anonymously.
            device: Target PyTorch device for :meth:`forward` tensor outputs.
            batch_size: Batch size used by :meth:`encode`.
        """
        super().__init__(device=device, batch_size=batch_size)
        self.url = url
        self.default_year = default_year
        self.fixed_year = fixed_year
        self.impute_nan = impute_nan
        self.workers = workers
        self.cache_dir = None if cache_dir is None else os.fspath(cache_dir)
        self.anon = anon
        self._array: Any = None

    def _get_array(self) -> Any:
        """Lazily open and cache the backing Zarr embeddings array."""
        if self._array is None:
            self._array = _open_aef_mosaic(self.url, anon=self.anon)
        return self._array

    def _encode(
        self,
        lon: NDArray[np.floating[Any]],
        lat: NDArray[np.floating[Any]],
        year: NDArray[np.floating[Any]] | None,
        month: NDArray[np.floating[Any]] | None = None,
    ) -> NDArray[np.float32]:
        """Fetch and dequantize AlphaEarth embeddings for one batch."""
        lon_f = np.asarray(lon, dtype=np.float64)
        lat_f = np.asarray(lat, dtype=np.float64)
        n = lon_f.shape[0]
        years = resolve_years(
            n,
            year,
            default_year=self.default_year,
            fixed_year=self.fixed_year,
            valid_years=AEF_YEARS,
        )

        cache_file: str | None = None
        if self.cache_dir is not None:
            cache_file = os.path.join(
                self.cache_dir, f'aef_{_cache_digest(lon_f, lat_f, years)}.npy'
            )
            if os.path.isfile(cache_file):
                return np.asarray(np.load(cache_file), dtype=np.float32)

        arr = self._get_array()
        _, _, height, width = arr.shape
        time_idx = years - AEF_YEARS[0]
        frow, fcol = aef_lonlat_to_pixel(lon_f, lat_f)
        row = np.clip(np.round(frow).astype(np.intp), 0, height - 1)
        col = np.clip(np.round(fcol).astype(np.intp), 0, width - 1)

        groups: dict[tuple[int, int, int], list[int]] = defaultdict(list)
        for i in range(n):
            groups[
                (
                    int(time_idx[i]),
                    int(row[i] // AEF_CHUNK_PX),
                    int(col[i] // AEF_CHUNK_PX),
                )
            ].append(i)

        raw_out = np.full((n, AEF_DIM), np.nan, dtype=np.float32)

        def _read_group(item: tuple[tuple[int, int, int], list[int]]) -> None:
            (t, gy, gx), idxs = item
            y0 = gy * AEF_CHUNK_PX
            x0 = gx * AEF_CHUNK_PX
            block = np.asarray(
                arr[t, :, y0 : y0 + AEF_CHUNK_PX, x0 : x0 + AEF_CHUNK_PX]
            )
            for idx in idxs:
                vec = block[:, row[idx] - y0, col[idx] - x0]
                if np.all(vec != AEF_NODATA):
                    raw_out[idx] = vec.astype(np.float32)

        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            list(pool.map(_read_group, groups.items()))

        embeddings = dequantize_aef(raw_out)
        if cache_file is not None:
            os.makedirs(os.path.dirname(cache_file), exist_ok=True)
            np.save(cache_file, embeddings)
        return embeddings

    def encode(
        self,
        lon: NDArray[np.floating[Any]],
        lat: NDArray[np.floating[Any]],
        year: NDArray[np.floating[Any]] | None = None,
        month: NDArray[np.floating[Any]] | None = None,
    ) -> NDArray[np.float32]:
        """Encode 1D ``(lon, lat)`` arrays with optional NaN row imputation.

        Args:
            lon: 1D array of longitudes in degrees.
            lat: 1D array of latitudes in degrees.
            year: Optional 1D array of observation years.
            month: Unused temporal input kept for :class:`LocationEncoder` parity.

        Returns:
            Float32 embedding array of shape ``(N, 64)``, or ``(0, 0)`` if empty.
        """
        out = super().encode(lon, lat, year=year, month=month)
        if self.impute_nan and out.shape[0] > 0:
            return impute_nan_rows(out)
        return out

    def forward(
        self, coords: Tensor, year: Tensor | None = None, month: Tensor | None = None
    ) -> Tensor:
        """Look up AlphaEarth embeddings for ``(lon, lat)`` coordinate tensors.

        Args:
            coords: Coordinate tensor of shape ``(B, 2)`` in ``(lon, lat)`` degrees.
            year: Optional tensor of calendar years with shape ``(B,)``.
            month: Unused temporal input kept for :class:`LocationEncoder` parity.

        Returns:
            Float32 embedding tensor of shape ``(B, 64)`` on *coords*'s device.
        """
        coords_np = coords.detach().cpu().numpy()
        year_np = None if year is None else year.detach().cpu().numpy()
        emb = self.encode(coords_np[:, 0], coords_np[:, 1], year=year_np)
        return torch.from_numpy(emb).to(device=coords.device, dtype=torch.float32)


class TESSERALocationEncoder(LocationEncoder):
    """TESSERA v1.1 128-d sharded Zarr v3 point-lookup encoder.

    Reads 10 m dequantized pixel embeddings directly from the per-UTM-zone sharded
    Zarr v3 store via HTTP Range requests, deduplicating repeated queries and
    optionally imputing off-coverage NaN rows with the per-channel mean.

    .. note::

       Requires the ``products`` extra (``pip install neuralftw[products]``).
    """

    name = 'tessera'
    embed_dim = TESSERA_DIM

    def __init__(
        self,
        *,
        url: str = TESSERA_ZARR_URL,
        default_year: int = 2021,
        fixed_year: int | None = None,
        impute_nan: bool = True,
        workers: int = 32,
        cache_dir: Path | None = None,
        device: str | torch.device = 'cpu',
        batch_size: int = 1_000_000,
    ) -> None:
        """Initialize a new TESSERALocationEncoder instance.

        Args:
            url: S3 URL prefix for the TESSERA v1.1 per-UTM-zone Zarr store.
            default_year: Default year used when *year* is omitted or NaN.
            fixed_year: Optional fixed year overriding per-point years.
            impute_nan: Whether to impute NaN rows with per-channel means.
            workers: Number of worker threads for concurrent HTTP Range reads.
            cache_dir: Optional directory for caching point-query ``.npy`` files.
            device: Target PyTorch device for :meth:`forward` tensor outputs.
            batch_size: Batch size used by :meth:`encode`.
        """
        super().__init__(device=device, batch_size=batch_size)
        self.url = url
        self.default_year = default_year
        self.fixed_year = fixed_year
        self.impute_nan = impute_nan
        self.workers = workers
        self.cache_dir = None if cache_dir is None else os.fspath(cache_dir)
        self._zone_cache: dict[int, Mapping[str, Any]] = {}

    def _get_zone(self, zone: int) -> Mapping[str, Any]:
        """Return cached grid metadata for UTM *zone*."""
        if zone not in self._zone_cache:
            self._zone_cache[zone] = _open_tessera_zone(zone, url=self.url)
        return self._zone_cache[zone]

    def _encode(
        self,
        lon: NDArray[np.floating[Any]],
        lat: NDArray[np.floating[Any]],
        year: NDArray[np.floating[Any]] | None,
        month: NDArray[np.floating[Any]] | None = None,
    ) -> NDArray[np.float32]:
        """Fetch and dequantize TESSERA v1.1 embeddings for one batch."""
        lon_f = np.asarray(lon, dtype=np.float64)
        lat_f = np.asarray(lat, dtype=np.float64)
        n = lon_f.shape[0]
        years = resolve_years(
            n,
            year,
            default_year=self.default_year,
            fixed_year=self.fixed_year,
            valid_years=TESSERA_YEARS,
        )

        cache_file: str | None = None
        if self.cache_dir is not None:
            cache_file = os.path.join(
                self.cache_dir, f'tessera_{_cache_digest(lon_f, lat_f, years)}.npy'
            )
            if os.path.isfile(cache_file):
                return np.asarray(np.load(cache_file), dtype=np.float32)

        keys = np.stack((lon_f, lat_f, years.astype(np.float64)), axis=1)
        uniq, inverse = np.unique(keys, axis=0, return_inverse=True)
        ulon = uniq[:, 0]
        ulat = uniq[:, 1]
        tidx = uniq[:, 2].astype(np.intp) - TESSERA_YEARS[0]
        zones = utm_zone(ulon)
        out = np.full((uniq.shape[0], TESSERA_DIM), np.nan, dtype=np.float32)

        shards: dict[
            tuple[int, int, int, int], dict[tuple[int, int], list[tuple[int, int, int]]]
        ] = defaultdict(lambda: defaultdict(list))

        for z in np.unique(zones):
            sel = np.flatnonzero(zones == z)
            meta = self._get_zone(int(z))
            a, c, e, f = meta['transform']
            east, north = _project_to_utm(meta['epsg'], ulon[sel], ulat[sel])
            col = np.floor((east - c) / a).astype(np.intp)
            row = np.floor((north - f) / e).astype(np.intp)
            ok = (row >= 0) & (row < meta['ny']) & (col >= 0) & (col < meta['nx'])
            for i, r, cc in zip(sel[ok], row[ok], col[ok], strict=True):
                ri = int(r)
                ci = int(cc)
                shard_key = (
                    int(z),
                    int(tidx[i]),
                    ri // TESSERA_SHARD_PX,
                    ci // TESSERA_SHARD_PX,
                )
                chunk_key = (
                    (ri % TESSERA_SHARD_PX) // TESSERA_CHUNK_PX,
                    (ci % TESSERA_SHARD_PX) // TESSERA_CHUNK_PX,
                )
                shards[shard_key][chunk_key].append((int(i), ri, ci))

        def _read_shard(
            item: tuple[
                tuple[int, int, int, int],
                dict[tuple[int, int], list[tuple[int, int, int]]],
            ],
        ) -> None:
            (z, t, sy, sx), chunks = item
            meta = self._get_zone(z)
            emb_url = f'{meta["emb_url"]}/{t}/0/{sy}/{sx}'
            scales_url = f'{meta["scales_url"]}/{t}/{sy}/{sx}'
            try:
                emb_idx = _parse_shard_index(
                    _range_read(emb_url, f'-{_TESSERA_INDEX_BYTES}')
                )
                scales_idx = _parse_shard_index(
                    _range_read(scales_url, f'-{_TESSERA_INDEX_BYTES}')
                )
            except (OSError, RuntimeError, ValueError):
                return
            if emb_idx is None or scales_idx is None:
                return

            for (cy, cx), pts in chunks.items():
                try:
                    block = _read_inner_chunk(
                        emb_url,
                        emb_idx,
                        cy,
                        cx,
                        (TESSERA_DIM, TESSERA_CHUNK_PX, TESSERA_CHUNK_PX),
                        'int8',
                    )
                    scale = _read_inner_chunk(
                        scales_url,
                        scales_idx,
                        cy,
                        cx,
                        (TESSERA_CHUNK_PX, TESSERA_CHUNK_PX),
                        '<f4',
                    )
                except (OSError, RuntimeError, ValueError):
                    continue
                if block is None or scale is None:
                    continue
                for idx, ri, ci in pts:
                    q = block[:, ri % TESSERA_CHUNK_PX, ci % TESSERA_CHUNK_PX]
                    s = scale[ri % TESSERA_CHUNK_PX, ci % TESSERA_CHUNK_PX]
                    if np.isfinite(s) and s != 0 and q.any():
                        out[idx] = q.astype(np.float32) * np.float32(s)

        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            list(pool.map(_read_shard, shards.items()))

        full = out[inverse]
        if cache_file is not None:
            os.makedirs(os.path.dirname(cache_file), exist_ok=True)
            np.save(cache_file, full)
        return full

    def encode(
        self,
        lon: NDArray[np.floating[Any]],
        lat: NDArray[np.floating[Any]],
        year: NDArray[np.floating[Any]] | None = None,
        month: NDArray[np.floating[Any]] | None = None,
    ) -> NDArray[np.float32]:
        """Encode 1D ``(lon, lat)`` arrays with optional NaN row imputation.

        Args:
            lon: 1D array of longitudes in degrees.
            lat: 1D array of latitudes in degrees.
            year: Optional 1D array of observation years.
            month: Unused temporal input kept for :class:`LocationEncoder` parity.

        Returns:
            Float32 embedding array of shape ``(N, 128)``, or ``(0, 0)`` if empty.
        """
        out = super().encode(lon, lat, year=year, month=month)
        if self.impute_nan and out.shape[0] > 0:
            return impute_nan_rows(out)
        return out

    def forward(
        self, coords: Tensor, year: Tensor | None = None, month: Tensor | None = None
    ) -> Tensor:
        """Look up TESSERA v1.1 embeddings for ``(lon, lat)`` coordinate tensors.

        Args:
            coords: Coordinate tensor of shape ``(B, 2)`` in ``(lon, lat)`` degrees.
            year: Optional tensor of calendar years with shape ``(B,)``.
            month: Unused temporal input kept for :class:`LocationEncoder` parity.

        Returns:
            Float32 embedding tensor of shape ``(B, 128)`` on *coords*'s device.
        """
        coords_np = coords.detach().cpu().numpy()
        year_np = None if year is None else year.detach().cpu().numpy()
        emb = self.encode(coords_np[:, 0], coords_np[:, 1], year=year_np)
        return torch.from_numpy(emb).to(device=coords.device, dtype=torch.float32)


__all__ = (
    'AEF_CHUNK_PX',
    'AEF_DIM',
    'AEF_LAT_ORIGIN',
    'AEF_LON_ORIGIN',
    'AEF_MOSAIC_URL',
    'AEF_NODATA',
    'AEF_PIXEL_DEG',
    'AEF_YEARS',
    'TESSERA_CHUNK_PX',
    'TESSERA_DIM',
    'TESSERA_SHARD_PX',
    'TESSERA_YEARS',
    'TESSERA_ZARR_URL',
    'AlphaEarthLocationEncoder',
    'TESSERALocationEncoder',
    'aef_lonlat_to_pixel',
    'dequantize_aef',
    'impute_nan_rows',
    'resolve_years',
    'utm_zone',
)
