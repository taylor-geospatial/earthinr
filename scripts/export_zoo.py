#!/usr/bin/env python3
# Copyright (c) Taylor Geospatial. All rights reserved.
# Licensed under the MIT License.

"""Export location-encoder-only safetensors checkpoints for taylor-geospatial/earthinr-zoo.

Downloads upstream checkpoints from Hugging Face Hub, strips all non-location
components (image/audio/text encoders, classification heads, task decoders,
pickle wrappers), verifies finiteness and expected tensor counts/shapes, writes
clean per-model `model.safetensors` + `config.json` files, generates the top-level
`README.md` model card with full attributions/licenses/SHA256s, and optionally
creates & uploads `taylor-geospatial/earthinr-zoo`.
"""

import argparse
import hashlib
import json
import urllib.request
import zipfile
from pathlib import Path
from typing import Any

import torch
from huggingface_hub import HfApi, hf_hub_download
from safetensors import safe_open
from safetensors.torch import load_file, save_file

ZOO_REPO_ID = 'taylor-geospatial/earthinr-zoo'
TERRANOVA_REVISION = 'ada28cc009cd52da6949981dce8819fd3ae7c8c0'
CSP_ZIP_URL = 'https://www.dropbox.com/s/qxr644rj1qxekn2/model_dir.zip?dl=1'
CSP_LICENSE = 'unlicensed (contact upstream authors; Mai et al., ICML 2023)'


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def verify_and_save(
    out_dir: Path,
    tensors: dict[str, torch.Tensor],
    config: dict[str, Any],
    *,
    metadata: dict[str, str] | None = None,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    clean: dict[str, torch.Tensor] = {}
    total_params = 0
    dtypes: set[str] = set()
    for k in sorted(tensors):
        t = tensors[k].detach().cpu().contiguous()
        if t.is_floating_point() and not torch.isfinite(t).all():
            raise ValueError(f'{out_dir.name}: non-finite values in tensor {k}')
        clean[k] = t
        total_params += t.numel()
        dtypes.add(str(t.dtype).replace('torch.', ''))

    st_path = out_dir / 'model.safetensors'
    save_file(clean, str(st_path), metadata=metadata or {'format': 'pt'})

    digest = sha256_file(st_path)
    size_bytes = st_path.stat().st_size

    cfg_path = out_dir / 'config.json'
    cfg_path.write_text(json.dumps(config, indent=2, sort_keys=True) + '\n')

    return {
        'model_id': out_dir.name,
        'num_tensors': len(clean),
        'num_params': total_params,
        'dtypes': sorted(dtypes),
        'size_bytes': size_bytes,
        'sha256': digest,
        'config': config,
    }


def resolve_revision(api: HfApi, repo_id: str, revision: str | None = None) -> str:
    info = api.repo_info(repo_id, revision=revision)
    assert info.sha is not None
    return info.sha


def export_satclip(
    api: HfApi, staging: Path, model_id: str, repo_id: str, ckpt_filename: str
) -> dict[str, Any]:
    rev = resolve_revision(api, repo_id)
    ckpt_path = hf_hub_download(repo_id, ckpt_filename, revision=rev)
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    hp = ckpt['hyper_parameters']
    raw_sd = ckpt['state_dict']

    tensors: dict[str, torch.Tensor] = {}
    for k, v in raw_sd.items():
        if 'nnet.' in k:
            norm_k = 'nnet.' + k.split('nnet.', 1)[1]
            # Store as float32 (SatCLIP location encoder runs SH in float64;
            # float32 SirenNet matches float64 SirenNet to <8e-6 and cuts size 2x).
            # Wait: check native dtype in metadata and preserve float64 if double!
            tensors[norm_k] = v.float() if v.dtype == torch.float64 else v

    assert len(tensors) == 6, (
        f'Expected 6 SatCLIP tensors, got {len(tensors)}: {list(tensors)}'
    )
    config = {
        'architecture': 'satclip',
        'coord_order': 'lonlat',
        'legendre_polys': int(hp['legendre_polys']),
        'capacity': int(hp['capacity']),
        'embed_dim': int(hp['embed_dim']),
        'output_dim': int(hp['embed_dim']),
        'num_hidden_layers': int(hp['num_hidden_layers']),
        'le_type': str(hp['le_type']),
        'pe_type': str(hp['pe_type']),
        'harmonics_calculation': str(hp['harmonics_calculation']),
        'source_repo': repo_id,
        'source_file': ckpt_filename,
        'source_revision': rev,
        'license': 'MIT',
        'upstream_url': 'https://github.com/microsoft/satclip',
    }
    return verify_and_save(
        staging / model_id,
        tensors,
        config,
        metadata={'format': 'pt', 'source_repo': repo_id, 'source_revision': rev},
    )


def export_rff_safetensors(
    api: HfApi,
    staging: Path,
    model_id: str,
    repo_id: str,
    *,
    coord_order: str,
    license_id: str,
    upstream_url: str,
) -> dict[str, Any]:
    rev = resolve_revision(api, repo_id)
    cfg_raw = json.loads(
        Path(hf_hub_download(repo_id, 'config.json', revision=rev)).read_text()
    )
    tensors = load_file(hf_hub_download(repo_id, 'model.safetensors', revision=rev))

    dim = int(cfg_raw['dim'])
    hidden_dim = int(tensors['LocEnc0.capsule.1.weight'].shape[0])
    output_dim = int(tensors['LocEnc0.head.0.weight'].shape[0])
    assert output_dim == dim

    config = {
        'architecture': 'geoclip',
        'coord_order': coord_order,
        'sigma': [
            float(s) if not float(s).is_integer() else int(s) for s in cfg_raw['sigma']
        ],
        'input_size': int(cfg_raw['input_size']),
        'encoded_size': int(cfg_raw['encoded_size']),
        'hidden_dim': hidden_dim,
        'output_dim': output_dim,
        'source_repo': repo_id,
        'source_file': 'model.safetensors',
        'source_revision': rev,
        'license': license_id,
        'upstream_url': upstream_url,
    }
    return verify_and_save(
        staging / model_id,
        tensors,
        config,
        metadata={'format': 'pt', 'source_repo': repo_id, 'source_revision': rev},
    )


def export_gair(api: HfApi, staging: Path) -> dict[str, Any]:
    repo_id = 'PingL/GAIR'
    rev = resolve_revision(api, repo_id)
    ckpt_path = hf_hub_download(repo_id, 'checkpoint.pth', revision=rev)
    state = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    for key in ('state_dict', 'model', 'model_state_dict'):
        if isinstance(state, dict) and key in state:
            state = state[key]

    prefix = next(k for k in state if 'LocEnc0.capsule.1.weight' in k).split('LocEnc0')[
        0
    ]
    tensors: dict[str, torch.Tensor] = {}
    for k, v in state.items():
        if k.startswith(prefix):
            # Keep exact GeoCLIP key layout: LocEnc{i}.capsule.{0.b, 1, 3, 5} & head.0
            tensors[k[len(prefix) :]] = v.float()

    assert len(tensors) == 27, f'Expected 27 GAIR tensors, got {len(tensors)}'
    hidden_dim = int(tensors['LocEnc0.capsule.1.weight'].shape[0])
    output_dim = int(tensors['LocEnc0.head.0.weight'].shape[0])
    encoded_size = int(tensors['LocEnc0.capsule.0.b'].shape[0])
    input_size = int(tensors['LocEnc0.capsule.0.b'].shape[1])

    config = {
        'architecture': 'geoclip',
        'coord_order': 'lonlat',
        'sigma': [1, 16, 256],
        'input_size': input_size,
        'encoded_size': encoded_size,
        'hidden_dim': hidden_dim,
        'output_dim': output_dim,
        'source_repo': repo_id,
        'source_file': 'checkpoint.pth',
        'source_revision': rev,
        'license': 'Apache-2.0',
        'upstream_url': 'https://huggingface.co/PingL/GAIR',
    }
    return verify_and_save(
        staging / 'gair',
        tensors,
        config,
        metadata={'format': 'pt', 'source_repo': repo_id, 'source_revision': rev},
    )


def export_sinr(api: HfApi, staging: Path) -> dict[str, Any]:
    repo_id = 'MVRL/sinr-location-encoder-1000-cls'
    rev = resolve_revision(api, repo_id)
    cfg_raw = json.loads(
        Path(hf_hub_download(repo_id, 'config.json', revision=rev)).read_text()
    )
    raw_tensors = load_file(hf_hub_download(repo_id, 'model.safetensors', revision=rev))

    # Strip the 47,375-class species head (48.5 MB) to keep location encoder feats.* only (2.11 MB).
    tensors = {k: v for k, v in raw_tensors.items() if k.startswith('feats.')}
    assert len(tensors) == 18, f'Expected 18 SINR tensors, got {len(tensors)}'

    config = {
        'architecture': 'sinr',
        'coord_order': 'lonlat',
        'num_inputs': int(cfg_raw['num_inputs']),
        'num_filts': int(cfg_raw['num_filts']),
        'depth': int(cfg_raw['depth']),
        'num_classes': 0,
        'output_dim': int(cfg_raw['num_filts']),
        'source_repo': repo_id,
        'source_file': 'model.safetensors',
        'source_revision': rev,
        'license': 'MIT',
        'upstream_url': 'https://github.com/elijahcole/sinr',
    }
    return verify_and_save(
        staging / 'sinr',
        tensors,
        config,
        metadata={'format': 'pt', 'source_repo': repo_id, 'source_revision': rev},
    )


def export_climplicit(api: HfApi, staging: Path) -> dict[str, Any]:
    repo_id = 'Jobedo/climplicit'
    rev = resolve_revision(api, repo_id)
    tensors = load_file(hf_hub_download(repo_id, 'model.safetensors', revision=rev))
    assert len(tensors) == 36, f'Expected 36 Climplicit tensors, got {len(tensors)}'

    config = {
        'architecture': 'climplicit',
        'coord_order': 'lonlat',
        'dim_in': 4,
        'dim_hidden': 512,
        'dim_out': 256,
        'num_layers': 16,
        'w0': 1.0,
        'w0_initial': 30.0,
        'h_siren': True,
        'residual_connections': True,
        'num_chelsa_vars': 11,
        'default_months': [3, 6, 9, 12],
        'output_dim': 1024,
        'source_repo': repo_id,
        'source_file': 'model.safetensors',
        'source_revision': rev,
        'license': 'CC-BY-4.0',
        'upstream_url': 'https://github.com/ecovision-uzh/climplicit',
    }
    return verify_and_save(
        staging / 'climplicit',
        tensors,
        config,
        metadata={'format': 'pt', 'source_repo': repo_id, 'source_revision': rev},
    )


def export_tte(api: HfApi, staging: Path) -> dict[str, Any]:
    repo_id = 'MVRL/TTE'
    rev = resolve_revision(api, repo_id)
    cfg_raw = json.loads(
        Path(hf_hub_download(repo_id, 'config.json', revision=rev)).read_text()
    )
    raw_tensors = load_file(hf_hub_download(repo_id, 'model.safetensors', revision=rev))

    drop = {
        'location.semantic_registers.img_to_logits.weight',
        'location.semantic_registers.img_to_logits.bias',
        'location.semantic_registers._img_temperature',
    }
    tensors = {k: v for k, v in raw_tensors.items() if k not in drop}
    assert len(tensors) == 22, f'Expected 22 TTE tensors, got {len(tensors)}'

    config = {
        'architecture': 'tte',
        'coord_order': 'latlon',
        'n_sites': int(cfg_raw['n_sites']),
        'site_embed_dim': int(cfg_raw['site_embed_dim']),
        'output_dim': int(cfg_raw['output_dim']),
        'hidden_dim': int(cfg_raw['hidden_dim']),
        'n_reslayers': int(cfg_raw['n_reslayers']),
        'n_registers': int(cfg_raw['n_registers']),
        'register_fixed_gate': float(cfg_raw['register_fixed_gate']),
        'use_cosine_attention': bool(cfg_raw['use_cosine_attention']),
        'normalize': bool(cfg_raw['normalize']),
        'source_repo': repo_id,
        'source_file': 'model.safetensors',
        'source_revision': rev,
        'license': 'MIT',
        'upstream_url': 'https://github.com/mvrl/TTE',
    }
    return verify_and_save(
        staging / 'tte',
        tensors,
        config,
        metadata={'format': 'pt', 'source_repo': repo_id, 'source_revision': rev},
    )


def export_terranova(api: HfApi, staging: Path) -> list[dict[str, Any]]:
    repo_id = 'crp94/terranova'
    rev = resolve_revision(api, repo_id, revision=TERRANOVA_REVISION)
    cfg_raw = json.loads(
        Path(hf_hub_download(repo_id, 'config.json', revision=rev)).read_text()
    )
    st_path = hf_hub_download(repo_id, 'model.safetensors', revision=rev)

    spatial_tensors: dict[str, torch.Tensor] = {}
    st_tensors: dict[str, torch.Tensor] = {}
    with safe_open(st_path, framework='pt', device='cpu') as f:
        for k in list(f.keys()):
            if k.startswith('location_encoder.'):
                t = f.get_tensor(k)
                spatial_tensors[k] = t
                st_tensors[k] = t
            elif k.startswith(('time_encoder.', 'st_fusion.')):
                st_tensors[k] = f.get_tensor(k)

    assert len(spatial_tensors) == 86, (
        f'Expected 86 spatial tensors, got {len(spatial_tensors)}'
    )
    assert len(st_tensors) == 221, f'Expected 221 ST tensors, got {len(st_tensors)}'

    base_loc_cfg = {
        'location_dim': int(cfg_raw['location_dim']),
        'output_dim': int(cfg_raw['location_dim']),
        'wire_widths': [int(w) for w in cfg_raw['wire_widths']],
        'wire_omega_0': float(cfg_raw['wire_omega_0']),
        'wire_sigma_0': float(cfg_raw['wire_sigma_0']),
        'wire_omega_0_initial': float(cfg_raw['wire_omega_0_initial']),
        'hash_n_levels': int(cfg_raw['hash_n_levels']),
        'hash_n_features': int(cfg_raw['hash_n_features']),
        'hash_log2_size': int(cfg_raw['hash_log2_size']),
        'hash_base_res': int(cfg_raw['hash_base_res']),
        'hash_finest_res': int(cfg_raw['hash_finest_res']),
        'hash_hidden': [int(w) for w in cfg_raw['hash_hidden']],
        'gate_init_bias': float(cfg_raw['gate_init_bias']),
    }

    spatial_cfg = {
        'architecture': 'terranova',
        'variant': 'spatial',
        'coord_order': 'lonlat',
        **base_loc_cfg,
        'source_repo': repo_id,
        'source_file': 'model.safetensors',
        'source_revision': rev,
        'license': 'CC-BY-4.0',
        'upstream_url': 'https://github.com/crp94/terranova-model',
    }
    st_cfg = {
        'architecture': 'terranova',
        'variant': 'st',
        'coord_order': 'lonlat',
        **base_loc_cfg,
        'time_dim': int(cfg_raw['time_dim']),
        'time_width': int(cfg_raw['time_width']),
        'time_depth': int(cfg_raw['time_depth']),
        'time_fourier': int(cfg_raw['time_fourier']),
        'year_min': int(cfg_raw['year_min']),
        'year_max': int(cfg_raw['year_max']),
        'default_year': 2020,
        'st_width': int(cfg_raw['st_width']),
        'st_heads': int(cfg_raw['st_heads']),
        'st_depth': int(cfg_raw['st_depth']),
        'alpha_residual_init': float(cfg_raw['alpha_residual_init']),
        'dropout': float(cfg_raw['dropout']),
        'source_repo': repo_id,
        'source_file': 'model.safetensors',
        'source_revision': rev,
        'license': 'CC-BY-4.0',
        'upstream_url': 'https://github.com/crp94/terranova-model',
    }

    meta = {'format': 'pt', 'source_repo': repo_id, 'source_revision': rev}
    return [
        verify_and_save(
            staging / 'terranova-spatial', spatial_tensors, spatial_cfg, metadata=meta
        ),
        verify_and_save(staging / 'terranova-st', st_tensors, st_cfg, metadata=meta),
    ]


def export_csp(
    api: HfApi, staging: Path, csp_zip: Path | None = None
) -> list[dict[str, Any]]:
    """Export CSP UNSUPER checkpoints (csp-inat, csp-fmow) from local zip or Dropbox."""
    raw_dir = staging / '_csp_raw'
    raw_dir.mkdir(parents=True, exist_ok=True)

    candidates = [csp_zip] if csp_zip is not None else []
    candidates.extend(
        [Path.home() / 'Downloads' / 'model_dir.zip', raw_dir / 'model_dir.zip']
    )
    zip_path = next((p for p in candidates if p is not None and p.exists()), None)
    if zip_path is None:
        zip_path = raw_dir / 'model_dir.zip'
        tmp = zip_path.with_suffix('.zip.tmp')
        urllib.request.urlretrieve(CSP_ZIP_URL, tmp)
        tmp.rename(zip_path)

    with zipfile.ZipFile(zip_path) as zf:
        for member in zf.namelist():
            if (
                '__MACOSX' not in member
                and 'UNSUPER' in member
                and member.endswith('.pth.tar')
            ):
                target = raw_dir / member
                if not target.exists():
                    zf.extract(member, raw_dir)

    specs = [('csp-inat', 'model_inat_2018'), ('csp-fmow', 'model_fmow')]
    results: list[dict[str, Any]] = []
    for model_id, subdir in specs:
        hits = sorted((raw_dir / 'model_dir' / subdir).glob('*UNSUPER*.pth.tar'))
        if not hits:
            raise FileNotFoundError(
                f'No CSP UNSUPER checkpoint found under {raw_dir / "model_dir" / subdir}'
            )
        ckpt_path = hits[0]
        ck = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        p = ck['params']
        sd = ck['state_dict']

        # Strip class_emb, user_emb, img_dec, and duplicated loc_enc.spa_enc.* alias;
        # keep only the 6 spa_enc.ffn.* tensors renamed cleanly to ffn.*.
        tensors: dict[str, torch.Tensor] = {}
        for k, v in sd.items():
            if k.startswith('spa_enc.ffn.'):
                tensors[k[len('spa_enc.') :]] = v.float()

        assert len(tensors) == 6, (
            f'Expected 6 CSP tensors for {model_id}, got {len(tensors)}: {list(tensors)}'
        )
        config = {
            'architecture': 'csp',
            'coord_order': 'lonlat',
            'normalize_coords': True,
            'spa_enc_type': str(p['spa_enc_type']),
            'coord_dim': int(p['num_loc_feats']),
            'frequency_num': int(p['frequency_num']),
            'max_radius': float(p['max_radius']),
            'min_radius': float(p['min_radius']),
            'freq_init': str(p['freq_init']),
            'num_hidden_layers': int(p['num_hidden_layer']),
            'hidden_dim': int(p['hidden_dim']),
            'output_dim': int(p['num_filts']),
            'activation': str(p['spa_f_act']),
            'use_layernorm': bool(p['use_layn']),
            'skip_connection': bool(p['skip_connection']),
            'source_repo': 'gengchenmai/csp',
            'source_file': ckpt_path.name,
            'source_revision': '67f34075c944ad3845499892a23c7363691cf3bf',
            'source_url': CSP_ZIP_URL,
            'license': CSP_LICENSE,
            'upstream_url': 'https://github.com/gengchenmai/csp',
        }
        results.append(
            verify_and_save(
                staging / model_id,
                tensors,
                config,
                metadata={
                    'format': 'pt',
                    'source_repo': 'gengchenmai/csp',
                    'source_file': ckpt_path.name,
                },
            )
        )
    return results


def export_osmgraphclip(
    api: HfApi,
    staging: Path,
    model_id: str,
    repo_id: str,
    ckpt_filename: str,
    *,
    multiscale: bool,
) -> dict[str, Any]:
    rev = resolve_revision(api, repo_id)
    ckpt_path = hf_hub_download(repo_id, ckpt_filename, revision=rev)
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    hp = ckpt['hyper_parameters']
    raw_sd = ckpt['state_dict']

    tensors: dict[str, torch.Tensor] = {}
    for k, v in raw_sd.items():
        if 'nnet.' in k:
            norm_k = 'nnet.' + k.split('nnet.', 1)[1]
            tensors[norm_k] = v.float() if v.dtype == torch.float64 else v

    assert len(tensors) == 6, (
        f'Expected 6 OSMGraphCLIP tensors, got {len(tensors)}: {list(tensors)}'
    )
    config = {
        'architecture': 'osmgraphclip',
        'coord_order': 'lonlat',
        'multiscale': multiscale,
        'legendre_polys': int(hp['legendre_polys']),
        'capacity': int(hp['capacity']),
        'embed_dim': int(hp['embed_dim']),
        'output_dim': int(hp['embed_dim']),
        'num_hidden_layers': int(hp['num_hidden_layers']),
        'le_type': str(hp['le_type']),
        'pe_type': str(hp['pe_type']),
        'harmonics_calculation': str(hp['harmonics_calculation']),
        'source_repo': repo_id,
        'source_file': ckpt_filename,
        'source_revision': rev,
        'license': 'Apache-2.0',
        'upstream_url': 'https://github.com/d-michail/osmgraphclip',
    }
    return verify_and_save(
        staging / model_id,
        tensors,
        config,
        metadata={'format': 'pt', 'source_repo': repo_id, 'source_revision': rev},
    )


def _temporal_summary(model_id: str, cfg: dict[str, Any]) -> str:
    if model_id == 'climplicit':
        return (
            'Optional `month` (1-12 -> 256-d; omitted -> `[3,6,9,12]` concat -> 1024-d)'
        )
    if model_id == 'terranova-st':
        return f'Optional `year` ({cfg["year_min"]}-{cfg["year_max"]}, default `{cfg["default_year"]}`)'
    return 'Time-invariant'


def build_readme(records: list[dict[str, Any]]) -> str:
    rows = []
    for rec in records:
        cfg = rec['config']
        size_mb = rec['size_bytes'] / (1024 * 1024)
        src_repo = cfg['source_repo']
        src_link = (
            f'[{src_repo}]({cfg["upstream_url"]})'
            if src_repo == 'gengchenmai/csp'
            else f'[{src_repo}](https://huggingface.co/{src_repo})'
        )
        temporal = _temporal_summary(rec['model_id'], cfg)
        rows.append(
            f'| `{rec["model_id"]}` | `{cfg["architecture"]}` | `(lon, lat)` deg (native `{cfg["coord_order"]}`) | '
            f'{temporal} | `{cfg["output_dim"]}` | {size_mb:.2f} MiB | {rec["num_tensors"]} | '
            f'{src_link} | {cfg["license"]} | `{rec["sha256"][:16]}...` |'
        )

    sha_lines = [
        f'- `{rec["model_id"]}/model.safetensors`: `{rec["sha256"]}`' for rec in records
    ]

    return f"""---
license: other
license_name: mixed-per-model-licenses
tags:
  - geospatial
  - location-encoder
  - implicit-neural-representation
  - earth-observation
  - safetensors
  - earthinr
---

# EarthINR Model Zoo (`taylor-geospatial/earthinr-zoo`)

Location-encoder-only `safetensors` checkpoints repackaged for **[EarthINR](https://github.com/taylor-geospatial/earthinr)**.

All non-location components (satellite/vision encoders, audio encoders, text encoders, large species classification heads, task decoders, and legacy pickle `.pth`/`.ckpt` wrappers) have been stripped so every model loads directly in pure PyTorch + `safetensors` with zero `rshf`, `geoclip`, `transformers`, `timm`, or `open_clip_torch` dependencies.

> **Notes:**
> - **MIND** weights remain at their canonical repository [`taylor-geospatial/MIND`](https://huggingface.co/taylor-geospatial/MIND) (`mind.safetensors`, MIT, 216.5 MiB fp16, `embed_dim=3072` pooled / `64` head, time-invariant).
> - **RANGE / RANGE+** are excluded (retrieval database rather than standalone location encoder).
> - **CSP** (`csp-inat`, `csp-fmow`) has no license file in its upstream repository (`gengchenmai/csp`); see Licenses & Attribution below.

## Repository Layout

Each model lives in its own directory `<model_id>/`:
- `<model_id>/model.safetensors`: location-encoder weights only.
- `<model_id>/config.json`: architecture hyperparameters and exact upstream source provenance (`source_repo`, `source_revision`, `license`, `upstream_url`).

## Available Checkpoints

All `earthinr` encoders expose a universal `(lon, lat)` WGS84 degrees public API and handle any native `(lat, lon)` coordinate swap internally.

| Model ID | Architecture | Spatial Input | Temporal Input | Output Dim | Size | Tensors | Upstream Source | License | SHA256 (prefix) |
|---|---|---|---|---:|---:|---:|---|---|---|
{chr(10).join(rows)}

## Full SHA256 Checksums

{chr(10).join(sha_lines)}

## Licenses & Attribution

This collection aggregates checkpoints released under their respective upstream licenses (**MIT**, **Apache-2.0**, **CC-BY-4.0**, plus **unlicensed** for CSP). Please respect the license and attribution terms of each individual model you use:

- **SatCLIP** (`satclip-*`): MIT License — Copyright (c) Microsoft Corporation. Klemmer et al., *SatCLIP: Global, General-Purpose Location Embeddings with Satellite Imagery*, AAAI 2025 (`arXiv:2311.17179`).
- **GeoCLIP** (`geoclip`): MIT License — Copyright (c) 2024 Vicente Vivanco (RFF adapted from `jmclong/random-fourier-features-pytorch`, MIT). Vivanco Cepeda et al., *GeoCLIP: Clip-Inspired Alignment between Locations and Images for Effective Worldwide Geo-localization*, NeurIPS 2023 (`arXiv:2309.16020`).
- **SINR** (`sinr`): MIT License — Copyright (c) 2023 Elijah Cole. Cole et al., *Spatial Implicit Neural Representations for Global-Scale Species Mapping*, ICML 2023 (`arXiv:2306.02564`).
- **Climplicit** (`climplicit`): Weights **CC-BY-4.0**, code MIT — UZH-EcoVision. Dollinger et al., *Climplicit: Climatic Implicit Embeddings for Global Ecological Tasks*, ICLR Workshops 2025 (`arXiv:2504.05089`).
- **GAIR** (`gair`): Apache-2.0 — Liu et al., *GAIR*, ISPRS 2026 (`arXiv:2503.16683`).
- **TaxaBind / EcoGeo** (`taxabind`): Apache-2.0 — MVRL. Sastry et al., *TaxaBind: A Unified Embedding Space for Ecological Applications*, WACV 2025.
- **SLED** (`sled-s1-s2-ls`, `sled-s2-ls`): Apache-2.0 — GeoHAI. Lane et al., *SLED* (`arXiv:2608.06612`).
- **TTE** (`tte`): MIT License — Copyright (c) 2026 Multimodal Vision Research Lab @ WashU. *Tessellating the Earth*, ECCV 2026.
- **TerraNova** (`terranova-spatial`, `terranova-st`): Weights **CC-BY-4.0**, code Apache-2.0 — Copyright 2026 Carlos Rodriguez-Pardo and Massimo Tavoni. Attribution: cite `arXiv:2607.29527`.
- **CSP** (`csp-inat`, `csp-fmow`): **Unlicensed** (no `LICENSE` file in `gengchenmai/csp`; contact upstream authors for reuse permissions). Mai et al., *CSP: Self-Supervised Contrastive Spatial Pre-Training for Geospatial-Visual Representations*, ICML 2023 (`arXiv:2305.01118`).
- **OSMGraphCLIP** (`osmgraphclip-ms-l40`, `osmgraphclip-ms-l10`, `osmgraphclip-a-l40`, `osmgraphclip-a-l10`): Weights **Apache-2.0**, code MIT — Michail et al., *OSMGraphCLIP: Learning Global Location Representations from OpenStreetMap Graphs*, ACM SIGSPATIAL 2026 (`doi:10.1145/3841645.3842950`, `arXiv:2606.08046`).
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--staging-dir', type=Path, required=True)
    parser.add_argument(
        '--csp-zip',
        type=Path,
        default=None,
        help='Optional local path to CSP model_dir.zip',
    )
    parser.add_argument(
        '--upload', action='store_true', help='Create repo and upload staged files'
    )
    parser.add_argument('--repo-id', default=ZOO_REPO_ID)
    args = parser.parse_args()

    staging: Path = args.staging_dir
    staging.mkdir(parents=True, exist_ok=True)
    api = HfApi()

    records: list[dict[str, Any]] = []

    satclip_specs = [
        ('satclip-vit16-l40', 'microsoft/SatCLIP-ViT16-L40', 'satclip-vit16-l40.ckpt'),
        ('satclip-vit16-l10', 'microsoft/SatCLIP-ViT16-L10', 'satclip-vit16-l10.ckpt'),
        (
            'satclip-resnet18-l40',
            'microsoft/SatCLIP-ResNet18-L40',
            'satclip-resnet18-l40.ckpt',
        ),
        (
            'satclip-resnet18-l10',
            'microsoft/SatCLIP-ResNet18-L10',
            'satclip-resnet18-l10.ckpt',
        ),
        (
            'satclip-resnet50-l40',
            'microsoft/SatCLIP-ResNet50-L40',
            'satclip-resnet50-l40.ckpt',
        ),
        (
            'satclip-resnet50-l10',
            'microsoft/SatCLIP-ResNet50-L10',
            'satclip-resnet50-l10.ckpt',
        ),
    ]
    for model_id, repo_id, fname in satclip_specs:
        print(f'Exporting {model_id} from {repo_id}...')
        records.append(export_satclip(api, staging, model_id, repo_id, fname))

    rff_specs = [
        (
            'geoclip',
            'MVRL/geoclip-location-encoder',
            'latlon',
            'MIT',
            'https://github.com/VicenteVivan/geo-clip',
        ),
        (
            'taxabind',
            'MVRL/ecogeo',
            'latlon',
            'Apache-2.0',
            'https://github.com/mvrl/TaxaBind',
        ),
        (
            'sled-s1-s2-ls',
            'geohai/sled-s1-s2-ls',
            'latlon',
            'Apache-2.0',
            'https://github.com/geohai/sled',
        ),
        (
            'sled-s2-ls',
            'geohai/sled-s2-ls',
            'latlon',
            'Apache-2.0',
            'https://github.com/geohai/sled',
        ),
    ]
    for model_id, repo_id, coord_order, lic, url in rff_specs:
        print(f'Exporting {model_id} from {repo_id}...')
        records.append(
            export_rff_safetensors(
                api,
                staging,
                model_id,
                repo_id,
                coord_order=coord_order,
                license_id=lic,
                upstream_url=url,
            )
        )

    print('Exporting sinr...')
    records.append(export_sinr(api, staging))

    print('Exporting climplicit...')
    records.append(export_climplicit(api, staging))

    print('Exporting gair...')
    records.append(export_gair(api, staging))

    print('Exporting tte...')
    records.append(export_tte(api, staging))

    print('Exporting terranova-spatial and terranova-st...')
    records.extend(export_terranova(api, staging))

    print('Exporting csp-inat and csp-fmow...')
    records.extend(export_csp(api, staging, csp_zip=args.csp_zip))

    osmgraphclip_specs = [
        (
            'osmgraphclip-ms-l40',
            'd-michail/OSMGraphCLIP-MS-L40',
            'osmgraphclip-ms-l40.ckpt',
            True,
        ),
        (
            'osmgraphclip-ms-l10',
            'd-michail/OSMGraphCLIP-MS-L10',
            'osmgraphclip-ms-l10.ckpt',
            True,
        ),
        (
            'osmgraphclip-a-l40',
            'd-michail/OSMGraphCLIP-A-L40',
            'osmgraphclip-a-l40.ckpt',
            False,
        ),
        (
            'osmgraphclip-a-l10',
            'd-michail/OSMGraphCLIP-A-L10',
            'osmgraphclip-a-l10.ckpt',
            False,
        ),
    ]
    for model_id, repo_id, fname, multiscale in osmgraphclip_specs:
        print(f'Exporting {model_id} from {repo_id}...')
        records.append(
            export_osmgraphclip(
                api, staging, model_id, repo_id, fname, multiscale=multiscale
            )
        )

    (staging / 'README.md').write_text(build_readme(records))
    manifest_path = staging / 'manifest.json'
    manifest_path.write_text(json.dumps(records, indent=2) + '\n')

    total_mb = sum(r['size_bytes'] for r in records) / (1024 * 1024)
    print(f'\nExported {len(records)} models ({total_mb:.2f} MiB total) into {staging}')
    for r in records:
        print(
            f'  {r["model_id"]:22s} | {r["size_bytes"] / (1024 * 1024):6.2f} MiB | '
            f'{r["num_tensors"]:3d} tensors | {r["dtypes"]} | {r["sha256"]}'
        )

    if args.upload:
        print(f'\nCreating/uploading {args.repo_id}...')
        api.create_repo(repo_id=args.repo_id, repo_type='model', exist_ok=True)
        api.upload_folder(
            repo_id=args.repo_id,
            repo_type='model',
            folder_path=str(staging),
            ignore_patterns=[
                'export_zoo.py',
                'manifest.json',
                '_csp_raw',
                '_csp_raw/*',
                '__pycache__/*',
            ],
            commit_message='Upload initial EarthINR location-encoder-only safetensors zoo (18 models)',
        )
        print(f'Uploaded cleanly to https://huggingface.co/{args.repo_id}')


if __name__ == '__main__':
    main()
