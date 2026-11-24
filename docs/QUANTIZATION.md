# ONNX Quantization for Geographic INRs

Exporting geographic Implicit Neural Representations (INRs) to ONNX enables CPU and browser WebGPU inference without PyTorch, but full FP32 checkpoints can exceed browser memory limits or slow downloads (`432.29 MiB` for `mind`, `54.16 MiB` for `sled-s1-s2-ls`, `36.20 MiB` for `geoclip`). Unlike standard vision or language backbones, coordinate networks are sensitive to quantization in their coordinate projections, trigonometric activations, and high-frequency first layers (`w0 = 30`). This note reports an empirical benchmark comparing five precision formats across five representative EarthINR encoders and documents how to reproduce the verified exports with [`scripts/export_onnx.py`](../scripts/export_onnx.py).

## Benchmark Setup

We evaluated five precision variants on five encoders spanning three architectural families—residual SIREN (`mind`), multi-scale RFF with ReLU (`sled-s1-s2-ls`, `geoclip`), 16-layer H-SIREN (`climplicit`), and spherical-harmonic SIREN (`satclip-vit16-l40`):

- **`fp32`**: Unmodified dynamic-batch opset 18 ONNX baseline (`mind` exported as `(lon, lat) -> (B, 3072)` with PyTorch cosine parity `> 0.999999999`).
- **`fp16_mixed`**: Hidden linear layers (`Gemm`) wrapped in `Cast(FP32 -> FP16) -> Gemm(FP16) -> Cast(FP16 -> FP32)` with FP16 weights and biases, leaving positional encodings, trigonometric operations, and first `w0 = 30` SIREN/H-SIREN layers strictly in FP32.
- **`int8_nbits`**: Weight-only 8-bit block quantization (`com.microsoft:MatMulNBits`, `bits=8`, asymmetric, `block_size=128`; plus `block_size=32` on `mind` WebGPU).
- **`int8_dyn`**: Dynamic activation and weight quantization via `onnxruntime.quantization.quantize_dynamic` (`QuantType.QInt8`), excluding the same protected layers.
- **`int4_nbits`**: Weight-only 4-bit block quantization (`com.microsoft:MatMulNBits`, `bits=4`, asymmetric, `block_size=128`, plus `block_size=32` on `mind`).

All non-excluded `Gemm(A, B, C, transB=1)` nodes were rewritten to `MatMul(A, B^T) + Add(C)` before `MatMulNBitsQuantizer` because `MatMulNBitsQuantizer` operates only on `MatMul` nodes. Identical transposed and quantized initializers were deduplicated so shared-weight graphs (such as `climplicit`'s four monthly passes) do not duplicate weights. Protected layers kept in FP32 across all quantized variants were coordinate projections (`EqualEarth`, RFF `MatMul`, spherical harmonics) and the first `w0 = 30` SIREN/H-SIREN layer (`first.linear` in `mind`, `layers.0` in `climplicit`, and `layers.0` in `satclip-vit16-l40`, with an ablation on `satclip-vit16-l40` that also quantizes `layers.0`).

Metrics were collected under three protocols:

- **Numerical fidelity and downstream probing**: `satclip-air-temp` (`N = 3,076`), measuring cosine similarity and max absolute difference against FP32, median CPU latency over 3 runs after 1 warmup on ONNX Runtime 1.32.0, and 5-fold Ridge $R^2$ (`earthinr.probes.linear_probe_score`, `random` and `spatial 10°` splits, `seed=0`).
- **Spatial scale and geometry**: `worldclim-bio1` land points (`N = 18,068`), 48 log-spaced geodesic lags from `1 km` to `8,000 km`, `pairs_per_lag = 1024`, `seed=0`, using identical anchor-offset pairs across all variants so `ΔHS (%)` has zero Monte Carlo noise. `Nugget 1 km` reports $\gamma(1\text{ km})$ and its percentage of the variogram sill; `PR` is the covariance participation ratio on base land embeddings.
- **Browser WebGPU and WASM**: Headless Chrome on Apple Silicon Metal WebGPU (`shader-f16=true`) over `N = 3,076` points, comparing `onnxruntime-web@1.31.1` against `1.20.1`.

## Results: MIND (64-d, 256-d, and 3072-d Prefixes)

| Slice | Variant | Size (MiB) | CPU (ms) | `min_cos` | `mean_cos` | `max_diff` | Rand $R^2$ (`Δ`) | Spat 10° $R^2$ (`Δ`) | Half-Sill (km) | `ΔHS (%)` | Nugget 1 km (`% sill`) | `PR` |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **64-d** | `fp32` | 432.29 | 358.8 | 1.00000 | 1.00000 | 0.0000 | 0.9450 (+0.0000) | 0.9429 (+0.0000) | 1006.7 | +0.00% | 0.0241 (0.00%) | 21.30 |
| **64-d** | `fp16_mixed` | 216.20 | 1524.3 | 0.99999 | 1.00000 | 0.0272 | 0.9450 (-0.0000) | 0.9429 (-0.0000) | 1006.7 | +0.00% | 0.0251 (0.00%) | 21.30 |
| **64-d** | `int8_nbits` | 112.49 | 387.1 | 0.99914 | 0.99995 | 0.3391 | 0.9449 (-0.0001) | 0.9427 (-0.0002) | 1006.8 | +0.01% | 0.0241 (0.00%) | 21.30 |
| **64-d** | `int8_dyn` | 108.26 | 277.6 | 0.85131 | 0.99419 | 6.1644 | 0.9439 (-0.0011) | 0.9417 (-0.0012) | 1005.8 | -0.09% | 0.5162 (0.05%) | 21.30 |
| **64-d** | `int4_nbits (bs=128)` | 58.07 | 955.1 | 0.81898 | 0.98734 | 4.3612 | 0.9391 (-0.0059) | 0.9358 (-0.0071) | 1005.4 | -0.13% | 0.0218 (0.00%) | 21.62 |
| **64-d** | `int4_nbits (bs=32)` | 69.46 | 980.7 | 0.83376 | 0.99069 | 3.6499 | 0.9424 (-0.0026) | 0.9402 (-0.0027) | 1002.0 | -0.47% | 0.0223 (0.00%) | 21.49 |
| **256-d** | `fp32` | 432.29 | 358.8 | 1.00000 | 1.00000 | 0.0000 | 0.9474 (+0.0000) | 0.9338 (+0.0000) | 603.3 | +0.00% | 0.1309 (0.00%) | 47.22 |
| **256-d** | `fp16_mixed` | 216.20 | 1524.3 | 0.99999 | 1.00000 | 0.0395 | 0.9474 (-0.0000) | 0.9338 (-0.0000) | 603.3 | +0.00% | 0.1370 (0.00%) | 47.22 |
| **256-d** | `int8_nbits` | 112.49 | 387.1 | 0.99887 | 0.99988 | 0.5263 | 0.9472 (-0.0002) | 0.9335 (-0.0003) | 603.2 | -0.01% | 0.1309 (0.00%) | 47.22 |
| **256-d** | `int8_dyn` | 108.26 | 277.6 | 0.84408 | 0.98788 | 6.1644 | 0.9458 (-0.0016) | 0.9343 (+0.0004) | 601.8 | -0.25% | 3.221 (0.12%) | 47.24 |
| **256-d** | `int4_nbits (bs=128)` | 58.07 | 955.1 | 0.81564 | 0.97001 | 5.5993 | 0.9405 (-0.0069) | 0.9291 (-0.0047) | 592.0 | -1.88% | 0.1241 (0.00%) | 48.11 |
| **256-d** | `int4_nbits (bs=32)` | 69.46 | 980.7 | 0.85111 | 0.97750 | 4.5035 | 0.9441 (-0.0033) | 0.9320 (-0.0019) | 593.7 | -1.58% | 0.1254 (0.00%) | 47.82 |
| **3072-d** | `fp32` | 432.29 | 358.8 | 1.00000 | 1.00000 | 0.0000 | 0.9271 (+0.0000) | 0.8771 (+0.0000) | 200.6 | +0.00% | 2.312 (0.01%) | 426.58 |
| **3072-d** | `fp16_mixed` | 216.20 | 1524.3 | 0.99999 | 1.00000 | 0.0517 | 0.9271 (+0.0000) | 0.8771 (-0.0000) | 200.6 | +0.00% | 2.435 (0.01%) | 426.58 |
| **3072-d** | `int8_nbits` | 112.49 | 387.1 | 0.99844 | 0.99965 | 0.7895 | 0.9272 (+0.0001) | 0.8771 (+0.0000) | 200.5 | -0.02% | 2.313 (0.01%) | 426.63 |
| **3072-d** | `int8_dyn` | 108.26 | 277.6 | 0.74856 | 0.96630 | 6.8546 | 0.9249 (-0.0022) | 0.8790 (+0.0019) | 201.1 | +0.27% | 59.52 (0.30%) | 427.20 |
| **3072-d** | `int4_nbits (bs=128)` | 58.07 | 955.1 | 0.80093 | 0.91984 | 6.8168 | 0.9217 (-0.0053) | 0.8697 (-0.0074) | 197.8 | -1.37% | 2.271 (0.01%) | 438.88 |
| **3072-d** | `int4_nbits (bs=32)` | 69.46 | 980.7 | 0.83112 | 0.93875 | 5.9777 | 0.9222 (-0.0048) | 0.8703 (-0.0068) | 198.7 | -0.93% | 2.276 (0.01%) | 434.90 |

## Results: Four Zoo Models

| Model | Variant | Size (MiB) | CPU (ms) | `min_cos` | `mean_cos` | `max_diff` | Rand $R^2$ (`Δ`) | Spat 10° $R^2$ (`Δ`) | Half-Sill (km) | `ΔHS (%)` | Nugget 1 km (`% sill`) | `PR` |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **`sled-s1-s2-ls`** | `fp32` | 54.16 | 50.9 | 1.00000 | 1.00000 | 0.0000 | 0.7226 (+0.0000) | 0.4347 (+0.0000) | 543.1 | +0.00% | 0.00110 (0.00%) | 14.57 |
| (RFF+ReLU) | `fp16_mixed` | 27.13 | 92.9 | 1.00000 | 1.00000 | 0.0018 | 0.7226 (+0.0001) | 0.4347 (-0.0000) | 543.1 | -0.00% | 0.00112 (0.00%) | 14.57 |
| | `int8_nbits` | 14.18 | 57.8 | 0.99992 | 0.99998 | 0.0230 | 0.7229 (+0.0003) | 0.4375 (+0.0028) | 543.1 | +0.01% | 0.00111 (0.00%) | 14.56 |
| | `int8_dyn` | 13.65 | 32.1 | 0.99851 | 0.99967 | 0.0844 | 0.7226 (+0.0000) | 0.4364 (+0.0017) | 544.5 | +0.25% | 0.00365 (0.00%) | 14.57 |
| | `int4_nbits` | 7.37 | 117.1 | 0.98033 | 0.99356 | 0.4177 | 0.7252 (+0.0026) | 0.4231 (-0.0117) | 546.7 | +0.66% | 0.00107 (0.00%) | 14.29 |
| **`geoclip`** | `fp32` | 36.20 | 33.1 | 1.00000 | 1.00000 | 0.0000 | 0.9285 (+0.0000) | 0.9153 (+0.0000) | 135.9 | +0.00% | 0.000778 (0.84%) | 57.54 |
| (RFF+ReLU) | `fp16_mixed` | 18.15 | 63.1 | 1.00000 | 1.00000 | 0.0001 | 0.9285 (-0.0000) | 0.9153 (-0.0000) | 135.9 | +0.00% | 0.000778 (0.84%) | 57.54 |
| | `int8_nbits` | 9.53 | 32.7 | 0.99978 | 0.99991 | 0.0009 | 0.9284 (-0.0001) | 0.9151 (-0.0001) | 135.8 | -0.05% | 0.000778 (0.84%) | 57.55 |
| | `int8_dyn` | 9.18 | 26.3 | 0.99927 | 0.99969 | 0.0016 | 0.9285 (-0.0000) | 0.9152 (-0.0000) | 135.7 | -0.08% | 0.000786 (0.85%) | 57.59 |
| | `int4_nbits` | 4.99 | 77.5 | 0.93150 | 0.97510 | 0.0161 | 0.9259 (-0.0026) | 0.9128 (-0.0025) | 132.2 | -2.67% | 0.000809 (0.85%) | 59.06 |
| **`climplicit`** | `fp32` | 16.17 | 164.1 | 1.00000 | 1.00000 | 0.0000 | 0.9858 (+0.0000) | 0.9822 (+0.0000) | 578.6 | +0.00% | 0.0286 (0.00%) | 7.88 |
| (16L H-SIREN) | `fp16_mixed` | 8.28 | 181.6 | 0.99999 | 1.00000 | 0.0208 | 0.9858 (+0.0000) | 0.9822 (+0.0000) | 578.6 | -0.00% | 0.0299 (0.00%) | 7.88 |
| | `int8_nbits` | 4.58 | 149.9 | 0.99770 | 0.99971 | 0.3994 | 0.9857 (-0.0000) | 0.9821 (-0.0001) | 578.1 | -0.09% | 0.0288 (0.00%) | 7.88 |
| | `int8_dyn` | 4.45 | 147.1 | 0.42069 | 0.96401 | 6.2067 | 0.9797 (-0.0061) | 0.9768 (-0.0054) | 568.5 | -1.74% | 1.074 (0.18%) | 7.99 |
| | `int4_nbits` | 2.62 | 226.3 | 0.73222 | 0.92878 | 3.3608 | 0.9807 (-0.0051) | 0.9765 (-0.0057) | 563.7 | -2.58% | 0.0266 (0.00%) | 8.34 |
| **`satclip-vit16-l40`** | `fp32` | 6.26 | 28.5 | 1.00000 | 1.00000 | 0.0000 | 0.8329 (+0.0000) | 0.7603 (+0.0000) | 446.7 | +0.00% | 0.00795 (0.00%) | 21.39 |
| (SH+SIREN) | `fp16_mixed` (L0 FP32) | 5.51 | 23.1 | 1.00000 | 1.00000 | 0.0046 | 0.8329 (+0.0000) | 0.7603 (-0.0000) | 446.7 | +0.00% | 0.00805 (0.00%) | 21.39 |
| | `fp16_mixed_incl_l0` | 3.94 | 26.7 | 1.00000 | 1.00000 | 0.0051 | 0.8329 (-0.0000) | 0.7603 (-0.0000) | 446.7 | +0.00% | 0.00810 (0.00%) | 21.39 |
| | `int8_nbits` (L0 FP32) | 5.15 | 28.1 | 1.00000 | 1.00000 | 0.0276 | 0.8328 (-0.0001) | 0.7603 (-0.0001) | 446.7 | +0.01% | 0.00795 (0.00%) | 21.39 |
| | `int8_nbits_incl_l0` | 2.87 | 27.0 | 0.99996 | 0.99999 | 0.0517 | 0.8328 (-0.0001) | 0.7603 (-0.0001) | 446.7 | +0.01% | 0.00795 (0.00%) | 21.39 |
| | `int8_dyn` | 2.79 | 26.0 | 0.99935 | 0.99995 | 0.2156 | 0.8325 (-0.0003) | 0.7595 (-0.0008) | 446.7 | +0.01% | 0.0341 (0.00%) | 21.39 |
| | `int4_nbits` (L0 FP32) | 4.96 | 26.2 | 0.99889 | 0.99927 | 0.4489 | 0.8316 (-0.0012) | 0.7591 (-0.0012) | 446.7 | +0.01% | 0.00796 (0.00%) | 21.37 |
| | `int4_nbits_incl_l0` | 2.27 | 30.6 | 0.99004 | 0.99808 | 0.8643 | 0.8292 (-0.0036) | 0.7555 (-0.0048) | 445.8 | -0.19% | 0.00796 (0.00%) | 21.38 |

## Results: Chrome WebGPU and WASM (`N = 3,076`)

| Model / Variant | Size (MiB) | `ort-web@1.31.1` WebGPU | `ort-web@1.31.1` WASM | `ort-web@1.20.1` WebGPU | `ort-web@1.20.1` WASM | WebGPU `cos` vs CPU FP32 |
|---|---:|---|---|---|---|---|
| `geoclip / fp32` | 36.20 | **20.3 ms** (OK) | 1131.4 ms (OK) | 167.4 ms (OK) | 1153.4 ms (OK) | 1.000000 |
| `geoclip / fp16_mixed` | 18.15 | **15.6 ms** (OK) | 1137.3 ms (OK) | 148.0 ms (OK) | 1106.4 ms (OK) | 0.999987 |
| `geoclip / int8_nbits` | 9.53 | **84.4 ms** (OK) | 1086.8 ms (OK) | FAIL (`bits=8` unsupported) | FAIL | 0.999915 |
| `geoclip / int8_dyn` | 9.18 | 1140.0 ms (WASM fallback) | 1124.7 ms (OK) | 1149.5 ms (WASM fallback) | 1150.3 ms (OK) | 0.999572 |
| `geoclip / int4_nbits` | 4.99 | **91.7 ms** (OK) | 1096.6 ms (OK) | 172.6 ms (**CORRUPT `cos=-0.154`**) | 1118.4 ms (OK) | 0.967955 (`1.31.1`) |
| `mind / fp32` | 432.29 | **FAIL (Chrome OOM)** | **FAIL (OOM)** | **FAIL (Chrome OOM)** | **FAIL (OOM)** | — |
| `mind / fp16_mixed` | 216.20 | **132.1 ms** (OK) | FAIL (`B=3076` OOM) | 1911.2 ms (OK) | FAIL (OOM) | 64d `0.99977` / 3072d `0.99862` |
| `mind / int8_nbits (bs=128)` | 112.49 | **836.1 ms** (OK) | FAIL (`B=3076`/`256` OOM) | FAIL (`bits=8` unsupported) | FAIL | 64d `0.99987` / 3072d `0.99918` |
| `mind / int8_nbits (bs=32)` | 125.15 | 2967.4 ms (OK) | — | — | — | 64d `0.99987` / 3072d `0.99940` |
| `mind / int8_dyn` | 108.26 | FAIL (WASM fallback OOM) | FAIL (OOM) | FAIL (WASM fallback OOM) | FAIL (OOM) | — |
| `mind / int4_nbits (bs=128)` | 58.07 | **788.2 ms** (OK) | FAIL (`B=3076` OOM) | 945.2 ms (OK) | FAIL (OOM) | 64d `0.96006` / 3072d `0.85650` |
| `mind / int4_nbits (bs=32)` | 69.46 | **243.2 ms** (OK) | **OK `1313.9 ms` (`B=256`)** | 649.8 ms (OK) | FAIL (`B=3076` OOM) | 64d `0.97471` / 3072d `0.87081` |

## Findings

### Weight-only `MatMulNBits` vs. dynamic activation quantization on SIREN nuggets

Dynamic INT8 (`int8_dyn`) quantizes intermediate activations per tensor into 8-bit integer grids before each linear layer. Because periodic SIREN and H-SIREN layers map small activation perturbations through `sin(x)` across many residual or sequential layers, step discontinuities in the activation grid inject local jitter between points separated by 1 km. On `mind`, `int8_dyn` increases the 1 km semivariogram nugget $\gamma(1\text{ km})$ by 21.4× on the 64-d slice (`0.0241` to `0.5162`), 24.6× on 256-d (`0.1309` to `3.221`), and 25.7× on 3072-d (`2.312` to `59.52`), while pulling worst-case cosine similarity down to `0.74856`. On 16-layer `climplicit`, the 1 km nugget increases by 37.5× (`0.0286` to `1.074`), `min_cos` falls to `0.42069`, and spatial 10° $R^2$ drops by `-0.0054`. Even shallow or ReLU models show smaller nugget increases under `int8_dyn` (`4.3×` on `satclip-vit16-l40`, `3.3×` on `sled-s1-s2-ls`).

Weight-only `MatMulNBits` avoids activation discretization by keeping activations in FP32 and dequantizing weights per block during matrix multiplication. Across all five models, `int8_nbits` reduces file size by ~3.8× while preserving the 1 km nugget (`2.313` vs. `2.312` on `mind` 3072-d; `0.0288` vs. `0.0286` on `climplicit`; `0.00795` vs. `0.00795` on `satclip-vit16-l40`), holding `|ΔHS| <= 0.09%`, keeping `mean_cos >= 0.99965`, and changing spatial 10° $R^2$ by at most `-0.0003` (and `+0.0028` on `sled-s1-s2-ls`). Weight-only `int4_nbits` also preserves the 1 km nugget without spiking, though 4-bit weight noise reduces cosine similarity on deep periodic models (`mean_cos = 0.92878` on `climplicit` and `0.91984` on `mind` 3072-d at `bs=128`, improving to `0.93875` at `bs=32`) and shifts spatial 10° $R^2$ by `-0.0012` to `-0.0117`.

Dynamic INT8 also lacks a WebGPU kernel in `onnxruntime-web`, so `geoclip / int8_dyn` falls back to WASM (`1140.0 ms` vs. `84.4 ms` for `int8_nbits` on WebGPU) and `mind / int8_dyn` runs out of memory in Chrome.

### Keeping positional encodings and first SIREN layers in FP32

Coordinate projections (`EqualEarth`, RFF `MatMul`, spherical harmonics), trigonometric activations (`Sin`, `Cos`, `Sinh`), and the first `w0 = 30` SIREN/H-SIREN layer amplify rounding error before downstream layers can average it out. Using `onnxconverter_common.float16.convert_float_to_float16` with `op_block_list` and `node_block_list` did not isolate those operations reliably because the converter still inserts FP16 casts on tensors passed between blocked nodes; on our benchmark, that behavior degraded `geoclip` to `min_cos = 0.4743` (`mean_cos = 0.9158`) and `mind` 3072-d to `min_cos = 0.9705` (`max_diff = 2.29`).

Wrapping only non-excluded hidden `Gemm` nodes with explicit FP32-to-FP16 input casts and FP16-to-FP32 output casts leaves every positional encoding, trigonometric op, and first-layer SIREN weight untouched in FP32. That hidden-only `fp16_mixed` graph cuts model size in half (`432.29` to `216.20 MiB` on `mind`, `36.20` to `18.15 MiB` on `geoclip`) while maintaining `min_cos >= 0.99999`, `mean_cos = 1.00000`, `0.00%` half-sill drift, and `0.0000` $R^2$ change on every tested model. On CPU, however, `fp16_mixed` is slower than FP32 (`1524.3 ms` vs. `358.8 ms` on `mind`) because CPU execution lacks native FP16 `Gemm` throughput and pays cast overhead; its speed advantage appears on WebGPU with `shader-f16`.

The `satclip-vit16-l40` ablation illustrates the size-versus-precision tradeoff for the first `w0 = 30` layer (`layers.0`, shape `512 x 1600`), which holds most of that model's weights. Quantizing `layers.0` under `int8_nbits_incl_l0` reduces file size from `5.15 MiB` to `2.87 MiB` with `mean_cos = 0.99999` and `ΔR^2 = -0.0001`, whereas quantizing `layers.0` at 4 bits (`int4_nbits_incl_l0`) widens the spatial 10° $R^2$ drop from `-0.0012` to `-0.0048`. Protecting `layers.0` by default preserves fidelity across all bit widths.

### Browser WebGPU runtime (`onnxruntime-web@1.31.1`) and block size

Upgrading `onnxruntime-web` from `1.20.1` to `1.31.1` yields 8–14× lower WebGPU latency on FP32 and Mixed-FP16 graphs: `geoclip / fp32` drops from `167.4 ms` to `20.3 ms` (8.2×), `geoclip / fp16_mixed` drops from `148.0 ms` to `15.6 ms` (9.5×), and `mind / fp16_mixed` drops from `1911.2 ms` to `132.1 ms` (14.5×). Version `1.31.1` also resolves two blockers in `1.20.1`: it fixes 4-bit WebGPU output corruption (`geoclip / int4_nbits` yielded `cos = -0.154` on `1.20.1` versus `0.967955` on `1.31.1`) and adds WebGPU and WASM support for 8-bit `MatMulNBits` (`bits=8`).

Because `mind / fp32` (`432.29 MiB`) triggers Chrome out-of-memory failures on both WebGPU and WASM, quantization is required to run MIND in a browser. On `onnxruntime-web@1.31.1`, `mind / fp16_mixed` (`216.20 MiB`) is fastest on WebGPU at `132.1 ms` (`cos = 0.99977` at 64-d), while `mind / int8_nbits (bs=128)` (`112.49 MiB`) halves the download size again at `836.1 ms` (`cos = 0.99987` at 64-d). For 4-bit browser deployment, `block_size=32` (`69.46 MiB`) is strongly preferable to `block_size=128` (`58.07 MiB`): it runs 3.2× faster on WebGPU (`243.2 ms` vs. `788.2 ms`), improves 64-d `mean_cos` from `0.98734` to `0.99069` and spatial 10° $R^2$ drift from `-0.0071` to `-0.0027`, and is the only MIND variant that also runs under WASM (`1313.9 ms` at `B = 256`). For 8-bit `MatMulNBits`, the reverse holds on WebGPU: `bs=128` (`836.1 ms`) is 3.5× faster than `bs=32` (`2967.4 ms`), so `bs=128` remains the default.

## Practical Defaults and Export Usage

- **CPU deployment (4× smaller, near-lossless)**: `--quantize int8` (`block_size=128`).
- **WebGPU deployment (2× smaller, fastest GPU latency, lossless accuracy)**: `--quantize fp16` with `onnxruntime-web>=1.31.1`.
- **Bandwidth- or memory-constrained WebGPU / WASM fallback (6–7× smaller)**: `--quantize int4 --block-size 32` with `onnxruntime-web>=1.31.1`.

[`scripts/export_onnx.py`](../scripts/export_onnx.py) implements the hidden-only FP16 `Gemm` wrapper and the `Gemm -> MatMul + Add` rewrite, initializer deduplication, and `MatMulNBitsQuantizer` pipeline with automatic protection of positional encodings and first `w0 = 30` SIREN/H-SIREN layers:

```bash
# Mixed FP16 for WebGPU (writes staging/mind/model_fp16.onnx)
uv run --with onnx --with onnxruntime --with onnxscript \
  python scripts/export_onnx.py --staging-dir /tmp/onnx_stage --models mind --quantize fp16

# Weight-only INT8 (block_size=128, writes staging/<model_id>/model_int8.onnx)
uv run --with onnx --with onnxruntime --with onnxscript \
  python scripts/export_onnx.py --staging-dir /tmp/onnx_stage --models mind geoclip climplicit --quantize int8

# Weight-only INT4 tuned for browser WebGPU & WASM (writes staging/mind/model_int4_bs32.onnx)
uv run --with onnx --with onnxruntime --with onnxscript \
  python scripts/export_onnx.py --staging-dir /tmp/onnx_stage --models mind --quantize int4 --block-size 32
```
