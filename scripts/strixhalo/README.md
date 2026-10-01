# Qwen3.8-Flash-Next on Strix Halo (gfx1151, ROCm 10) — hb12

This branch is the build that serves Qwen3.8-Flash-Next UD-Q4_K_XL in production (release `hb12`, 2026-10-01).
It is upstream llama.cpp plus the `qwen4exp` architecture and HIP prefill/decode work for RDNA3.5.

## Measured (same machine, production config, MTP draft on)

| | 8K prefill | 32K prefill | 60K prefill | 200K prefill | code decode | 64K-deep decode |
|---|---|---|---|---|---|---|
| hb12 | 1333 | 1425–1440 | 1450 | 1400 | 40.5 | 37.5 |

## Build

```bash
cmake -S . -B build-opt -DGGML_HIP=ON -DGPU_TARGETS=gfx1151 -DCMAKE_BUILD_TYPE=Release -DLLAMA_BUILD_TESTS=ON \
      -DCMAKE_PREFIX_PATH=/home/zhangde/rocm-10.0 -DCMAKE_HIP_COMPILER=/home/zhangde/rocm-10.0/llvm/bin/clang++
cmake --build build-opt --target llama-server llama-bench test-backend-ops -j 32
```

ROCm 10 was used (`/home/zhangde/rocm-10.0`); `HSA_OVERRIDE_GFX_VERSION=11.5.1`.

## Run

`llama-server-wrapper.sh` is the release wrapper: it sets the runtime environment and falls back to the previous
release on a crash. `server-args.txt` holds the production command line, and `hblt.db` is the hipBLASLt
solution table (`GGML_CUDA_HBLT_DB`).

| Environment variable | Production value | Meaning |
|---|---|---|
| `GGML_CUDA_DISABLE_GRAPHS` | 1 | HIP graphs off (decode +2.4%) |
| `GGML_CUDA_MMB_MIN_T_ID` | 64 | MoE uses the mmb kernels from 64 tokens |
| `LLAMA_SPEC_DRAFT_UBATCH` | 8192 | MTP draft context micro-batch |
| `LLAMA_MTP_PREFILL_TAIL` | 4096 | Prompt ubatches ending more than 4096 tokens before the prompt end run the MTP head K/V-only |
| `LLAMA_ATTN_ROT_DISABLE` | 1 | No attention rotation |

## Kill switches for the fused paths (all on by default)

| Variable | Disables |
|---|---|
| `LLAMA_QSA5=0` | Per-query sparse attention (falls back to the 4-query union kernel) |
| `GGML_CUDA_SIGMUL_ADD=0`, `GGML_CUDA_ATTN_GATE=0`, `GGML_CUDA_RMS_SCALE=0` | Elementwise fusions |
| `GGML_CUDA_NORM_ROPE=0`, `GGML_CUDA_GDN_QKNORM=0` | q/k norm + (i)mrope fusion; GDN in-kernel q/k norm |
| `GGML_CUDA_MOE_SIGMERGE=0` | Shared-expert merge into the MoE reduction |
| `LLAMA_PLE_FUSED=0` | PLE custom kernels |
| `LLAMA_QSA_SETSTRIP=0` | In-place indexer strips |

## Tests

```bash
build-opt/bin/test-backend-ops test -b ROCm0 -o SIGMUL_ADD   # also ATTN_GATE RMS_SCALE NORM_ROPE GDN_QKNORM LIGHTNING_INDEXER
```
