#pragma once
#include "common.cuh"

// bf16 x bf16 -> f32 GEMM through hipBLASLt with a per-shape algorithm picked by timing (HIP builds only).
// W is row-major [M, K], X row-major [T, K], D row-major [T, M] (D = X * W^T). Returns false when not available.
bool ggml_cuda_hblt_gemm_bf16(ggml_backend_cuda_context & ctx, const uint16_t * W, const uint16_t * X, float * D,
                              int M, int K, int T, cudaStream_t stream);
