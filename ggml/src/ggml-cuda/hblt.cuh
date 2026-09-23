#pragma once
#include "common.cuh"

// bf16 x bf16 GEMM through hipBLASLt with a per-shape solution picked by timing (HIP builds only).
// W is row-major [M, K], X row-major [T, K], D row-major [T, M] (D = X * W^T). Returns false when not available.
bool ggml_cuda_hblt_gemm(ggml_backend_cuda_context & ctx, const uint16_t * W, const uint16_t * X, void * D, bool out_bf16,
                         int M, int K, int T, cudaStream_t stream);
// same with a bf16 result widened into an f32 D (hipBLASLt's f32-output solutions are far slower on gfx1151)
bool ggml_cuda_hblt_gemm_f32out(ggml_backend_cuda_context & ctx, const uint16_t * W, const uint16_t * X, float * D,
                                int M, int K, int T, cudaStream_t stream);
