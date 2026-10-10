#pragma once
#include "common.cuh"

// Chunked (WY / UT-transform) prefill path for GATED_DELTA_NET with scalar gates, S_v = 128, one sequence.
// Strides in floats. q/k use head iq1 = h % neqk1. State layout [H][col][row] (as the recurrent kernels).
struct ggml_cuda_gdn_chunk_args {
    const float * q; const float * k; int64_t sq1, sq2; int neqk1;
    bool qkn; float q_eps, q_scale, k_eps, k_scale;     // in-kernel l2 norm of raw q/k (RMS_NORM -> SCALE)
    const float * v; int64_t sv1, sv2;
    const float * g; const float * beta; int64_t sb1, sb2;
    const float * s_in;                                 // initial state
    float * dst;                                        // attention output [t][H][128]
    float * state; int64_t slot_stride; int K;          // K > 1: snapshot slot j = state after token n-1-j; else final state
    int H; int n_tokens; float scale;
};

bool ggml_cuda_gdn_chunk_enabled(int64_t n_tokens);
void ggml_cuda_gdn_chunk(ggml_backend_cuda_context & ctx, const ggml_cuda_gdn_chunk_args & a);
