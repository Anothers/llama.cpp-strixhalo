#pragma once
#include "common.cuh"

// out = a + b * sigmoid(g), with g either the shape of b or one value per row (broadcast over ne0).
// Replaces the UNARY(SIGMOID) -> MUL -> ADD chain of a gated residual (Qwen3-Next shared expert:
// ffn_out = moe_out + ffn_shexp * sigmoid(shared_gate)): three kernels and two extra passes over [n, T].
struct ggml_cuda_sigmul_add_args {
    const ggml_tensor * a   = nullptr;
    const ggml_tensor * b   = nullptr;
    const ggml_tensor * g   = nullptr;
    ggml_tensor *       dst = nullptr;
};
// returns the number of graph nodes covered (3) or 0
int  ggml_cuda_match_sigmul_add(const ggml_cgraph * cgraph, int i, ggml_cuda_sigmul_add_args & args);
void ggml_cuda_op_sigmul_add(ggml_backend_cuda_context & ctx, const ggml_cuda_sigmul_add_args & args);

// out = attn * sigmoid(gate), where gate is a strided view (Qwen3-Next's gate half of the fused Q projection) that
// the graph first copies contiguous. Replaces CONT(view) -> UNARY(SIGMOID) -> MUL: the gate is read in place, and the
// product is written as BF16 when the only readers are MMB matmuls (see ggml_cuda_mmb_is_bf16_only).
struct ggml_cuda_attn_gate_args {
    const ggml_tensor * gate_view = nullptr; // src of the CONT: [D, H, T] strided
    const ggml_tensor * attn      = nullptr; // contiguous, same element count
    ggml_tensor *       dst       = nullptr; // the MUL node
};
int  ggml_cuda_match_attn_gate(const ggml_cgraph * cgraph, int i, ggml_cuda_attn_gate_args & args);
void ggml_cuda_op_attn_gate(ggml_backend_cuda_context & ctx, const ggml_cuda_attn_gate_args & args);
bool ggml_cuda_attn_gate_enabled();

// out = rms_norm(x, eps) * s, a warp per row (the l2 norm of GDN q/k heads is built as RMS_NORM -> SCALE on short
// rows, where the generic rms_norm kernel launches a 256-thread block per 128-element row).
struct ggml_cuda_rms_scale_args {
    const ggml_tensor * x     = nullptr; // RMS_NORM src, rows contiguous (nb0 == 4), any row strides
    float               eps   = 0.0f;
    float               scale = 1.0f;
    ggml_tensor *       dst   = nullptr; // the SCALE node
};
int  ggml_cuda_match_rms_scale(const ggml_cgraph * cgraph, int i, ggml_cuda_rms_scale_args & args);
void ggml_cuda_op_rms_scale(ggml_backend_cuda_context & ctx, const ggml_cuda_rms_scale_args & args);
