#include "common.cuh"
#include "ggml.h"

// fused-kernel recurrent-state output; strides in elements (per-seq stride is always D, set in-kernel)
struct ggml_cuda_gated_delta_net_fused_cache {
    float * data;        // rollback slot 0
    int64_t slot_stride; // between rollback slots (0 when K==1)
};

void ggml_cuda_op_gated_delta_net(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

// same op, but writes the snapshot(s) into the cache instead of dst (see ggml_cuda_try_gdn_cache_fusion)
void ggml_cuda_op_gated_delta_net_fused_cache(ggml_backend_cuda_context & ctx, ggml_tensor * dst,
                                              ggml_cuda_gated_delta_net_fused_cache cache);

// raw q/k views and the RMS_NORM -> SCALE parameters for a GATED_DELTA_NET whose q/k norms are applied in-kernel
struct ggml_cuda_gdn_qknorm {
    const float * q = nullptr; const float * k = nullptr;
    int64_t sq2 = 0, sq3 = 0;                 // token / sequence strides of the raw views (floats); head stride unchanged
    float q_eps = 0.0f, q_scale = 1.0f, k_eps = 0.0f, k_scale = 1.0f;
};
bool ggml_cuda_gdn_qknorm_eligible(const ggml_tensor * gdn);
void ggml_cuda_gdn_qknorm_put(const ggml_tensor * gdn, const ggml_cuda_gdn_qknorm & e);
bool ggml_cuda_gdn_qknorm_take(const ggml_tensor * gdn, ggml_cuda_gdn_qknorm & e);
