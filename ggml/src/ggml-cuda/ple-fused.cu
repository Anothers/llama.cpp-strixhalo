#include "ple-fused.cuh"

// qwen4exp PLE mix as three GGML_OP_CUSTOM nodes (tagged through the custom op's userdata, see qwen4exp.cpp):
//   GATE: gate[h,t] = sigmoid(sgn(s) * sqrt(max(|s|, 1e-6))),  s = <rmsnorm(key)*w_k, rmsnorm(hidden)*w_q>_h / sqrt(E)
//   NORM: out[e,h,t] = rmsnorm_e(value[e,t] * gate[h,t]) * w_c[h*E + e]
//   OUT : out[e,h,t] = hidden[e,h,t] + (value[e,t] * gate[h,t] + conv[e,h,t])
// The graph built these from ~25 elementwise/reduce nodes over [E*hc, T] tensors (each a full pass).

struct ple_custom_params { void * fun; int n_tasks; void * userdata; float eps; };

static float ple_eps(const ggml_tensor * t) {
    const ple_custom_params * p = (const ple_custom_params *) t->op_params;
    return p->eps;
}

intptr_t ggml_cuda_ple_tag(const ggml_tensor * t) {
    if (t->op != GGML_OP_CUSTOM) return 0;
    const ple_custom_params * p = (const ple_custom_params *) t->op_params;
    const intptr_t tag = (intptr_t) p->userdata;
    return (tag == GGML_PLE_TAG_GATE || tag == GGML_PLE_TAG_NORM || tag == GGML_PLE_TAG_OUT) ? tag : 0;
}

template <int BS>
static __device__ __forceinline__ float ple_block_sum(float v, float * sh) {
    v = warp_reduce_sum(v);
    const int w = threadIdx.x / WARP_SIZE, l = threadIdx.x % WARP_SIZE;
    if (l == 0) sh[w] = v;
    __syncthreads();
    v = l < BS / WARP_SIZE ? sh[l] : 0.0f;
    v = warp_reduce_sum(v);
    __syncthreads();
    return v;
}

// one block per (h, t)
static __global__ void __launch_bounds__(256) ple_gate_kernel(const float * __restrict__ key, const float * __restrict__ hid,
        const float * __restrict__ wk, const float * __restrict__ wq, float * __restrict__ gate, const int E, const int hc, const float eps) {
    __shared__ float sh[32];
    const int h = blockIdx.x, t = blockIdx.y;
    const float * k = key + ((size_t) t * hc + h) * E;
    const float * q = hid + ((size_t) t * hc + h) * E;
    const float * a = wk + (size_t) h * E, * b = wq + (size_t) h * E;
    float kk = 0.f, qq = 0.f, kq = 0.f;
    for (int e = threadIdx.x; e < E; e += 256) {
        const float x = k[e], y = q[e];
        kk += x * x; qq += y * y; kq += (x * a[e]) * (y * b[e]);
    }
    kk = ple_block_sum<256>(kk, sh); qq = ple_block_sum<256>(qq, sh); kq = ple_block_sum<256>(kq, sh);
    if (threadIdx.x == 0) {
        const float s = kq * rsqrtf(kk / E + eps) * rsqrtf(qq / E + eps) * (1.0f / sqrtf((float) E));
        const float mag = sqrtf(fmaxf(fabsf(s), 1e-6f));
        const float sg = s > 0.f ? 1.f : (s < 0.f ? -1.f : 0.f);
        gate[(size_t) t * hc + h] = 1.0f / (1.0f + expf(-(sg * mag)));
    }
}

// one block per (h, t)
static __global__ void __launch_bounds__(256) ple_norm_kernel(const float * __restrict__ val, const float * __restrict__ gate,
        const float * __restrict__ wc, float * __restrict__ out, const int E, const int hc, const float eps) {
    __shared__ float sh[32];
    const int h = blockIdx.x, t = blockIdx.y;
    const float g = gate[(size_t) t * hc + h];
    const float * v = val + (size_t) t * E;
    float ss = 0.f;
    for (int e = threadIdx.x; e < E; e += 256) { const float x = v[e] * g; ss += x * x; }
    ss = ple_block_sum<256>(ss, sh);
    const float r = rsqrtf(ss / E + eps);
    float * o = out + ((size_t) t * hc + h) * E;
    const float * w = wc + (size_t) h * E;
    for (int e = threadIdx.x; e < E; e += 256) o[e] = ((v[e] * g) * r) * w[e];
}

static __global__ void ple_out_kernel(const float4 * __restrict__ hid, const float * __restrict__ val, const float * __restrict__ gate,
        const float4 * __restrict__ conv, float4 * __restrict__ out, const int E4, const int hc, const int64_t n4) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n4) return;
    const int64_t row = i / E4; const int e4 = (int) (i - row * E4);   // row = t*hc + h
    const int64_t t = row / hc;
    const float g = gate[row];
    const float4 v = ((const float4 *) (val + t * (int64_t) E4 * 4))[e4];
    const float4 c = conv[i], x = hid[i];
    out[i] = make_float4(x.x + (v.x * g + c.x), x.y + (v.y * g + c.y), x.z + (v.z * g + c.z), x.w + (v.w * g + c.w));
}

bool ggml_cuda_ple_supported(const ggml_tensor * t) {
    const intptr_t tag = ggml_cuda_ple_tag(t);
    if (!tag) return false;
    for (int i = 0; i < GGML_MAX_SRC && t->src[i]; ++i) {
        if (t->src[i]->type != GGML_TYPE_F32 || !ggml_is_contiguous(t->src[i])) return false;
    }
    return t->type == GGML_TYPE_F32 && ggml_is_contiguous(t);
}

void ggml_cuda_op_ple(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const intptr_t tag = ggml_cuda_ple_tag(dst);
    const float eps = ple_eps(dst);
    cudaStream_t st = ctx.stream();
    if (tag == GGML_PLE_TAG_GATE) {
        // src: key [E*hc, T], hidden [E*hc, T], w_k [E*hc], w_q [E*hc]; dst [hc, T]
        const int hc = (int) dst->ne[0]; const int64_t T = dst->ne[1]; const int E = (int) (dst->src[0]->ne[0] / hc);
        ple_gate_kernel<<<dim3(hc, (unsigned) T), 256, 0, st>>>((const float *) dst->src[0]->data, (const float *) dst->src[1]->data,
            (const float *) dst->src[2]->data, (const float *) dst->src[3]->data, (float *) dst->data, E, hc, eps);
    } else if (tag == GGML_PLE_TAG_NORM) {
        // src: value [E, T], gate [hc, T], w_c [E*hc]; dst [E*hc, T]
        const int hc = (int) dst->src[1]->ne[0]; const int64_t T = dst->ne[1]; const int E = (int) dst->src[0]->ne[0];
        ple_norm_kernel<<<dim3(hc, (unsigned) T), 256, 0, st>>>((const float *) dst->src[0]->data, (const float *) dst->src[1]->data,
            (const float *) dst->src[2]->data, (float *) dst->data, E, hc, eps);
    } else {
        // src: hidden [E*hc, T], value [E, T], gate [hc, T], conv [E*hc, T]; dst [E*hc, T]
        const int hc = (int) dst->src[2]->ne[0]; const int E = (int) dst->src[1]->ne[0];
        GGML_ASSERT(E % 4 == 0);
        const int64_t n4 = ggml_nelements(dst) / 4;
        ple_out_kernel<<<(unsigned) ((n4 + 255) / 256), 256, 0, st>>>((const float4 *) dst->src[0]->data, (const float *) dst->src[1]->data,
            (const float *) dst->src[2]->data, (const float4 *) dst->src[3]->data, (float4 *) dst->data, E / 4, hc, n4);
    }
    CUDA_CHECK(cudaGetLastError());
}
