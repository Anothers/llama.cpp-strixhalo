#include "fused-gate.cuh"

static bool sigmul_add_enabled() {
    static const bool on = [] { const char * e = getenv("GGML_CUDA_SIGMUL_ADD"); return !e || atoi(e) != 0; }();
    return on;
}

// 4 elements per thread along ne0 (ne0 % 4 == 0 checked on the host)
static __global__ void sigmul_add_f32(const float4 * __restrict__ a, const float4 * __restrict__ b, const float * __restrict__ g,
                                      float4 * __restrict__ dst, const int64_t n4, const int64_t ne0_4, const bool g_row) {
    const int64_t i = (int64_t) blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= n4) {
        return;
    }
    const float4 av = a[i];
    const float4 bv = b[i];
    float4 o;
    if (g_row) {
        const float s = 1.0f/(1.0f + expf(-g[i / ne0_4]));
        o = make_float4(av.x + bv.x*s, av.y + bv.y*s, av.z + bv.z*s, av.w + bv.w*s);
    } else {
        const float4 gv = ((const float4 *) g)[i];
        o = make_float4(av.x + bv.x/(1.0f + expf(-gv.x)), av.y + bv.y/(1.0f + expf(-gv.y)),
                        av.z + bv.z/(1.0f + expf(-gv.z)), av.w + bv.w/(1.0f + expf(-gv.w)));
    }
    dst[i] = o;
}

int ggml_cuda_match_sigmul_add(const ggml_cgraph * cgraph, int i, ggml_cuda_sigmul_add_args & args) {
    if (!sigmul_add_enabled() || i + 2 >= cgraph->n_nodes) {
        return 0;
    }
    const ggml_tensor * sig = cgraph->nodes[i];
    const ggml_tensor * mul = cgraph->nodes[i + 1];
    const ggml_tensor * add = cgraph->nodes[i + 2];
    if (sig->op != GGML_OP_UNARY || ggml_get_unary_op(sig) != GGML_UNARY_OP_SIGMOID || mul->op != GGML_OP_MUL || add->op != GGML_OP_ADD) {
        return 0;
    }
    if (!ggml_node_has_n_uses(cgraph, i, 1) || !ggml_node_has_n_uses(cgraph, i + 1, 1)) {
        return 0;
    }
    for (const ggml_tensor * t : { sig, mul, add }) {
        if ((t->flags & GGML_TENSOR_FLAG_COMPUTE) == 0 || t->type != GGML_TYPE_F32) {
            return 0;
        }
    }
    const ggml_tensor * b = mul->src[0] == sig ? mul->src[1] : mul->src[0];
    const ggml_tensor * a = add->src[0] == mul ? add->src[1] : add->src[0];
    const ggml_tensor * g = sig->src[0];
    if ((mul->src[0] != sig && mul->src[1] != sig) || (add->src[0] != mul && add->src[1] != mul)) {
        return 0;
    }
    if (a->type != GGML_TYPE_F32 || b->type != GGML_TYPE_F32 || g->type != GGML_TYPE_F32) {
        return 0;
    }
    if (!ggml_is_contiguous(a) || !ggml_is_contiguous(b) || !ggml_is_contiguous(g) || !ggml_is_contiguous(add)) {
        return 0;
    }
    if (!ggml_are_same_shape(a, b) || !ggml_are_same_shape(b, mul) || !ggml_are_same_shape(mul, add) || b->ne[0] % 4 != 0) {
        return 0;
    }
    // g: same shape as b, or a single column [1, ne1, ne2, ne3]
    const bool g_same = ggml_are_same_shape(g, b);
    const bool g_row  = g->ne[0] == 1 && g->ne[1] == b->ne[1] && g->ne[2] == b->ne[2] && g->ne[3] == b->ne[3];
    if (!g_same && !g_row) {
        return 0;
    }
    args.a = a; args.b = b; args.g = g; args.dst = const_cast<ggml_tensor *>(add);
    return 3;
}

void ggml_cuda_op_sigmul_add(ggml_backend_cuda_context & ctx, const ggml_cuda_sigmul_add_args & args) {
    const int64_t n4    = ggml_nelements(args.b) / 4;
    const int64_t ne0_4 = args.b->ne[0] / 4;
    const bool    g_row = args.g->ne[0] == 1 && args.b->ne[0] != 1;
    const int bs = 256;
    sigmul_add_f32<<<(n4 + bs - 1)/bs, bs, 0, ctx.stream()>>>(
        (const float4 *) args.a->data, (const float4 *) args.b->data, (const float *) args.g->data, (float4 *) args.dst->data,
        n4, ne0_4, g_row);
}

#include "mmb.cuh"

bool ggml_cuda_attn_gate_enabled() {
    static const bool on = [] { const char * e = getenv("GGML_CUDA_ATTN_GATE"); return !e || atoi(e) != 0; }();
    return on;
}

static __device__ __forceinline__ uint16_t attn_gate_f2bf(const float f) {
    uint32_t u = __float_as_uint(f);
    u += 0x7fffu + ((u >> 16) & 1u);
    return (uint16_t) (u >> 16);
}

// one thread per element pair along D; D % 2 == 0
template <bool BF16_OUT>
static __global__ void attn_gate_kernel(const float * __restrict__ attn, const char * __restrict__ gate, void * __restrict__ dst,
                                        const int D, const int H, const int64_t n2, const int64_t gnb0, const int64_t gnb1, const int64_t gnb2) {
    const int64_t i2 = (int64_t) blockIdx.x*blockDim.x + threadIdx.x;
    if (i2 >= n2) {
        return;
    }
    const int64_t e  = 2*i2;
    const int64_t DH = (int64_t) D*H;
    const int64_t t  = e / DH;
    const int64_t j  = e - t*DH;
    const int64_t h  = j / D;
    const int64_t d  = j - h*D;
    const char * gp = gate + d*gnb0 + h*gnb1 + t*gnb2;
    const float g0 = *(const float *) gp;
    const float g1 = *(const float *) (gp + gnb0);
    const float2 a = *(const float2 *) (attn + e);
    const float o0 = a.x/(1.0f + expf(-g0));
    const float o1 = a.y/(1.0f + expf(-g1));
    if constexpr (BF16_OUT) {
        ((uint32_t *) dst)[i2] = (uint32_t) attn_gate_f2bf(o0) | ((uint32_t) attn_gate_f2bf(o1) << 16);
    } else {
        ((float2 *) dst)[i2] = make_float2(o0, o1);
    }
}

int ggml_cuda_match_attn_gate(const ggml_cgraph * cgraph, int i, ggml_cuda_attn_gate_args & args) {
    if (!ggml_cuda_attn_gate_enabled() || i + 2 >= cgraph->n_nodes) {
        return 0;
    }
    const ggml_tensor * cont = cgraph->nodes[i];
    const ggml_tensor * sig  = cgraph->nodes[i + 1];
    const ggml_tensor * mul  = cgraph->nodes[i + 2];
    if (cont->op != GGML_OP_CONT || sig->op != GGML_OP_UNARY || ggml_get_unary_op(sig) != GGML_UNARY_OP_SIGMOID || mul->op != GGML_OP_MUL) {
        return 0;
    }
    if (sig->src[0] != cont || (mul->src[0] != sig && mul->src[1] != sig)) {
        return 0;
    }
    if (!ggml_node_has_n_uses(cgraph, i, 1) || !ggml_node_has_n_uses(cgraph, i + 1, 1)) {
        return 0;
    }
    for (const ggml_tensor * t : { cont, sig, mul }) {
        if ((t->flags & GGML_TENSOR_FLAG_COMPUTE) == 0 || t->type != GGML_TYPE_F32) {
            return 0;
        }
    }
    const ggml_tensor * gv   = cont->src[0];
    const ggml_tensor * attn = mul->src[0] == sig ? mul->src[1] : mul->src[0];
    if (gv->type != GGML_TYPE_F32 || attn->type != GGML_TYPE_F32 || gv->nb[0] != sizeof(float)) {
        return 0;
    }
    if (gv->ne[3] != 1 || gv->ne[0] % 2 != 0 || !ggml_is_contiguous(attn) || !ggml_is_contiguous(mul) || !ggml_is_contiguous(cont)) {
        return 0;
    }
    if (ggml_nelements(attn) != ggml_nelements(gv) || !ggml_are_same_shape(attn, mul) || !ggml_are_same_shape(cont, sig)) {
        return 0;
    }
    // the element order of attn/mul must match the view's [D, H, T] order: the CONT keeps that order, so attn must
    // have the CONT's shape (possibly reshaped) — same element count and same row length suffices for contiguous tensors
    if (attn->ne[0] != gv->ne[0] && attn->ne[0] != gv->ne[0]*gv->ne[1]) {
        return 0;
    }
    args.gate_view = gv; args.attn = attn; args.dst = const_cast<ggml_tensor *>(mul);
    return 3;
}

void ggml_cuda_op_attn_gate(ggml_backend_cuda_context & ctx, const ggml_cuda_attn_gate_args & args) {
    const ggml_tensor * gv = args.gate_view;
    const int D = gv->ne[0], H = gv->ne[1];
    const int64_t n2 = ggml_nelements(args.attn) / 2;
    const int bs = 256;
    const dim3 grid((n2 + bs - 1)/bs);
    if (ggml_cuda_mmb_is_bf16_only(args.dst)) {
        attn_gate_kernel<true><<<grid, bs, 0, ctx.stream()>>>((const float *) args.attn->data, (const char *) gv->data, args.dst->data,
            D, H, n2, gv->nb[0], gv->nb[1], gv->nb[2]);
    } else {
        attn_gate_kernel<false><<<grid, bs, 0, ctx.stream()>>>((const float *) args.attn->data, (const char *) gv->data, args.dst->data,
            D, H, n2, gv->nb[0], gv->nb[1], gv->nb[2]);
    }
}

static bool rms_scale_enabled() {
    static const bool on = [] { const char * e = getenv("GGML_CUDA_RMS_SCALE"); return !e || atoi(e) != 0; }();
    return on;
}

// NPL = ncols/32 elements per lane, ROWS rows (warps) per block
template <int NPL, int ROWS>
static __global__ void rms_scale_f32(const char * __restrict__ x, float * __restrict__ dst, const int64_t nrows,
                                     const int ne1, const int ne2, const int64_t nb1, const int64_t nb2, const int64_t nb3,
                                     const float eps, const float scale) {
    const int64_t row = (int64_t) blockIdx.x*ROWS + threadIdx.y;
    if (row >= nrows) {
        return;
    }
    const int lane = threadIdx.x;
    const int64_t i1 = row % ne1;
    const int64_t i2 = (row / ne1) % ne2;
    const int64_t i3 = row / ((int64_t) ne1*ne2);
    const float * xr = (const float *) (x + i1*nb1 + i2*nb2 + i3*nb3);
    float v[NPL];
    float s = 0.0f;
#pragma unroll
    for (int k = 0; k < NPL; ++k) {
        v[k] = xr[k*WARP_SIZE + lane];
        s += v[k]*v[k];
    }
    s = warp_reduce_sum(s);
    const float r = rsqrtf(s/(NPL*WARP_SIZE) + eps)*scale;
    float * dr = dst + row*(NPL*WARP_SIZE);
#pragma unroll
    for (int k = 0; k < NPL; ++k) {
        dr[k*WARP_SIZE + lane] = v[k]*r;
    }
}

int ggml_cuda_match_rms_scale(const ggml_cgraph * cgraph, int i, ggml_cuda_rms_scale_args & args) {
    if (!rms_scale_enabled() || i + 1 >= cgraph->n_nodes) {
        return 0;
    }
    const ggml_tensor * rms = cgraph->nodes[i];
    const ggml_tensor * sc  = cgraph->nodes[i + 1];
    if (rms->op != GGML_OP_RMS_NORM || sc->op != GGML_OP_SCALE || sc->src[0] != rms || !ggml_node_has_n_uses(cgraph, i, 1)) {
        return 0;
    }
    if ((rms->flags & GGML_TENSOR_FLAG_COMPUTE) == 0 || (sc->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
        return 0;
    }
    const ggml_tensor * x = rms->src[0];
    if (x->type != GGML_TYPE_F32 || rms->type != GGML_TYPE_F32 || sc->type != GGML_TYPE_F32 || x->nb[0] != sizeof(float)) {
        return 0;
    }
    const int64_t n = x->ne[0];
    if (n != 64 && n != 128 && n != 256 || !ggml_is_contiguous(sc) || !ggml_are_same_shape(x, sc)) {
        return 0;
    }
    float s, b;
    memcpy(&s, (const float *) sc->op_params + 0, sizeof(float));
    memcpy(&b, (const float *) sc->op_params + 1, sizeof(float));
    if (b != 0.0f) {
        return 0;
    }
    memcpy(&args.eps, rms->op_params, sizeof(float));
    args.x = x; args.scale = s; args.dst = const_cast<ggml_tensor *>(sc);
    return 2;
}

void ggml_cuda_op_rms_scale(ggml_backend_cuda_context & ctx, const ggml_cuda_rms_scale_args & args) {
    const ggml_tensor * x = args.x;
    const int64_t nrows = ggml_nrows(x);
    constexpr int ROWS = 8;
    const dim3 grid((nrows + ROWS - 1)/ROWS), block(WARP_SIZE, ROWS);
    auto go = [&](auto kern) {
        kern<<<grid, block, 0, ctx.stream()>>>((const char *) x->data, (float *) args.dst->data, nrows, (int) x->ne[1], (int) x->ne[2],
            x->nb[1], x->nb[2], x->nb[3], args.eps, args.scale);
    };
    switch (x->ne[0]) {
        case 64:  go(rms_scale_f32<64/WARP_SIZE,  ROWS>); break;
        case 128: go(rms_scale_f32<128/WARP_SIZE, ROWS>); break;
        default:  go(rms_scale_f32<256/WARP_SIZE, ROWS>); break;
    }
}
