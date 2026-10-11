// Chunked prefill for Gated DeltaNet (RDNA3.5 WMMA). Two kernels:
//  gdn_chunk_prep : parallel over (head, chunk) - everything that does not depend on the state:
//                   Gram K.K^T / K.Q^T, the 16x16 unit-lower-triangular inverse, decay factors.
//  gdn_chunk_main : sequential over chunks, one 256-thread block per head. Wave w owns value columns
//                   [16w, 16w+16); the 128x128 f32 state lives in registers as 8 WMMA accumulator tiles per
//                   wave (lane l, vgpr e of tile i <-> S[row 16i+2e+(l>>4)][col 16w+(l&15)]) and is turned into
//                   B operands with permlanex16 + cvt, never touching LDS or memory inside the loop.
// Per chunk (gcs = inclusive cumsum of g inside the chunk):
//  L[s][r] = beta_s (k_s.k_r) e^{gcs_s-gcs_r} (r<s),  T' = (I+L)^{-1} diag(beta)
//  D       = T' (V - diag(e^gcs) K S)
//  out     = diag(e^gcs) Q S + QKm D,   QKm[s][r] = (q_s.k_r) e^{gcs_s-gcs_r} (r<=s)
//  S'      = e^{gcs_last} S + K^T diag(e^{gcs_last-gcs}) D
// Chunks shorter than 16 are zero-padded (k = q = v = 0, g = beta = 0), which leaves S unchanged, so the last K
// tokens can be run as single-token chunks to produce the per-token rollback snapshots.
#include "gdn-chunk.cuh"

// host side: HIP builds only (RDNA3 is a device-pass macro); kernel bodies are compiled for RDNA3 only
#if defined(GGML_USE_HIP)
#define GDN_CHUNK_AVAILABLE 1
#endif
#if !defined(__HIP_DEVICE_COMPILE__) || defined(RDNA3)
#define GDN_CHUNK_DEVICE 1
#endif

#ifdef GDN_CHUNK_AVAILABLE

#define GDC_SV 128
#define GDC_CS 16
#define GDC_KP (GDC_SV + 8)   // padded f16 row stride of K/Q in LDS

typedef _Float16 gdc_v16h __attribute__((ext_vector_type(16)));
typedef float    gdc_v8f  __attribute__((ext_vector_type(8)));
typedef _Float16 gdc_v8h  __attribute__((ext_vector_type(8)));

#define GDC_WMMA(a, b, c) __builtin_amdgcn_wmma_f32_16x16x16_f16_w32((a), (b), (c))
#define GDC_SCHED_FENCE() __builtin_amdgcn_sched_barrier(0)
#define GDC_PHASE() asm volatile("" ::: "memory")

struct gdc_aux {
    _Float16 Tp[GDC_CS][GDC_CS];   // T' row-major [s][r]
    _Float16 Qm[GDC_CS][GDC_CS];   // QKm row-major [s][r]
    float    ea[GDC_CS];           // e^{gcs_s}
    float    eb[GDC_CS];           // e^{gcs_last - gcs_s}
};

struct gdc_sched {   // chunk c -> token range; the last nsingle chunks hold one token each
    int nA, n1;
    __device__ __forceinline__ void get(int c, int & start, int & len) const {
        if (c < nA) { start = c*GDC_CS; len = min(GDC_CS, n1 - start); }
        else        { start = n1 + (c - nA); len = 1; }
    }
};

static __device__ __forceinline__ float gdc_xhalf(float x) {   // same vgpr of lane l^16
    return __int_as_float(__builtin_amdgcn_permlanex16(__float_as_int(x), __float_as_int(x), 0x76543210, 0xfedcba98, false, false));
}

// accumulator tile X (lane: col n, rows 2e+h) -> operand fragment holding X[0..15][n]
static __device__ __forceinline__ gdc_v16h gdc_c_to_op(const gdc_v8f x, const int h) {
    gdc_v16h b;
#pragma unroll
    for (int e = 0; e < 8; ++e) {
        const float own = x[e], oth = gdc_xhalf(own);
        b[2*e]   = (_Float16)(h ? oth : own);
        b[2*e+1] = (_Float16)(h ? own : oth);
    }
    return b;
}

static __device__ __forceinline__ gdc_v16h gdc_ld_frag(const _Float16 * p) {
    const uint4 a = *(const uint4 *) p, b = *(const uint4 *)(p + 8);
    return __builtin_bit_cast(gdc_v16h, (uint4[2]){a, b});
}

// 16 f32 -> f16 fragment, scaled
static __device__ __forceinline__ gdc_v16h gdc_ld_frag_f32(const float * p, const float sc) {
    gdc_v16h r;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const float4 x = *(const float4 *)(p + 4*j);
        r[4*j+0] = (_Float16)(x.x*sc); r[4*j+1] = (_Float16)(x.y*sc); r[4*j+2] = (_Float16)(x.z*sc); r[4*j+3] = (_Float16)(x.w*sc);
    }
    return r;
}

// one wave per (head, chunk), 4 waves per block
template <bool NQK>
static __global__ void __launch_bounds__(128) gdn_chunk_prep(const ggml_cuda_gdn_chunk_args a, const gdc_sched sc, const int nch, gdc_aux * __restrict__ aux0) {
#ifdef GDN_CHUNK_DEVICE
    __shared__ float Lsh[4][GDC_CS][GDC_CS+1];
    const int w = threadIdx.x >> 5, l = threadIdx.x & 31, n = l & 15, h = l >> 4;
    const int c = blockIdx.y*4 + w, head = blockIdx.x;
    if (c >= nch) {
        return;
    }
    int start, len;
    sc.get(c, start, len);
    gdc_aux * aux = aux0 + (int64_t) head*nch + c;

    const bool rowok = n < len;
    const int  t     = start + (rowok ? n : 0);
    const int64_t gb = (int64_t) t*a.sb2 + (int64_t) head*a.sb1;
    const float gm = rowok ? a.g[gb] : 0.f, bm = rowok ? a.beta[gb] : 0.f;
    float gcs = gm;
#pragma unroll
    for (int d = 1; d < GDC_CS; d <<= 1) { const float o = __shfl_up(gcs, d, 16); if (n >= d) gcs += o; }
    const float glast = __shfl(gcs, GDC_CS-1);

    const int64_t roff = (int64_t) t*a.sq2 + (int64_t)(head % a.neqk1)*a.sq1;
    const float * qr = a.q + roff, * kr = a.k + roff;
    float rq = 1.f, rk = 1.f;
    if constexpr (NQK) {
        float sq = 0.f, sk = 0.f;
#pragma unroll
        for (int j = 0; j < 16; ++j) {   // lanes n and n+16 split the row
            const float4 x = *(const float4 *)(qr + 8*j + 4*h), y = *(const float4 *)(kr + 8*j + 4*h);
            sq += x.x*x.x + x.y*x.y + x.z*x.z + x.w*x.w;
            sk += y.x*y.x + y.y*y.y + y.z*y.z + y.w*y.w;
        }
        sq += gdc_xhalf(sq); sk += gdc_xhalf(sk);
        rq = rsqrtf(sq/GDC_SV + a.q_eps)*a.q_scale;
        rk = rsqrtf(sk/GDC_SV + a.k_eps)*a.k_scale;
    }
    if (!rowok) { rq = 0.f; rk = 0.f; }

    gdc_v8f KK = {}, KQ = {};
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const gdc_v16h kf = gdc_ld_frag_f32(kr + 16*i, rk), qf = gdc_ld_frag_f32(qr + 16*i, rq);
        KK = GDC_WMMA(kf, kf, KK); KQ = GDC_WMMA(kf, qf, KQ);
    }
    float gc[8], bt[8];
#pragma unroll
    for (int e = 0; e < 8; ++e) { gc[e] = __shfl(gcs, 2*e + h); bt[e] = __shfl(bm, 2*e + h); }
#pragma unroll
    for (int e = 0; e < 8; ++e) {
        const int s = 2*e + h;   // C element (row s, col n)
        Lsh[w][s][n] = n < s ? bt[e]*KK[e]*__expf(gc[e] - gcs) : 0.f;
        aux->Qm[n][s] = (_Float16)(s <= n ? KQ[e]*__expf(gcs - gc[e]) : 0.f);   // KQ(row s, col n) = k_s.q_n
    }
    __builtin_amdgcn_wave_barrier(); __builtin_amdgcn_fence(__ATOMIC_SEQ_CST, "wavefront");
    if (h == 0) {
        float tc[GDC_CS];
#pragma unroll
        for (int s = 0; s < GDC_CS; ++s) {
            float acc = s == n ? 1.f : 0.f;
#pragma unroll
            for (int j = 0; j < s; ++j) acc -= Lsh[w][s][j]*tc[j];
            tc[s] = acc;
        }
#pragma unroll
        for (int s = 0; s < GDC_CS; ++s) aux->Tp[s][n] = (_Float16)(tc[s]*bm);
        aux->ea[n] = __expf(gcs);
        aux->eb[n] = __expf(glast - gcs);
    }
#else
    GGML_UNUSED_VARS(a, sc, nch, aux0);
    NO_DEVICE_CODE;
#endif
}

template <bool NQK, bool PACK_PREFETCH>
static __global__ void __launch_bounds__(256) __attribute__((amdgpu_waves_per_eu(6)))
gdn_chunk_main(const ggml_cuda_gdn_chunk_args a, const gdc_sched sc, const int nch, const gdc_aux * __restrict__ aux0) {
#ifdef GDN_CHUNK_DEVICE
    __shared__ __align__(16) _Float16 Ksh[GDC_CS][GDC_KP];
    __shared__ __align__(16) _Float16 Qsh[GDC_CS][GDC_KP];
    __shared__ __align__(16) _Float16 KT[GDC_SV][GDC_CS];

    const int head = blockIdx.x;
    const float * qh = a.q + (int64_t)(head % a.neqk1)*a.sq1;
    const float * kh = a.k + (int64_t)(head % a.neqk1)*a.sq1;
    const float * vh = a.v + (int64_t) head*a.sv1;
    const gdc_aux * aux = aux0 + (int64_t) head*nch;
    const int64_t hst = (int64_t) head*GDC_SV*GDC_SV;

    const int tid0 = threadIdx.x;
    gdc_v8f S[8];
    {
        const int w = tid0 >> 5, n = tid0 & 15, h = (tid0 >> 4) & 1, col = 16*w + n;
        const float * s0 = a.s_in + hst + col*GDC_SV + h;
#pragma unroll
        for (int i = 0; i < 8; ++i)
#pragma unroll
            for (int e = 0; e < 8; ++e) S[i][e] = s0[16*i + 2*e];
    }
    GDC_SCHED_FENCE();

    // staging: thread -> row tid>>4, elements [8*(tid&15), +8); V: own accumulator elements
    float4 pk[2], pq[2]; float pv[8];
    gdc_v8h pk16, pq16;
    auto prefetch = [&](int c, int tid, int h, int col) {
        int start, len;
        sc.get(c, start, len);
        const int s = tid >> 4;
        const bool ok = s < len;
        const int64_t off = (int64_t)(start + (ok ? s : 0))*a.sq2 + 8*(tid & 15);
        const float m = ok ? 1.f : 0.f;
#pragma unroll
        for (int j = 0; j < 2; ++j) {
            const float4 x = *(const float4 *)(kh + off + 4*j), y = *(const float4 *)(qh + off + 4*j);
            pk[j] = make_float4(x.x*m, x.y*m, x.z*m, x.w*m);
            pq[j] = make_float4(y.x*m, y.y*m, y.z*m, y.w*m);
        }
        if constexpr (PACK_PREFETCH) {
            float rk = 1.f, rq = 1.f;
            if constexpr (NQK) {
                float sk = 0.f, sq = 0.f;
#pragma unroll
                for (int j = 0; j < 2; ++j) {
                    sk += pk[j].x*pk[j].x + pk[j].y*pk[j].y + pk[j].z*pk[j].z + pk[j].w*pk[j].w;
                    sq += pq[j].x*pq[j].x + pq[j].y*pq[j].y + pq[j].z*pq[j].z + pq[j].w*pq[j].w;
                }
#pragma unroll
                for (int d = 1; d < 16; d <<= 1) { sk += __shfl_xor(sk, d, 16); sq += __shfl_xor(sq, d, 16); }
                rk = s < len ? rsqrtf(sk/GDC_SV + a.k_eps)*a.k_scale : 0.f;
                rq = s < len ? rsqrtf(sq/GDC_SV + a.q_eps)*a.q_scale : 0.f;
            }
            const float kk[8] = {pk[0].x, pk[0].y, pk[0].z, pk[0].w, pk[1].x, pk[1].y, pk[1].z, pk[1].w};
            const float qq[8] = {pq[0].x, pq[0].y, pq[0].z, pq[0].w, pq[1].x, pq[1].y, pq[1].z, pq[1].w};
#pragma unroll
            for (int u = 0; u < 8; ++u) { pk16[u] = (_Float16)(kk[u]*rk); pq16[u] = (_Float16)(qq[u]*rq); }
        }
#pragma unroll
        for (int e = 0; e < 8; ++e) {
            const int s2 = 2*e + h;
            pv[e] = vh[(int64_t)(start + (s2 < len ? s2 : 0))*a.sv2 + col] * (s2 < len ? 1.f : 0.f);
        }
    };
    prefetch(0, tid0, (tid0 >> 4) & 1, 16*(tid0 >> 5) + (tid0 & 15));

    for (int c = 0; c < nch; ++c) {
        // re-derive per-thread indices each iteration: stops LICM from hoisting dozens of loop-invariant
        // addresses that would otherwise live (and spill) across the whole loop
        int tid_ = tid0; asm volatile("" : "+v"(tid_));
        const int tid = tid_, w = __builtin_amdgcn_readfirstlane(tid >> 5), l = tid & 31, n = l & 15, h = l >> 4;
        const int col = 16*w + n;
        int start, len;
        sc.get(c, start, len);

        // ---- stage K, Q, K^T (l2-normalized if NQK) into LDS ----
        {
            const int s = tid >> 4, cc = 8*(tid & 15);
            gdc_v8h kv, qv;
            if constexpr (PACK_PREFETCH) {
                kv = pk16; qv = pq16;
            } else {
            float rk = 1.f, rq = 1.f;
            if constexpr (NQK) {
                float sk = 0.f, sq = 0.f;
#pragma unroll
                for (int j = 0; j < 2; ++j) {
                    sk += pk[j].x*pk[j].x + pk[j].y*pk[j].y + pk[j].z*pk[j].z + pk[j].w*pk[j].w;
                    sq += pq[j].x*pq[j].x + pq[j].y*pq[j].y + pq[j].z*pq[j].z + pq[j].w*pq[j].w;
                }
#pragma unroll
                for (int d = 1; d < 16; d <<= 1) { sk += __shfl_xor(sk, d, 16); sq += __shfl_xor(sq, d, 16); }
                rk = s < len ? rsqrtf(sk/GDC_SV + a.k_eps)*a.k_scale : 0.f;   // padded rows are zero: avoid 0*inf
                rq = s < len ? rsqrtf(sq/GDC_SV + a.q_eps)*a.q_scale : 0.f;
            }
            const float kk[8] = {pk[0].x, pk[0].y, pk[0].z, pk[0].w, pk[1].x, pk[1].y, pk[1].z, pk[1].w};
            const float qq[8] = {pq[0].x, pq[0].y, pq[0].z, pq[0].w, pq[1].x, pq[1].y, pq[1].z, pq[1].w};
#pragma unroll
            for (int u = 0; u < 8; ++u) { kv[u] = (_Float16)(kk[u]*rk); qv[u] = (_Float16)(qq[u]*rq); }
            }
            *(uint4 *)&Ksh[s][cc] = __builtin_bit_cast(uint4, kv);
            *(uint4 *)&Qsh[s][cc] = __builtin_bit_cast(uint4, qv);
#pragma unroll
            for (int u = 0; u < 8; ++u) KT[cc+u][s] = kv[u];
        }
        float vreg[8];
#pragma unroll
        for (int e = 0; e < 8; ++e) vreg[e] = pv[e];
        __syncthreads();
        if (c + 1 < nch) {
            prefetch(c + 1, tid, h, col);
        }

        const gdc_aux * ax = aux + c;
        const gdc_v16h tf = gdc_ld_frag(&ax->Tp[n][0]);
        const gdc_v16h qm = gdc_ld_frag(&ax->Qm[n][0]);
        float ea[8], eb[8];
#pragma unroll
        for (int e = 0; e < 8; ++e) { ea[e] = ax->ea[2*e+h]; eb[e] = ax->eb[2*e+h]; }
        const float al = ax->ea[GDC_CS-1];

        // ---- K.S, Q.S ----
        gdc_v8f kS = {}, qS = {};
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const gdc_v16h kf = gdc_ld_frag(&Ksh[n][16*i]);
            const gdc_v16h qf = gdc_ld_frag(&Qsh[n][16*i]);
            const gdc_v16h sb = gdc_c_to_op(S[i], h);
            kS = GDC_WMMA(kf, sb, kS);
            qS = GDC_WMMA(qf, sb, qS);
            GDC_SCHED_FENCE();
        }
        GDC_PHASE();

        // ---- D = T' (V - e^gcs kS);  out = e^gcs qS + QKm D ----
        gdc_v8f rhs;
#pragma unroll
        for (int e = 0; e < 8; ++e) {
            rhs[e] = vreg[e] - ea[e]*kS[e]; qS[e] *= ea[e];
        }
        gdc_v8f D = {};
        D = GDC_WMMA(tf, gdc_c_to_op(rhs, h), D);
        const gdc_v8f o = GDC_WMMA(qm, gdc_c_to_op(D, h), qS);
#pragma unroll
        for (int e = 0; e < 8; ++e) {
            const int s = 2*e + h;
            if (s < len) {
                a.dst[((int64_t)(start + s)*a.H + head)*GDC_SV + col] = o[e]*a.scale;
            }
        }

        // ---- S = e^{glast} S + K^T (e^{glast-gcs} D) ----
        gdc_v8f Dsc;
#pragma unroll
        for (int e = 0; e < 8; ++e) Dsc[e] = D[e]*eb[e];
        const gdc_v16h Dsb = gdc_c_to_op(Dsc, h);
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const gdc_v8f cS = S[i]*al;
            S[i] = GDC_WMMA(gdc_ld_frag(&KT[16*i + n][0]), Dsb, cS);
            GDC_SCHED_FENCE();
        }
        if (c >= sc.nA) {   // single-token tail chunk: rollback snapshot of the state after token `start`
            const int slot = a.n_tokens - 1 - start;
            if (slot >= 0 && slot < a.K) {
                float * sp = a.state + (int64_t) slot*a.slot_stride + hst + col*GDC_SV + h;
#pragma unroll
                for (int i = 0; i < 8; ++i)
#pragma unroll
                    for (int e = 0; e < 8; ++e) sp[16*i + 2*e] = S[i][e];
            }
        }
        __syncthreads();
    }
    if (a.K <= 1) {
        int base = ((tid0 >> 5)*16 + (tid0 & 15))*GDC_SV + ((tid0 >> 4) & 1);
        asm volatile("" : "+v"(base));   // recompute the 64 addresses here, not keep them live from the load
        float * sp = a.state + hst + base;
#pragma unroll
        for (int i = 0; i < 8; ++i)
#pragma unroll
            for (int e = 0; e < 8; ++e) sp[16*i + 2*e] = S[i][e];
    }
#else
    GGML_UNUSED_VARS(a, sc, nch, aux0);
    NO_DEVICE_CODE;
#endif
}

bool ggml_cuda_gdn_chunk_enabled(int64_t n_tokens) {
    static const int min_tokens = [] { const char * e = getenv("GGML_CUDA_GDN_CHUNK"); return e ? atoi(e) : 64; }();
    return min_tokens > 0 && n_tokens >= min_tokens;
}

void ggml_cuda_gdn_chunk(ggml_backend_cuda_context & ctx, const ggml_cuda_gdn_chunk_args & a) {
    const int nsingle = a.K > 1 ? a.K : 0;
    gdc_sched sc;
    sc.n1 = a.n_tokens - nsingle;
    sc.nA = (sc.n1 + GDC_CS - 1)/GDC_CS;
    const int nch = sc.nA + nsingle;
    ggml_cuda_pool_alloc<gdc_aux> aux(ctx.pool(), (size_t) a.H*nch);
    cudaStream_t stream = ctx.stream();
    const dim3 pgrid(a.H, (nch + 3)/4);
    static const bool packed = [] { const char * e = getenv("GGML_CUDA_GDN_PACK_PREFETCH"); return e && atoi(e) != 0; }();
    if (a.qkn) {
        gdn_chunk_prep<true><<<pgrid, 128, 0, stream>>>(a, sc, nch, aux.get());
        if (packed) gdn_chunk_main<true, true><<<a.H, 256, 0, stream>>>(a, sc, nch, aux.get());
        else        gdn_chunk_main<true, false><<<a.H, 256, 0, stream>>>(a, sc, nch, aux.get());
    } else {
        gdn_chunk_prep<false><<<pgrid, 128, 0, stream>>>(a, sc, nch, aux.get());
        gdn_chunk_main<false, false><<<a.H, 256, 0, stream>>>(a, sc, nch, aux.get());
    }
    CUDA_CHECK(cudaGetLastError());
}

#else

bool ggml_cuda_gdn_chunk_enabled(int64_t) { return false; }
void ggml_cuda_gdn_chunk(ggml_backend_cuda_context &, const ggml_cuda_gdn_chunk_args &) { GGML_ABORT("gdn chunk path not available"); }

#endif
