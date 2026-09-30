#include "common.cuh"
#include "qsa.cuh"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <algorithm>
#include "qsa-decode.cuh"

typedef short v16s __attribute__((ext_vector_type(16)));
typedef float v8f  __attribute__((ext_vector_type(8)));

#define QSA3_L2E 1.4426950408889634f
#define QSA3_G 4

static __device__ __forceinline__ uint16_t qsa3_f2h(const float f) { return __builtin_bit_cast(uint16_t, (_Float16) f); }
static __device__ __forceinline__ float qsa3_h2f(const uint16_t h) { return (float) __builtin_bit_cast(_Float16, h); }

// ------------------------------------------------------------------------------------------------------------
// union builder, kernel A: per query row, check that the (validity-transformed) row is non-decreasing; if not,
// rank-sort it into srow[q] and set sflag[q]. Sentinel 0x7FFFFFFF for invalid (-1 / >= nk) entries.
#define QSA3_SENT 0x7FFFFFFF
// (the unsorted rows - the top-k of a shallow context comes back in score order - are sorted with a bitonic network
// over the row padded to np = next power of two; the rank sort it replaces was O(ns^2) per row, ~7 ms per layer)
__global__ __launch_bounds__(256) void qsa3_rows_kernel(
        const int * __restrict__ ids, const size_t i1, const int ns, const int nk, int * __restrict__ srow, int * __restrict__ sflag,
        const int np) {
    extern __shared__ int ent[];
    __shared__ int unsorted;
    const int q = blockIdx.x, tid = threadIdx.x;
    const int * row = reinterpret_cast<const int *>(reinterpret_cast<const char *>(ids) + (size_t) q * i1);
    for (int j = tid; j < np; j += 256) {
        int key = QSA3_SENT;
        if (j < ns) { key = row[j]; key = (key >= 0 && key < nk) ? key : QSA3_SENT; }
        ent[j] = key;
    }
    if (tid == 0) { unsorted = 0; }
    __syncthreads();
    for (int j = tid; j + 1 < ns; j += 256) { if (ent[j] > ent[j+1]) { unsorted = 1; } }
    __syncthreads();
    if (unsorted) {
        for (int size = 2; size <= np; size <<= 1) {
            for (int stride = size >> 1; stride > 0; stride >>= 1) {
                for (int i = tid; i < np / 2; i += 256) {
                    const int lo = 2*i - (i & (stride - 1)), hi = lo + stride;
                    const bool up = (lo & size) == 0;
                    const int a = ent[lo], b = ent[hi];
                    if ((a > b) == up) { ent[lo] = b; ent[hi] = a; }
                }
                __syncthreads();
            }
        }
        int * dst = srow + (size_t) q * ns;
        for (int j = tid; j < ns; j += 256) { dst[j] = ent[j]; }
    }
    if (tid == 0) { sflag[q] = unsorted; }
}

// kernel B: one workgroup (128 lanes) per group of 4 queries. Lane = key sub-range; the range boundaries are the
// quantiles of the first non-empty row (block aligned), so the work per lane is balanced for the real (sorted,
// similar) rows. Each lane merges the 4 sorted rows inside its range: pass 1 counts, block-wide prefix sum,
// pass 2 writes.
#define QSA3_MERGE_LANES 128
static __device__ __forceinline__ int qsa3_get(const int * __restrict__ p, const int j, const int nk, const bool raw) {
    const int key = p[j];
    return raw ? ((key >= 0 && key < nk) ? key : QSA3_SENT) : key;
}
template <int G>
__global__ __launch_bounds__(QSA3_MERGE_LANES) void qsa3_merge_kernel(
        const int * __restrict__ ids, const size_t i1, const int n_q, const int ns, const int nk,
        const int * __restrict__ srow, const int * __restrict__ sflag,
        uint16_t * __restrict__ ublk, uint16_t * __restrict__ umask, int * __restrict__ ucount, const int cap) {
    extern __shared__ int rows_s[];
    __shared__ int scan_s[QSA3_MERGE_LANES];
    const int lane = threadIdx.x;
    const int g = blockIdx.x;
#pragma unroll
    for (int qi = 0; qi < G; ++qi) {
        const int q = G * g + qi;
        int * dst = rows_s + qi * ns;
        if (q < n_q) {
            const bool rw = sflag[q] == 0;
            const int * src = rw ? reinterpret_cast<const int *>(reinterpret_cast<const char *>(ids) + (size_t) q * i1) : srow + (size_t) q * ns;
            for (int j = lane; j < ns; j += QSA3_MERGE_LANES) { dst[j] = qsa3_get(src, j, nk, rw); }
        } else {
            for (int j = lane; j < ns; j += QSA3_MERGE_LANES) { dst[j] = QSA3_SENT; }
        }
    }
    __syncthreads();
    const int * rp[G]; bool raw[G]; int cnt[G];
#pragma unroll
    for (int qi = 0; qi < G; ++qi) {
        raw[qi] = false; rp[qi] = rows_s + qi * ns;
        int a = 0, b = ns;
        while (a < b) { const int mid = (a + b) >> 1; if (rp[qi][mid] < QSA3_SENT) { a = mid + 1; } else { b = mid; } }
        cnt[qi] = a;
    }
    int kmax = -1, ref = -1;
#pragma unroll
    for (int qi = 0; qi < G; ++qi) {
        if (cnt[qi] > 0) { kmax = max(kmax, qsa3_get(rp[qi], cnt[qi] - 1, nk, raw[qi])); if (ref < 0) { ref = qi; } }
    }
    uint16_t * ob = ublk + (size_t) g * cap, * om = umask + (size_t) g * cap;
    if (kmax < 0) { if (lane == 0) { ucount[g] = 0; } return; }
    // lane range [lo, hi) in key space, block aligned, from the reference row's quantiles
    const int rc = cnt[ref];
    const int lo = lane == 0 ? 0 : (qsa3_get(rp[ref], (int) (((long long) rc * lane) / QSA3_MERGE_LANES), nk, raw[ref]) & ~3);
    const int hi = lane == QSA3_MERGE_LANES - 1 ? ((kmax + 4) & ~3)
                 : (qsa3_get(rp[ref], (int) (((long long) rc * (lane + 1)) / QSA3_MERGE_LANES), nk, raw[ref]) & ~3);
    int p[G], pe[G];
#pragma unroll
    for (int qi = 0; qi < G; ++qi) {
        int a = 0, b = cnt[qi];
        while (a < b) { const int mid = (a + b) >> 1; if (qsa3_get(rp[qi], mid, nk, raw[qi]) < lo) { a = mid + 1; } else { b = mid; } }
        p[qi] = a; b = cnt[qi];
        while (a < b) { const int mid = (a + b) >> 1; if (qsa3_get(rp[qi], mid, nk, raw[qi]) < hi) { a = mid + 1; } else { b = mid; } }
        pe[qi] = a;
    }
    // pass 1: count union blocks in range
    int n = 0;
    {
        int pp[G]; int head[G];
#pragma unroll
        for (int qi = 0; qi < G; ++qi) { pp[qi] = p[qi]; head[qi] = pp[qi] < pe[qi] ? qsa3_get(rp[qi], pp[qi], nk, raw[qi]) : QSA3_SENT; }
        while (true) {
            int b = QSA3_SENT;
#pragma unroll
            for (int qi = 0; qi < G; ++qi) { b = min(b, head[qi]); }
            if (b == QSA3_SENT) { break; }
            b >>= 2;
#pragma unroll
            for (int qi = 0; qi < G; ++qi) {
                if (head[qi] != QSA3_SENT && (head[qi] >> 2) == b) {
                    const int * rr = rp[qi];
                    int pn = pp[qi];
                    if ((head[qi] & 3) == 0 && pn + 3 < pe[qi] && rr[pn+3] == head[qi] + 3 && rr[pn+1] == head[qi] + 1 && rr[pn+2] == head[qi] + 2) { pn += 4; }
                    else { do { ++pn; } while (pn < pe[qi] && (rr[pn] >> 2) == b); }
                    pp[qi] = pn; head[qi] = pn < pe[qi] ? rr[pn] : QSA3_SENT;
                }
            }
            ++n;
        }
    }
    scan_s[lane] = n;
    __syncthreads();
    for (int off = 1; off < QSA3_MERGE_LANES; off <<= 1) {
        const int v = lane >= off ? scan_s[lane - off] : 0;
        __syncthreads();
        scan_s[lane] += v;
        __syncthreads();
    }
    const int total = scan_s[QSA3_MERGE_LANES - 1];
    int o = scan_s[lane] - n;
    {
        int head[G];
#pragma unroll
        for (int qi = 0; qi < G; ++qi) { head[qi] = p[qi] < pe[qi] ? qsa3_get(rp[qi], p[qi], nk, raw[qi]) : QSA3_SENT; }
        while (true) {
            int b = QSA3_SENT;
#pragma unroll
            for (int qi = 0; qi < G; ++qi) { b = min(b, head[qi]); }
            if (b == QSA3_SENT) { break; }
            b >>= 2;
            uint32_t mk = 0;
#pragma unroll
            for (int qi = 0; qi < G; ++qi) {
                if (head[qi] != QSA3_SENT && (head[qi] >> 2) == b) {
                    const int * rr = rp[qi];
                    int pn = p[qi];
                    if ((head[qi] & 3) == 0 && pn + 3 < pe[qi] && rr[pn+3] == head[qi] + 3 && rr[pn+1] == head[qi] + 1 && rr[pn+2] == head[qi] + 2) { mk |= 0xFu << (4*qi); pn += 4; }
                    else { do { mk |= 1u << (4*qi + (rr[pn] & 3)); ++pn; } while (pn < pe[qi] && (rr[pn] >> 2) == b); }
                    p[qi] = pn; head[qi] = pn < pe[qi] ? rr[pn] : QSA3_SENT;
                }
            }
            ob[o] = (uint16_t) b; om[o] = (uint16_t) mk; ++o;
        }
    }
    if (lane == QSA3_MERGE_LANES - 1) {
        int t = total;
        while (t & 3) { ob[t] = 0xFFFFu; om[t] = 0; ++t; }
        ucount[g] = t;
    }
}

struct qsa3_layout {
    size_t q1, q2, o1, o2, m1;
    int nk, n_q, gqa, cap;
    float scale;
    size_t k1, k2, v1, v2;   // DIRECT: element strides of the unpacked K/V cache views (cell, kv head)
};

// DIRECT reads the K/V cache in place: a K fragment row is 16 contiguous dims of one cell, a V fragment gathers
// one dim of the 4 cells of a block. Without it the graph repacks the whole cache every ubatch (4 full copies
// per QSA layer, linear in context depth).
template <bool DIRECT, bool DIRECT_V>
__global__ __launch_bounds__(256) void qsa3_attn_kernel(
        const float * __restrict__ q, const uint16_t * __restrict__ pk, const uint16_t * __restrict__ pv,
        const uint16_t * __restrict__ mask, const uint16_t * __restrict__ ublk, const uint16_t * __restrict__ umask,
        const int * __restrict__ ucount, float * __restrict__ out, const qsa3_layout s) {
    const int tid = threadIdx.x, lane = tid & 31, w = tid >> 5, r = lane & 15, hi = lane >> 4;
    const int g = blockIdx.x, kvh = blockIdx.y;
    __shared__ float4 part[3][8][64];
    __shared__ uint4  ptile[3][32];
    __shared__ float  alpha_s[48];
    __shared__ float  l_s[48];

    const size_t nblk = (size_t) s.nk / 4;
    const uint16_t * pkg = pk + (size_t) kvh * nblk * 1024;
    const uint16_t * pvg = pv + (size_t) kvh * nblk * 1024;
    const uint16_t * gblk = ublk + (size_t) g * s.cap;
    const uint16_t * gmsk = umask + (size_t) g * s.cap;
    const int nchunks = ucount[g] >> 2;

    v16s qf[3][2];
    {
        uint16_t * stage = reinterpret_cast<uint16_t *>(&part[0][0][0]) + w * (48 * 32);   // 3 KB per wave, row stride 64 B
        const _Float16 hs = (_Float16) s.scale;
#pragma unroll
        for (int c = 0; c < 12; ++c) {
            const int row = 4*c + (lane >> 3), dq = 4 * (lane & 7);
            const int qi = row / 12, h = row - 12*qi;
            const int query = QSA3_G * g + qi;
            float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
            if (query < s.n_q) {
                v = *reinterpret_cast<const float4 *>(reinterpret_cast<const char *>(q) + (size_t) query * s.q1 + (size_t) (kvh * s.gqa + h) * s.q2 + (size_t) (32*w + dq) * 4);
            }
            const uint16_t h0 = __builtin_bit_cast(uint16_t, (_Float16) v.x * hs), h1 = __builtin_bit_cast(uint16_t, (_Float16) v.y * hs);
            const uint16_t h2 = __builtin_bit_cast(uint16_t, (_Float16) v.z * hs), h3 = __builtin_bit_cast(uint16_t, (_Float16) v.w * hs);
            *reinterpret_cast<uint2 *>(stage + row * 32 + dq) = make_uint2((uint32_t) h0 | ((uint32_t) h1 << 16), (uint32_t) h2 | ((uint32_t) h3 << 16));
        }
        __syncthreads();
#pragma unroll
        for (int i = 0; i < 3; ++i)
#pragma unroll
            for (int t = 0; t < 2; ++t) {
                const uint16_t * src = stage + (16*i + r) * 32 + 16*t;
                qf[i][t] = __builtin_bit_cast(v16s, (uint4[2]){*reinterpret_cast<const uint4 *>(src), *reinterpret_cast<const uint4 *>(src + 8)});
            }
        __syncthreads();
    }
    v8f O[3][2];
#pragma unroll
    for (int i = 0; i < 3; ++i)
#pragma unroll
        for (int t = 0; t < 2; ++t)
#pragma unroll
            for (int e = 0; e < 8; ++e) { O[i][t][e] = 0.f; }
    float m = -INFINITY, l = 0.f;                       // owner state (waves 0..2): row 16w + r
    const int own_row = 16*w + r, own_qi = own_row / 12;
    const int own_query = QSA3_G * g + own_qi;
    const uint16_t * maskq = (mask && w < 3 && own_query < s.n_q)
        ? reinterpret_cast<const uint16_t *>(reinterpret_cast<const char *>(mask) + (size_t) own_query * s.m1) : nullptr;

    auto load_desc = [&](const int c, uint32_t & b01, uint32_t & b23, uint32_t & m01, uint32_t & m23) {
        const uint2 bb = *reinterpret_cast<const uint2 *>(gblk + 4*c);
        const uint2 mm = *reinterpret_cast<const uint2 *>(gmsk + 4*c);
        b01 = bb.x; b23 = bb.y; m01 = mm.x; m23 = mm.y;
    };
    auto blk_of = [&](const uint32_t b01, const uint32_t b23, const int j) -> int {
        const uint32_t v = (j < 2 ? b01 : b23) >> (16 * (j & 1)) & 0xFFFFu;
        return v == 0xFFFFu ? 0 : (int) v;
    };
    auto load_k = [&](const uint32_t b01, const uint32_t b23, v16s * kf) {
        const int kb = blk_of(b01, b23, r >> 2);
        if constexpr (DIRECT) {
            const uint16_t * kc = pk + (size_t) kvh * s.k2 + (size_t) (4*kb + (r & 3)) * s.k1 + 32*w;
#pragma unroll
            for (int t = 0; t < 2; ++t) {
                const uint4 lo  = *reinterpret_cast<const uint4 *>(kc + 16*t);
                const uint4 hi4 = *reinterpret_cast<const uint4 *>(kc + 16*t + 8);
                kf[t] = __builtin_bit_cast(v16s, (uint4[2]){lo, hi4});
            }
            return;
        }
        const uint16_t * krow = pkg + (size_t) kb * 1024 + (r & 3) * 16 + w * 128;
#pragma unroll
        for (int t = 0; t < 2; ++t) {
            const uint4 lo  = *reinterpret_cast<const uint4 *>(krow + t*64);
            const uint4 hi4 = *reinterpret_cast<const uint4 *>(krow + t*64 + 8);
            kf[t] = __builtin_bit_cast(v16s, (uint4[2]){lo, hi4});
        }
    };

    uint32_t b01 = 0, b23 = 0, m01 = 0, m23 = 0;
    v16s kf[2];
    if (nchunks > 0) { load_desc(0, b01, b23, m01, m23); load_k(b01, b23, kf); }

    for (int c = 0; c < nchunks; ++c) {
        v8f sc[3];
#pragma unroll
        for (int i = 0; i < 3; ++i) {
#pragma unroll
            for (int e = 0; e < 8; ++e) { sc[i][e] = 0.f; }
#pragma unroll
            for (int t = 0; t < 2; ++t) { sc[i] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(kf[t], qf[i][t], sc[i]); }
        }
        v16s vf[2];
        {
            const int kb0 = blk_of(b01, b23, 0), kb1 = blk_of(b01, b23, 1), kb2 = blk_of(b01, b23, 2), kb3 = blk_of(b01, b23, 3);
            if constexpr (DIRECT_V) {
                const uint16_t * vb = pv + (size_t) kvh * s.v2;
                const int kbs[4] = {kb0, kb1, kb2, kb3};
#pragma unroll
                for (int t = 0; t < 2; ++t) {
                    const int d = 32*w + 16*t + r;
                    uint32_t wv[8];
#pragma unroll
                    for (int j = 0; j < 4; ++j) {
                        const uint16_t * vc = vb + (size_t) (4*kbs[j]) * s.v1 + d;
                        const uint32_t e0 = vc[0], e1 = vc[s.v1], e2 = vc[2*s.v1], e3 = vc[3*s.v1];
                        wv[2*j] = e0 | (e1 << 16); wv[2*j + 1] = e2 | (e3 << 16);
                    }
                    vf[t] = __builtin_bit_cast(v16s, (uint4[2]){make_uint4(wv[0], wv[1], wv[2], wv[3]), make_uint4(wv[4], wv[5], wv[6], wv[7])});
                }
            } else
#pragma unroll
            for (int t = 0; t < 2; ++t) {
                const int d = 32*w + 16*t + r;
                const uint2 v0 = *reinterpret_cast<const uint2 *>(pvg + (size_t) kb0 * 1024 + d * 4);
                const uint2 v1 = *reinterpret_cast<const uint2 *>(pvg + (size_t) kb1 * 1024 + d * 4);
                const uint2 v2 = *reinterpret_cast<const uint2 *>(pvg + (size_t) kb2 * 1024 + d * 4);
                const uint2 v3 = *reinterpret_cast<const uint2 *>(pvg + (size_t) kb3 * 1024 + d * 4);
                vf[t] = __builtin_bit_cast(v16s, (uint2[4]){v0, v1, v2, v3});
            }
        }
        // --- prefetch next chunk's descriptor and K fragments ---
        uint32_t nb01 = 0, nb23 = 0, nm01 = 0, nm23 = 0;
        v16s kfn[2];
        if (c + 1 < nchunks) { load_desc(c + 1, nb01, nb23, nm01, nm23); load_k(nb01, nb23, kfn); }
        else { kfn[0] = kf[0]; kfn[1] = kf[1]; }
#pragma unroll
        for (int i = 0; i < 3; ++i) {
            if (i != w) {
                part[i][w][lane*2+0] = make_float4(sc[i][0], sc[i][1], sc[i][2], sc[i][3]);
                part[i][w][lane*2+1] = make_float4(sc[i][4], sc[i][5], sc[i][6], sc[i][7]);
            }
        }
        __syncthreads();
        if (w < 3) {
            float scf[8];
#pragma unroll
            for (int e = 0; e < 8; ++e) { scf[e] = 0.f; }
#pragma unroll
            for (int w2 = 0; w2 < 8; ++w2) {
                if (w2 == w) {
#pragma unroll
                    for (int e = 0; e < 8; ++e) { scf[e] += sc[w][e]; }
                } else {
                    const float4 a = part[w][w2][lane*2+0], b = part[w][w2][lane*2+1];
                    scf[0] += a.x; scf[1] += a.y; scf[2] += a.z; scf[3] += a.w; scf[4] += b.x; scf[5] += b.y; scf[6] += b.z; scf[7] += b.w;
                }
            }
#pragma unroll
            for (int e = 0; e < 8; ++e) {
                const int j = e >> 1, cc = 2*(e & 1) + hi;
                const uint32_t mk = ((j < 2 ? m01 : m23) >> (16 * (j & 1))) & 0xFFFFu;
                const bool kv = (mk >> (own_qi * 4 + cc)) & 1u;
                if (maskq && kv) { scf[e] += qsa3_h2f(maskq[4 * blk_of(b01, b23, j) + cc]); }
                if (!kv) { scf[e] = -INFINITY; }
            }
            float mloc = scf[0];
#pragma unroll
            for (int e = 1; e < 8; ++e) { mloc = fmaxf(mloc, scf[e]); }
            mloc = fmaxf(mloc, __shfl_xor(mloc, 16));
            const float mnew  = fmaxf(m, mloc);
            const float alpha = mnew == -INFINITY ? 1.f : exp2f((m - mnew) * QSA3_L2E);
            float p[8], ls = 0.f;
#pragma unroll
            for (int e = 0; e < 8; ++e) { p[e] = mnew == -INFINITY ? 0.f : exp2f((scf[e] - mnew) * QSA3_L2E); ls += p[e]; }
            ls += __shfl_xor(ls, 16);
            l = l * alpha + ls;
            m = mnew;
            uint32_t pp[4], po[4];
#pragma unroll
            for (int e2 = 0; e2 < 4; ++e2) {
                pp[e2] = (uint32_t) qsa3_f2h(p[2*e2]) | ((uint32_t) qsa3_f2h(p[2*e2+1]) << 16);
                po[e2] = (uint32_t) __shfl_xor((int) pp[e2], 16);
            }
            uint16_t ph[16];
#pragma unroll
            for (int e = 0; e < 8; ++e) {
                const uint16_t mine  = (e & 1) ? (uint16_t) (pp[e>>1] >> 16) : (uint16_t) (pp[e>>1] & 0xffffu);
                const uint16_t other = (e & 1) ? (uint16_t) (po[e>>1] >> 16) : (uint16_t) (po[e>>1] & 0xffffu);
                ph[2*e]   = hi ? other : mine;
                ph[2*e+1] = hi ? mine  : other;
            }
            if (hi == 0) {
                ptile[w][r*2+0] = make_uint4((uint32_t) ph[0] | ((uint32_t) ph[1] << 16), (uint32_t) ph[2] | ((uint32_t) ph[3] << 16),
                                             (uint32_t) ph[4] | ((uint32_t) ph[5] << 16), (uint32_t) ph[6] | ((uint32_t) ph[7] << 16));
                ptile[w][r*2+1] = make_uint4((uint32_t) ph[8] | ((uint32_t) ph[9] << 16), (uint32_t) ph[10] | ((uint32_t) ph[11] << 16),
                                             (uint32_t) ph[12] | ((uint32_t) ph[13] << 16), (uint32_t) ph[14] | ((uint32_t) ph[15] << 16));
                alpha_s[own_row] = alpha;
            }
        }
        __syncthreads();
        float al[3];
#pragma unroll
        for (int i = 0; i < 3; ++i) { al[i] = alpha_s[16*i + r]; }
        if (__ballot(al[0] != 1.f || al[1] != 1.f || al[2] != 1.f)) {
#pragma unroll
            for (int i = 0; i < 3; ++i)
#pragma unroll
                for (int t = 0; t < 2; ++t)
#pragma unroll
                    for (int e = 0; e < 8; ++e) { O[i][t][e] *= al[i]; }
        }
#pragma unroll
        for (int i = 0; i < 3; ++i) {
            const v16s pf = __builtin_bit_cast(v16s, (uint4[2]){ptile[i][r*2+0], ptile[i][r*2+1]});
#pragma unroll
            for (int t = 0; t < 2; ++t) { O[i][t] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(vf[t], pf, O[i][t]); }
        }
        kf[0] = kfn[0]; kf[1] = kfn[1];
        b01 = nb01; b23 = nb23; m01 = nm01; m23 = nm23;
    }

    if (w < 3 && hi == 0) { l_s[own_row] = l; }
    __syncthreads();
    // output: per tile, stage [16 rows x 32 dims] F32 per wave in LDS (2 KB), then coalesced 16-byte row stores
    float * ostage = reinterpret_cast<float *>(&part[0][0][0]) + w * (16 * 32);
#pragma unroll
    for (int i = 0; i < 3; ++i) {
        const float li = l_s[16*i + r];
#pragma unroll
        for (int t = 0; t < 2; ++t)
#pragma unroll
            for (int e = 0; e < 8; ++e) { ostage[r * 32 + 16*t + 2*e + hi] = li > 0.f ? O[i][t][e] / li : 0.f; }
        __syncthreads();
#pragma unroll
        for (int c = 0; c < 4; ++c) {
            const int rr = 4*c + (lane >> 3), dq = 4 * (lane & 7);
            const int row = 16*i + rr, qi = row / 12, h = row - 12*qi;
            const int query = QSA3_G * g + qi;
            if (query < s.n_q) {
                float * dst = reinterpret_cast<float *>(reinterpret_cast<char *>(out) + (size_t) query * s.o2 + (size_t) (kvh * s.gqa + h) * s.o1) + 32*w + dq;
                *reinterpret_cast<float4 *>(dst) = *reinterpret_cast<const float4 *>(ostage + rr * 32 + dq);
            }
        }
        __syncthreads();
    }
}

// qsa5: one block per (query, kv head). The 12 heads of the query are one 16-row tile (4 pad rows); the query's own
// block list (no union with neighbours: the work stays 512 blocks per query at any depth). K comes packed per
// 4-cell block like V (qsa3_pack_k_kernel): reading 32 B of 16 different cache rows per load wasted 3/4 of every line is split over KS waves in
// interleaved 16-key chunks; the waves merge once at the end. Per wave: QK 16 wmma, softmax in registers, PV 16 wmma.
#define QSA5_QS  264   // Q row stride in LDS (halves)
#define QSA5_GRP 4     // K/V loads in flight per wave
#define QSA5_KS  4     // waves per block, each takes every 4th chunk
template <int KS>
__global__ __launch_bounds__(32*KS) void qsa5_attn_kernel(
        const float * __restrict__ q, const uint16_t * __restrict__ pk, const uint16_t * __restrict__ pv,
        const uint16_t * __restrict__ mask, const uint16_t * __restrict__ qblk, const uint16_t * __restrict__ qmask,
        const int * __restrict__ qcount, float * __restrict__ out, const qsa3_layout s) {
    const int tid = threadIdx.x, lane = tid & 31, ks = tid >> 5, r = lane & 15, hi = lane >> 4;
    const int query = blockIdx.x, kvh = blockIdx.y;
    __shared__ __attribute__((aligned(16))) uint16_t qs[16 * QSA5_QS];   // 8.4 KB, rows padded against bank conflicts
    __shared__ __attribute__((aligned(16))) float xs[KS > 1 ? 4 * 8 * 32 : 1];  // 4 KB merge buffer
    __shared__ float ml_s[KS][16][2];

    const size_t nblk = (size_t) s.nk / 4;
    const uint16_t * pvg = pv + (size_t) kvh * nblk * 1024;
    const uint16_t * gblk = qblk + (size_t) query * s.cap;
    const uint16_t * gmsk = qmask + (size_t) query * s.cap;
    const int nchunks = qcount[query] >> 2;
    {
        const _Float16 hs = (_Float16) s.scale;
        for (int idx = tid; idx < 16 * 64; idx += 32*KS) {
            const int h = idx >> 6, d4 = (idx & 63) * 4;
            float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
            if (h < 12) {
                v = *reinterpret_cast<const float4 *>(reinterpret_cast<const char *>(q) + (size_t) query * s.q1 + (size_t) (kvh * s.gqa + h) * s.q2 + (size_t) d4 * 4);
            }
            const uint16_t h0 = __builtin_bit_cast(uint16_t, (_Float16) v.x * hs), h1 = __builtin_bit_cast(uint16_t, (_Float16) v.y * hs);
            const uint16_t h2 = __builtin_bit_cast(uint16_t, (_Float16) v.z * hs), h3 = __builtin_bit_cast(uint16_t, (_Float16) v.w * hs);
            *reinterpret_cast<uint2 *>(qs + h * QSA5_QS + d4) = make_uint2((uint32_t) h0 | ((uint32_t) h1 << 16), (uint32_t) h2 | ((uint32_t) h3 << 16));
        }
    }
    __syncthreads();
    const uint16_t * qrow = qs + r * QSA5_QS;
    const uint16_t * maskq = mask ? reinterpret_cast<const uint16_t *>(reinterpret_cast<const char *>(mask) + (size_t) query * s.m1) : nullptr;

    v8f O[16];
#pragma unroll
    for (int t = 0; t < 16; ++t)
#pragma unroll
        for (int e = 0; e < 8; ++e) { O[t][e] = 0.f; }
    float m = -INFINITY, l = 0.f;

    for (int c = ks; c < nchunks; c += KS) {
        const uint2 bb = *reinterpret_cast<const uint2 *>(gblk + 4*c);
        const uint2 mm = *reinterpret_cast<const uint2 *>(gmsk + 4*c);
        auto blk = [&](const int j) -> int { const uint32_t v = ((j < 2 ? bb.x : bb.y) >> (16 * (j & 1))) & 0xFFFFu; return v == 0xFFFFu ? 0 : (int) v; };
        // packed K: [kvh][blk][16 dim chunks][4 cells][16 dims] -> the 4 cells of a block share one 128 B line per chunk
        const uint16_t * kc0 = pk + ((size_t) kvh * nblk + blk(r >> 2)) * 1024 + (r & 3) * 16;
        v8f sc;
#pragma unroll
        for (int e = 0; e < 8; ++e) { sc[e] = 0.f; }
#pragma unroll
        for (int t = 0; t < 16; t += 2) {
            const uint4 k0 = *reinterpret_cast<const uint4 *>((kc0 + 64*(t + hi))), k1 = *reinterpret_cast<const uint4 *>((kc0 + 64*(t + hi)) + 8);
            const uint32_t x[8] = {k0.x, k0.y, k0.z, k0.w, k1.x, k1.y, k1.z, k1.w};
            uint32_t a[8], b[8];
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                const uint32_t y = __builtin_amdgcn_permlanex16(x[i], x[i], 0x76543210u, 0xfedcba98u, false, false);
                a[i] = hi ? y : x[i]; b[i] = hi ? x[i] : y;
            }
            const v16s ka = __builtin_bit_cast(v16s, (uint4[2]){make_uint4(a[0], a[1], a[2], a[3]), make_uint4(a[4], a[5], a[6], a[7])});
            const v16s kb = __builtin_bit_cast(v16s, (uint4[2]){make_uint4(b[0], b[1], b[2], b[3]), make_uint4(b[4], b[5], b[6], b[7])});
            const v16s q0 = __builtin_bit_cast(v16s, (uint4[2]){*reinterpret_cast<const uint4 *>(qrow + 16*t), *reinterpret_cast<const uint4 *>(qrow + 16*t + 8)});
            const v16s q1 = __builtin_bit_cast(v16s, (uint4[2]){*reinterpret_cast<const uint4 *>(qrow + 16*t + 16), *reinterpret_cast<const uint4 *>(qrow + 16*t + 24)});
            sc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(ka, q0, sc);
            sc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(kb, q1, sc);
            if (((t >> 1) & (QSA5_GRP/2 - 1)) == QSA5_GRP/2 - 1) { asm volatile("" ::: "memory"); }
        }
        float scf[8];
#pragma unroll
        for (int e = 0; e < 8; ++e) {
            const int j = e >> 1, cc = 2*(e & 1) + hi;
            const uint32_t mk = ((j < 2 ? mm.x : mm.y) >> (16 * (j & 1))) & 0xFFFFu;
            const bool kv = (mk >> cc) & 1u;
            scf[e] = sc[e];
            if (maskq && kv) { scf[e] += qsa3_h2f(maskq[4 * blk(j) + cc]); }
            if (!kv) { scf[e] = -INFINITY; }
        }
        float mloc = scf[0];
#pragma unroll
        for (int e = 1; e < 8; ++e) { mloc = fmaxf(mloc, scf[e]); }
        mloc = fmaxf(mloc, __shfl_xor(mloc, 16));
        const float mnew  = fmaxf(m, mloc);
        const float alpha = mnew == -INFINITY ? 1.f : exp2f((m - mnew) * QSA3_L2E);
        float p[8], ls = 0.f;
#pragma unroll
        for (int e = 0; e < 8; ++e) { p[e] = mnew == -INFINITY ? 0.f : exp2f((scf[e] - mnew) * QSA3_L2E); ls += p[e]; }
        ls += __shfl_xor(ls, 16);
        l = l * alpha + ls;
        m = mnew;
        uint32_t pp[4], po[4];
#pragma unroll
        for (int e2 = 0; e2 < 4; ++e2) {
            pp[e2] = (uint32_t) qsa3_f2h(p[2*e2]) | ((uint32_t) qsa3_f2h(p[2*e2+1]) << 16);
            po[e2] = (uint32_t) __shfl_xor((int) pp[e2], 16);
        }
        uint32_t pw[8];
#pragma unroll
        for (int e = 0; e < 8; ++e) {
            const uint32_t mine  = (e & 1) ? (pp[e>>1] >> 16) : (pp[e>>1] & 0xffffu);
            const uint32_t other = (e & 1) ? (po[e>>1] >> 16) : (po[e>>1] & 0xffffu);
            pw[e] = hi ? (other | (mine << 16)) : (mine | (other << 16));
        }
        const v16s pf = __builtin_bit_cast(v16s, (uint4[2]){make_uint4(pw[0], pw[1], pw[2], pw[3]), make_uint4(pw[4], pw[5], pw[6], pw[7])});
        if (__ballot(alpha != 1.f)) {
#pragma unroll
            for (int t = 0; t < 16; ++t)
#pragma unroll
                for (int e = 0; e < 8; ++e) { O[t][e] *= alpha; }
        }
        const int kb0 = blk(0), kb1 = blk(1), kb2 = blk(2), kb3 = blk(3);
        // lanes 16-31 duplicate lanes 0-15 in a wmma operand: the upper half loads tile t+1 and the halves swap
#pragma unroll
        for (int t = 0; t < 16; t += 2) {
            const int d = 16*(t + hi) + r;
            const uint2 v0 = *reinterpret_cast<const uint2 *>(pvg + (size_t) kb0 * 1024 + d * 4);
            const uint2 v1 = *reinterpret_cast<const uint2 *>(pvg + (size_t) kb1 * 1024 + d * 4);
            const uint2 v2 = *reinterpret_cast<const uint2 *>(pvg + (size_t) kb2 * 1024 + d * 4);
            const uint2 v3 = *reinterpret_cast<const uint2 *>(pvg + (size_t) kb3 * 1024 + d * 4);
            const uint32_t x[8] = {v0.x, v0.y, v1.x, v1.y, v2.x, v2.y, v3.x, v3.y};
            uint32_t a[8], b[8];
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                const uint32_t y = __builtin_amdgcn_permlanex16(x[i], x[i], 0x76543210u, 0xfedcba98u, false, false);
                a[i] = hi ? y : x[i]; b[i] = hi ? x[i] : y;
            }
            const v16s va = __builtin_bit_cast(v16s, (uint4[2]){make_uint4(a[0], a[1], a[2], a[3]), make_uint4(a[4], a[5], a[6], a[7])});
            const v16s vb = __builtin_bit_cast(v16s, (uint4[2]){make_uint4(b[0], b[1], b[2], b[3]), make_uint4(b[4], b[5], b[6], b[7])});
            O[t]     = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(va, pf, O[t]);
            O[t + 1] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(vb, pf, O[t + 1]);
            if (((t >> 1) & (QSA5_GRP/2 - 1)) == QSA5_GRP/2 - 1) { asm volatile("" ::: "memory"); }
        }
    }
    if (hi == 0) { ml_s[ks][r][0] = m; ml_s[ks][r][1] = l; }
    __syncthreads();
    float mt = -INFINITY;
#pragma unroll
    for (int k = 0; k < KS; ++k) { mt = fmaxf(mt, ml_s[k][r][0]); }
    const float myf = m == -INFINITY ? 0.f : exp2f((m - mt) * QSA3_L2E);
    float lt = 0.f;
#pragma unroll
    for (int k = 0; k < KS; ++k) { const float mk = ml_s[k][r][0]; lt += mk == -INFINITY ? 0.f : ml_s[k][r][1] * exp2f((mk - mt) * QSA3_L2E); }
#pragma unroll
    for (int t = 0; t < 16; ++t)
#pragma unroll
        for (int e = 0; e < 8; ++e) { O[t][e] *= myf; }
    if constexpr (KS > 1) {
        // tree-free: waves 1..KS-1 add into wave 0 one after another, 4 tiles at a time
        for (int k = 1; k < KS; ++k) {
#pragma unroll
            for (int tb = 0; tb < 16; tb += 4) {
                if (ks == k) {
#pragma unroll
                    for (int t = 0; t < 4; ++t)
#pragma unroll
                        for (int e = 0; e < 8; ++e) { xs[(t*8 + e)*32 + lane] = O[tb + t][e]; }
                }
                __syncthreads();
                if (ks == 0) {
#pragma unroll
                    for (int t = 0; t < 4; ++t)
#pragma unroll
                        for (int e = 0; e < 8; ++e) { O[tb + t][e] += xs[(t*8 + e)*32 + lane]; }
                }
                __syncthreads();
            }
        }
    }
    if (ks != 0) { return; }
    const float inv = lt > 0.f ? 1.f / lt : 0.f;
    float * st = reinterpret_cast<float *>(qs);    // [16 rows x 64 dims] f32 = 4 KB
#pragma unroll
    for (int tb = 0; tb < 16; tb += 4) {
#pragma unroll
        for (int t = 0; t < 4; ++t)
#pragma unroll
            for (int e = 0; e < 8; ++e) { st[r * 64 + 16*t + 2*e + hi] = O[tb + t][e] * inv; }
        __builtin_amdgcn_s_waitcnt(0); __builtin_amdgcn_wave_barrier();
#pragma unroll
        for (int c = 0; c < 6; ++c) {    // 12 rows x 64 dims: 2 rows per pass
            const int rr = 2*c + (lane >> 4), dq = 4 * (lane & 15);
            float * dst = reinterpret_cast<float *>(reinterpret_cast<char *>(out) + (size_t) query * s.o2 + (size_t) (kvh * s.gqa + rr) * s.o1) + 64*(tb/4) + dq;
            *reinterpret_cast<float4 *>(dst) = *reinterpret_cast<const float4 *>(st + rr * 64 + dq);
        }
        __builtin_amdgcn_s_waitcnt(0); __builtin_amdgcn_wave_barrier();
    }
}

// V cache view -> [block of 4 cells][256 dims][4 cells] in one pass: each thread gathers 2 dims of the 4 cells
// (four 4-byte loads, coalesced across the dims) and writes 16 bytes
static __global__ void qsa3_pack_v_kernel(const uint16_t * __restrict__ v, uint16_t * __restrict__ out,
        const size_t v1, const size_t v2, const int nblk, const int nkvh) {
    const size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x;   // (head, blk, dim pair)
    if (i >= (size_t) nkvh * nblk * 128) return;
    const int dp = (int) (i % 128); const size_t hb = i / 128; const int kb = (int) (hb % nblk), h = (int) (hb / nblk);
    const uint16_t * src = v + (size_t) h * v2 + (size_t) (4*kb) * v1 + 2*dp;
    uint32_t c[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) c[j] = *reinterpret_cast<const uint32_t *>(src + j * v1);
    // out[(h*nblk + kb)*1024 + d*4 + cell]: dims 2dp and 2dp+1
    const uint32_t d0a = (c[0] & 0xffffu) | (c[1] << 16), d0b = (c[2] & 0xffffu) | (c[3] << 16);
    const uint32_t d1a = (c[0] >> 16) | (c[1] & 0xffff0000u), d1b = (c[2] >> 16) | (c[3] & 0xffff0000u);
    *reinterpret_cast<uint4 *>(out + hb * 1024 + (size_t) (2*dp) * 4) = make_uint4(d0a, d0b, d1a, d1b);
}

// K cache view -> [kv head][block of 4 cells][16 dim chunks][4 cells][16 dims]: each thread moves 16 bytes
static __global__ void qsa3_pack_k_kernel(const uint16_t * __restrict__ k, uint16_t * __restrict__ out,
        const size_t k1, const size_t k2, const int nblk, const int nkvh) {
    const size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x;   // (head, blk, chunk, cell, half)
    if (i >= (size_t) nkvh * nblk * 128) return;
    const int hf = (int) (i & 1), cell = (int) ((i >> 1) & 3), c = (int) ((i >> 3) & 15); const size_t hb = i >> 7;
    const int kb = (int) (hb % nblk), h = (int) (hb / nblk);
    const uint4 v = *reinterpret_cast<const uint4 *>(k + (size_t) h * k2 + (size_t) (4*kb + cell) * k1 + 16*c + 8*hf);
    *reinterpret_cast<uint4 *>(out + hb * 1024 + c * 64 + cell * 16 + 8*hf) = v;
}

bool ggml_cuda_flash_attn_ext_qsa_supported(ggml_backend_cuda_context & ctx, const ggml_tensor * dst) {
    const auto * q = dst->src[0], * k = dst->src[1], * v = dst->src[2], * m = dst->src[3], * ids = dst->src[5];
    const auto * packed = dst->src[6], * pv = dst->src[7];
    if (!q || !k || !v || !ids || dst->src[4] || !packed || !pv ||
        !GGML_CUDA_CC_IS_RDNA3_5(ggml_cuda_info().devices[ctx.device].cc)) { return false; }
    if (q->ne[1] < 128) { return false; }
    float bias, softcap; memcpy(&bias, (const char *) dst->op_params + 4, 4); memcpy(&softcap, (const char *) dst->op_params + 8, 4);
    if (bias != 0 || softcap != 0 || q->type != GGML_TYPE_F32 || k->type != GGML_TYPE_F16 || v->type != GGML_TYPE_F16 ||
        dst->type != GGML_TYPE_F32 || ids->type != GGML_TYPE_I32 || q->ne[0] != 256 || k->ne[0] != 256 || v->ne[0] != 256 ||
        q->ne[1] < 1 || q->ne[1] > INT_MAX || k->ne[1] < 1 || k->ne[1] > 262140 || k->ne[1] % 4 ||
        k->ne[2] < 1 || q->ne[2] != 12*k->ne[2] || v->ne[2] != k->ne[2] || v->ne[1] != k->ne[1] ||
        q->ne[3] != 1 || k->ne[3] != 1 || v->ne[3] != 1 || q->nb[0] != 4 || k->nb[0] != 2 || v->nb[0] != 2 ||
        ids->nb[0] != 4 || ids->ne[1] < q->ne[1] || ids->ne[2] != 1 || ids->ne[3] != 1 || ids->ne[0] < 1 || ids->ne[0] > 2560 ||
        ids->nb[1] % 4 || uintptr_t(ids->data) % 4 || q->nb[1] % 16 || q->nb[2] % 16 || uintptr_t(q->data) % 16 ||
        !ggml_is_contiguous(dst)) { return false; }
    // DIRECT: src[6] (and/or src[7]) are the unpacked K (V) cache views themselves
    if (packed->ne[0] == 256) {
        if (packed->data != k->data || k->nb[1] % 16 || k->nb[2] % 16 || uintptr_t(k->data) % 16) { return false; }
    } else if (packed->type != GGML_TYPE_F16 || !ggml_is_contiguous(packed) || packed->ne[0] != 16 || packed->ne[1] != 4 ||
        packed->ne[2] != 16 || packed->ne[3] != k->ne[1]/4*k->ne[2] || uintptr_t(packed->data) % 16) { return false; }
    if (pv->ne[0] == 256) {
        if (pv->data != v->data || v->nb[1] < 512 || v->nb[1] % 4 || v->nb[2] % 4 || uintptr_t(v->data) % 4) { return false; }
    } else if (pv->type != GGML_TYPE_F16 || !ggml_is_contiguous(pv) || pv->ne[0] != 4 || pv->ne[1] != 256 ||
        pv->ne[2] != k->ne[1]/4*k->ne[2] || pv->ne[3] != 1 || uintptr_t(pv->data) % 8) { return false; }
    return !m || (m->type == GGML_TYPE_F16 && m->nb[0] == 2 && m->ne[0] >= k->ne[1] && m->ne[1] >= q->ne[1] && m->ne[2] == 1 && m->ne[3] == 1);
}

void ggml_cuda_flash_attn_ext_qsa(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const auto * q = dst->src[0], * k = dst->src[1], * m = dst->src[3], * ids = dst->src[5], * pk = dst->src[6], * pv = dst->src[7];
    float scale; memcpy(&scale, dst->op_params, 4);
    const int n_q = (int) q->ne[1], ns = (int) ids->ne[0], nk = (int) k->ne[1];
    // LLAMA_QSA5=0/1 forces the 4-query union kernel (qsa3) / the per-query kernel (qsa5)
    static const int want5 = [] { const char * e = getenv("LLAMA_QSA5"); return e ? atoi(e) : -1; }();
    static const bool vgather = [] { const char * e = getenv("LLAMA_QSA_VGATHER"); return e && atoi(e) != 0; }();
    const bool use5 = want5 != 0 && pk->ne[0] == 256 && !(dst->src[7]->ne[0] == 256 && vgather);   // qsa5: direct K, packed V
    const int G = use5 ? 1 : QSA3_G;
    const int ngroups = (n_q + G - 1) / G;
    const int cap = (G * ns + 3) & ~3;
    ggml_cuda_pool_alloc<uint16_t> ublk(ctx.pool(), (size_t) ngroups * cap);
    ggml_cuda_pool_alloc<uint16_t> umask(ctx.pool(), (size_t) ngroups * cap);
    ggml_cuda_pool_alloc<int>      ucount(ctx.pool(), (size_t) ngroups);
    ggml_cuda_pool_alloc<int>      srow(ctx.pool(), (size_t) n_q * ns);
    ggml_cuda_pool_alloc<int>      sflag(ctx.pool(), (size_t) n_q);
    {
        int np = 1; while (np < ns) { np <<= 1; }
        const ggml_cuda_kernel_launch_params launch(dim3(n_q), dim3(256), (size_t) np * sizeof(int), ctx.stream());
        ggml_cuda_kernel_launch(qsa3_rows_kernel, launch, (const int *) ids->data, ids->nb[1], ns, nk, srow.get(), sflag.get(), np);
        CUDA_CHECK(cudaGetLastError());
        const ggml_cuda_kernel_launch_params launch2(dim3(ngroups), dim3(QSA3_MERGE_LANES), (size_t) G * ns * sizeof(int), ctx.stream());
        ggml_cuda_kernel_launch(use5 ? qsa3_merge_kernel<1> : qsa3_merge_kernel<QSA3_G>, launch2, (const int *) ids->data, ids->nb[1], n_q, ns, nk,
                                (const int *) srow.get(), (const int *) sflag.get(), ublk.get(), umask.get(), ucount.get(), cap);
        CUDA_CHECK(cudaGetLastError());
    }
    const auto * v = dst->src[2];
    const bool direct = pk->ne[0] == 256;
    // an unpacked V view is packed here (one vectorized pass) unless LLAMA_QSA_VGATHER=1 gathers it in the kernel
    const bool direct_v = pv->ne[0] == 256 && vgather;
    const uint16_t * pv_data = (const uint16_t *) pv->data;
    ggml_cuda_pool_alloc<uint16_t> vpack(ctx.pool());
    if (pv->ne[0] == 256 && !vgather) {
        const int nblk = nk / 4, nkvh = (int) k->ne[2];
        vpack.alloc((size_t) nkvh * nblk * 1024);
        const size_t n = (size_t) nkvh * nblk * 128;
        qsa3_pack_v_kernel<<<(unsigned) ((n + 255) / 256), 256, 0, ctx.stream()>>>((const uint16_t *) pv->data, vpack.get(), v->nb[1] / 2, v->nb[2] / 2, nblk, nkvh);
        CUDA_CHECK(cudaGetLastError());
        pv_data = vpack.get();
    }
    static const bool stats = [] { const char * e = getenv("LLAMA_QSA3_STATS"); return e && atoi(e) != 0; }();
    hipStreamCaptureStatus qcs = hipStreamCaptureStatusNone;
    if (stats && hipStreamIsCapturing(ctx.stream(), &qcs) == hipSuccess && qcs == hipStreamCaptureStatusNone) {   // needs GGML_CUDA_DISABLE_GRAPHS=1 in prefill
        std::vector<int> uc(ngroups);
        CUDA_CHECK(cudaMemcpyAsync(uc.data(), ucount.get(), ngroups * sizeof(int), cudaMemcpyDeviceToHost, ctx.stream()));
        CUDA_CHECK(cudaStreamSynchronize(ctx.stream()));
        double sum = 0; int mx = 0; for (int x : uc) { sum += x; mx = std::max(mx, x); }
        fprintf(stderr, "qsa3: n_q=%d nk=%d ns=%d groups=%d  union blocks mean %.0f max %d  -> cells/query %.0f (ideal %d)\n",
                n_q, nk, ns, ngroups, sum / ngroups, mx, sum / ngroups * 4 / G, ns);
    }
    qsa3_layout layout{q->nb[1], q->nb[2], dst->nb[1], dst->nb[2], m ? m->nb[1] : 0, nk, n_q, 12, cap, scale,
                       k->nb[1] / 2, k->nb[2] / 2, v->nb[1] / 2, v->nb[2] / 2};
    if (use5) {
        const int nblk = nk / 4, nkvh = (int) k->ne[2];
        ggml_cuda_pool_alloc<uint16_t> kpack(ctx.pool(), (size_t) nkvh * nblk * 1024);
        const size_t n = (size_t) nkvh * nblk * 128;
        qsa3_pack_k_kernel<<<(unsigned) ((n + 255) / 256), 256, 0, ctx.stream()>>>((const uint16_t *) pk->data, kpack.get(), k->nb[1] / 2, k->nb[2] / 2, nblk, nkvh);
        CUDA_CHECK(cudaGetLastError());
        const ggml_cuda_kernel_launch_params launch5(dim3(n_q, k->ne[2]), dim3(32*QSA5_KS), 0, ctx.stream());
        ggml_cuda_kernel_launch(qsa5_attn_kernel<QSA5_KS>, launch5, (const float *) q->data, (const uint16_t *) kpack.get(), pv_data,
                                m ? (const uint16_t *) m->data : nullptr, (const uint16_t *) ublk.get(), (const uint16_t *) umask.get(),
                                (const int *) ucount.get(), (float *) dst->data, layout);
        CUDA_CHECK(cudaGetLastError());
        return;
    }
    const ggml_cuda_kernel_launch_params launch(dim3(ngroups, k->ne[2]), dim3(256), 0, ctx.stream());
    ggml_cuda_kernel_launch(direct ? (direct_v ? qsa3_attn_kernel<true, true> : qsa3_attn_kernel<true, false>)
                                   : (direct_v ? qsa3_attn_kernel<false, true> : qsa3_attn_kernel<false, false>), launch, (const float *) q->data, (const uint16_t *) pk->data, pv_data,
                            m ? (const uint16_t *) m->data : nullptr, (const uint16_t *) ublk.get(), (const uint16_t *) umask.get(),
                            (const int *) ucount.get(), (float *) dst->data, layout);
    CUDA_CHECK(cudaGetLastError());
}
