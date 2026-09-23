#include "hblt.cuh"

#if defined(GGML_USE_HIP) && defined(GGML_HIP_HIPBLASLT)
#include <hipblaslt/hipblaslt.h>
#include <hipblaslt/hipblaslt-ext.hpp>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <mutex>
#include <string>
#include <tuple>
#include <vector>

// On gfx1151 hipBLASLt ships ~750 bf16 x bf16 -> bf16 solutions (TRANSA=T) but only 9 for an f32 output, and the
// heuristic rarely ranks the fast ones first. Every supported solution is timed once per (M, K, T-bucket, out type)
// and the winner's index is kept in a small text database (GGML_CUDA_HBLT_DB, default ~/.cache/ggml-hblt.db) so a
// restarted server does not tune again.
namespace {
using hblt_key = std::tuple<int, int, int, int>;   // M, K, T bucket, out bf16
struct hblt_plan { hipblasLtMatmulAlgo_t algo; bool ok; };

std::mutex g_mtx;
std::map<hblt_key, hblt_plan> g_plans;
std::map<hblt_key, int> g_db;                        // key -> solution index
bool g_db_loaded = false;
hipblasLtHandle_t g_handle[GGML_CUDA_MAX_DEVICES] = {nullptr};
constexpr size_t HBLT_WS = 64u << 20;

int hblt_bucket(int T) {
    if (T <= 1024) return (T + 63) / 64 * 64;
    if (T <= 8192) return (T + 511) / 512 * 512;
    return (T + 2047) / 2048 * 2048;
}

std::string hblt_db_path() {
    if (const char * e = getenv("GGML_CUDA_HBLT_DB")) return e;
    const char * home = getenv("HOME");
    return std::string(home ? home : "/tmp") + "/.cache/ggml-hblt.db";
}

void hblt_db_load() {
    if (g_db_loaded) return;
    g_db_loaded = true;
    FILE * f = fopen(hblt_db_path().c_str(), "r");
    if (!f) return;
    int M, K, Tb, ob, idx;
    while (fscanf(f, "%d %d %d %d %d", &M, &K, &Tb, &ob, &idx) == 5) g_db[{M, K, Tb, ob}] = idx;
    fclose(f);
}

void hblt_db_append(const hblt_key & k, int idx) {
    FILE * f = fopen(hblt_db_path().c_str(), "a");
    if (!f) return;
    fprintf(f, "%d %d %d %d %d\n", std::get<0>(k), std::get<1>(k), std::get<2>(k), std::get<3>(k), idx);
    fclose(f);
}

struct hblt_desc {
    hipblasLtMatmulDesc_t op = nullptr; hipblasLtMatrixLayout_t la = nullptr, lb = nullptr, lc = nullptr;
    hblt_desc(int M, int K, int T, hipDataType out) {
        hipblasLtMatmulDescCreate(&op, HIPBLAS_COMPUTE_32F, HIP_R_32F);
        hipblasOperation_t ta = HIPBLAS_OP_T, tb = HIPBLAS_OP_N;
        hipblasLtMatmulDescSetAttribute(op, HIPBLASLT_MATMUL_DESC_TRANSA, &ta, sizeof(ta));
        hipblasLtMatmulDescSetAttribute(op, HIPBLASLT_MATMUL_DESC_TRANSB, &tb, sizeof(tb));
        // column-major view: A = W^T stored as K x M, B = X^T stored as K x T, C = D^T stored as M x T
        hipblasLtMatrixLayoutCreate(&la, HIP_R_16BF, K, M, K);
        hipblasLtMatrixLayoutCreate(&lb, HIP_R_16BF, K, T, K);
        hipblasLtMatrixLayoutCreate(&lc, out,        M, T, M);
    }
    ~hblt_desc() { hipblasLtMatrixLayoutDestroy(la); hipblasLtMatrixLayoutDestroy(lb); hipblasLtMatrixLayoutDestroy(lc); hipblasLtMatmulDescDestroy(op); }
};

__global__ void hblt_bf16_to_f32(const uint16_t * __restrict__ x, float * __restrict__ y, const size_t n) {
    const size_t i = ((size_t) blockIdx.x * blockDim.x + threadIdx.x) * 4;
    if (i + 4 <= n) {
        const uint2 v = *(const uint2 *) (x + i);
        *(float4 *) (y + i) = make_float4(__uint_as_float(v.x << 16), __uint_as_float(v.x & 0xffff0000u),
                                          __uint_as_float(v.y << 16), __uint_as_float(v.y & 0xffff0000u));
    } else {
        for (size_t j = i; j < n; ++j) y[j] = __uint_as_float((uint32_t) x[j] << 16);
    }
}
}

bool ggml_cuda_hblt_gemm(ggml_backend_cuda_context & ctx, const uint16_t * W, const uint16_t * X, void * D, bool out_bf16,
                         int M, int K, int T, cudaStream_t stream) {
    static const bool enabled = [] { const char * e = getenv("GGML_CUDA_HBLT"); return !e || atoi(e) != 0; }();
    if (!enabled) return false;
    const int dev = ctx.device;
    if (!g_handle[dev] && hipblasLtCreate(&g_handle[dev]) != HIPBLAS_STATUS_SUCCESS) return false;
    const hipDataType out = out_bf16 ? HIP_R_16BF : HIP_R_32F;
    hblt_desc d(M, K, T, out);
    const float alpha = 1.0f, beta = 0.0f;
    ggml_cuda_pool_alloc<uint8_t> ws(ctx.pool(), HBLT_WS);
    const hblt_key key{M, K, hblt_bucket(T), out_bf16 ? 1 : 0};

    hblt_plan plan{};
    int db_idx = -1;
    {
        std::lock_guard<std::mutex> lk(g_mtx);
        auto it = g_plans.find(key);
        if (it != g_plans.end()) plan = it->second;
        hblt_db_load();
        auto jt = g_db.find(key);
        if (jt != g_db.end()) db_idx = jt->second;
    }
    if (!plan.ok && db_idx >= 0) {
        std::vector<int> idx{db_idx};
        std::vector<hipblasLtMatmulHeuristicResult_t> res;
        size_t need = 0;
        if (hipblaslt_ext::getAlgosFromIndex(g_handle[dev], idx, res) == HIPBLAS_STATUS_SUCCESS && !res.empty() &&
            hipblaslt_ext::matmulIsAlgoSupported(g_handle[dev], d.op, &alpha, d.la, d.lb, &beta, d.lc, d.lc, res[0].algo, need) == HIPBLAS_STATUS_SUCCESS &&
            need <= HBLT_WS) {
            plan = { res[0].algo, true };
            std::lock_guard<std::mutex> lk(g_mtx);
            g_plans[key] = plan;
        }
    }
    if (!plan.ok) {
        // tuning times kernels on the stream, which is not allowed while it is being captured
        hipStreamCaptureStatus cs = hipStreamCaptureStatusNone;
        if (hipStreamIsCapturing(stream, &cs) != hipSuccess || cs != hipStreamCaptureStatusNone) return false;
        std::vector<hipblasLtMatmulHeuristicResult_t> all;
        hipblaslt_ext::getAllAlgos(g_handle[dev], hipblaslt_ext::GemmType::HIPBLASLT_GEMM, HIPBLAS_OP_T, HIPBLAS_OP_N,
                                   HIP_R_16BF, HIP_R_16BF, out, out, HIPBLAS_COMPUTE_32F, all);
        // the scratch output keeps the timing runs from touching D, which may alias an input of this op
        ggml_cuda_pool_alloc<uint8_t> scratch(ctx.pool(), (size_t) M * T * (out_bf16 ? 2 : 4));
        hipEvent_t e0, e1; hipEventCreate(&e0); hipEventCreate(&e1);
        float best = 1e30f; int bi = -1, nsup = 0;
        for (size_t i = 0; i < all.size(); ++i) {
            size_t need = 0;
            if (hipblaslt_ext::matmulIsAlgoSupported(g_handle[dev], d.op, &alpha, d.la, d.lb, &beta, d.lc, d.lc, all[i].algo, need) != HIPBLAS_STATUS_SUCCESS || need > HBLT_WS) continue;
            ++nsup;
            if (hipblasLtMatmul(g_handle[dev], d.op, &alpha, W, d.la, X, d.lb, &beta, scratch.get(), d.lc, scratch.get(), d.lc, &all[i].algo, ws.get(), HBLT_WS, stream) != HIPBLAS_STATUS_SUCCESS) continue;
            hipEventRecord(e0, stream);
            for (int r = 0; r < 2; ++r) hipblasLtMatmul(g_handle[dev], d.op, &alpha, W, d.la, X, d.lb, &beta, scratch.get(), d.lc, scratch.get(), d.lc, &all[i].algo, ws.get(), HBLT_WS, stream);
            hipEventRecord(e1, stream); hipEventSynchronize(e1);
            float ms = 0; hipEventElapsedTime(&ms, e0, e1);
            if (ms < best) { best = ms; bi = (int) i; }
        }
        hipEventDestroy(e0); hipEventDestroy(e1);
        (void) hipGetLastError();
        if (bi < 0) return false;
        plan = { all[bi].algo, true };
        const int idx = hipblaslt_ext::getIndexFromAlgo(all[bi].algo);
        GGML_LOG_INFO("hblt: M=%d K=%d T=%d out=%s -> solution %d (%d supported), %.3f ms (%.1f TFLOPS)\n", M, K, T,
                      out_bf16 ? "bf16" : "f32", idx, nsup, best / 2, 2.0 * M * K * T / (best / 2 * 1e-3) / 1e12);
        std::lock_guard<std::mutex> lk(g_mtx);
        g_plans[key] = plan;
        g_db[key] = idx;
        hblt_db_append(key, idx);
    }
    return hipblasLtMatmul(g_handle[dev], d.op, &alpha, W, d.la, X, d.lb, &beta, D, d.lc, D, d.lc, &plan.algo, ws.get(), HBLT_WS, stream) == HIPBLAS_STATUS_SUCCESS;
}

bool ggml_cuda_hblt_gemm_f32out(ggml_backend_cuda_context & ctx, const uint16_t * W, const uint16_t * X, float * D,
                                int M, int K, int T, cudaStream_t stream) {
    ggml_cuda_pool_alloc<uint16_t> tmp(ctx.pool(), (size_t) M * T);
    if (!ggml_cuda_hblt_gemm(ctx, W, X, tmp.get(), true, M, K, T, stream)) return false;
    const size_t n = (size_t) M * T;
    hblt_bf16_to_f32<<<(unsigned) ((n / 4 + 255) / 256 + 1), 256, 0, stream>>>(tmp.get(), D, n);
    return true;
}
#else
bool ggml_cuda_hblt_gemm(ggml_backend_cuda_context &, const uint16_t *, const uint16_t *, void *, bool, int, int, int, cudaStream_t) { return false; }
bool ggml_cuda_hblt_gemm_f32out(ggml_backend_cuda_context &, const uint16_t *, const uint16_t *, float *, int, int, int, cudaStream_t) { return false; }
#endif
