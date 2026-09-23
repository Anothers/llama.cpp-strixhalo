#include "hblt.cuh"

#if defined(GGML_USE_HIP) && defined(GGML_HIP_HIPBLASLT)
#include <hipblaslt/hipblaslt.h>
#include <map>
#include <mutex>
#include <tuple>
#include <vector>
#include <cstdlib>

namespace {
struct hblt_plan { hipblasLtMatmulAlgo_t algo; size_t ws; bool ok; };
std::mutex g_mtx;
std::map<std::tuple<int,int,int,int>, hblt_plan> g_plans;   // (device, M, K, T)
hipblasLtHandle_t g_handle[GGML_CUDA_MAX_DEVICES] = {nullptr};
constexpr size_t HBLT_WS = 64u << 20;

struct hblt_desc {
    hipblasLtMatmulDesc_t op = nullptr; hipblasLtMatrixLayout_t la = nullptr, lb = nullptr, lc = nullptr;
    hblt_desc(int M, int K, int T) {
        hipblasLtMatmulDescCreate(&op, HIPBLAS_COMPUTE_32F, HIP_R_32F);
        hipblasOperation_t ta = HIPBLAS_OP_T, tb = HIPBLAS_OP_N;
        hipblasLtMatmulDescSetAttribute(op, HIPBLASLT_MATMUL_DESC_TRANSA, &ta, sizeof(ta));
        hipblasLtMatmulDescSetAttribute(op, HIPBLASLT_MATMUL_DESC_TRANSB, &tb, sizeof(tb));
        // column-major view: A = W^T stored as K x M, B = X^T stored as K x T, C = D^T stored as M x T
        hipblasLtMatrixLayoutCreate(&la, HIP_R_16BF, K, M, K);
        hipblasLtMatrixLayoutCreate(&lb, HIP_R_16BF, K, T, K);
        hipblasLtMatrixLayoutCreate(&lc, HIP_R_32F,  M, T, M);
    }
    ~hblt_desc() { hipblasLtMatrixLayoutDestroy(la); hipblasLtMatrixLayoutDestroy(lb); hipblasLtMatrixLayoutDestroy(lc); hipblasLtMatmulDescDestroy(op); }
};
}

bool ggml_cuda_hblt_gemm_bf16(ggml_backend_cuda_context & ctx, const uint16_t * W, const uint16_t * X, float * D,
                              int M, int K, int T, cudaStream_t stream) {
    static const bool enabled = [] { const char * e = getenv("GGML_CUDA_HBLT"); return !e || atoi(e) != 0; }();
    if (!enabled) return false;
    const int dev = ctx.device;
    if (!g_handle[dev] && hipblasLtCreate(&g_handle[dev]) != HIPBLAS_STATUS_SUCCESS) return false;
    hblt_desc d(M, K, T);
    const float alpha = 1.0f, beta = 0.0f;
    ggml_cuda_pool_alloc<uint8_t> ws(ctx.pool(), HBLT_WS);

    hblt_plan plan{};
    {
        std::lock_guard<std::mutex> lk(g_mtx);
        auto it = g_plans.find({dev, M, K, T});
        if (it != g_plans.end()) plan = it->second;
    }
    if (!plan.ok) {
        // tuning times kernels on the stream, which is not allowed while it is being captured
        hipStreamCaptureStatus cs = hipStreamCaptureStatusNone;
        if (hipStreamIsCapturing(stream, &cs) != hipSuccess || cs != hipStreamCaptureStatusNone) return false;
        hipblasLtMatmulPreference_t pref; hipblasLtMatmulPreferenceCreate(&pref);
        size_t wsz = HBLT_WS;
        hipblasLtMatmulPreferenceSetAttribute(pref, HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &wsz, sizeof(wsz));
        std::vector<hipblasLtMatmulHeuristicResult_t> res(16); int got = 0;
        hipblasLtMatmulAlgoGetHeuristic(g_handle[dev], d.op, d.la, d.lb, d.lc, d.lc, pref, (int) res.size(), res.data(), &got);
        hipblasLtMatmulPreferenceDestroy(pref);
        if (got <= 0) return false;
        hipEvent_t e0, e1; hipEventCreate(&e0); hipEventCreate(&e1);
        float best = 1e30f; int bi = -1;
        for (int i = 0; i < got; ++i) {
            if (hipblasLtMatmul(g_handle[dev], d.op, &alpha, W, d.la, X, d.lb, &beta, D, d.lc, D, d.lc, &res[i].algo, ws.get(), HBLT_WS, stream) != HIPBLAS_STATUS_SUCCESS) continue;
            hipEventRecord(e0, stream);
            for (int r = 0; r < 3; ++r) hipblasLtMatmul(g_handle[dev], d.op, &alpha, W, d.la, X, d.lb, &beta, D, d.lc, D, d.lc, &res[i].algo, ws.get(), HBLT_WS, stream);
            hipEventRecord(e1, stream); hipEventSynchronize(e1);
            float ms = 0; hipEventElapsedTime(&ms, e0, e1);
            if (ms < best) { best = ms; bi = i; }
        }
        hipEventDestroy(e0); hipEventDestroy(e1);
        if (bi < 0) return false;
        plan = { res[bi].algo, res[bi].workspaceSize, true };
        GGML_LOG_INFO("hblt: M=%d K=%d T=%d -> algo %d of %d, %.3f ms (%.1f TFLOPS)\n", M, K, T, bi, got, best / 3,
                      2.0 * M * K * T / (best / 3 * 1e-3) / 1e12);
        std::lock_guard<std::mutex> lk(g_mtx);
        g_plans[{dev, M, K, T}] = plan;
    }
    return hipblasLtMatmul(g_handle[dev], d.op, &alpha, W, d.la, X, d.lb, &beta, D, d.lc, D, d.lc, &plan.algo, ws.get(), HBLT_WS, stream) == HIPBLAS_STATUS_SUCCESS;
}
#else
bool ggml_cuda_hblt_gemm_bf16(ggml_backend_cuda_context &, const uint16_t *, const uint16_t *, float *, int, int, int, cudaStream_t) { return false; }
#endif
