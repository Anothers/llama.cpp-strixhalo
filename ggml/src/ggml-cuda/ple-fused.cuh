#pragma once
#include "common.cuh"

// userdata tags of the qwen4exp PLE custom ops (keep in sync with src/models/qwen4exp.cpp)
#define GGML_PLE_TAG_GATE ((intptr_t) 0x504c4531)
#define GGML_PLE_TAG_NORM ((intptr_t) 0x504c4532)
#define GGML_PLE_TAG_OUT  ((intptr_t) 0x504c4533)

intptr_t ggml_cuda_ple_tag(const ggml_tensor * t);
bool     ggml_cuda_ple_supported(const ggml_tensor * t);
void     ggml_cuda_op_ple(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
