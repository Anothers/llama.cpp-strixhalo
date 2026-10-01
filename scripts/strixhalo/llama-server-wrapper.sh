#!/usr/bin/env bash
# hb12c: hb12b + per-context mmb BF16 cache/slots (fixes the cross-stream race behind nondeterministic MTP output). hb12b: per-context mmb BF16 marks (fixes the target corruption the K/V-only MTP prefill exposed); K/V-only prefill back on. hb12a: hb12 + Anthropic /v1/messages accepts system/developer roles inside messages. hb12: hb11 + MTP draft head runs K/V-only for prompt ubatches more than 4096 tokens before the prompt end (+1.2% long prefill).
# 安全网:若崩溃(abort 134 / segfault 139),写标记文件并改用 hb11 继续服务;删掉标记文件后重启服务即回到 hb12。
export HSA_OVERRIDE_GFX_VERSION=11.5.1 GGML_HIP_ENABLE_UNIFIED_MEMORY=1 LLAMA_ATTN_ROT_DISABLE=1
export GGML_CUDA_DISABLE_GRAPHS=1 GGML_CUDA_MMB_MIN_T_ID=64
D="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FLAG=/home/zhangde/kvcache/.hb12c_crashed
if [ ! -e "$FLAG" ]; then
  LD_LIBRARY_PATH="$D:/home/zhangde/rocm-10.0/lib" GGML_CUDA_HBLT_DB="$D/hblt.db" LLAMA_SPEC_DRAFT_UBATCH=8192 LLAMA_MTP_PREFILL_TAIL=4096 \
    "$D/llama-server.real" "$@"
  rc=$?
  [ $rc -ne 134 ] && [ $rc -ne 139 ] && exit $rc
  echo "hb12c 退出码 $rc,切换到 hb11" >&2; date > "$FLAG"
fi
exec /home/zhangde/releases/llama-qwen4exp-hip10-20261001-hb11/bin/llama-server "$@"
