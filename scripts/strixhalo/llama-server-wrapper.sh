#!/usr/bin/env bash
# Validated hb21 (two slots); fatal exits fall back to hb17 with ONE slot at 262144, batch 8K. hb22 = hb21 + prompt cache evicts by system MemAvailable (LLAMA_CACHE_RAM_RESERVE_MIB, default 10240). hb23 = hb22 + dense-QSA ubatch cap for image sequences (LLAMA_DENSE_UBATCH_BUDGET_MIB, default 3072). hb24 = hb23 + text after an image keeps scalar QSA (LLAMA_QSA_IMAGE_SCALAR=0 restores dense).
export HSA_OVERRIDE_GFX_VERSION=11.5.1
export GGML_HIP_ENABLE_UNIFIED_MEMORY=1
export LLAMA_ATTN_ROT_DISABLE=1
export GGML_CUDA_DISABLE_GRAPHS=1
export GGML_CUDA_MMB_MIN_T_ID=64
export LLAMA_BATCH_EMBD_POOL=1
export LLAMA_SPEC_DRAFT_UBATCH=8192
export LLAMA_MTP_PREFILL_TAIL=4096
export LLAMA_MTP_PREFILL_KV_ONLY=1
export LLAMA_MTP_PREFILL_SYNC=0
export LLAMA_LAZY_CACHE_MB=0
export LLAMA_PLE_STAGING_POOL=0
export GGML_CUDA_GDN_PACK_PREFETCH=1
export GGML_CUDA_HC_NORM_PROJECT=1
hb21_bin="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
hb21_flag=/home/zhangde/kvcache/.hb24_crashed
hb21_fallback=/home/zhangde/releases/llama-qwen4exp-hip10-20261010-hb17/bin/llama-server
hb21_stopping=0
hb21_child=
hb21_stop() {
  hb21_stopping=1
  if [[ -n "$hb21_child" ]]; then kill -TERM "$hb21_child" 2>/dev/null || true; fi
}
trap hb21_stop TERM INT HUP
if [[ ! -e "$hb21_flag" ]]; then
  LD_LIBRARY_PATH="$hb21_bin:/home/zhangde/rocm-10.0/lib" GGML_CUDA_HBLT_DB="$hb21_bin/hblt.db" \
    "$hb21_bin/llama-server.real" "$@" &
  hb21_child=$!
  wait "$hb21_child"
  hb21_rc=$?
  if (( hb21_stopping )); then
    wait "$hb21_child" 2>/dev/null || true
    exit 0
  fi
  case "$hb21_rc" in 0|129|130|143) exit "$hb21_rc" ;; esac
  printf 'hb21 exit=%s at %s\n' "$hb21_rc" "$(date -u +%FT%TZ)" > "$hb21_flag"
  printf 'hb21 exited %s; falling back to hb17, one slot, context 262144, batch/ubatch 8192\n' "$hb21_rc" >&2
fi
# hb17 cannot serve two non-unified slots (it asserts at startup): drop the slot, context and cache-size arguments
hb21_args=()
while (( $# )); do
  case "$1" in
    -b|--batch-size|-ub|--ubatch-size|-np|--parallel|-c|--ctx-size|-cram|--cache-ram)
      if (( $# < 2 )); then exit 64; fi
      shift 2 ;;
    --batch-size=*|--ubatch-size=*|--parallel=*|--ctx-size=*|--cache-ram=*|-b[0-9]*|-ub[0-9]*) shift ;;
    *) hb21_args+=("$1"); shift ;;
  esac
done
hb21_args+=(-np 1 -c 262144 --cache-ram 8192 -b 8192 -ub 8192)
exec "$hb21_fallback" "${hb21_args[@]}"
