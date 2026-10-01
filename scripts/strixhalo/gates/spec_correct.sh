#!/usr/bin/env bash
# Is MTP speculative decoding exact? hb12a build: A = no MTP (greedy + top-5 logprobs), B = production MTP config,
# B0 = MTP with LLAMA_MTP_PREFILL_TAIL unset. Same prompts, temperature 0; divergences are judged by A's logprob gap.
set -u
S=/home/zhangde/bench/_scratch; R=${R:-/home/zhangde/releases/llama-qwen4exp-hip10-20261001-hb12a/bin}
M=/data/qwen38-flash-next/UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf
gate(){ sudo /home/zhangde/tools/RyzenAdj/build/ryzenadj --stapm-limit=115000 --fast-limit=130000 --slow-limit=115000 --tctl-temp=88 >/dev/null 2>&1
  sudo sh -c 'for d in /sys/class/drm/card*/device; do [ -w $d/pp_od_clk_voltage ] || continue; echo "s 1 2800" > $d/pp_od_clk_voltage; echo c > $d/pp_od_clk_voltage; done' 2>/dev/null; }
stop_srv(){ [ -n "${SPID:-}" ] && { kill $SPID 2>/dev/null; for i in $(seq 20); do kill -0 $SPID 2>/dev/null || break; sleep 1; done; kill -9 $SPID 2>/dev/null; }; SPID=; sleep 3; }
restore(){ stop_srv; sudo systemctl start llama-ds4; gate
  for i in $(seq 100); do curl -s -m 2 localhost:8080/health | grep -q '"ok"' && { echo "OK 生产已恢复"; break; }; sleep 3; done; }
trap restore EXIT
sudo systemctl stop llama-ds4; sleep 3; gate
BASE="-m $M -dev ROCm0 -ngl 999 -c 131072 -fa on -fit off -np 1 -b 8192 -ub 8192 -ctk f16 -ctv f16 --no-context-shift --cache-ram 0
  --jinja --chat-template-file /home/zhangde/chat-templates/qwen38-flash-next-midsystem.jinja --reasoning-format auto -t 16
  --load-mode none --lazy-mode on-direct --host 127.0.0.1 --port 8094"
MTP="-md /data/qwen38-flash-next/mtp/mtp-shared-Q8_0.gguf --spec-type draft-mtp --spec-draft-n-max ${NMAX:-2} -ngld 99 --spec-draft-device ROCm0"
COMMON="HSA_OVERRIDE_GFX_VERSION=11.5.1 GGML_HIP_ENABLE_UNIFIED_MEMORY=1 LLAMA_ATTN_ROT_DISABLE=1 GGML_CUDA_DISABLE_GRAPHS=1 GGML_CUDA_MMB_MIN_T_ID=64
  LD_LIBRARY_PATH=$R:/home/zhangde/rocm-10.0/lib GGML_CUDA_HBLT_DB=$R/hblt.db LLAMA_SPEC_DRAFT_UBATCH=8192"
for cfg in ${CFGS:-A B B0}; do
  case $cfg in
    A)  extra=""; envx="X=1";;
    B)  extra="$MTP"; envx="LLAMA_MTP_PREFILL_TAIL=4096";;
    B0) extra="$MTP"; envx="X=1";;
  esac
  env $COMMON $envx $R/llama-server.real $BASE $extra ${SRV_EXTRA:-} > $S/speccorr_$cfg.log 2>&1 & SPID=$!
  up=0; for i in $(seq 200); do kill -0 $SPID 2>/dev/null || break; curl -s -m 2 localhost:8094/health | grep -q '"ok"' && { up=1; break; }; sleep 3; done
  [ $up = 1 ] || { echo "!! test server $cfg failed to start:"; tail -3 $S/speccorr_$cfg.log; exit 1; }
  python3 $S/spec_correct.py $cfg
  stop_srv
done
python3 $S/spec_correct.py compare
