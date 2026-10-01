#!/usr/bin/env bash
# Determinism: the 29K summary prompt, temperature 0, fresh prefill (cache_prompt false), 3 runs per config.
set -u
S=/home/zhangde/bench/_scratch; R=${R:-/home/zhangde/releases/llama-qwen4exp-hip10-20261001-hb12b/bin}
M=/data/qwen38-flash-next/UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf
gate(){ sudo /home/zhangde/tools/RyzenAdj/build/ryzenadj --stapm-limit=115000 --fast-limit=130000 --slow-limit=115000 --tctl-temp=88 >/dev/null 2>&1
  sudo sh -c 'for d in /sys/class/drm/card*/device; do [ -w $d/pp_od_clk_voltage ] || continue; echo "s 1 2800" > $d/pp_od_clk_voltage; echo c > $d/pp_od_clk_voltage; done' 2>/dev/null; }
stop_srv(){ [ -n "${SPID:-}" ] && { kill $SPID 2>/dev/null; for i in $(seq 20); do kill -0 $SPID 2>/dev/null || break; sleep 1; done; kill -9 $SPID 2>/dev/null; }; SPID=; sleep 3; }
restore(){ stop_srv; sudo systemctl start llama-ds4; gate
  for i in $(seq 100); do curl -s -m 2 localhost:8080/health | grep -q '"ok"' && { echo "OK 生产已恢复"; break; }; sleep 3; done; }
trap restore EXIT
sudo systemctl stop llama-ds4; sleep 3; gate
MTP="-md /data/qwen38-flash-next/mtp/mtp-shared-Q8_0.gguf --spec-type draft-mtp -ngld 99 --spec-draft-device ROCm0"
for cfg in ${CFGS:-A B2 B3 B3t0}; do
  case $cfg in
    A)    extra="";                              envx="X=1";;
    B2)   extra="$MTP --spec-draft-n-max 2";     envx="LLAMA_MTP_PREFILL_TAIL=4096";;
    B3)   extra="$MTP --spec-draft-n-max 3";     envx="LLAMA_MTP_PREFILL_TAIL=4096";;
    B3t0) extra="$MTP --spec-draft-n-max 3";     envx="X=1";;
    B2t0) extra="$MTP --spec-draft-n-max 2";     envx="X=1";;
  esac
  env HSA_OVERRIDE_GFX_VERSION=11.5.1 GGML_HIP_ENABLE_UNIFIED_MEMORY=1 LLAMA_ATTN_ROT_DISABLE=1 GGML_CUDA_DISABLE_GRAPHS=1 GGML_CUDA_MMB_MIN_T_ID=64 \
    LD_LIBRARY_PATH=$R:/home/zhangde/rocm-10.0/lib GGML_CUDA_HBLT_DB=$R/hblt.db LLAMA_SPEC_DRAFT_UBATCH=8192 $envx ${DET_ENV:-} \
    $R/llama-server.real -m $M -dev ROCm0 -ngl 999 -c 131072 -fa on -fit off -np 1 -b 8192 -ub 8192 -ctk f16 -ctv f16 --no-context-shift --cache-ram 0 \
    --jinja --chat-template-file /home/zhangde/chat-templates/qwen38-flash-next-midsystem.jinja --reasoning-format auto -t 16 \
    --load-mode none --lazy-mode on-direct --host 127.0.0.1 --port 8094 $extra ${SRV_EXTRA:-} > $S/det_$cfg.log 2>&1 & SPID=$!
  up=0; for i in $(seq 200); do kill -0 $SPID 2>/dev/null || break; curl -s -m 2 localhost:8094/health | grep -q '"ok"' && { up=1; break; }; sleep 3; done
  [ $up = 1 ] || { echo "!! $cfg failed to start"; tail -3 $S/det_$cfg.log; exit 1; }
python3 - $cfg <<'PYEOF'
import json,urllib.request,sys
real=open('/home/zhangde/bench/_scratch/real_text.txt',encoding='utf-8',errors='replace').read()
outs=[]
for rep in range(3):
    body={"messages":[{"role":"user","content":real[:96000]+"\n\n请用三句话总结上文的主要内容。"}],"max_tokens":48,"temperature":0,
          "cache_prompt":False,"chat_template_kwargs":{"enable_thinking":False}}
    r=urllib.request.Request("http://127.0.0.1:8094/v1/chat/completions",data=json.dumps(body).encode(),headers={"Content-Type":"application/json"})
    d=json.load(urllib.request.urlopen(r,timeout=1800)); outs.append(d["choices"][0]["message"]["content"].replace("\n"," "))
for o in outs: print(f"  [{sys.argv[1]}] {o[:60]}",flush=True)
print(f"  [{sys.argv[1]}] distinct outputs: {len(set(outs))}",flush=True)
PYEOF
  stop_srv
done
