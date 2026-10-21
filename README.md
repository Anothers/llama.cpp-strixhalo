# llama.cpp-strixhalo：Qwen3.8-Flash-Next 在 Strix Halo（gfx1151）上的优化版

这个仓库是 llama.cpp 的分支，专门为 **AMD Ryzen AI Max+ 395（Strix Halo，Radeon 8060S，gfx1151，RDNA3.5，128 GB 统一内存）** 运行 **Qwen3.8-Flash-Next UD-Q4_K_XL** 做了优化，后端为 HIP（ROCm 10）。

- `main` / `hb24-prod`：当前生产版本，即 hb24 加上草稿头生成脚本。
- `hb25-draft-head`：词表子集草稿头实验。实测更慢，生产没有使用。

本文先列出与上游相比改了哪些算子、各自的实测收益，然后是运行方式、开关和验证方法。后面附带上游原版 README。

---

## 1. 结果

测试机器：同一台 EVO-X3 / Ryzen AI Max+ 395，GPU 频率上限 2800 MHz，生产配置，开 MTP 投机解码。

| 指标 | 09-28（hb7，开始做算子优化之前） | 现在（hb24 + Q4_0 草稿头） |
|---|---|---|
| 预填充 32K | 约 1280 t/s | 约 1520 t/s |
| 预填充 131K | — | 约 1520 t/s |
| 预填充 244K（单槽，接近 256K 上限） | — | 1553 t/s |
| 解码：写代码，短上下文 | 37 t/s | 48.5~49 t/s |
| 解码：写代码，8K 上下文 | — | 48.5 t/s |
| 解码：写代码，128K 上下文 | — | 41.3 t/s（hb24 加 Q4_0 头之前；Q4_0 头在 128K 下未单独测） |
| 解码：散文，短上下文 | — | 34~35 t/s |
| 200K 文本中夹图片 | 会 OOM 或退回稠密路径（约 206 t/s） | 1521 t/s |
| 并发 | 单槽 | 双槽，每槽 262144 上下文 |

关于这些数字：
- 解码数据在 temperature 1.0 下测得，接受率会随内容变化，写代码时最高。
- 预填充数据包含视觉编码器之外的完整流程。
- 作为参照，同一台机器上 Halogen 的 32K 预填充为 1276~1412 t/s。

## 2. 代码来源（请先读）

本仓库以 upstream llama.cpp `718f7b417`（2026-09-12）为基础，在上面依次叠加了以下几部分。

| 来源 | 内容 |
|---|---|
| **Piotr Wilkin（pwilkin）的 qwen4exp 分支**，约 48 个提交 | Qwen3.8-Flash-Next 架构支持；第一版 HIP 内核：稀疏选择注意力 qsa3、WMMA lightning indexer、分块 GATED_DELTA_NET、HC（hyper-connection）融合、radix top-k、head-size-256 WMMA flash attention 等；后端调度器修复 |
| 上游 PR | #27694（MTP 概率草稿采样）、#29924（temp>0 时 n-gram 草稿被截断的修复）、#30020、#29633 |
| **本仓库的工作**，作者 Anothers，49 个提交 | 下文第 3~5 节列出的内容 |

下文只列**我们自己做的改动**。pwilkin 的内核是我们的起点，凡是在它基础上继续改的地方都会注明。

## 3. 改了哪些算子

文中路径默认在 `ggml/src/ggml-cuda/` 下。HIP 也走这套 CUDA 代码，编译时用 hipify 转换。收益都是同机 A/B 测得的。

### 3.1 MoE 和稠密矩阵乘（mmb 内核与 hipBLASLt），预填充的大头

| 改动 | 说明 | 收益 |
|---|---|---|
| Q4_K 路由专家 gate+up 反量化分摊到 256 线程（`c551e6079`） | 原来使用 64 行 tile 时，每个 k 步只有 64 个线程在反量化，其余线程都在 barrier 处等待 | 路由 GLU 在 8K 时从 23.6 提升到 24.8 TFLOPS，32K 时从 29.4 提升到 31.0；结果逐位相同 |
| 所有 mmb 量化类型的路由专家 down 都输出 bf16（`0259cdce4`） | UD GGUF 里有 Q5_1 专家，原来只有部分类型能输出 bf16 | 下游少一次 f32 读写 |
| 高瘦 Q8_0 权重（HC down，320×10240）使用 bf16 影子权重，并加 384 行 tile（`336cc385f`、`f12ef64dd`） | UD 版的 hc_*_down 是 Q8_0 | — |
| **hipBLASLt bf16 GEMM + 按形状自动调优**（`11d0acdf1`、`5baa382b2`） | 枚举全部 hipBLASLt 解，bf16 输出时 gfx1151 上约有 380 个（f32 输出只有 9 个）；按 (M, K, T 分档) 计时，最优解写进 `GGML_CUDA_HBLT_DB` | HC down 在 T=8192 时从 3.0 降到 1.6 ms |
| 行数 ≤4 的 f32 权重改走流式 kernel（`5baa382b2`） | 用于 hc inject 投影 | 从 3.4 降到 1.8 ms |
| MoE 加权求和融合（`1a03dbfa1`、`5b5d780b3`） | 不再为 `ffn_moe_weighted` 单独分配张量，改为原地计算；同时修正融合匹配器，原地情形下原来会悄悄退回 MUL+ADD，导致预填充慢 9~10% | 16K 计算缓冲从 6138 降到 5338 MiB |
| **共享专家的门控残差折进路由 MoE 归约**（`759d70829`） | 在归约 kernel 里一次算完 `Σ w_e·expert_e + shexp·sigmoid(gate)`，moe_out 不用写出再读回 | pp32768 从 1478 提升到 1491 |
| MoE 走 mmb 的最小 token 数可配置（`d7f4206f5`） | `GGML_CUDA_MMB_MIN_T_ID`，生产设为 64 | — |
| 试过后回退的 | 稠密高 tile 的 split-K（`8c74285b4`）、32×64 短输出 tile（`283e9431e`） | 都更慢，已 revert |

### 3.2 GATED_DELTA_NET（线性注意力层）

| 改动 | 说明 | 收益 |
|---|---|---|
| **分块 WMMA 预填充路径**（`db296ca3a`，新文件 `gdn-chunk.cu`） | 拆成两个 kernel。`gdn_chunk_prep` 按 heads×chunks 并行，算出与状态无关的项：Gram 矩阵、(I+L)⁻¹、衰减系数。`gdn_chunk_main` 把 128×128 的状态常驻在 WMMA 累加寄存器里，每 16 token 的块执行 26 次 WMMA。最后 K 个 token 按单 token 块处理，保证投机解码回滚用的快照精确 | 解决的是"逐 token 顺序递归"的瓶颈，相关分析见 Halogen 差距计划 |
| GDN 的 q/k l2 norm 放进 tiled kernel 内部（`ab9236c84`） | `try_fuse` 先把原始视图挂起，交给消费它们的 GATED_DELTA_NET 在 tile 内归一化 | GDN 每次调用多 1.0 ms，但每层省掉 rms_scale 的 1.25 ms |
| Q/K 归一化与打包的预取融合（`671961b45`） | `GGML_CUDA_GDN_PACK_PREFETCH` | — |
| GDN 链改为 bf16（`5baa382b2`） | 只有融合 conv 读取 qkv 投影时，直接写 bf16；gated norm 读 bf16 的 gate、写 bf16 的输出，供 out 投影使用 | — |

### 3.3 QSA 稀疏注意力（qwen4exp 的长上下文注意力）

| 改动 | 说明 | 收益 |
|---|---|---|
| qsa3 原地读 K，在 launcher 内打包 V（`47affae6c`） | 原来每个 ubatch、每个 QSA 层都要把整个 K/V cache 重排一遍，200K 时每层约 21 ms | — |
| **逐 query 稀疏注意力 kernel qsa5**（`928925988`） | 每个 (query, kv head) 一个 block，只读这个 query 自己的 512 个块（原来是读 4 个 query 的并集）。K 按块重排成 `[blk][16 维块][4 cell][16 维]`，每次加载都用满 128 B cache line（原来 16 行里每行只用 32 B）。t 和 t+1 分别放在两个半 wave，用 `permlanex16` 交换 | 64K 深度时每次调用从 43.0 降到 27.0 ms；pp32768 从 1409 提升到 1449 |
| lightning indexer 提速（`340475f40`、`984c87605`，在 pwilkin 的 WMMA indexer 上继续改） | 64 个块 key 常驻寄存器，跨 8 个 16-query tile 复用；下一个 query tile 用展开的 16 字节加载与当前 tile 的乘法重叠；head 权重和可见性边界随 query 预取一起放进 LDS；key 行改用无分支的 16 字节加载 | 64K 深度时每次 8K 预填充从 531 降到 396 ms；pp8192@131K 从 1147 提升到 1180 |
| 宽行 top-k（≥4096 列）（`340475f40`） | 一遍 12-bit 直方图加并行后缀搜索，阈值桶压缩到 LDS 里解决；超容量时退回 8-bit 循环 | 同上 |
| 各 strip 的 top-k 原地写入（`984c87605`） | 用 `ggml_set_2d_inplace` 代替 concat 树，原来每行要被复制 log2(strips) 次 | — |
| qsa3_rows 改用 bitonic 排序（`984c87605`） | 原来是 O(n²) 的秩排序，每个浅层约 7 ms | pp8192 从 1317 提升到 1336 |
| **双槽（非统一 KV）**（`81ab4b9b1`、`51bfc21e2`） | 每个 ubatch 只放一个序列，所以单流的块选择、无 mask 和增量 QSA 快路径全部可用；图预留按 1 个流计算；增量 QSA 状态和块 key 缓存按流分开 | 双槽的计算缓冲与单槽完全相同，只多一份 KV（7.83 GiB） |
| 有图片时的 QSA 路径（`a45d024be`） | 修复前：序列里只要有图片 cell，后面所有 ubatch 都会退回稠密 mask（200K 时约 20 GB，主机和 GTT 各一份，会 OOM）。修复后：只有当图片 cell 的位置落在本批查询范围内才走稠密路径，图片之后的文本继续走标量 QSA；稠密路径按 n_kv 限制 ubatch 大小 | 40K 加图片从 847 提升到 1490 t/s；200K 加图片 1521 t/s，不再 OOM |

### 3.4 逐元素算子融合（预填充）

| 融合 | 替代的节点链 | 开关 |
|---|---|---|
| `sigmul_add`：a + b·sigmoid(g)（`a4d4147e8`） | 共享专家残差的 SIGMOID→MUL→ADD | `GGML_CUDA_SIGMUL_ADD=0` |
| `attn_gate`：attn·sigmoid(gate)（`a4d4147e8`） | 原地读取 Q 里的 gate 那一半，直接输出 BF16 给 mmb | `GGML_CUDA_ATTN_GATE=0` |
| `rms_scale`（`a4d4147e8`） | GDN q/k head 的 RMS_NORM→SCALE，改为每行一个 warp | `GGML_CUDA_RMS_SCALE=0` |
| **q/k norm + (i)mrope**（`ab9236c84`） | 注意力 q/k 上的 RMS_NORM→MUL(w)→ROPE(multi/imrope)，一遍完成，每行一个 wave，旋转在寄存器里做，归约顺序与 rms_rows_f32 一致 | `GGML_CUDA_NORM_ROPE=0` |
| HC norm + 4 路输出投影（`671961b45`，`hc-cn.cu`） | hyper-connection 的 norm 和 f32 inject 投影 | `GGML_CUDA_HC_NORM_PROJECT` |

`a4d4147e8` 一组三个融合合计：pp32768 从 1373 提升到 1406 t/s。

### 3.5 PLE（per-layer embedding，28.8 GB 的 n-gram 表）

| 改动 | 说明 | 收益 |
|---|---|---|
| **PLE 混合改成 3 个自定义算子**（`5da15b71a`，新文件 `ple-fused.cu`） | 门控 `sigmoid(sgn(s)·sqrt(max(\|s\|,1e-6)))`、gated-value norm、最终的 hidden+gated+conv 残差，各用一个 kernel。原来是约 25 个逐元素/归约节点，作用在 [E·hc, T] 上。`qwen4exp.cpp` 里有 CPU 参考实现 | 29K 和 71K 下贪婪输出不变 |
| 惰性读取器：常驻线程池，每个任务读一行（`48438f048`，`src/llama-lazy-reader.h`） | PLE 表不进内存，按需 pread 读盘。解码时一次 gather 只读 50~200 行，每行约 100 B 的随机读，原来全落在一个线程上，GPU 空等。现在用 2×核数的常驻池并行读取 | 每次 gather 从 9.6 降到 1.7 ms；**MTP 解码从 37.0 提升到 41.5 t/s** |

### 3.6 MTP 投机解码

| 改动 | 说明 | 收益 |
|---|---|---|
| **草稿头单独量化成 Q4_0**（`36f569bd1`，`scripts/qwen4exp-mtp-draft-head.py`） | 草稿每步约 80% 的时间花在主模型的 output.weight（Q8_0，644 MiB）上。把它反量化后重新量化为 Q4_0（357 MiB），作为 `blk.48.nextn.shared_head_head.weight` 写进 MTP GGUF，代码不用改 | 草稿每步从 4.3 降到 2.86 ms；**解码提升 6~13%**；正确性门结果与 Q8_0 头相同 |
| 远离 prompt 末尾的预填充只算 K/V（`8f17b2b5d`、`04346cd26`） | 草稿头是单层，它的 K/V 只取决于层输入，所以只写 K/V 和 indexer K 就是精确的，可以跳过 attention、FFN 和 LM head。改成按 token 而不是按批次决定 | 226K 预填充从 1376 提升到 1395；草稿计算缓冲从 1744 降到 1063.5 MiB |
| eh_proj 中共享的 token 那一半只投影一次（`e31a64bf6`） | 拆成 W_h·h（遍历所有流）加上广播的 W_e·e，不再生成 repeat 加 concat 的中间结果（8192 token 时 640 MiB） | 8192 token 时从 80.8 降到 24.9 ms；pinned 内存从 1280.7 降到 640.7 MiB |
| 草稿提示阶段的主机拷贝（`7782bc6a0`） | 8 线程拷贝 hidden state（从 18 降到 7 ms）；`verify_h` 只保留前 64 行和最后一行（原来整份拷贝要 33 ms） | — |
| 草稿 micro-batch 单独设置（`97d5f443e`） | `LLAMA_SPEC_DRAFT_UBATCH` | — |

### 3.7 内存（为了 256K 双槽）

| 改动 | 收益 |
|---|---|
| 只以 BF16 写入的 F32 张量按一半大小分配（`4bd21862b`） | 计算缓冲在 ub8192 时从 3352 降到 2880 MiB，ub16384 时从 6704 降到 5760 MiB |
| 前面提到的 MoE 原地加权、MTP K/V-only 和 eh 拆分 | 见 3.1 和 3.6 |
| 计算缓冲按当前批次预留；embd ubatch 缓冲复用（`671961b45`） | — |
| prompt 缓存按 MemAvailable 淘汰（`a45d024be`） | Linux 超额分配时 bad_alloc 永远不会触发，所以改为先查可用内存 |

## 4. 修复的正确性 bug（都会让 MTP 输出和不开 MTP 时不一致）

- `1dab94272`、`877fc67a4`：mmb 的 BF16 标记、BF16 激活缓存和 producer slot 原来是进程级共享的。MTP 草稿上下文会清掉或覆盖目标模型正在读的数据，导致第二个长请求开始输出不确定。现在改成每个后端上下文各有一份。
- `667a892c4`：`flash_attn_mask_to_KV_max` 在最后一个 tile 越界读 mask，导致 Memory Fault 和服务退出。
- `2fecc1eb9`：服务端恢复采样器副本时把 rng 也回绕了，下一次验证会重复使用同一组随机数。
- `671961b45`：集成 GPU 上 host buffer 分配失败时返回 null，不再返回未映射的 CPU 内存。

## 5. 服务端（与算子无关）

- Anthropic `/v1/messages` 兼容（`022cbfe8a`、`375abf2a0`、`8ee1cf26a`）：tool_result 的顺序、tool_choice、thinking、在 `<think>` 内不匹配 stop 字符串、document 块（文本，PDF 按页渲染成图片）、messages 里出现的 system turn。

## 6. 构建和运行

```bash
cmake -S . -B build-opt -DGGML_HIP=ON -DGPU_TARGETS=gfx1151 -DCMAKE_BUILD_TYPE=Release -DLLAMA_BUILD_TESTS=ON \
      -DCMAKE_PREFIX_PATH=$ROCM -DCMAKE_HIP_COMPILER=$ROCM/llvm/bin/clang++
cmake --build build-opt --target llama-server llama-bench test-backend-ops -j 32
```

### 生产环境变量

| 变量 | 值 | 作用 |
|---|---|---|
| `HSA_OVERRIDE_GFX_VERSION` | 11.5.1 | |
| `GGML_HIP_ENABLE_UNIFIED_MEMORY` | 1 | |
| `GGML_CUDA_DISABLE_GRAPHS` | 1 | 关闭 HIP graphs，解码反而快 1~4% |
| `GGML_CUDA_MMB_MIN_T_ID` | 64 | MoE 从 64 个 token 起走 mmb |
| `GGML_CUDA_HBLT_DB` | `scripts/strixhalo/hblt.db` | hipBLASLt 调优表 |
| `LLAMA_ATTN_ROT_DISABLE` | 1 | |
| `LLAMA_BATCH_EMBD_POOL` | 1 | |
| `LLAMA_SPEC_DRAFT_UBATCH` | 8192 | |
| `LLAMA_MTP_PREFILL_TAIL` | 4096 | 距 prompt 末尾 4096 以外的部分只算 K/V |
| `LLAMA_MTP_PREFILL_KV_ONLY` | 1 | |
| `GGML_CUDA_GDN_PACK_PREFETCH` | 1 | |
| `GGML_CUDA_HC_NORM_PROJECT` | 1 | |

### 生产参数（双槽，每槽 256K）

```bash
llama-server -m Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf \
  --mmproj mmproj-F16.gguf --mmproj-device ROCm0 \
  -md mtp-shared-Q8_0-headQ4_0.gguf \
  --spec-type ngram-mod,draft-mtp --spec-draft-n-max 3 -ngld 99 --spec-draft-device ROCm0 --spec-draft-sampling probabilistic \
  -dev ROCm0 -ngl 999 -c 524288 -np 2 -b 16384 -ub 16384 -fa on -fit off \
  -ctk f16 -ctv f16 --no-context-shift --cache-ram 2048 \
  --jinja --reasoning-format auto -t 16 --load-mode none --lazy-mode on-direct
```

- 双槽时**不要加 `-bs`**（后端采样）。它在双槽下和 MTP+ngram 一起用会串扰出乱码，而且对速度没有帮助。
- `--lazy-mode on-direct` 表示 PLE 表留在磁盘上按需 pread，不占内存。

生成草稿头 MTP 文件：

```bash
python3 scripts/qwen4exp-mtp-draft-head.py Q4_0 mtp-shared-Q8_0.gguf <主模型目录> mtp-shared-Q8_0-headQ4_0.gguf
```

### 关闭开关（默认全开，用于 A/B 测试）

| 变量 | 关掉什么 |
|---|---|
| `LLAMA_QSA5=0` | qsa5，退回 qsa3 |
| `GGML_CUDA_GDN_CHUNK=0` | GDN 分块 WMMA 路径 |
| `GGML_CUDA_SIGMUL_ADD=0`、`GGML_CUDA_ATTN_GATE=0`、`GGML_CUDA_RMS_SCALE=0` | 逐元素融合 |
| `GGML_CUDA_NORM_ROPE=0`、`GGML_CUDA_GDN_QKNORM=0` | norm+rope 融合；GDN 内的 q/k norm |
| `GGML_CUDA_MOE_SIGMERGE=0` | 共享专家并入 MoE 归约 |
| `LLAMA_MOE_WEIGHTED_INPLACE=0` | MoE 加权原地计算 |
| `GGML_CUDA_TOPK_RADIX2=0` | 宽行 12-bit top-k |
| `LLAMA_PLE_FUSED=0` | PLE 自定义 kernel |
| `LLAMA_LAZY_SERIAL=1` | 惰性读取器退回单线程 |
| `LLAMA_QSA_SETSTRIP=0` | strip 原地写入 |
| `LLAMA_QSA_IMAGE_SCALAR=0` | 图片之后的文本恢复走稠密路径（旧行为） |
| `LLAMA_MTP_EH_SPLIT=0`、`LLAMA_MTP_PREFILL_TAIL_TOKENS=0` | MTP eh 拆分；按 token 判定 K/V-only |
| `LLAMA_UBATCH_PER_SEQ=0` | 双槽时允许一个 ubatch 混放多个序列 |

## 7. 验证

每次发布前都要跑：

- `scripts/strixhalo/gates/spec_correct.sh`：贪婪解码下，开 MTP 和不开 MTP 的输出必须逐字相同。唯一允许的分歧是近似并列：MTP 选的 token 在不开 MTP 时的 top-5 里排名 ≤2，且差距在 0.15 nats 以内。
- `scripts/strixhalo/gates/det_test.sh`：同一个服务进程里连续发 3 个全新的 29K 请求，输出必须完全一致。
- 算子测试：
  ```bash
  build-opt/bin/test-backend-ops test -b ROCm0 -o SIGMUL_ADD   # 另有 ATTN_GATE RMS_SCALE NORM_ROPE GDN_QKNORM LIGHTNING_INDEXER MTP_EH_PROJ
  ```

## 8. 试过但没有采用的

| 方案 | 结果 |
|---|---|
| 词表子集草稿头（10.6 万 id，分支 `hb25-draft-head`） | 写代码场景慢 4% |
| `--spec-draft-n-max 4` | 散文慢 10% |
| `--spec-p-min 0.5` | 变慢；0.3 时持平 |
| HIP graphs / `GGML_CUDA_GRAPH_OPT` | 慢 1~4% |
| 换用 Strata 引擎（同一模型、同一测试脚本） | 单请求解码快 5~12%，但预填充慢 35%（984 对 1500 t/s），上下文只有 131K，而且不支持我们这种每槽都带完整 MTP 的双槽 |
| hipBLASLt 做路由矩阵乘、加大批次 | 都更慢 |

---

# 以下为上游 llama.cpp 原版 README

# llama.cpp

![llama](https://raw.githubusercontent.com/ggml-org/llama.brand/refs/heads/master/cover/llama-cpp/cover-llama-cpp-dark.svg)

<div align="center">

<b>LLM inference in C/C++</b>

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Release](https://img.shields.io/github/v/release/ggml-org/llama.cpp?filter=v*&color=brightgreen)](https://github.com/ggml-org/llama.cpp/releases?q=tag:v0)
[![Nightly](https://img.shields.io/github/v/release/ggml-org/llama.cpp?label=nightly&filter=b*&color=orange)](https://github.com/ggml-org/llama.cpp/releases?q=b)
[![Server](https://img.shields.io/github/actions/workflow/status/ggml-org/llama.cpp/server.yml?label=Server)](https://github.com/ggml-org/llama.cpp/actions/workflows/server.yml)
[![Docker](https://img.shields.io/github/actions/workflow/status/ggml-org/llama.cpp/docker.yml?label=Docker)](https://github.com/ggml-org/llama.cpp/actions/workflows/docker.yml)
[![Winget](https://img.shields.io/github/actions/workflow/status/ggml-org/llama.cpp/winget.yml?label=Winget)](https://github.com/ggml-org/llama.cpp/actions/workflows/winget.yml)

[ggml](https://github.com/ggml-org/ggml) / [ops](https://github.com/ggml-org/llama.cpp/blob/master/docs/ops.md) / [maintainer PRs](https://github.com/ggml-org/llama.cpp/issues?q=is%3Apr%20is%3Aopen%20draft%3AFalse%20(author%3Argerganov%20OR%20author%3AKitaitiMakoto%20OR%20author%3Adanbev%20OR%20author%3Aaldehir%20OR%20author%3Amax-krasnyansky%20OR%20author%3ACISC%20OR%20author%3Aggerganov%20OR%20author%3Aam17an%20OR%20author%3Ajhen0409%20OR%20author%3Abartowski1182%20OR%20author%3Anikwen%20OR%20author%3Ahipudding%20OR%20author%3Aravi9%20OR%20author%3AServeurpersoCom%20OR%20author%3Apwilkin%20OR%20author%3Areeselevine%20OR%20author%3Angxson%20OR%20author%3Ajeffbolznv%20OR%20author%3Amarty1885%20OR%20author%3A0cc4m%20OR%20author%3ATitaniumtown%20OR%20author%3Aangt%20OR%20author%3AIMbackK%20OR%20author%3Aarthw%20OR%20author%3AJohannesGaessler%20OR%20author%3AORippler%20OR%20author%3Aruixiang63%20OR%20author%3Axctan%20OR%20author%3Aallozaur%20OR%20author%3Ayomaytk%20OR%20author%3Aaendk%20OR%20author%3Awine99%20OR%20author%3Agaugarg-nv%20OR%20author%3Ataronaeo%20OR%20author%3Aforforever73%20OR%20author%3Alhez%20OR%20author%3Anetrunnereve%20OR%20author%3Afairydreaming)%20sort%3Aupdated-desc) / [dev stats](https://github.com/ggml-org/llama.cpp-dev) / [lib llama API](https://github.com/ggml-org/llama.cpp/issues/9289) / [llama-server REST API](https://github.com/ggml-org/llama.cpp/issues/9291)

</div>

## Quick start

A few options to get `llama.cpp` installed on your machine:

- Visit https://llama.app and follow the instructions
- Run with Docker - see our [Docker documentation](docs/docker.md)
- Download pre-built binaries from the [releases page](https://github.com/ggml-org/llama.cpp/releases)
- Build from source by cloning this repository - check out [our build guide](docs/build.md)

Once installed:

```sh
# Download and run a model directly from Hugging Face
llama cli -hf ggml-org/Qwen3.5-0.8B-GGUF

# Launch OpenAI-compatible API server
llama serve -hf ggml-org/Qwen3.5-0.8B-GGUF
```

<table align="center">
    <tr>
        <td align="center" width=50%>
            <img width="1310" height="888" alt="VLM session with `llama cli`" src="https://github.com/user-attachments/assets/88726b48-1713-48aa-a525-95a02e78afc4" />
            <i>VLM session with <b>llama cli</b></i>
        </td>
        <td align="center">
            <img width="1392" height="958" alt="Built-in web UI against `llama serve` running Qwen 3.6" src="https://github.com/user-attachments/assets/b402f972-2e32-4def-8771-8d849f08cf2e" />
            <i>Built-in web UI against <b>llama serve</b></i>
        </td>
    </tr>
<table>

## Description

The main goal of `llama.cpp` is to enable LLM (and VLM) inference with minimal setup and state-of-the-art performance on
a wide range of hardware - locally and in the cloud.

- Plain C/C++ implementation without any dependencies
- Apple silicon is a first-class citizen - optimized via ARM NEON, Accelerate and Metal frameworks
- AVX, AVX2, AVX512 and AMX support for x86 architectures
- RVV, ZVFH, ZFH, ZICBOP and ZIHINTPAUSE support for RISC-V architectures
- 1.5-bit, 2-bit, 3-bit, 4-bit, 5-bit, 6-bit, and 8-bit integer quantization for faster inference and reduced memory use
- Custom CUDA kernels for running LLMs on NVIDIA GPUs (support for AMD GPUs via HIP and Moore Threads GPUs via MUSA)
- Vulkan and SYCL backend support
- CPU+GPU hybrid inference to partially accelerate models larger than the total VRAM capacity

The `llama.cpp` project is build on top of the [ggml](https://github.com/ggml-org/ggml) library.

## Supported backends

| Backend | Target devices |
| --- | --- |
| [BLAS](docs/build.md#blas-build) | All |
| [BLIS](docs/backend/BLIS.md) | All |
| [CANN](docs/build.md#cann) | Ascend NPU |
| [CUDA](docs/build.md#cuda) | Nvidia GPU |
| [HIP](docs/build.md#hip) | AMD GPU |
| [Hexagon](docs/backend/snapdragon/README.md) | Snapdragon |
| [IBM zDNN](docs/backend/zDNN.md) | IBM Z & LinuxONE |
| [MUSA](docs/build.md#musa) | Moore Threads GPU |
| [Metal](docs/build.md#metal-build) | Apple Silicon |
| [OpenCL](docs/backend/OPENCL.md) | Adreno GPU |
| [OpenVINO [In Progress]](docs/backend/OPENVINO.md) | Intel CPUs, GPUs, and NPUs |
| [RPC](https://github.com/ggml-org/llama.cpp/tree/master/tools/rpc) | All |
| [SYCL](docs/backend/SYCL.md) | Intel GPU |
| [VirtGPU](docs/backend/VirtGPU.md) | VirtGPU APIR |
| [Vulkan](docs/build.md#vulkan) | GPU |
| [WebGPU](docs/build.md#webgpu) | All |
| [ZenDNN](docs/build.md#zendnn) | AMD CPU |

## Documentation

#### Tools

- [cli](tools/cli/README.md)
- [completion](tools/completion/README.md)
- [server](tools/server/README.md)
- [GBNF grammars](grammars/README.md)

#### Development

- [How to build](docs/build.md)
- [Running on Docker](docs/docker.md)
- [Build on Android](docs/android.md)
- [Multi-GPU usage](docs/multi-gpu.md)
- [Performance troubleshooting](docs/development/token_generation_performance_tips.md)
- [GGML tips & tricks](https://github.com/ggml-org/llama.cpp/wiki/GGML-Tips-&-Tricks)
- [XCFramework](docs/xcframework.md)
- [Completions](docs/completions.md)
- [Models](docs/models.md)
- [Release process](docs/release.md)

## Contributing

- Contributors can open PRs
- Collaborators will be invited based on contributions
- Maintainers can push to branches in the `llama.cpp` repo and merge PRs into the `master` branch
- Any help with managing issues, PRs and projects is very appreciated!
- Read the [CONTRIBUTING.md](CONTRIBUTING.md) for more information

## Acknowledgements

- [yhirose/cpp-httplib](https://github.com/yhirose/cpp-httplib) - Single-header HTTP server, used by `llama-server` - MIT license
- [nothings/stb](https://github.com/nothings/stb) - Single-header image format decoder, used by multimodal subsystem - Public domain
- [nlohmann/json](https://github.com/nlohmann/json) - Single-header JSON library, used by various tools/examples - MIT License
- [mackron/miniaudio](https://github.com/mackron/miniaudio) - Single-header audio format decoder, used by multimodal subsystem - Public domain
- [sheredom/subprocess.h](https://github.com/sheredom/subprocess.h) - Single-header process launching solution for C and C++ - Public domain
