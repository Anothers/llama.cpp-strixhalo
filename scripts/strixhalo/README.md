# Strix Halo 发布辅助文件

说明文档在仓库根目录 [README.md](../../README.md)。

- `hblt.db`：hipBLASLt 调优表（`GGML_CUDA_HBLT_DB`）
- `gates/`：正确性门（`spec_correct.sh`：MTP 与不开 MTP 逐字比对；`det_test.sh`：确定性）
- `llama-server-wrapper.sh`：当前生产（hb24）的发布 wrapper，设置运行时环境变量，崩溃时回退到 hb17 单槽
- `server-args.txt`：hb12c 时期的旧参数，当前生产参数以根 README 为准
