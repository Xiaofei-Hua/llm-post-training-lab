# Planned entrypoints

后续只提供薄入口，不在脚本中隐藏实验逻辑：

- `profile_*`：C0/C5 硬件与显存闭合；
- `prepare_*`：数据 registry、去污染和 manifest；
- `train_*`：SFT/GRPO/OPD，两阶段 token gate；
- `evaluate_*`：frozen decoding 与 sealed evaluator；
- `analyze_*`：预注册统计、成本表和 claim audit。

每个入口均需支持 `--dry-run`、打印 resolved config，并拒绝缺失 revision/hash 的正式运行。

## 已实现

- `sync_environment_mirror.sh`：将 frozen `uv.lock` 导出为含 hashes 的 requirements，经清华镜像安装到独立环境；可用 `POSTTRAIN_PYPI_INDEX` 指定镜像，不改锁定版本。
- `validate_accelerator.py`：D13 单 GPU tiny 模型验收，支持 `--dry-run`（CPU）、`--witness-only` 与完整数值/续训/profile。GPU 当前被其他任务占用，本次只运行 dry-run，未生成 GPU 结果。

- `train_cpu_example.py`：D12 合成五臂两阶段三 seed 学习入口，支持 `--dry-run`、`--output`、`--plot-only`；输出逐步指标、held-out context 评测与错误 Teacher 诊断图。
- `profile_cpu.py`：D11 独立子进程 CPU 基准入口，支持 `--dry-run`、`--output`、`--plot-only`；执行两档 CE/KL、完整 OPD step 与 profiler。CPU 示例和 microbenchmark 不受正式训练入口的 revision/hash gate 约束。
- `smoke_model_adapter.py`：D09 的薄 CPU 示例入口；支持 `--dry-run` 打印配置与 `--output` 保存四组 CE/KL × text/LoRA 的参数更新对照。只用本地随机 tiny 模型与合成整数序列，运行逻辑在 `models/smoke.py`；这不是正式训练入口，不要求预先 commit 或新建 hash 审计。
- `audit_verifier.py`：D05 的薄入口；读取 strict JSONL attack corpus，执行 canonical verifier，并原子写出带 corpus/policy/backend/source/lock/Git provenance 的 JSON report。它拒绝未跟踪或与 HEAD 不一致的实现、lock 与 corpus 输入；不是训练或 benchmark entrypoint，因此没有 experiment `--dry-run`，运行本身就是只读 CPU audit。
- `audit_data_trust.py`：D06 的正式 CPU audit 入口；接收 repository-relative source/transform registry、candidate/evaluation JSONL、split policy、expectation 和可选 `--parent-ledger`，在内部重算 split、dirty scan、family quarantine、clean rescan 与 manifest。实现、`uv.lock`、全部输入和 transform code/config 必须被 Git 跟踪并与捕获的 revision 一致；输出不含原始题目、答案或轨迹。
- `audit_evaluator.py`：D07 的正式 CPU audit 入口；分开加载 public prompts 与 sealed references，用冻结 greedy/sampling protocol 和 synthetic backend 生成完整 item×sample records，再由 D05 verifier 重算逐题指标。实际 `__main__`/module source origin、loader-consumed bytes、实现、环境文件与全部 fixture/expectation 必须与单一 Git revision 一致；输出只含 hashes、版本、计数和 metrics。它不下载 benchmark/模型，不执行 GPU，也不代替 D15/G1。
- `audit_paired_statistics.py`：D08 的正式 CPU audit 入口；加载 text-free five-arm/three-seed paired panel、冻结统计 protocol 与 expectation，执行完整 10,000 次分层 item bootstrap、100,000 次 complete-vector sign-flip、Holm 和 sequential TOST。runtime source、loader-consumed bytes、环境与 inputs 必须绑定同一 Git revision；synthetic decision 只验证机制，不代替 D23 真实分析。
