# Planned entrypoints

后续只提供薄入口，不在脚本中隐藏实验逻辑：

- `profile_*`：C0/C5 硬件与显存闭合；
- `prepare_*`：数据 registry、去污染和 manifest；
- `train_*`：SFT/GRPO/OPD，两阶段 token gate；
- `evaluate_*`：frozen decoding 与 sealed evaluator；
- `analyze_*`：预注册统计、成本表和 claim audit。

每个入口均需支持 `--dry-run`、打印 resolved config，并拒绝缺失 revision/hash 的正式运行。

## 已实现

- `audit_verifier.py`：D05 的薄入口；读取 strict JSONL attack corpus，执行 canonical verifier，并原子写出带 corpus/policy/backend/source/lock/Git provenance 的 JSON report。它拒绝未跟踪或与 HEAD 不一致的实现、lock 与 corpus 输入；不是训练或 benchmark entrypoint，因此没有 experiment `--dry-run`，运行本身就是只读 CPU audit。
- `audit_data_trust.py`：D06 的正式 CPU audit 入口；接收 repository-relative source/transform registry、candidate/evaluation JSONL、split policy、expectation 和可选 `--parent-ledger`，在内部重算 split、dirty scan、family quarantine、clean rescan 与 manifest。实现、`uv.lock`、全部输入和 transform code/config 必须被 Git 跟踪并与捕获的 revision 一致；输出不含原始题目、答案或轨迹。
