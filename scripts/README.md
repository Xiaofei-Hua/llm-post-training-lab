# Planned entrypoints

后续只提供薄入口，不在脚本中隐藏实验逻辑：

- `profile_*`：C0/C5 硬件与显存闭合；
- `prepare_*`：数据 registry、去污染和 manifest；
- `train_*`：SFT/GRPO/OPD，两阶段 token gate；
- `evaluate_*`：frozen decoding 与 sealed evaluator；
- `analyze_*`：预注册统计、成本表和 claim audit。

每个入口均需支持 `--dry-run`、打印 resolved config，并拒绝缺失 revision/hash 的正式运行。
