# Artifact policy

这里仅存放可公开、体积可控的图表、数据卡、reward card、model/result cards 和报告。模型权重、原始数据与敏感笔记不得提交；它们的地址、revision 和 checksum 只写入 manifest。

## 已实现

- `cpu/d11_performance.json` 及同名 PNG/PDF：两档 CE/KL 和完整 OPD step 的 5×30 CPU 计时、独立进程 RSS、dense/最后位置 head 的 profiler 形状。重计算在部分 workload 更慢，完整 step 没有稳定加速结论。
- `cpu/d12_learning.json`、`d12_learning` / `d12_diagnostics` PNG/PDF：合成 copy-bit 的五臂两阶段三 seed、逐步预算/行为/成本、held-out context 指标与错误 Teacher 对照。不是正式 Gemma 五臂或 C1/C2 结果。
- `cpu/d09_model_adapter.json`：D09 合成固定 prefix 的 CPU 模型接入实测；保存配置、seed、合成整数输入、CE/KL × text/LoRA 的 12 次更新 loss、dense reference 误差与参数存储估算。由 `scripts/smoke_model_adapter.py` 生成，不含模型权重或真实数据；不是 on-policy、benchmark 准确率或峰值内存/加速结果。
- `audits/D05_VERIFIER_AUDIT.json`：对公开合成 attack corpus 的确定性 CPU audit；只包含分类计数、依赖版本、policy/corpus/source/lock/Git hashes 和失败记录，不包含模型或业务数据。正式生成要求 verifier、audit、CLI、`uv.lock` 与本次 corpus 均已被 Git 跟踪且和记录 revision 一致。
- `audits/D06_DATA_TRUST_AUDIT.json`：D06 synthetic data trust fixture 的正式 CPU audit；只包含 source/transform/split/contamination/manifest hashes、计数、quarantine IDs、失败与 Git provenance，不包含题目、答案或轨迹。它证明机制与冻结 fixture 可复现，不代表 OpenR1 或真实 benchmark 已 materialize/去污染，也不通过 G1。
- `audits/D07_EVALUATOR_AUDIT.json`：D07 synthetic sealed-evaluator fixture 的正式 CPU audit；绑定 implementation `c948fe2eae50289b78513a3a9513e188bff54a98`，canonical report hash `a6c1bed7e74fcc9bf3448fa095535237c27f183cd67938799d9d96bc40b8abf4`。报告只包含 implementation/runtime/fixture/evaluator/record/report hashes 与 greedy/sampling 计数，不含 prompt、prediction、candidate 或 reference。它证明冻结评测机制可复现，不是 Gemma 4/MATH-500 结果，也不代表官方 adapter、人审或 G1 已完成。
- `audits/D08_PAIRED_STATISTICS_AUDIT.json`：D08 synthetic paired-statistics fixture 的正式 CPU audit；绑定 implementation `9a5cee946c617acca6d9e5a167fa725d67798eef`，canonical audit hash `099b4251e2056f990aa7175485334506d4c1a03a3c3b5d914928fe15f406c5d9`。报告包含 outcome-independent RNG design、完整 10k/100k analysis、runtime/input/Git hashes 与 synthetic decisions，不含 prompt、generation 或 reference。8-item effect/null/equivalence 是 regression oracle，不是算法效果，也不完成真实 STAT-C1/C2 或 G5/G6。

D13 的 GPU 验证目前暂停，尚无 `gpu/d13_runtime.json` 或 GPU 性能图表；后续仅实际运行通过后保存。训练 checkpoint 写入忽略的 `outputs/d13/checkpoints/`，不提交。
