# Artifact policy

这里仅存放可公开、体积可控的图表、数据卡、reward card、model/result cards 和报告。模型权重、原始数据与敏感笔记不得提交；它们的地址、revision 和 checksum 只写入 manifest。

## 已实现

- `audits/D05_VERIFIER_AUDIT.json`：对公开合成 attack corpus 的确定性 CPU audit；只包含分类计数、依赖版本、policy/corpus/source/lock/Git hashes 和失败记录，不包含模型或业务数据。正式生成要求 verifier、audit、CLI、`uv.lock` 与本次 corpus 均已被 Git 跟踪且和记录 revision 一致。
- `audits/D06_DATA_TRUST_AUDIT.json`：D06 synthetic data trust fixture 的正式 CPU audit；只包含 source/transform/split/contamination/manifest hashes、计数、quarantine IDs、失败与 Git provenance，不包含题目、答案或轨迹。它证明机制与冻结 fixture 可复现，不代表 OpenR1 或真实 benchmark 已 materialize/去污染，也不通过 G1。
- `audits/D07_EVALUATOR_AUDIT.json`：D07 synthetic sealed-evaluator fixture 的正式 CPU audit；只包含 implementation/runtime/fixture/evaluator/record/report hashes 与 greedy/sampling 计数，不含 prompt、prediction、candidate 或 reference。它证明冻结评测机制可复现，不是 Gemma 4/MATH-500 结果，也不代表官方 adapter、人审或 G1 已完成。
