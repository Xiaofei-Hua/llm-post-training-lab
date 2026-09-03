# Result contract

`results/raw/` 被 Git 忽略；可公开的聚合结果必须由 immutable raw generations/logs 生成，并保存：run ID、arm/stage/seed、checkpoint/config/data/evaluator hashes、逐题 correctness、训练行为指标、E1/E2 成本和失败状态。

任何进入 README、报告或简历的数字都必须能反向定位到一份聚合 JSON、一份 frozen config 和生成脚本版本。
