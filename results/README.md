# Result contract

`results/raw/` 被 Git 忽略；可公开的聚合结果必须由 immutable raw generations/logs 生成，并保存：run ID、arm/stage/seed、checkpoint/config/data/evaluator hashes、逐题 correctness、训练行为指标、E1/E2 成本和失败状态。

任何进入 README、报告或简历的数字都必须能反向定位到一份聚合 JSON、一份 frozen config 和生成脚本版本。

D07 已将 raw generation 与 sanitized evaluation report 分开：raw bundle 保存 prompt hash、generated text/token IDs、finish/error 和逐 record hash；evaluation report 只保存 generation-record hash、逐 sample/item correctness/status、metric/evaluator hashes。D08 统计只能联合读取后者的 correctness 与由其 benchmark/public-item hashes 精确绑定的 public snapshot strata，不得重解析生成原文。D07 synthetic audit 数字只属于 `artifacts/audits/` 的机制验证，不得进入模型结果表。
