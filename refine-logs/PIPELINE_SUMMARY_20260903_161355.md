# Pipeline Summary

## 当前结论

前期规划已完成并通过第四轮独立审查（9.06/10）。核心是 2026 Gemma 4 E2B Student 与 same-lineage E4B Teacher 上的对称五臂 GRPO–OPD 研究。当前没有训练结果；执行状态有意保持 CONDITIONAL。

## 建议阅读顺序

1. `README.md`：一分钟理解项目；
2. `refine-logs/FINAL_PROPOSAL.md`：研究主张与完整设计；
3. `refine-logs/EXPERIMENT_PLAN.md`：执行合同与 go/no-go；
4. `refine-logs/EXPERIMENT_TRACKER.md`：拿到算力后的逐 run 队列；
5. `docs/planning/LEARNING_CURRICULUM.md`：求职准备的知识与验收；
6. `docs/architecture/SYSTEM_DESIGN.md`、`docs/data/DATA_PLAN.md`、`docs/evaluation/BENCHMARK_PLAN.md`：实现细节；
7. `refine-logs/REVIEW_SUMMARY.md`、`refine-logs/REFINEMENT_REPORT.md`：四轮审查与设计依据。

## 下一次开始执行时

先填写 GPU 型号/数量/可用时长与公开边界；随后只运行 E0 的兼容性、loss oracle、evaluator 和100-step profile。C0/C3/C4/C5 未通过前，不创建15个正式 run。

## 不应提前做的事

不要下载全量数据后才做污染审计；不要先看 test 选超参；不要用 E4B-it/QLoRA/top-k KL 静默救场；不要因单 seed 结果删 arm；不要把工程吞吐写成算法贡献。
