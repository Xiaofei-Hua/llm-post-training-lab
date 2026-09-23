# 项目技术主线与下一步

2026-09-09 已按用户要求将主线调整为算法、训练框架、性能优化和指标分析。研究问题仍是 Gemma 4 E2B Student 与 same-lineage E4B Teacher 上的 SFT/GRPO/OPD 五臂两阶段对照；C1/C2 与三 paired seeds 保留。历史四轮评审属于此前方法设计记录，不代表本次计划已获同样评审。

截至 2026-09-23 完成 D01–D14，共 14/24；已有 CE、GRPO、reverse-KL、tiny 模型/LoRA 训练循环、CPU 合成学习和单卡 CUDA 数值/恢复/性能实测。性能结果按具体 workload 和测量范围表述，见 D11/D13 报告；另已完成真实 E2B/E4B Base 的两步 LoRA 接入与内存测量，见 D14 报告；没有正式训练或 benchmark 准确率结果。

## 下一步

D01–D13 回顾已完成，最后两轮无新增问题。2026-09-23 授权的 D14 真实模型接入也已验收；下一步为 D15 真实数据与 Base 评测准备。tiny 学习实验和真实 Base 两步 smoke 均不替代正式 SFT/GRPO/OPD、Teacher 能力门槛或 C1/C2。

已有校验设施直接复用；近期交付围绕实际参数更新、学习行为、性能瓶颈与指标解释。CPU 数值实验和 microbenchmark 不需要另建研究 claim。

## 阅读入口

1. `README.md`：技术主线和已完成范围。
2. `docs/planning/DEVELOPMENT_MODULES.md`、`docs/planning/ROADMAP.md`：执行顺序与各模块技术验收。
3. `docs/planning/PERFORMANCE_PLAN.md`：CPU/GPU 性能实验、测量范围和优化对照。
4. `refine-logs/FINAL_PROPOSAL.md`、`refine-logs/EXPERIMENT_PLAN.md`：研究设计、正式实验与开发实验。
5. `refine-logs/EXPERIMENT_TRACKER.md`：开发与训练队列。
6. `docs/evaluation/BENCHMARK_PLAN.md`、`docs/planning/PORTFOLIO_CHECKLIST.md`：指标、结果解释和求职交付。

日期命名的 refine-logs 快照与历史 review 报告保留原状；当前工作以无日期的 canonical 方案和 2026-09-09 修订文档为准。
