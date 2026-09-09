# 项目技术主线与下一步

2026-09-09 已按用户要求将主线调整为算法、训练框架、性能优化和指标分析。研究问题仍是 Gemma 4 E2B Student 与 same-lineage E4B Teacher 上的 SFT/GRPO/OPD 五臂两阶段对照；C1/C2 与三 paired seeds 保留。历史四轮评审属于此前方法设计记录，不代表本次计划已获同样评审。

当前完成 D01–D08，共 8/24；已有 CE、GRPO、reverse-KL 和数据/评测/统计支持。没有真实模型训练或 benchmark 结果，没有已测性能加速。

## 下一步

按 D09 模型适配与可训练参数接入、D10 统一训练循环、D11 性能剖析与优化、D12 CPU 合成学习实验推进。当前只用本地 tiny 模型和合成输入，不下载真实模型/数据、不运行 MPS/CUDA/GPU。GPU 授权与资源在 D13 进入前确认。

已有校验设施直接复用；近期交付围绕实际参数更新、学习行为、性能瓶颈与指标解释。CPU 数值实验和 microbenchmark 不需要另建研究 claim。

## 阅读入口

1. `README.md`：技术主线和已完成范围。
2. `docs/planning/DEVELOPMENT_MODULES.md`、`docs/planning/ROADMAP.md`：执行顺序与各模块技术验收。
3. `docs/planning/PERFORMANCE_PLAN.md`：CPU/GPU 性能实验、测量范围和优化对照。
4. `refine-logs/FINAL_PROPOSAL.md`、`refine-logs/EXPERIMENT_PLAN.md`：研究设计、正式实验与开发实验。
5. `refine-logs/EXPERIMENT_TRACKER.md`：开发与训练队列。
6. `docs/evaluation/BENCHMARK_PLAN.md`、`docs/planning/PORTFOLIO_CHECKLIST.md`：指标、结果解释和求职交付。

日期命名的 refine-logs 快照与历史 review 报告保留原状；当前工作以无日期的 canonical 方案和 2026-09-09 修订文档为准。
