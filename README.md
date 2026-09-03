# LLM Post-Training Lab

一个以算法归因为中心的个人后训练研究仓库，目标是为基模/后训练算法实习准备一套真实、可复现、能解释失败的作品，而不是再搭一个训练工程脚手架。

## 一句话主线

固定同一学生模型、训练 prompt 分布、评测集与 **Student backward loss-token** 预算，建立五臂两阶段对照：

```text
Base
  ├─ E2B ── SFT anchor ─┬─ SFT  → SFT   (A0)
  │                     ├─ GRPO → GRPO  (A1)
  │                     ├─ OPD  → OPD   (A2)
  │                     ├─ OPD  → GRPO  (A3)
  │                     └─ GRPO → OPD   (A4)
  └─ E4B ── 同源 SFT ───── frozen Teacher for OPD

每一阶段固定 2M Student loss tokens；所有臂在阶段边界统一重置 optimizer/scheduler。
```

项目不预设某个顺序一定最好，而是用受控实验回答：

1. SFT、GRPO、OPD 分别改变了什么？
2. 改善来自算法，还是更多训练 token、数据泄漏或解码差异？
3. reward 与离线 benchmark 一致时，模型是否仍出现长度投机、格式投机或能力遗忘？
4. OPD 应位于 RL 前还是 RL 后，它与 Teacher 上限、Student 容量的关系是什么？

## 首期范围

- 任务：以可程序验证的数学推理为主，指令遵循与通用能力作为 retention 检查。
- 学生模型：`google/gemma-4-E2B`，2026 年发布的最新小型 Gemma 4；smoke 与 main 使用同一模型，只改变数据量和序列长度。
- Teacher：`google/gemma-4-E4B` Base 使用与 Student 相同的 `D_anchor` 和模板完成同源 SFT 后冻结；E4B-it 只能作为标注清楚的敏感性实验，不能替代主 Teacher。
- 核心算法：SFT continuation、GRPO、on-policy distillation（OPD/GKD）及两种顺序。
- Shadow 算法：DPO 只在五臂与三 seed 全部完成后运行，不进入主结论。
- 扩展算法：ORPO、GSPO/TIS、PRM/process reward，仅在核心结论稳定后进入。
- 主确认性评测：MATH-500；GSM8K、MathArena ArXivMath 06/2026、AIME 2026、IFEval 与 MMLU-Pro 用作次要、freshness、hardness 与 retention 证据。

## 明确不做

- 不从零预训练基模。
- 不把 vLLM、FSDP、显存优化或吞吐提升包装成算法贡献。
- 不在首期同时比较多个模型架构；固定 Gemma 4 hybrid local/global attention 后研究后训练算法。
- 不追求榜单 SOTA；追求实验归因、复现质量和技术表达。
- 不使用或外传内部数据与未公开实现。

## 项目地图

| 主题 | 文档 |
|---|---|
| 为什么做、成功标准 | `docs/planning/PROJECT_CHARTER.md` |
| 训练与评测系统边界 | `docs/architecture/SYSTEM_DESIGN.md` |
| Gemma 4 前沿模型选择 | `docs/architecture/FRONTIER_MODEL_MATRIX.md` |
| 算法公式与对照关系 | `docs/algorithms/ALGORITHM_MAP.md` |
| 数据来源、质量、去污染 | `docs/data/DATA_PLAN.md` |
| benchmark、统计与防泄漏 | `docs/evaluation/BENCHMARK_PLAN.md` |
| 算力分档与成本 gate | `docs/planning/COMPUTE_BUDGET.md` |
| Gemma 4/TRL/vLLM 兼容性 | `docs/planning/COMPATIBILITY_GATES.md` |
| 12 周路线图 | `docs/planning/ROADMAP.md` |
| 算法学习与验收课程 | `docs/planning/LEARNING_CURRICULUM.md` |
| 完整实验计划 | `refine-logs/EXPERIMENT_PLAN.md` |
| 待运行矩阵 | `refine-logs/EXPERIMENT_TRACKER.md` |
| 最终研究方案 | `refine-logs/FINAL_PROPOSAL.md` |
| 四轮评审摘要 | `refine-logs/REVIEW_SUMMARY.md` |
| 设计迭代报告 | `refine-logs/REFINEMENT_REPORT.md` |
| 最短阅读入口 | `refine-logs/PIPELINE_SUMMARY.md` |
| 面试交付物 | `docs/planning/PORTFOLIO_CHECKLIST.md` |

## 当前状态

前期规划已完成四轮独立审查，最终 9.06/10，**Planning/Method READY**。尚未下载数据、安装训练环境或启动 GPU 实验，因此 **Execution CONDITIONAL**；下一道 gate 是确认 GPU、做 C0/C3/C4/C5 correctness/profile 并冻结精确版本。
