# LLM Post-Training Lab

一个面向基模/后训练算法实习的技术作品：实现 SFT、GRPO 与 on-policy distillation，打通训练框架，分析学习动态和计算瓶颈，用能力、稳定性与效率指标解释方法的收益和限制。

## 技术主线与交付

| 方向 | 核心技术问题 | 计划交付 |
|---|---|---|
| 算法机制 | 稀疏 reward 与稠密 Teacher 信号如何改变梯度、探索和顺序效应？ | 公式与梯度分析、CE/GRPO/OPD 实现、五臂对照、失败案例 |
| 训练框架 | 如何组织当前策略采样、Teacher/old-policy、有效 token 更新与阶段切换？ | 可运行的统一训练循环、模型/LoRA 适配、CPU 学习示例及后续真实模型训练 |
| 性能优化 | 瓶颈位于 rollout、LM head、KL、backward 还是通信？ | baseline/profile、分块与重计算实验、吞吐/内存对照、端到端成本分析 |
| 指标分析 | 准确率、探索性、长度、能力保持和成本如何共同变化？ | accuracy/pass@k、entropy/KL/有效组率曲线、retention、accuracy–cost 图 |

这些交付均按实际进度标注。数据隔离、环境复现和现有校验是基础支持；研发和成果展示优先围绕上表展开。性能提升可作为框架/系统成果单独呈现，准确率收益由受控算法实验回答。

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
2. 相同 Student 更新预算下，哪些方法更有效，端到端成本又是多少？
3. reward 与离线 benchmark 一致时，模型是否仍出现长度投机、格式投机或能力遗忘？
4. OPD 应位于 RL 前还是 RL 后，它与 Teacher 上限、Student 容量的关系是什么？
5. 分块、重计算、batch 与 rollout 优化能减少多少内存或时间，代价是什么？

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
- 不建设通用实验治理、哈希审计平台或生产服务；框架工作直接服务于后训练算法和性能实验。
- 不在首期同时比较多个模型架构；固定 Gemma 4 hybrid local/global attention 后研究后训练算法。
- 不追求榜单 SOTA；追求算法理解、可执行框架、实测优化和有解释力的结果。
- 不使用或外传内部数据与未公开实现。

## 项目地图

| 主题 | 文档 |
|---|---|
| 为什么做、成功标准 | `docs/planning/PROJECT_CHARTER.md` |
| 训练与评测系统边界 | `docs/architecture/SYSTEM_DESIGN.md` |
| Gemma 4 前沿模型选择 | `docs/architecture/FRONTIER_MODEL_MATRIX.md` |
| 算法公式与对照关系 | `docs/algorithms/ALGORITHM_MAP.md` |
| D01 loss mask 与精确预算实现 | `docs/algorithms/LOSS_TOKEN_BUDGET.md` |
| D02 masked causal CE 实现 | `docs/algorithms/MASKED_CAUSAL_CE.md` |
| D03 exact-reward Dr.GRPO surrogate 实现 | `docs/algorithms/GRPO_SURROGATE.md` |
| D04 OPD full-vocabulary reverse-KL 实现 | `docs/algorithms/OPD_REVERSE_KL.md` |
| D05 exact/symbolic verifier 与 reward audit | `docs/algorithms/EXACT_MATH_VERIFIER.md` |
| D06 data registry、family split 与 contamination gate | `docs/data/DATA_REGISTRY_AND_CONTAMINATION.md` |
| D07 sealed evaluator、generation/result 与 metric contracts | `docs/evaluation/SEALED_EVALUATOR.md` |
| D08 paired bootstrap、randomization、Holm 与 TOST | `docs/evaluation/PAIRED_STATISTICS.md` |
| D01–D24 完整训练链路与 X01–X08 扩展总表 | `docs/planning/DEVELOPMENT_MODULES.md` |
| 数据来源、质量、去污染 | `docs/data/DATA_PLAN.md` |
| benchmark、统计与防泄漏 | `docs/evaluation/BENCHMARK_PLAN.md` |
| 算力分档与成本 gate | `docs/planning/COMPUTE_BUDGET.md` |
| 性能瓶颈、baseline 与优化测量 | `docs/planning/PERFORMANCE_PLAN.md` |
| Gemma 4/TRL/vLLM 兼容性 | `docs/planning/COMPATIBILITY_GATES.md` |
| CPU 近期计划与后续训练路线 | `docs/planning/ROADMAP.md` |
| 算法学习与验收课程 | `docs/planning/LEARNING_CURRICULUM.md` |
| 完整实验计划 | `refine-logs/EXPERIMENT_PLAN.md` |
| 待运行矩阵 | `refine-logs/EXPERIMENT_TRACKER.md` |
| 最终研究方案 | `refine-logs/FINAL_PROPOSAL.md` |
| 四轮评审摘要 | `refine-logs/REVIEW_SUMMARY.md` |
| 设计迭代报告 | `refine-logs/REFINEMENT_REPORT.md` |
| 最短阅读入口 | `refine-logs/PIPELINE_SUMMARY.md` |
| 面试交付物 | `docs/planning/PORTFOLIO_CHECKLIST.md` |

## 当前状态

2026-09-09 起，研发计划按算法、框架、性能、指标重新聚焦。核心进度仍为 **8/24**：D01–D04 已完成 loss-token 预算、masked CE、Dr.GRPO surrogate 与 full-vocab reverse-KL 的 CPU 实现和数值/梯度验证；D05–D08 已提供 verifier、数据隔离、评测指标和配对统计。已有实现继续复用，详细验证记录放在对应模块文档中。

接下来依次完成 **D09 模型适配与可训练参数接入 → D10 统一训练循环 → D11 性能剖析与优化 → D12 CPU 端到端学习实验**。验收将关注实际 forward/backward/update、学习曲线、瓶颈定位和测量结果。

D13–D20 规划真实模型接入、GPU 性能和训练 pilots；D21–D22 为五臂两阶段三 seed 训练；D23–D24 为指标分析、技术报告与求职交付。X01–X08 扩展仍延后。当前尚未下载模型或真实训练数据、未执行 MPS/CUDA/GPU，尚无模型准确率或训练加速结果；本地 tiny 模型的 CPU 验证也不能代替 Gemma 实验。

开发环境通过 `uv.lock` 复现。四轮历史方法评审见 `refine-logs/REVIEW_SUMMARY.md`，不作为本次修订或未运行实验的验收结论。

```bash
uv sync --frozen --all-groups
uv run ruff check .
uv run pytest -q
```
