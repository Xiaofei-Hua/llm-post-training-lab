# Research Proposal：PostTrainLab——受控后训练信号与阶段顺序研究

## Problem Anchor

- **Bottom-line problem**：在有限算力和完全公开、可追溯的数据条件下，搭建一个能因果区分 SFT、offline preference optimization、GRPO 与 on-policy distillation 作用的后训练实验仓库，并产出可复现结果、失败分析和面试级技术报告。
- **Must-solve bottleneck**：多数个人项目同时更换模型、数据、训练 token、reward、解码和训练框架，因此无法判断收益来自算法本身、数据泄漏、更多计算还是 evaluator 投机。
- **Non-goals**：从零预训练、训练系统吞吐优化、生产部署、榜单 SOTA、多模态或大规模 Agent RL。
- **Constraints**：GPU 型号、数量和可用时长尚未确认；前沿模型是硬约束，首期使用 Qwen3.5/3.6，不退回 Qwen3；支持单卡 24 GB smoke，并给出 80 GB/多卡扩展路径；只使用公开模型与可发布数据；不使用内部资产。
- **Success condition**：完成 Base/SFT/DPO/GRPO/OPD 的 matched-budget 公平对照，至少两个决定性消融与一个负结果复盘，并形成可复现配置、主表、技术报告和面试材料。

## Technical Gap

公开 recipe 能分别训练 SFT、DPO、GRPO 或 distillation 模型，但一个面向学习者的仓库通常缺少共同的数据 lineage、冻结 evaluator 和预算匹配，导致跨算法数字不可比。本项目要补的是“训练信号的因果归因”，不是另造一个训练器。朴素地叠加更多算法只会扩大 confounder；最小充分介入是固定 backbone、prompt pool、训练 token 与 evaluator，建立分支式 checkpoint graph，并用 reward/length/entropy/retention 联合诊断。

## Method Thesis

- **One-sentence thesis**：在同一小型基模、不可变数据划分和匹配训练预算下，用分支式 `SFT / DPO / GRPO / OPD` 对照与顺序实验，建立不同后训练信号对能力、探索、分布迁移和遗忘的可解释因果地图。
- **Why smallest adequate**：复用 TRL/Open-R1，不新增模型模块，只新增严格的实验控制、reward audit 和 stage-order matrix。
- **Why timely**：verifiable RL 与 on-policy distillation 已有可用公开实现，但二者如何组合、何时各自有效，正是后训练岗位需要理解的机制问题。

## Contribution Focus

- **Dominant contribution**：一个 matched-backbone、matched-data、matched-token 的训练信号与阶段顺序实验协议。
- **Optional supporting contribution**：reward 与独立 evaluator 的偏差诊断，包括格式/长度投机和能力 retention。
- **Explicit non-contributions**：不声称提出新的 RL/KD loss，不声称基础设施创新，不把组合收益归功于单个算法。

## Proposed Method

### Complexity Budget

- **Frozen/reused**：Qwen3.5 hybrid-attention backbone、TRL trainer、Open-R1 数据、lm-eval、数学 verifier。
- **New trainable components**：无；仅训练 Student policy，Teacher/reference 冻结。
- **Intentionally excluded**：PRM、learned reward model、MoE、多模态、Agent tool environment、复杂 curriculum。

### System Overview

```text
public data → license/schema → quality → decontamination → immutable splits
                                      ├→ response data: SFT
                                      ├→ paired data: DPO
                                      └→ prompt-only: GRPO / OPD

Base → SFT ─┬→ DPO
            ├→ GRPO
            ├→ OPD
            ├→ OPD → GRPO
            └→ GRPO → OPD
                    ↓
           frozen evaluator + paired statistics + failure taxonomy
```

### Core Mechanism

- **Input/output**：数学 prompt 输入，推理过程与最终可解析答案输出。
- **Architecture/policy**：Qwen3.5-0.8B-Base 用于 smoke，Qwen3.5-2B-Base 为 Main Student，Qwen3.5-4B/9B 为 Teacher；Qwen3.6-27B 作为高预算 stretch Teacher。文本阶段冻结视觉塔，固定 hybrid-attention 语言主干。
- **Training signals**：SFT CE；DPO pairwise preference；GRPO group-normalized verifier reward；OPD generalized JSD/KL on Student-generated prefixes。
- **Main novelty/learning value**：不是 loss 新颖性，而是用预算、数据和评测控制隔离 loss 的实际作用，并直接比较训练阶段顺序。

### Supporting Diagnostic

- 独立 evaluator 复算 accuracy，不信任 trainer reward；
- 记录 entropy、KL、有效 advantage group、长度、格式与 truncation；
- IFEval/通用小集监控 retention；
- 对 reward hack 和 parser false positive 建 adversarial tests。

### Modern Primitive Usage

- **GRPO**：作为无需 critic 的 online verifiable RL；
- **OPD/GKD**：Teacher 在 Student 自身 prefix 上给 dense distribution，缓解 off-policy exposure mismatch；
- 两者是研究对象，不是装饰；必要性通过 DPO/off-policy KD 对照证明。

### Integration

训练器与评测器通过不可变 checkpoint manifest 连接。每个 run 保存 parent checkpoint、数据 revision、配置 hash、训练 token、seed 与 evaluator revision。最终只比较同一适配策略和 decoding config 的 checkpoint。

### Training Plan

1. Base evaluation 与 evaluator audit；
2. Qwen3.5-0.8B 上完成 SFT 64-sample overfit 与 2k smoke，再转 Qwen3.5-2B 的 10k main；
3. 同 prompt 多采样构造 DPO pair；
4. 从 SFT checkpoint 分支 GRPO 与 OPD；
5. 只对通过单 seed gate 的配置比较两种 stage order；
6. 最终配置在 Qwen3.5-2B 上跑三 seed；预算允许时用 Qwen3.6 Teacher 做 portability。

### Failure Modes and Diagnostics

- **数据泄漏**：exact hash + normalized n-gram/MinHash，保存 removed pairs；
- **reward hacking**：独立 evaluator、adversarial parser tests、reward/accuracy gap；
- **零 advantage**：记录 group reward variance，动态检查有效 prompt；
- **OPD 不兼容**：tokenizer/vocab gate；不兼容时降级为单独命名的 sequence KD；
- **遗忘**：IFEval/通用 benchmark 和 KL-to-SFT；
- **算力失控**：100-step calibration、单 seed 淘汰、Qwen3.5-0.8B 先闭环；资源不足不退回旧代架构。

### Novelty and Elegance Argument

这个项目不包装新的算法名。其价值是把常被混在一起的“数据质量、offline preference、online reward、dense teacher distribution、训练顺序”压缩成同一受控图，并要求每个收益经过 anti-claim 检验。相比增加 reward model、PRM 或 Agent 环境，这一方案更小、更适合有限算力，也更能展示算法判断力。

## Claim-Driven Validation Sketch

### Claim 1：不同训练信号产生可分辨的能力与行为变化

- **Minimal experiment**：Base、SFT、SFT→DPO、SFT→GRPO、SFT→OPD 的 matched-token 对照。
- **Baselines/ablations**：SFT continuation（额外训练 token 对照）、off-policy KD。
- **Metrics**：MATH-500/GSM8K accuracy、AIME pass@k、IFEval、entropy/length/format。
- **Expected evidence**：不是预设全都提升，而是能稳定辨认每种信号的收益与失败边界。

### Claim 2：reward specification 与阶段顺序影响收益可靠性

- **Minimal experiment**：exact reward vs exact+format；`OPD→GRPO` vs `GRPO→OPD`。
- **Baselines/ablations**：过宽 parser/错误 reward 作为受控负例；matched total optimized tokens。
- **Metrics**：independent accuracy、reward gap、training stability、retention、错误类型迁移。
- **Expected evidence**：至少识别一个 reward 或顺序导致的可靠差异，并用机制指标解释。

## Experiment Handoff Inputs

- **Must-prove claims**：C1、C2。
- **Must-run ablations**：SFT continuation、off-policy KD、reward deletion、stage order。
- **Critical datasets/metrics**：MATH-500、GSM8K、AIME、IFEval；accuracy、pass@k、retention、length、entropy。
- **Highest-risk assumptions**：小模型存在足够 RL 信号；Teacher 与 Student vocab 兼容；可用算力足以多 seed。

## Compute & Timeline Estimate

- **GPU-hours**：档位 A 约 100–220；推荐档位 B 约 250–550；Qwen3.6 Teacher 扩展需 600 GPUh 以上；均须由 G0 实测修正。
- **Data/annotation**：公开数据；约 100–300 evaluator edge cases 和至少 100 条模型输出人工审计。
- **Timeline**：12 周，前 2 周只做环境、数据和评测可信度。
