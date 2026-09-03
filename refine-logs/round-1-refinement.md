# Round 1 Refinement

## Problem Anchor

- **Bottom-line problem**：在有限算力和完全公开、可追溯的数据条件下，搭建一个能因果区分关键后训练信号作用的实验仓库，并产出可复现结果、失败分析和面试级技术报告。
- **Must-solve bottleneck**：多数个人项目同时更换模型、数据、训练 token、reward、解码和训练框架，无法判断收益来自算法信号、数据泄漏、额外计算还是 evaluator 投机。
- **Frontier constraint**：用户明确要求不用 Qwen3，必须采用项目启动时最前沿且开放、可训练的模型；截至 2026-09 主线选择 Gemma 4 E2B/E4B。
- **Non-goals**：从零预训练、训练系统吞吐创新、生产部署、榜单 SOTA、首期多模态或大规模 Agent RL。
- **Constraints**：GPU 型号、数量和可用时长尚未确认；只使用公开模型与可发布数据，不使用内部资产；算法矩阵必须先经 100-step profile 证明预算闭合。
- **Success condition**：从同一 SFT checkpoint 出发，完成 sparse verifier reward 与 dense on-policy Teacher signal 的五臂受控对照、双预算核算、关键失败分析，并形成可复现配置、主表、技术报告和面试材料。

## Anchor Check

- **Original bottleneck**：后训练 recipe 同时变化导致无法归因。
- **Why preserved**：修订后只研究两类核心信号及顺序，模型、parent、prompt registry、解码和 Student 更新预算固定。
- **Anchor correction**：Round 0 把用户的“最前沿且不用 Qwen3”过度具体化为 Qwen3.5/3.6。核对 2026 发布后，Gemma 4 E2B/E4B 更新且可由 TRL 支持；改为 Gemma 4 是恢复用户原意，不是改变研究问题。
- **Rejected drift**：不因 Gemma 4 原生多模态而加入视觉/音频训练；那会把项目变成 VLM recipe。

## Simplicity Check

- **Dominant contribution after revision**：稀疏 verifier reward（GRPO）与稠密 on-policy Teacher distribution（OPD）的单独作用及顺序交互。
- **Removed/merged**：DPO 降为 shadow baseline；多个 reward、多个 KL、多个 Teacher 和跨架构 scaling 移出核心；Qwen3.6 不再进入关键路径。
- **Rejected complexity**：PRM、learned reward model、Agent 环境、多模态 retention、大规模超参搜索。
- **Why smallest adequate**：五臂是回答“单独作用 + 两种顺序”所需的最小闭合图，SFT continuation 是必需的 Student-update control。

## Changes Made

### 1. 收紧研究问题

- **Reviewer said**：SFT/DPO/GRPO/OPD、多个顺序和多个 reward 过宽。
- **Action**：主矩阵只留 SFT continuation、GRPO、OPD、OPD→GRPO、GRPO→OPD。
- **Reasoning**：DPO 不提供本项目核心的 online sparse/dense 信号，只需作为 offline 参照。
- **Impact**：主叙事从“所有算法地图”变为“GRPO–OPD interaction”。

### 2. 替换 matched-token

- **Reviewer said**：相同 token 不能控制 rollout、Teacher/reference forward 和 Student backward。
- **Action**：预注册 Signal Efficiency 与 Practical Compute 两个 estimand。
- **Reasoning**：一个回答学习信号，一个回答实际成本，不强行压成单一公平数字。
- **Impact**：避免把 Student token 相同误写为总算力相同。

### 3. 固定算法合同

- **Reviewer said**：OPD divergence、GRPO group/reward/clip 和 adapter 都不够具体。
- **Action**：OPD 固定 fully on-policy full-vocab chunked reverse KL；GRPO 固定 exact reward、group 8、clip 0.2；主矩阵统一 text-only LoRA。
- **Reasoning**：先回答主问题，再做 loss variant。
- **Impact**：实现和预注册都可操作。

### 4. 升级并收紧前沿模型

- **Reviewer said**：Teacher capability 与多模态 text-only 边界需 gate。
- **Action**：锁定 Gemma 4 E2B Base Student、E4B-it Teacher；增加能力/NLL、tokenizer、non-text zero-gradient gate。
- **Reasoning**：Gemma 4 是 2026 最新小型架构之一，同时 E2B/E4B 允许完成多臂实验。
- **Impact**：满足前沿约束但不扩大到多模态任务。

## Revised Proposal

# Research Proposal：Sparse Reward × Dense Teacher Signal in Gemma 4 Post-Training

## Problem Anchor

- **Bottom-line problem**：在有限算力和完全公开、可追溯的数据条件下，搭建一个能因果区分关键后训练信号作用的实验仓库，并产出可复现结果、失败分析和面试级技术报告。
- **Must-solve bottleneck**：多数个人项目同时更换模型、数据、训练 token、reward、解码和训练框架，无法判断收益来自算法信号、数据泄漏、额外计算还是 evaluator 投机。
- **Frontier constraint**：不用 Qwen3；采用截至 2026-09 前沿、开放且可训练的 Gemma 4。
- **Non-goals**：从零预训练、训练系统吞吐创新、生产部署、榜单 SOTA、首期多模态或大规模 Agent RL。
- **Constraints**：硬件待确认；公开资产；先 profile 再承诺主矩阵。
- **Success condition**：完成五臂受控对照、双预算核算与失败分析，产出可复现且可审计的技术作品。

## Technical Gap

现有 recipe 能单独训练 SFT、GRPO 和 distillation，但通常没有从同一 parent、同一 canonical prompt registry 和双预算口径比较 sparse outcome signal 与 dense token distribution。尤其是 `OPD→GRPO` 和 `GRPO→OPD` 常被当作经验配方，而非受控交互问题。项目的缺口不是新 trainer，而是一个最小、可证伪的 recipe-intervention protocol。

## Method Thesis

> 从同一个 Gemma 4 E2B SFT anchor 出发，在固定 prompt、解码、adapter 与 Student 更新预算下，比较 sparse verifier reward 与 dense on-policy Teacher distribution 单独及按不同顺序施加时，对正确率、探索性、训练稳定性和 retention 的可重复影响。

## Contribution Focus

- **Dominant contribution**：GRPO–OPD 单独/顺序交互的五臂受控协议。
- **Supporting contribution**：Student signal efficiency 与端到端 practical compute 的双视角，以及 reward–accuracy gap 诊断。
- **Claim boundary**：比较的是完整、受控的 recipe intervention，不声称隔离抽象 loss 的纯因果效应，也不提出新 loss。

## Proposed Method

### Complexity Budget

- **Frozen/reused**：Gemma 4、TRL、Open-R1、math-verify、lm-eval、MathArena。
- **Trainable**：Gemma 4 E2B text attention/MLP projection LoRA；其余参数冻结。
- **Excluded**：视觉/音频训练、PRM、learned reward、Agent tools、多 Teacher 主矩阵。

### Models

- **Student**：`google/gemma-4-E2B`，35 text layers，262,144 vocab，hybrid local/global attention。
- **Teacher**：`google/gemma-4-E4B-it`，42 text layers，262,144 vocab；必须通过 calibration accuracy/NLL gate。
- **Non-text modules**：不输入 image/audio token；视觉/音频 encoder `requires_grad=False`，训练前后 checksum 不变。
- **Adapter**：统一 LoRA，候选 targets 为 text attention 与 MLP projection，rank 32、alpha 64、dropout 0；准确 module names 在 C0 introspection 后冻结。

### Data Contract

- `D_anchor`：去污染后 10k verified reasoning traces，用于产生唯一 SFT anchor。
- `D_core`：2k canonical prompts，按 source/problem/template family 与 anchor/dev/test 分离；所有五臂使用相同 IDs。
- `D_calib`：500 prompts，用于 Teacher gate 和不超过两档的 objective-specific learning-rate pilot。
- `D_dev`：500 prompts，用于 early-stop 和预注册 threshold，不触碰 test。
- 每个 arm 只能读取允许字段：SFT control 可见 gold trace；GRPO 只见 prompt/reference answer；OPD 只见 prompt 和 Teacher distribution。
- 训练前对 prompt、reference solution 和 reasoning trace 都做 normalized exact/near-duplicate audit。

### Five Core Arms

| Arm | Objective | Allocation |
|---|---|---|
| A0 | SFT continuation | 100% Student update budget |
| A1 | GRPO | 100% |
| A2 | OPD | 100% |
| A3 | OPD → GRPO | 50% + 50% |
| A4 | GRPO → OPD | 50% + 50% |

所有 arm 从同一 SFT anchor 启动。每个新 stage 重置 AdamW optimizer/scheduler，固定 paired seed、prompt exposure、max length、sampling 和 stop rule。DPO 只在最后用 frozen rollout bank 做单 seed shadow baseline。

### Budget Estimands

#### E1: Signal Efficiency

匹配 canonical prompt IDs、Student non-padding loss tokens、Student forward/backward FLOPs 和总更新配额。暂定每臂 `U=4M` Student loss tokens；顺序臂每阶段 `2M+2M`。G0 可以按吞吐等比例下调 U，但必须在运行前对五臂同时冻结。

#### E2: Practical Compute

记录全部 Student/old/reference/Teacher forward、rollout tokens、Student backward、accelerator-seconds、GPU-hours、峰值显存。画 accuracy/retention–compute Pareto，不宣称与 E1 相同。

### GRPO Contract

- group size 8；clip range 0.2；主 reward 仅 exact/symbolic correctness `0/1`；不使用 learned reward 或正长度奖励。
- completion cap 先比较 2k/4k 的 P95 截断率，再在正式 run 前冻结。
- 零方差 group 不更新并记录；正式 run 不通过动态换题掩盖其比例。
- 若有效 group 比例低于 30%，停止主训练，重新选择难度混合或重新评估 Student readiness。
- 独立 evaluator 复算 accuracy；trainer reward 不作为主结果。

### OPD Contract

- fully on-policy：`y ~ π_student`，Student sampling token IDs stop-gradient。
- Teacher 冻结，在 Student-generated prefix 上给分布。
- 主 loss：temperature 1.0 的 full-vocabulary chunked reverse KL `KL(π_student || π_teacher)`。
- 不持久化完整 `[B,L,V]` logits；chunked loss 与 tiny full-reference 数值/梯度单测对齐。
- Teacher 不通过能力/NLL gate或 tokenizer 不兼容，则 OPD 主臂停止并如实记录；不能静默换 Teacher/数据。

### Teacher Gate

- calibration accuracy 相对 SFT Student 的 paired bootstrap CI 下界大于 0；
- 绝对提升至少 5pp，或覆盖至少 20% Student 错题；
- verified solution token NLL 更低；
- parse rate 不低于 Student；
- tokenizer files hash 与随机文本 token IDs 相同。

### Evaluation

- **Primary**：MATH-500、GSM8K、MathArena ArXivMath 06/2026。
- **Hard**：AIME 2026，报告 pass@1/pass@8 和题目级结果。
- **Retention**：IFEval 与预注册通用 benchmark 子集。
- **Behavior**：entropy、KL、clip fraction、effective groups、length、truncation、format failures。
- **Statistics**：paired item bootstrap 95% CI；五臂单 seed screening 后，最终配置运行三 paired seeds；只声称项目后训练数据去污染，不能证明预训练数据未见公开 benchmark。

### Failure Modes

- **Reward rises, accuracy flat/down**：判定 reward misspecification/parser gaming，停止扩大训练。
- **Teacher weak**：按 Teacher-correct/incorrect slice 分析，不过滤主数据掩盖问题。
- **OPD collapse**：检查 KL direction、Teacher entropy、temperature、chunk equivalence 与 Student support。
- **GRPO zero variance**：检查题目难度和 group size，不用格式 reward制造虚假方差。
- **Retention loss**：报告 Pareto，不用只挑数学指标的方式掩盖。
- **Compute overrun**：同时下调所有 arm 的 U 或减少 nice-to-have，不删 SFT control/两个顺序臂。

## Claim-Driven Validation

### Claim 1：Sparse 与 dense signal 产生可重复、可诊断的不同作用

- **Experiment**：A0/A1/A2，E1 matched Student budget。
- **Minimum evidence**：A1/A2 至少一个相对 A0 在 primary composite 上达到预注册 +2pp 或 paired CI 正向，且 IFEval 下降不超过 2pp；若不成立，负结论仍有效，但不能声称提升。
- **Anti-claim controls**：A0 控制额外 Student 更新；E2 揭示总计算差异；独立 evaluator 排除 reward 投机。

### Claim 2：Stage order 产生可解释的交互，或可证伪为影响很小

- **Experiment**：A3 vs A4，并与 A1/A2 比较。
- **Minimum evidence**：A3/A4 在三 seed 上有一致方向，或差异 CI/最小效应达到预注册阈值；同时 entropy、Teacher-correct slice 或 retention 给出机制解释。
- **Falsification**：若两顺序差异落在 ±2pp practical-equivalence band 内，则结论是该预算下顺序影响有限，不追加模块追逐正结果。

## Compute and Timeline

- E2B/E4B 名称是 effective size，实际总参数约 5.1B/8B；预算按实测权重与 forward/backward 核算。
- G0 必测 SFT tokens/s、GRPO rollout tokens/s、Student backward tokens/s、Teacher KL tokens/s、峰值显存。
- 推荐资源初估 2×80 GB 或 4×48 GB、450–900 GPUh；未经 profile 不承诺该数字。
- 12 周：2 周可信评测、3 周 SFT/校准、4 周 GRPO/OPD、1 周顺序、2 周重复与报告。

## Experiment Handoff

- **Must-run**：A0–A4、Teacher gate、reward/parser audit、dual-budget accounting、fresh benchmark。
- **Shadow**：DPO 单 seed。
- **Nice-to-have**：off-policy KD、forward KL/JSD、larger Teacher、多模态/跨架构。
- **Hard stop**：Teacher/tokenizer gate 失败、有效 GRPO group <30%、evaluator audit <99%、预算无法覆盖五臂。
