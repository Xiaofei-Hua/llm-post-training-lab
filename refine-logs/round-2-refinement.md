# Round 2 Refinement — Same-lineage Teacher 与对称五臂

## Problem Anchor

- **Bottom-line problem**：从同一个 2026 前沿 Gemma 4 E2B SFT checkpoint 出发，因果区分 sparse verifier reward（GRPO）与 dense on-policy Teacher signal（OPD）的单独作用和阶段顺序。
- **Must-solve bottleneck**：模型、数据谱系、训练预算、scheduler、reward、解码或 Teacher 既有后训练混在一起时，算法收益无法归因。
- **Non-goals**：从零预训练、推理服务优化、SOTA 竞赛、多模态训练、Agent RL、PRM、在主矩阵中比较 DPO。
- **Frontier constraint**：排除 Qwen3；Student/Teacher 使用 2026 Gemma 4 E2B/E4B Base pair。
- **Resources**：主设计要求 2×80 GB 或 4×48 GB 的候选资源，但在 C0/C5 实测前不声称预算闭合；单卡 24 GB 仅支持代码路径。
- **Success condition**：A0–A4 全部 3 paired seeds，预注册 C1/C2 统计检验、E1/E2 双账本、一次 reward-hacking 负例、可复现配置与面试材料。

## Anchor Check

**Preserved.** 本轮没有增加任务或算法，而是修复同一个后训练归因问题中的三个混杂：Teacher 数据谱系、stage/scheduler 不对称和多重“matched”口径。Gemma 4 仍是前沿模型硬约束，研究对象仍是 GRPO–OPD 交互。

## Simplicity Check

**通过 deletion test。** 五臂是回答两个 claims 的最小闭合图：A0 控制继续训练，A1/A2识别单信号，A3/A4识别顺序。删除任一臂都会失去一个必要反事实。DPO、E4B-it、larger Teacher、top-k KL、其他 divergence 和格式 reward 均移出 must-run。

## Round 2 Review → 具体修订

| Reviewer gap | 修订 |
|---|---|
| E4B-it 混入未知 post-training | Primary Teacher 改为 E4B Base，使用与 E2B 相同的 `D_anchor`、模板、mask、顺序、LoRA module classes 和 token/epoch budget 完成 SFT |
| 单臂与顺序臂 stage count 不一致 | 五臂统一为两个 2M Student loss-token stage，全部在中点重置 optimizer/scheduler |
| prompt/loss-token/FLOPs 无法同时匹配 | E1 只严格匹配 Student backward loss tokens；其余只审计；E2 记录所有端到端成本 |
| GRPO 合同缺关键字段 | 冻结 `dr_grpo, beta=0, ε=0.2, G=8, num_iterations=1`、token IS、group scaling、每 generation batch refresh/sync、双 RNG |
| single-seed arm screening 偏差 | A0–A4 全部 3 paired seeds；pilot 只能选 objective config |
| primary composite 未定义 | MATH-500 成为唯一 endpoint；C1 Holm、C2 唯一 order contrast、TOST ±2pp、pass@8 estimator 全部预注册 |
| Teacher 成本隐藏 | 拆分 `C_anchor/C_teacher/C_arm`，同时报告 marginal、cold-start、campaign total |

## 修订后的完整方案

### 1. 模型与同源 anchor

Student 从 `google/gemma-4-E2B` Base 开始，Teacher 从 `google/gemma-4-E4B` Base 开始。二者只使用去污染后的 10k `D_anchor` verified traces，完全共享 chat template、assistant-only loss mask、max length、样本顺序和 SFT token/epoch budget；LoRA 均为相同 module classes、rank 32、alpha 64、dropout 0。学习率只可从同一预注册两档集合中分别选择，checkpoint 只由 `D_calib` 决定。

E2B 产出所有实验共享的 frozen Student anchor；E4B 通过 gate 后产出所有 OPD stages 共享的 frozen Teacher。Gate 同时要求 paired 95% CI 下界 >0、accuracy 点估计至少高 5pp、verified-solution NLL 更低、parse rate 不下降、tokenizer/hash/token IDs 对齐。失败则 primary OPD 停止；E4B-it 只能新建 sensitivity family。

### 2. 数据与隔离

- `D_anchor`：10k verified traces，只用于两个 Base 模型的同源 SFT；
- `D_core`：2k canonical prompts，五个主臂从相同分层分布和冻结循环顺序抽样；
- `D_calib`：500，Teacher/anchor checkpoint gate；
- `D_dev`：500，objective 内两档学习率和 smoke 配置；
- sealed evaluation：MATH-500、GSM8K、MathArena 06/2026、AIME 2026、IFEval、MMLU-Pro。

四个训练/校准 split 在来源、题型和 template family 层面互斥；prompt、reference solution 和 reasoning trace 都做 exact/fuzzy dedup。不同 objective 的 exposure 数可以不同，但其来源分布和顺序规则相同，并完整报告。

### 3. 对称五臂

| Arm | Stage 1 | Stage 2 | 角色 |
|---|---|---|---|
| A0 | SFT 2M | SFT 2M | continued-training control |
| A1 | GRPO 2M | GRPO 2M | sparse signal |
| A2 | OPD 2M | OPD 2M | dense signal |
| A3 | OPD 2M | GRPO 2M | dense→sparse order |
| A4 | GRPO 2M | OPD 2M | sparse→dense order |

2M 指 Student non-padding backward loss tokens。所有 arm 从同一 anchor 初始化，所有 arm 在 2M 中点重置 optimizer/scheduler，保存中点与终点 checkpoint。A0–A4 都跑三 paired seeds。

### 4. GRPO 执行合同

主 reward 只有 frozen exact/symbolic verifier：正确 1，错误、不可解析、截断 0。配置为 `group_size=8`、`loss_type=dr_grpo`、`epsilon=0.2`、`beta=0`、`num_iterations=1`、token-level importance sampling、group reward scaling、`temperature=1.0`。old policy 与 rollout weights 每个 generation batch 刷新/同步。prompt/data RNG 与 rollout RNG 独立。

正式 run 不重采样零方差 group；effective-group rate <30% 触发 stop gate。completion cap 默认 2048；若 G0 冻结 pilot 的 truncation >5%，所有正式臂统一采用 4096。监控 reward、独立 accuracy、entropy、diagnostic anchor-KL、clip fraction、length、truncation、effective-group rate 和 sync lag。

### 5. OPD 执行合同

Student 当前策略在 `D_core` prompt 上生成 prefix，采样 token stop-gradient；frozen E4B Teacher 在完全相同 prefix 上输出分布。主 loss 是 temperature 1.0 的 full-vocabulary chunked reverse KL `KL(Student || Teacher)`。不得持久化完整 `[B,L,V]` logits；chunked 实现必须先与 tiny-tensor full reference 做数值、极限和梯度对齐。top-k、forward KL、JSD 或 off-policy KD 都不进入首个主结果。

### 6. 两个成本 estimand

- **E1 Signal efficiency**：唯一严格匹配量是每阶段 2M Student backward loss tokens。prompt exposure 和 Student FLOPs报告但不称 matched。
- **E2 Practical efficiency**：记录 rollout tokens、所有 Student/old/reference/Teacher forward、Student backward、accelerator-seconds、GPU-hours、峰值显存和可获得的能耗。

成本拆为共享 `C_anchor`、共享 `C_teacher` 与逐臂 `C_arm`。报告 warm-start `C_arm`、cold-start pipeline 和 campaign total；Teacher inference 属于含 OPD 的 `C_arm`，Teacher SFT 在 campaign 中只计一次。

### 7. Claims 与统计

- **C1**：A1−A0 与 A2−A0 是两个确认性 contrasts；唯一 endpoint 为 MATH-500 greedy item-level accuracy。三 seed 与 item 配对，按 seed 和 MATH level 内 item 做 10,000 次 two-way cluster bootstrap，报 95% CI；item-level seed-mean 差异做 100,000 次 paired sign-flip randomization，p-value 用 Holm 控制 family-wise α=0.05。统计 superiority 还需点估计至少 +2pp 才达到项目 practical success。
- **C2**：A3−A4 是唯一确认性 order contrast。superiority 需 95% CI 不含 0 且绝对点估计 ≥2pp；否则只有 TOST 90% CI 完全落入 [-2,+2]pp 才能称 practical equivalence。
- pass@8 每题固定 n=8，使用 `1-C(n-c,k)/C(n,k)`；generation seeds 成对。GSM8K、MathArena、AIME、IFEval、MMLU-Pro 都是 supporting evidence，不拼 composite。

### 8. 可行性硬 gate

C0/C5 分别 profile E2B LoRA backward、group-8 rollout、E4B SFT 和 E2B+E4B full-vocab KL。用 5 arms × 3 seeds × 4M = 60M Student loss tokens 外加 `C_anchor/C_teacher` 重算工时，保留 30% buffer。若可用资源不是至少 2×80 GB/4×48 GB 且 profile 不闭合，则停止主矩阵；只能对所有臂等比例减 `U` 或统一 completion cap 后重测，不能量化 Teacher、改近似 KL、换旧模型或删臂。

## Remaining Uncertainty

唯一尚不能在规划阶段消除的关键不确定性是实际 GPU 型号/数量、Gemma 4 当前框架 revision 的正确性，以及 full-vocab KL steady-state throughput。这些已被转化为执行前 C0/C5 hard gate，不再伪装成已解决的可行性。
