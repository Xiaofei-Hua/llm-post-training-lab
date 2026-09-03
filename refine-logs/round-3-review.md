# Round 3 Raw Review

> Reviewer output preserved by the primary agent. No edits were made by the reviewer.

CALIBRATION: none

COMPOSITE: **8.37 / 10**

| 维度 | 权重 | 分数 | 加权分 |
|---|---:|---:|---:|
| Problem Fidelity | 15% | 8.5 | 1.275 |
| Method Specificity | 25% | 8.7 | 2.175 |
| Contribution Quality | 25% | 8.5 | 2.125 |
| Frontier Leverage | 15% | 9.0 | 1.350 |
| Feasibility | 10% | 6.5 | 0.650 |
| Validation Focus | 5% | 7.8 | 0.390 |
| Venue/Portfolio Readiness | 5% | 8.0 | 0.400 |

**GAP:** 没有人工精选的 proposal anchors，因此无法进行 exemplar-relative calibration。相对 READY=9，本轮已解决 Teacher 来源、stage/scheduler 不对称、matched-token 误称和主终点发散，核心五臂也已达到较高的聚焦度。剩余差距不需要增加模块，而是要关闭四个精确性缺口：Teacher checkpoint selection 与 capability gate 仍共用 `D_calib`；统计协议对 training seed 与 item 的层级关系定义错误；OPD 尚未冻结 rollout-refresh、loss mask/normalization 和 `U` 计数语义；C0/C5 尚无真实硬件结果。最后一项意味着当前最多是 methodologically near-ready、execution-conditionally ready，而不是无条件 READY。

## Problem Anchor / Drift

**科学问题实质上 preserved。** 本轮仍然研究同一 Student anchor 下 GRPO 与 OPD 的单独作用和顺序，没有加入多模态、DPO 主矩阵或新模块。

但存在一个 **anchor hygiene 问题**：本轮把 Bottom-line problem 重写成了包含 Gemma 4、GRPO 和 OPD 的方法定义，而不是逐字保留上一轮更一般的 immutable problem anchor。这样会让“问题”依赖当前解法。

修复方式：

- 恢复上一轮经用户意图校正后的通用 Problem Anchor 原文；
- 把 Gemma 4 pair、五臂和具体 objectives 放入 Constraints、Method Thesis 和 Success Condition；
- 恢复一句明确 claim boundary：结论是“指定 Teacher 和固定 recipe 在等 Student backward-token 预算下的 intervention effect”，不是抽象 loss、Teacher 容量或所有 dense/sparse signal 的普遍因果效应。

**Drift Warning:** **NONE substantively**；但上述 anchor protocol violation 应在 READY 前修正。

## Same-lineage E4B Teacher 审计

### 是否足以归因

**对当前限定后的主张足够，但不能支持更强主张。**

E4B Base 使用与 E2B 完全相同的 `D_anchor`、模板、mask、样本顺序、SFT token/epoch budget 和 LoRA 类别，确实消除了 E4B-it 未知 instruction/post-training recipe 这一主要混杂。Teacher 在所有 OPD stages 中冻结且唯一，也使 A3/A4 的顺序比较成立。

它足以支持：

> 在这个经 gate 的 same-lineage E4B Teacher 下，加入 OPD recipe 以及改变 OPD/GRPO 顺序的效果。

它不能支持：

- 效果只来自参数量；
- E2B/E4B 除容量外完全相同；
- reverse-KL OPD 对任意 Teacher 都有效；
- “dense signal”一般优于或不同于“sparse signal”。

E2B/E4B 仍有层数、per-layer embedding 数量和容量结构差异。当前 `SYSTEM_DESIGN.md` 已正确承认这一点，不需要增加额外 Teacher baseline。

### 当前阻塞缺口：selection/gate double use

`D_calib` 同时用于 E2B/E4B checkpoint selection，又用于 Teacher capability gate。这样 +5pp、CI 和 NLL gate 都是在被用于选择 checkpoint 的数据上计算，存在乐观偏差。

必须拆成：

- `D_select`：只用于 Student/Teacher SFT checkpoint selection；
- `D_teacher_gate`：完全 sealed，只用于最终 Teacher qualification；
- `D_dev`：继续只用于 objective LR/smoke configuration。

建议保留 500 条 `D_select`，另增加或划出 500 条 family-disjoint `D_teacher_gate`。Gate 的 accuracy、NLL、parse rate 全部只能在后者计算。若不拆分，same-lineage 设计仍然好，但“qualified stronger Teacher”这一前提没有得到独立验证。

## 五臂两阶段公平性

**结构上已经公平，是本轮最强的改进。**

A0–A4 均有两个 2M stage、相同中点 reset、相同总 Student loss-token budget，解决了上一轮 scheduler restart 与 stage-count 混杂。A3−A4 现在是干净的顺序 contrast。

仍需冻结两项实现语义：

1. 同一个 objective 在所有出现位置必须复用同一配置。例如 A1 stage 1/2、A3 stage 2、A4 stage 1 的 GRPO 使用完全相同 LR、scheduler family、batch construction 和 stopping semantics。
2. 精确定义 `U`：

> `U` 是经过 EOS/padding/objective mask 后、实际进入一次已执行 optimizer update 的 completion loss positions 数。

GRPO 被跳过的 zero-variance groups 不计入 `U`；rollout prompt 和纯生成 token 不计入 E1；OPD 每个有 KL loss 的 completion position 计一次；SFT 只计 assistant target tokens。否则不同实现可能都声称“2M loss tokens”，但实际梯度机会不一致。

E1 仍只代表固定 Student gradient-token opportunity，不代表数据量、prompt exposure 或总计算相同；文档目前对此表述正确。

## GRPO / OPD 可执行性

### GRPO

GRPO 合同已经接近执行级：`dr_grpo`、`beta=0`、clip、group size、importance-sampling level、reward scaling、policy refresh、weight sync、双 RNG、zero-variance gate 和 completion-cap rule 均已给出。

剩余主要是版本验证，不是方法缺失。C2 必须确认固定 TRL revision 下：

- `steps_per_generation=1` 与“每 generation batch 刷新 old policy”语义一致；
- `dr_grpo`、token IS、group scaling 的实现确实匹配文档；
- skipped group 不被 dataloader 隐式重采样；
- vLLM/training model sync 后的 policy age 恰为一个 generation batch。

### OPD

OPD 的 divergence 与 Teacher 接口已经明确，但尚缺三个会影响结果的配置：

- **Refresh:** 固定 `num_iterations=1`，每次 update 使用当前 policy 新生成的 completion；
- **Mask:** KL 只作用于 Student-generated completion positions；prompt、padding 和 EOS 后位置必须 mask；
- **Normalization:** 先对所有有效 vocabulary 求每 token reverse KL，再按全 batch 有效 completion token 数求均值；不能先按序列等权平均。

Chunked exact KL 还需说明是 sequence-chunk、vocab-chunk 两遍 log-normalizer，还是 fused kernel；但只要 C3 数值、极限和梯度对齐，并由 C5 证明吞吐，方法上不要求新增 approximate variant。

## 统计检验审计

**当前并非完全无歧义。** Training seed 不是嵌套在 item 内；每个 checkpoint seed 同时影响全部 500 个 item，因此这是 crossed structure。把 seed 在每个 item 内独立重采样会产生伪重复。

最简单且不增加训练成本的定义是条件于三个预注册 seed：

```text
d_i = (1/3) Σ_s (Y_a,s,i - Y_b,s,i)
Delta_hat = (1/500) Σ_i d_i
```

然后：

- bootstrap 只重采样 500 个 item；
- 每次重采样必须携带该 item 的完整三-seed prediction vector；
- paired randomization 也以 item 为单位交换完整 arm/control seed vector；
- C1 两个 p-value 再做 Holm；
- 单独报告三个 paired-seed effect，要求方向一致或明确标注不稳定；
- C2 TOST 使用同一 item-level bootstrap estimand。

该 CI 推断的是“这三个预注册训练 seed 平均模型在 MATH item population 上的差异”，不是对所有可能 training seeds 的总体推断。若要把 training seed 当随机总体做强推断，三个 seed 不足；seed-level exact sign permutation 只有 8 个排列，不可能支持常规双侧 `p<0.05`。本项目不应为此扩展大量 seeds，限定推断范围更合适。

其余统计设计是清晰的：MATH-500 是唯一 confirmatory endpoint；C1 Holm family 明确；C2 是唯一 order contrast；superiority 与 ±2pp TOST equivalence 区分正确；supporting benchmarks 不拼 composite；pass@8 的固定采样数和公式明确。

## 硬件与可行性

硬件未知使当前只能给 **conditional readiness**。

规划对风险的处理是诚实且合理的：60M Student loss tokens 并未伪装成总 token；`C_anchor/C_teacher/C_arm` 和三种成本视角也定义正确。但 E2B/E4B、262K-vocab exact KL、group-8 长 rollout 和五臂三 seed 是否能在 450–900 GPUh 内完成，只能由 C5 回答。

### 低于 7 分的维度

#### Feasibility — 6.5/10

- **Weakness:** 2×80 GB/4×48 GB 尚只是候选资源；E4B SFT、E2B+E4B exact reverse-KL 和 GRPO 2k/4k rollout 没有 steady-state profile。若 C5 失败，Success Condition 中五臂三 seed 无法满足。
- **Concrete fix:** 实际执行 C0–C5；按测得吞吐重算 `C_anchor+C_teacher+ΣC_arm` 并保留 30% buffer。若预算不足，只允许所有臂同步降低 `U` 或统一 cap 后重新 profile。只有资源已确认且 C5 通过，才把 status 从 conditional 改为 READY。
- **Priority:** **CRITICAL**

## 其他阻塞 READY 的一致性问题

1. **配置中的 QLoRA fallback 与主方案冲突。** 应改为 `same_model_lower_U_or_shorter_context`；若研究 QLoRA，必须是新 experiment family。**Priority: IMPORTANT**
2. **Stretch 的“architecture portability”措辞不一致。** 统一为 Teacher/checkpoint transfer。**Priority: MINOR**
3. **Anchor 和 claim boundary 未在本轮完整方案中保持。** 恢复一般问题表述，并明确 recipe-level estimand。**Priority: IMPORTANT**
4. **Teacher gate 必须从 checkpoint selection 数据中独立。** **Priority: CRITICAL**
5. **统计 bootstrap 必须改为 item-conditional inference。** **Priority: CRITICAL**
6. **OPD refresh、mask、normalization 和 `U` 计数必须冻结。** **Priority: IMPORTANT**

## Simplicity Opportunities

**NONE for the core design.** 五臂已是回答两个 claims 的最小闭合图。DPO、E4B-it、larger Teacher、top-k KL 和其他 divergence 必须继续停留在核心结果之后；不建议为统计问题增加新的训练臂。

## Modernization Opportunities

**NONE.** 当前需要的是实验单位、Teacher qualification 和实现语义更严格，而不是再加入 PRM、Agent RL 或异步系统贡献。

## Verdict

**REVISE**

这是一个方法学上接近完成、贡献高度聚焦的方案，但还不能 READY。先修复独立 Teacher gate、统计单位、OPD mask/refresh/normalization、anchor 边界和配置冲突；随后必须由真实 C0/C5 关闭 Gemma 4 full-vocab OPD 的资源风险。文档修正后可以称 **method-ready**，但在硬件与吞吐实测前只能称 **execution-conditionally ready**。
