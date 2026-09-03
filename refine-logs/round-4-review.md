# Round 4 Raw Review

> Reviewer output preserved by the primary agent. No edits were made by the reviewer.

CALIBRATION: none

COMPOSITE: **9.06 / 10**

| 维度 | 权重 | 分数 | 加权分 |
|---|---:|---:|---:|
| Problem Fidelity | 15% | 9.5 | 1.425 |
| Method Specificity | 25% | 9.4 | 2.350 |
| Contribution Quality | 25% | 8.8 | 2.200 |
| Frontier Leverage | 15% | 9.2 | 1.380 |
| Feasibility | 10% | 8.0 | 0.800 |
| Validation Focus | 5% | 9.2 | 0.460 |
| Venue/Portfolio Readiness | 5% | 8.8 | 0.440 |

**GAP:** 没有人工精选的 proposal anchors，因此无法进行 exemplar-relative calibration。相对于 READY=9，当前规划已经关闭了此前所有核心文档缺口：Problem Anchor 恢复为解法无关表述，Teacher selection 与 qualification 隔离，五臂采用完全对称的 stage/reset 结构，GRPO/OPD 的数据流、mask、refresh、归一化与预算计数均已冻结，统计推断单位也不再伪造 training-seed 精度。剩余差距主要不是可继续“写方案”消除的问题，而是尚未获授权执行的外部证据：Gemma 4 框架兼容性、same-lineage Teacher 是否通过 C4、full-vocab OPD 吞吐以及总 GPU 预算是否通过 C5。文档内只剩少量数据计划措辞清理，不构成 method-ready blocker。

## Anchor / Drift / Simplicity

- **Problem Anchor:** preserved。Round 3 的 immutable anchor 与 `PROJECT_CHARTER.md` 一致，且不再把 Gemma 4、GRPO 或 OPD 写成问题本身。
- **Claim boundary:** 清晰。结论被限制为指定 E4B Teacher、冻结 recipe 和等 Student backward-token 预算下对 E2B Student 的 intervention effect。
- **Drift Warning:** **NONE**。
- **Simplicity:** **通过**。A0–A4 是两个 claims 所需的最小闭合反事实；没有必要删除任何核心臂。
- **Contribution focus:** 单一且稳定，即 GRPO sparse reward 与 OPD dense Teacher signal 的单独作用和顺序交互。
- **Modernity:** 足够，不需要再加入 PRM、Agent RL、其他 divergence 或多模态任务。

## 核心方法审计

### Same-lineage E4B Teacher

设计已经足以支持限定后的归因：

- E2B/E4B 都从 Base 出发；
- 使用相同 `D_anchor`、template、mask、样本顺序、token/epoch budget 和 LoRA 类别；
- `D_select` 只选 checkpoint；
- 独立 sealed `D_teacher_gate` 才执行 +5pp、CI、NLL、parse 与 tokenizer qualification；
- 所有 OPD stages 只读同一个冻结 Teacher；
- Gate 失败时停止 primary OPD，不以 E4B-it 静默救场。

这不能隔离纯参数量效应，但文档已经明确不作该主张，因此不存在额外 Teacher baseline 的必要性。

### 五臂两阶段公平性

当前结构成立：全部是两个 2M stage、全部在中点重置 optimizer/scheduler、每臂总预算均为 4M、同一 objective 跨 arm/stage 复用相同 config hash、A3−A4 只改变信号顺序、A0–A4 全部运行三个预注册 seeds。

`U` 也已定义为实际进入已执行 optimizer update 的有效 objective positions，并明确排除 prompt、padding、EOS 后位置、纯 forward token 和被跳过的 zero-variance group。末批稳定排序 mask 使预算精确闭合。

### GRPO 合同

规划层已经可执行。`dr_grpo`、group size、clip、beta、更新次数、IS 粒度、reward scaling、temperature、policy refresh、weight sync、RNG、zero-variance 和 completion-cap gate 均已冻结。剩余工作属于 C2 的实现验证，而非方法补写。

### OPD 合同

规划层已经可执行：每次 update 前生成新 rollout、Student sampling IDs stop-gradient、Teacher 接收相同 prefix、completion-through-first-EOS mask、full-vocab reverse KL、batch-token mean、Teacher 永久冻结、exact chunked/fused 实现必须对齐 tiny oracle、不允许以 top-k 近似救场。剩余风险完全被 C3/C5 显式承接。

### 统计协议

现在定义无歧义：

- 唯一 confirmatory endpoint 是 MATH-500 greedy accuracy；
- 推断明确条件于三个预注册 training seeds；
- 每个 item 携带完整三-seed vector；
- bootstrap 和 randomization 都只在 item 层执行；
- C1 两个 contrasts 使用 Holm；
- C2 是唯一 order contrast；
- superiority 与 TOST equivalence 分开；
- supporting benchmarks 不进入 composite；
- seed-specific effects 必须单列并披露方向不一致。

该协议不声称推广到所有训练随机种子，避免了三 seed 下不可能成立的 seed-population 显著性主张。

## 当前规划阶段仍可修复的非阻塞问题

1. `DATA_PLAN.md` 中“GSM8K / MATH-500 / MathArena / AIME 只作评测和 dev calibration”与 sealed evaluation 定义冲突。应改成“只作评测，不用于 dev calibration”。**Priority: MINOR**
2. `DATA_PLAN.md` 的 `D_dev` 描述仍写“与 D_calib 隔离”；`D_calib` 已拆分，应改为 `D_select` 与 `D_teacher_gate`。**Priority: MINOR**
3. `DATA_PLAN.md` 的 sealed evaluation 表未列出 `MMLU-Pro`，而 Charter、Benchmark 和 config 都把它列为 retention benchmark。应补入 E 层。**Priority: MINOR**

这些是跨文档一致性清理，不改变方法、分数或规划 verdict。

## Simplification Opportunities

**NONE.** 核心图已经最小。DPO、E4B-it、larger Teacher、QLoRA、top-k KL 和其他 divergence 应继续保持 deferred。

## Modernization Opportunities

**NONE.** 当前技术路线已经适当现代化；继续叠加组件只会降低归因质量。

## Readiness Verdicts

### Planning / Method Verdict: **READY**

当前授权范围是前期规划；在这个范围内，方法、数据、预算口径、统计协议、失败条件和执行 gates 已足够具体且相互一致。三个 minor 文档措辞可在最终整理时修正，但不是方法 blocker。

### Execution Verdict: **CONDITIONAL — NOT YET READY**

未下载模型、未占用 GPU、未执行 profile，因此不能声称执行就绪。至少需要：

- **C0:** Gemma 4 text path、processor、tokenizer、冻结非文本模块和版本组合通过；
- **C3:** exact full-vocab reverse-KL 数值与梯度通过；
- **C4:** same-lineage E4B Teacher 在独立 gate 上通过；
- **C5:** E4B SFT、group-8 rollout、E2B+E4B OPD 和 60M Student-token campaign 在真实资源下闭合，并保留 30% buffer。

这些是当前授权外的外部条件，不应通过继续改文档或虚构 profile 来消除。
