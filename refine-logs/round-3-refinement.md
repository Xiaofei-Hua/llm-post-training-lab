# Round 3 Refinement — Selection/Gate 隔离与精确实验单位

## Immutable Problem Anchor

在有限算力和完全公开、可追溯的数据条件下，从同一前沿学生模型的 SFT checkpoint 出发，受控比较不同后训练学习信号的单独作用和顺序交互，并把收益与模型、数据、训练预算、评测及计算成本混杂区分开。

## Anchor Check

**Preserved.** Gemma 4 E2B/E4B、GRPO、OPD 和五臂属于 Constraints/Method Thesis，不再写进 immutable problem 本身。本轮只修复评测独立性、实验单位与实现语义，没有扩展研究问题。

## Claim Boundary

确认性 estimand 是：**指定 same-lineage E4B Teacher 与 frozen GRPO/OPD recipe，在相等 Student backward-token 预算下，对这个 E2B Student 的 intervention effect。** 不声称效果只来自参数量，不推广到任意 Teacher、任意 KL 或所有 dense/sparse signal。

## Simplicity Check

**无可删除核心项。** A0–A4 仍是最小闭合反事实；DPO、E4B-it、larger Teacher、QLoRA、top-k KL、其他 divergence、PRM 和 Agent RL 都不进入 must-run。本轮没有新增训练 arm。

## Round 3 Review → 修订

| Gap | Resolution |
|---|---|
| Problem Anchor 依赖当前解法 | 恢复通用 immutable anchor；具体模型/算法移至 Method Thesis |
| Teacher selection 与 qualification 双重使用 `D_calib` | 拆为 500 `D_select` 与 500 始终 sealed、family-disjoint `D_teacher_gate` |
| training seed 错误嵌套于 item | 推断条件于三个预注册 seeds，只重采样 item 并携带完整三-seed vector |
| `U` 含义不精确 | 定义为实际进入已执行 optimizer update 的有效 objective positions；冻结最后一批精确 budget mask |
| 同一 objective 跨 arm/stage 可能漂移 | 同一 objective 必须加载相同 config hash，除 parent/stage/RNG 外不得变化 |
| OPD refresh/mask/normalization 缺失 | 固定 `num_iterations=1`、每 update 新 rollout、completion-through-EOS mask、batch-token mean |
| QLoRA fallback 与 LoRA 主线冲突 | 删除 fallback；QLoRA 必须新建 experiment family |
| Stretch 描述不一致 | 统一为 Teacher/checkpoint transfer，不称 architecture portability |

## 冻结后的方法合同

### Data independence

`D_anchor=10k` 用于 E2B/E4B 同源 SFT；`D_select=500` 只选 SFT checkpoint；`D_teacher_gate=500` 在选择结束后只解封一次并只做 Teacher +5pp/CI/NLL/parse qualification；`D_dev=500` 只做 objective config pilot；`D_core=2k` 只做五臂 intervention。所有 split 在 source/problem/template family 层面隔离。

若 gate 失败，不得用 gate 结果返回选择 checkpoint。任何重选都需新建 experiment family 和新的 sealed gate split。

### Exact `U`

`U` 是 objective mask 为 1、真正进入一次已执行 optimizer update 的 Student loss positions 数，每个 position 每次 update 计一次：

- SFT：assistant target 首 token 至模板 EOS（含 EOS）；
- GRPO：实际参与 policy loss 的 generated completion positions；zero-variance skipped group 为 0；
- OPD：有 reverse-KL loss 的 Student-generated completion 首 token至第一个 EOS（含 EOS）；
- prompt、padding、EOS 后、只做 rollout/Teacher/old-policy forward 的 token 均为 0。

最后一个 update 的候选 positions 按 `(sample_id, generation_index, token_index)` 稳定排序；SFT 的 `generation_index=0`。只保留填满剩余预算的前缀 budget mask，使每个 stage 精确结束于 `U=2,000,000`。manifest 保存候选数、保留数和累计数。

### Objective invariance

同一 objective 在所有 arm/stage 使用相同 resolved config hash，包括 LR、scheduler family、batch construction、loss mask/normalization、generation cap 和 stopping semantics。阶段边界统一 reset；只允许 parent checkpoint、stage ID 和预注册 RNG seed 不同。

### OPD

- `num_iterations=1`；每次 optimizer update 前由当前 Student 生成新 completion；
- Teacher 对同一 prefix 做 frozen forward，sampling IDs stop-gradient；
- mask 只包含 completion 至首个 EOS（含 EOS）；
- 每个有效 token 上计算 full-vocab reverse KL，再用全 batch 有效 token 总数归一化；
- full-vocab exact chunked/fused kernel 必须与 tiny full-tensor oracle 做 value/limit/gradient 对齐；C5 决定 kernel 是否够快，不以 top-k 近似救场。

### Conditional item inference

对 arm `a` 与 `b`，先固定三个预注册训练 seeds：

```text
d_i = (1/3) * sum_s(Y[a,s,i] - Y[b,s,i])
Delta = (1/500) * sum_i(d_i)
```

bootstrap 在 MATH level 内只重采样 item，每个 item 携带完整三-seed prediction vectors。paired randomization 也以 item 为单位交换完整 arm/control vector。该区间只推广到 MATH item population 条件下的这三个平均模型，不推广到所有 training seeds。每个 seed 的 paired effect 单列；方向不一致必须标注 instability。C1 用 Holm，C2 用同一 unit 的 superiority/TOST。

## Readiness Boundary

至此文档层方法可以 method-ready。执行仍以 C0/C5 为 hard gate：只有确认真实 GPU、Gemma 4 revision、E4B SFT、group-8 rollout 和 exact full-vocab KL 的 steady-state profile，并为 `C_anchor+C_teacher+ΣC_arm` 留 30% 余量后，才可标记 execution READY。用户当前只要求前期规划，因此不下载模型、不占用 GPU、不伪造 profile。
