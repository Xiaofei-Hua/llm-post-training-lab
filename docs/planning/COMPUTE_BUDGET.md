# 算力与成本规划

## 为什么先校准再报预算

GRPO 和 OPD 的成本主要由 rollout 长度、group size、Teacher forward 和实现吞吐决定。规划阶段直接写一个精确 GPU-hour 数会产生虚假确定性，因此先给预算档位，并把 100-step 实测作为 G0。

## 预算档位

| 档位 | 资源假设 | Student / Teacher | 能完成的范围 | 暂定总量 |
|---|---|---|---|---|
| A：代码路径 | 1×24 GB | Gemma 4 E2B / E4B | 仅做单模型 SFT/GRPO 与小型精确 full-logit/chunked-KL 正确性测试；full-vocab OPD 主实验不闭合 | 不给主实验工时承诺 |
| B：推荐主线 | 2×80 GB 或 4×48 GB | Gemma 4 E2B / same-lineage E4B | 10k 双模型 SFT、五臂全部 3 seeds、完整评测 | profile 前暂估 450–900 GPUh |
| C：前沿扩展 | 4–8×80 GB | E2B / Gemma 4 12B或31B | 强 Teacher、长上下文与 Teacher/checkpoint transfer | 1000 GPUh 以上 |

Gemma 4 的 E2B/E4B 名称指 effective size，官方 BF16 推理内存估计约 11.4/17.9 GB；两模型静态加载即约 29 GB，尚未计激活、KV cache、训练状态和 262K-vocab divergence。范围仅用于排期；C5 的 100-step calibration 后必须重算并保留 30% 余量。

## 成本公式

```text
SFT cost ≈ trained_tokens / measured_train_tokens_per_second
GRPO rollout cost ≈ prompts × group_size × avg_generated_tokens / rollout_tokens_per_second
GRPO update cost ≈ optimized_tokens / measured_train_tokens_per_second
OPD cost ≈ student rollout + student forward/backward + teacher forward
```

主矩阵固定 5 arms × 3 seeds × 2 stages × 2M Student backward loss tokens，即 post-anchor 共 60M Student loss tokens。它不是 60M end-to-end tokens：GRPO 的 group rollout、old-policy forward 与 OPD Teacher forward 必须另计。

## 三类成本与三种视角

- `C_anchor`：一次 E2B Base→10k `D_anchor` SFT，所有臂共享；
- `C_teacher`：一次 E4B Base→同一 `D_anchor` SFT，并通过 gate 后冻结；
- `C_arm`：一个 post-anchor arm 的两阶段训练，包括 rollout、old/reference/Teacher forward 和 Student backward。

| 视角 | 非 OPD arm | 含 OPD arm | 用途 |
|---|---|---|---|
| Warm-start / marginal | `C_arm` | `C_arm`（含 Teacher inference） | 已有 anchor/Teacher 时选 recipe |
| Cold-start pipeline | `C_anchor+C_arm` | `C_anchor+C_teacher+C_arm` | 从公开 Base 开始复现单条路线 |
| Campaign total | \- | `C_anchor+C_teacher+ΣC_arm` | 整个研究项目真实成本 |

`C_teacher` 在 campaign total 中只计一次；不能隐藏，也不能在 A2/A3/A4 中重复三次。

## G0 校准任务

固定 E2B Student 与 E4B Base/same-lineage Teacher，先跑：

1. 20 step inference warm-up；
2. 100 step SFT；
3. 20 prompt × 4 generations 的 rollout；
4. 100 step E4B LoRA SFT；
5. 20 step E2B+E4B full-vocab chunked reverse-KL distillation；
6. 记录 steady-state tokens/s、峰值显存、平均输出长度、weight-sync 开销与 checkpoint 大小。

据此确认是否能执行五臂主矩阵，并把预算误差控制在 ±30%。若不闭合，所有臂按同一比例降低 `U` 或 completion cap 后重新 profile；不得量化 Teacher、改 top-k KL、换旧模型或删除关键对照来静默改变问题。

## 省算力原则

- 先用 64–256 样本做 overfit 和 reward 单测。
- 单 seed 只筛 objective 内的学习率/实现配置；A0–A4 五臂全部进入 3 paired seeds，禁止按初步效果淘汰 arm。
- smoke 和 main 都使用 Gemma 4 E2B；smoke 只缩短序列和数据，避免跨尺度选参。
- 不用大规模超参 sweep；每个关键算法只允许 2–3 个有理论依据的配置。
- Nice-to-have 实验不能挤占主对照与重复性预算。
