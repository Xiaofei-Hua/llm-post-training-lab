# Development Module D03 — Exact-Reward Dr.GRPO Surrogate

## 模块边界

D03 实现项目冻结的 exact-reward Dr.GRPO 数值核心：全局 group-relative advantage、zero-variance group 排除、token-level importance ratio、PPO clipping、Dr.GRPO 固定常数归一化，以及 gradient accumulation/DDP 的精确缩放。

本模块不实现 reward parser/verifier、rollout、old-policy 生命周期、vLLM、模型 forward、optimizer 或 GPU 训练。它是之后 C2 trainer adapter 要调用的张量层，不等于 `C2-001` 或 G3 已完成。

## Deletion test：为什么 D02 不足

对 target token 的 masked CE 有：

```text
∂L_CE / ∂logit_target = p_target - 1 ≤ 0
```

gradient descent 因而只能提高被选 target 的概率。GRPO 在 on-policy ratio `ρ=1` 附近的 sampled-token surrogate gradient 与 `-A` 成正比：正 advantage 要提高概率，负 advantage 必须降低概率。masked CE 无法表达后一种更新方向，也不包含 old-policy ratio、PPO clipping 或 Dr.GRPO 的固定分母。

因此删除 D03 并用 D02 替代，会直接失去 signed outcome feedback 和 trust-region 语义；现有最小方案不足，新增独立模块成立。

## 冻结数学合同

对同一 prompt 的完整全局 group `g`，group size 为 `K=8`，exact reward `r_i∈{0,1}`。先计算 sample standard deviation：

```text
μ_g = (1/K) Σ_i r_i
s_g = sqrt((1/(K-1)) Σ_i (r_i-μ_g)^2)
A_i = (r_i-μ_g)/(s_g+1e-4),  if s_g>0
A_i = 0,                       if s_g=0
```

group 必须先完整聚合，再切成 microbatch 或 DDP rank。`s_g=0` 的 group 不重采样，其所有 token 从 D01 mask 排除。这样 skipped group 不产生梯度，也不消耗 Student backward loss-token 预算。

对 active completion 的 target-aligned token log probability：

```text
ρ_i,t = exp(log πθ(y_i,t|x_i,y_i,<t) - log πold(y_i,t|x_i,y_i,<t))
l_i,t = -min(ρ_i,t A_i, clip(ρ_i,t, 1-ε, 1+ε) A_i)
L = (1/(B_active L_max)) Σ_i,t M_i,t l_i,t
```

其中 `ε=0.2`，`M` 是 zero-variance filter 与 D01 精确预算共同决定的最终 Boolean mask；`B_active` 是预算末批截断前、全局 logical update 中 active completion 的数量；`L_max` 是冻结的 maximum completion length。`ΣM` 是跨算法匹配的预算 `U`，但不是 Dr.GRPO 分母。

### 项目特有的 zero-variance 语义

公开 TRL 的 Dr.GRPO 以 batch completion 数乘 `max_completion_length` 归一化；其零 advantage 行通常仍留在 batch 维度。本项目已经预注册“zero-variance group 不计入 backward token”，因此 D03 进一步把整个 skipped group 从 active normalizer 排除。也就是说，D03 不是对 TRL 默认 trainer 的逐字节复刻，而是公式相同、skip 语义显式化的项目 adapter。未来 C2 parity 测试必须分别锁定两种分母，禁止静默混用。

reference KL 固定 `beta=0`，不进入 D03 公式。

## 公开 API

实现位于 `src/posttrain_lab/train/grpo_surrogate.py`，并从 `posttrain_lab.train` 导出：

- `compute_exact_group_advantages(...) -> ExactGroupAdvantageOutput`：验证完整 exact-reward groups，返回 advantage、group mean/sample std、active/skipped group IDs、过滤后的 mask、completion 计数和有效组率；
- `dr_grpo_token_surrogate(...) -> DrGRPOSurrogateOutput`：消费已经 target-aligned 的 current/old log probabilities、advantage 与最终 mask，返回可反向传播的 loss 和可聚合诊断；它不做 causal shift，也不从 logits 内部 gather target；
- `normalization_denominator` 固定为 `global_active_completion_count × max_completion_length`；
- `importance_ratio_{sum,min,max}`、clip counts 与 `local_token_count` 都是 local shard 指标，不执行隐式 collective。

`current_log_probs` 保留梯度；`old_log_probs`、reward 与 advantage 必须 stop-gradient。空 local shard 返回与 current graph 相连的可微零值，以支持不均匀 DDP 分片。

## Gradient accumulation 与 DDP

设 world size 为 `W`，rank `r` 的 microbatch `m` 上 raw token loss sum 为 `S_r,m`。D03 返回：

```text
L_r,m = W × S_r,m / (B_active L_max)
```

同一 logical update 内直接累加所有 microbatch gradient；default DDP 再对 `W` 个 rank 求平均，最终得到：

```text
(1/W) Σ_r Σ_m ∇L_r,m
= ∇[Σ_r Σ_m S_r,m / (B_active L_max)]
```

因此调用方不得再按 microbatch 数或 accumulation steps 除 loss。若外层 trainer 会自动做该除法，adapter 必须显式抵消。`B_active` 必须是所有 rank 共享的 global 值并可被 `K` 整除；D03 不替调用方执行 all-reduce。

## 数值与诊断合同

- FP16/BF16/FP32 log probability 默认在 FP32 计算 ratio、clip 和 loss sum；FP64 输入默认保持 FP64；显式 override 只允许 FP32/FP64；
- selected log probabilities 必须有限且不大于 0；unselected sentinel 不参与计算；
- ratio overflow 或 underflow 到 0 都直接失败，不做会改变目标的静默 clamp；
- clip count 只统计真正进入饱和梯度区的 token：`A>0, ρ>1+ε` 或 `A<0, ρ<1-ε`；
- group IDs 在输出中稳定排序，active/skipped completion 与 group 计数用于记录 effective-group rate；
- CUDA 上完整 grouping、collective 与确定性仍属于后续 integration gate，本模块只完成 CPU 数值核心。

## Hard failures

D03 会拒绝以下输入，而不是猜测或修复：

- 非 `0/1`、非有限、带梯度的 reward；不完整 group、负 group ID 或错误 group size；
- 任一生成 completion 没有 candidate token；mask 的 rank、shape、dtype 或 device 不一致；
- old log probability 或 advantage 带梯度；selected token 的 advantage 为零；
- selected log probability 为正或非有限，importance ratio 非有限或不严格为正；
- global active completion 数小于 local contributing rows、不能被 group size 整除；
- 单行 selected token 数超过冻结的 `max_completion_length`；非法 epsilon、world size 或 compute dtype。

## 可复现验证

```bash
uv sync --frozen --all-groups
uv run pytest -q tests/test_grpo_surrogate.py
uv run pytest -q
```

D03 有 21 个 CPU tests，其中包含 100 个 Hypothesis 生成 batch；覆盖独立 advantage/loss oracle、sample std、任意顺序 group IDs、zero-variance active denominator、on-policy signed gradient、双向 clip、autograd gradcheck、固定分母、BF16→FP32、不均匀 accumulation/DDP、空 rank、D01→D03 接线及全部 hard failures。仓库当前全量共 108 个 tests、575 个生成案例。

验证没有下载模型或数据，没有访问 MPS/CUDA。D03 通过不等于 `C2-001` 或 G3 已完成：真实模型 forward、rollout freshness、old-policy/weight-sync、reward parser/verifier 与两步集成测试仍未运行。

## 上游语义审计

- [Hugging Face TRL GRPO 文档](https://huggingface.co/docs/trl/grpo_trainer)：核对 `dr_grpo`、token-level importance sampling、clipping 与 reward scaling 的公开定义；
- [TRL `grpo_trainer.py` 固定审计版本](https://github.com/huggingface/trl/blob/4506f4af8ccc5b7f8b337f589a696a19913d8a4a/trl/trainer/grpo_trainer.py)：核对 sample std、`1e-4` 稳定项及实现级分母语义。

仓库不引入 TRL 运行时依赖；这些来源是语义审计基线，D03 的数值 oracle 独立实现并明确记录上述项目差异。
