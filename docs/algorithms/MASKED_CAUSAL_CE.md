# Development Module D02 — Production Masked Causal Cross-Entropy

## 模块边界

D02 实现 SFT 的 masked causal cross-entropy 数值核心，并把 D01 的绝对 target-position mask 直接转换为可反向传播的 loss。它不实现 Transformer forward、optimizer、trainer 或模型集成；GRPO surrogate 与 OPD reverse-KL 后续已由独立 D03/D04 实现，不属于本模块。

- 实现：`src/posttrain_lab/train/masked_ce.py`
- 公开导出：`src/posttrain_lab/train/__init__.py`
- 验证：`tests/test_masked_ce.py`
- 上游 mask/budget：`docs/algorithms/LOSS_TOKEN_BUDGET.md`

仓库只保留当前张量 API，没有平行的旧 CE 实现或占位 trainer。

## 目标位置与归一化

`loss_mask[b, t]` 描述的是绝对 **target token position**。因果模型用 position `t-1` 的输出预测 target `labels[b, t]`，因此 position 0 永远不能进入 loss。

令 `M` 为 Boolean loss mask，`z[b,t-1]` 为预测 target `t` 的 logits，整个 logical optimizer update 的有效 token 数为 `N`：

```text
S = - Σ_b Σ_{t=1}^{T-1} M[b,t] log softmax(z[b,t-1])[labels[b,t]]
L = S / N
N = Σ_b Σ_t M[b,t]
```

归约严格按全体有效 token 求均值，不先做 sequence mean，也不让长短样本获得相同权重。mask 外 label 可以是 `ignore_index` 或其他 sentinel；mask 内 label 必须是合法 vocabulary ID。

## 两条生产路径

### `masked_causal_cross_entropy`

输入已经生成的 `[batch, sequence, vocabulary]` logits。实现只 gather mask 选中的 prediction rows，并按 `max_tokens_per_chunk` 分块计算 CE。适用于模型已经返回 logits、需要独立验证 loss 接线的路径。

### `chunked_masked_causal_linear_cross_entropy`

输入 `[batch, sequence, hidden]` hidden states 与 `[vocabulary, hidden]` LM-head weight。实现先选择预测有效 targets 的 hidden rows，再逐块执行 LM-head projection 和 CE，不构造 `[batch, sequence, vocabulary]` 的完整 logits 张量；weight tying 不影响梯度流。需要梯度时，每个 chunk 使用 PyTorch 非重入 activation checkpoint，backward 重算 LM-head/CE，而不是持久保存 vocabulary logits。

`max_tokens_per_chunk=128` 是当前默认值。D02 保证每次 LM-head projection 的 token 维不超过该值，并用额外一次 LM-head 计算换取更小的保存张量；真实模型峰值显存仍取决于 PyTorch backend、Transformer activations 与编译策略，必须在 C5 profile 后才能声称硬件闭合。

## Logical update、gradient accumulation 与 DDP

不传显式归一化参数时，`loss` 是当前调用中有效 token 的均值。若一个 logical update 被拆成多个 microbatches 或 data-parallel shards，调用方必须先从 D01 selection 得到全局 `N`，并对每个 shard 传入：

- `global_token_count=N`；
- `ddp_world_size=W`，单进程 accumulation 时为 1，DDP 时为真实 world size。

每个 shard 反向传播：

```text
L_shard = W × S_shard / N
```

把同一 rank 的 microbatch gradients 相加，再经过默认 DDP gradient averaging，可得：

```text
(1/W) Σ_rank Σ_microbatch ∇(W × S_shard / N) = ∇(Σ S_shard / N)
```

因此调用方不得再按 gradient-accumulation steps 等权除一次。某个 local shard 没有有效 token 时，只有在提供正的全局 `N` 和 `W` 后才允许返回可微零值，以保证所有 rank 参与同一 backward；单独调用的空 mask 会 hard fail。

`loss_sum` 只是本 shard 的未归一化 CE 和，`loss` 是用于 backward 的缩放标量。跨 rank 记录训练 loss 时应 all-reduce `loss_sum` 和 token count 后相除，不能平均各 rank 的 `loss`。

## 精度合同

- float16、bfloat16、float32 logits 默认在 float32 中执行 CE/log-sum-exp 与累加；
- float64 输入默认保留 float64，用于高精度 oracle；
- `loss_compute_dtype` 只允许显式选择 float32 或 float64；
- LM-head projection 使用输入/参数的 forward dtype，随后再把 logits 转为 loss compute dtype；
- hidden、weight、bias 通常必须同 dtype；活动 autocast 下允许 float32 参数与配置的低精度 activation dtype 组合。

这一区分避免把“稳定的 CE reduction”误写成“所有 projection 都强制 FP32”。

## 与 D01 的唯一接线

```text
完整 logical-update objective mask
  → plan_torch_loss_budget(...)
  → 将 selection.loss_mask 随 microbatch/rank 切分
  → 每个 shard 用同一 selection.selected_tokens 作为 global_token_count
  → loss.backward()，执行 optimizer step
  → 仅在 step 成功后 commit_torch_loss_budget(..., True)
```

D02 不自行重算预算，也不根据 label 是否等于 `ignore_index` 静默改变 D01 已冻结的 selection。若 selection 选中了无效 target，D02 直接失败。

## Hard failures

- logits/hidden/LM-head、labels 与 mask 的 rank、shape 或 device 不一致；
- logits/hidden/LM-head 不是支持的浮点 dtype，或 mask 不是 Boolean；
- mask 选择 position 0、`ignore_index`、负 label 或越界 label；
- batch、sequence、vocabulary 或 hidden dimension 为零；
- 未启用兼容 autocast却混用 LM-head forward dtype；
- chunk size、global token count 或 world size 不是正整数；
- 只传 global token count/world size 之一，或全局 count 小于 local count；
- 单独调用时 mask 没有任何可训练 causal target。

## 可复现验证

```bash
uv sync --frozen --all-groups
uv run ruff check .
uv run pytest -q
```

D02 有 23 个 CPU tests，其中包含 100 个 Hypothesis 随机 batch；覆盖独立 log-softmax/gather value oracle、logits/hidden/weight/bias gradients、autograd gradcheck、causal shift、token mean、分块宽度、backward saved-tensor 边界、bfloat16 极值、CPU autocast、不均匀 microbatch、DDP 不均匀 rank 与空 local shard。仓库当前全量共 108 个 tests、575 个生成案例。

验证没有下载模型或数据，没有访问 MPS/CUDA。D02 通过不等于 tracker 的 `C1-001` 已完成：后者仍依赖 C0 的真实 Gemma 4 tokenizer/model/LoRA 集成与 8-sample overfit。

## D02 范围外

- Hugging Face/Gemma 4 forward 与 LoRA target 接线；
- AMP scaler、optimizer step、gradient clipping 与 distributed collectives；
- packed/unpacked logits parity 与 8-sample overfit；
- exact-reward GRPO surrogate、old-policy ratio 与 reward advantage（已由 D03 实现）；
- OPD full-vocabulary reverse-KL（已由 D04 实现）。
