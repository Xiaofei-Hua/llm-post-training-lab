# Development Module D01 — PyTorch Loss Mask 与精确 Token Budget

## 模块边界

D01 实现 SFT、GRPO、OPD 共享的 loss-position 语义和阶段预算事务，不实现模型 forward、具体 objective 数值、optimizer 或训练循环。

- 张量实现：`src/posttrain_lab/train/torch_loss_budget.py`
- 预算状态：`src/posttrain_lab/train/loss_budget.py`
- 张量测试：`tests/test_torch_loss_budget.py`
- 状态测试：`tests/test_loss_budget.py`
- 环境锁定：`pyproject.toml`、`uv.lock`、`.python-version`

仓库只保留这一套公开实现；不存在旧的标量 mask API 或占位 trainer。

## Canonical `U`

`U` 是 objective mask 为 1，并实际进入一次成功 optimizer update 的 Student loss positions 数。每个 position 每次 update 只计一次。

| Objective | 计入 | 排除 |
|---|---|---|
| SFT | attended assistant targets，含模板 EOS | prompt、`ignore_index`、padding |
| GRPO | 非零 reward-variance group 中实际参与 policy loss 的 completion positions | prompt、纯 rollout、zero-variance groups |
| OPD | 有 exact reverse-KL 的 Student-generated completion 至首个 EOS（含） | prompt、Teacher-only forward、padding、EOS 后 |

failed、overflowed 或被跳过的 update 不增加 `U`。调用方只有在确认 optimizer step 确实执行后，才能以 `optimizer_step_executed=True` 提交 reservation。

## 执行合同

```text
batched objective mask
  → objective-specific filtering
  → plan_torch_loss_budget(...)
  → 使用 selection.loss_mask 计算 loss/backward
  → optimizer step 是否真正执行？
      ├─ yes: commit_torch_loss_budget(..., optimizer_step_executed=True)
      └─ no:  commit_torch_loss_budget(..., optimizer_step_executed=False)，U 不变
```

一次 plan 必须覆盖一次完整的 **logical optimizer update**：

- gradient accumulation：先汇总该 update 的全部 microbatch masks，再统一 plan；
- data parallel：先汇总全局 token metadata，在全局做一次选择，再把 row masks 分发回各 rank；
- 不允许每个 microbatch 或 rank 用独立 counter 近似全局 `U`。

跨 rank 的 collect/scatter 属于后续 trainer integration；D01 已冻结它必须满足的全局接口与一致性条件。

## 精确末批算法

如果候选 positions 超过 remaining `U`，先按 `(sample_id, generation_index)` 稳定排序 row，再按 `token_index` 保留前 `remaining` 个有效 positions。选择在 tensor 上通过 prefix cumsum 完成，不构造逐 token Python 对象。

SFT 的 `generation_index=0`；GRPO/OPD 的同 prompt 多 generation 必须显式编号。同一 logical update 中 row key 重复会 hard fail。

每个 reservation 同时绑定：

- target、consumed、counter version；
- candidate 与 selected token 数；
- canonical row metadata；
- 完整 candidate/selected Boolean mask bytes 的 SHA-256。

因此 batch row 顺序改变不会改变 `selection_id`，但候选集合、mask、预算状态或选择结果任一变化都会改变 ID。旧 reservation 在成功 update、retry 状态变化或 checkpoint 恢复后不能重复提交。

## 公开 API

- `torch_completion_loss_mask`：batched absolute completion start、right padding、first-EOS；
- `torch_assistant_target_loss_mask`：单 assistant completion 的 label/attention/EOS mask；
- `torch_intersect_masks`：严格合取同 shape、同 device 的 binary masks；
- `torch_exclude_zero_variance_grpo_groups`：在 tensor batch 上排除 exact zero-variance groups；
- `plan_torch_loss_budget`：生成确定性 `candidate_mask`、最终 `loss_mask` 与 reservation；
- `commit_torch_loss_budget`：复核 mask、digest 和状态后提交；
- `LossTokenBudget.state_dict/from_state_dict`：带 schema version 的 checkpoint 状态；
- `BudgetStepRecord.to_dict`：可直接写入 JSONL ledger。

## Hard failures

- 输入不是 rank-2 integer/bool tensor，或 mask 含非 0/1 值；
- tensor shape/device 不一致；
- completion attention 在 padding 后重新激活；
- completion start 越界；
- reward 非有限数，或 group ID 非整数/为负；
- 同一 update 的 `(sample_id, generation_index)` 重复；
- returned loss mask 被原地修改或不再是 canonical prefix；
- reservation digest、target/version/consumed state 不一致；
- budget 已完成或没有有效 token，却声称执行了 optimizer step；
- checkpoint schema 不匹配。

## 可复现验证

```bash
uv sync --frozen --all-groups
uv run ruff check .
uv run pytest -q
```

当前环境锁定 Python 3.12、PyTorch 2.14.0、NumPy 2.5.2。27 个 CPU tests 全部通过，其中包含 275 个 Hypothesis 生成案例；测试未访问 MPS/CUDA，也未下载模型或数据。

本机 Apple arm64 CPU 的非门槛诊断中，`[64, 4096]` mask（196,608 candidates、100,000 selected）的 budget plan 约 1.14 ms。该数字只用于防止实现退化，不作为跨硬件性能结论。

## D01 范围外

- masked CE 数值与梯度已由 D02 `MASKED_CAUSAL_CE.md` 实现；
- exact-reward Dr.GRPO surrogate 已由 D03 `GRPO_SURROGATE.md` 实现；
- OPD full-vocabulary reverse-KL 已由 D04 `OPD_REVERSE_KL.md` 实现；
- optimizer/AMP、gradient accumulation 和 distributed trainer 接线；
- 模型/tokenizer/LoRA 集成。

所以 D01–D04 的张量层完成不等于实验 tracker 的 C1/C2/C3 已通过。
