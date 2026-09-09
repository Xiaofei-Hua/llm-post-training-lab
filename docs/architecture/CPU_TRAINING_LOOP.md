# D10：SFT / GRPO / OPD 统一 CPU 训练循环

2026-09-09 完成。实现位于 `src/posttrain_lab/train/loop.py` 与 `rollout.py`，复用 D01 的预算事务、D02 CE、D03 Dr.GRPO、D04 exact reverse-KL 和 D09 模型接口。不是占位 trainer；三种 objective 均执行实际模型 forward、backward 和 AdamW 更新。

## 更新路径

```text
start_stage(objective, loss_tokens, seed)
  └─ 保留 Student 参数，创建新 AdamW / token-based LR scheduler / 设备对应 RNG

step(training_examples)
  ├─ SFT: prompt + reference completion
  └─ GRPO / OPD: 当前 Student eval/no_grad 同步采样
       └─ 保存实际采样分布的 detached old log probabilities
  ↓
score
  ├─ GRPO: exact 0/1 callback → 完整 group advantage → 排除零方差组
  └─ OPD: 冻结 Teacher 在相同 token IDs / prefixes 上 forward
  ↓
D01: 对完整 logical batch 统一选择剩余预算内的有效 loss positions
  ↓
Student features → CE / clipped Dr.GRPO / full-vocab KL(Student || Teacher)
  ↓
finite loss → backward → finite gradient / norm → clip → AdamW step
  ↓
仅成功更新提交 D01 预算；推进 policy version 和 LR scheduler
```

`TrainingExample` 只包含当前训练样本的 ID、prompt 和 reference tokens。reward 由调用方提供；D12 使用合成 token 序列 exact equality。需要数学文本 reward 时可接已有 D05 verifier，不读取 sealed benchmark answers。

## 关键语义

- `sample_completions` 使用与模型同设备的 `torch.Generator`、全词表、temperature 1，无 top-k/top-p、重试或隐式补采。每行生成到首个 EOS（含）或 cap；生成出的 pad ID 是真实 token，只有 attention mask 为零的槽位才是 padding。
- 同一 GRPO prompt 完整采样 `group_size` 个 completion。old log probabilities 在采样当刻保存，不由更新后的 Student 重算；每次更新后都刷新 rollout。当前只做每批一次更新，因此正常的初始 importance ratio 约为 1、clip fraction 为 0，这不是 clipping 未接入。
- reward、advantage、old-policy 概率和 Teacher 均无梯度。OPD 的 Teacher 必须预先冻结、eval，随后由 Trainer 迁移到配置设备；Student/Teacher 消费相同 prefix。D14 再验证真实 tokenizer 与模型特有 logit transforms。
- Dr.GRPO 分母仍是**预算截断前 active completion 数 × 固定 cap**。D01 末批可以只留下一个 group 的部分 token，不能据此缩小分母；这是与独立 clipped objective 对照的测试重点。
- 零方差 group 被排除；整批无有效 token 时记录 skip、实际 rollout 成本和有效组率，不更新参数、不消耗预算、不推进 scheduler。`run_stage` 有 attempt 上限，预算无法完成时抛出 `StageBudgetIncomplete`，不会无限补采。
- 非有限 loss/gradient 时跳过更新、清空 gradient、不计预算，记录明确失败状态。后续有效 attempt 可以继续。optimizer 本身抛异常属于致命执行失败，调用方不能把它当成正常 skip 后继续假定状态完好。
- LR 随已成功提交的阶段 token 比例线性从 objective LR 降到 10%；阶段切换仅在当前预算闭合后进行，重新创建 AdamW moments、scheduler 与 RNG，保留 Student 权重和全局 policy version。
- D11 的默认 rollout head 只投影最后一个有效 hidden state；`RolloutConfig(head_projection="dense")` 保留同条件性能对照。两条路径均无 KV cache。

每个 step 记录 loss、reward、entropy、KL、clip/ratio、有效组率、生成长度/截断率、gradient norm、LR、候选/实际 loss tokens、rollout tokens、阶段累计预算，以及 sample/score/loss/backward/update/total 毫秒。未定义指标使用 `None`，不能把 SFT reference 的正确性当成训练 reward。计时边界间的预算和记录开销包含在 total 中，total 不要求等于分段之和。

## 可执行示例与验证

```bash
uv run --frozen pytest -q tests/test_cpu_training_loop.py
uv run --frozen python scripts/train_cpu_example.py
```

`Trainer.start_stage(...)`、`step(...)`、`run_stage(...)` 由 D12 的同一五臂脚本实际调用。D10/D11 的 15 项集成测试覆盖：

- SFT/GRPO/OPD × text/LoRA 的真实更新和 17-token 精确末批；
- 首个 EOS、sampled pad token、不同 prompt 长度；
- 采样概率与更新前模型一致，更新后 policy version 刷新；
- 完整 GRPO AdamW step 与独立公式对照，包括截断前分母；
- 全 0/全 1 reward 不更新、不扣预算且有界退出；
- NaN gradient skip 后恢复；阶段切换保留权重并清空 moments/scheduler；
- last-position head 与 dense 的采样 token、mask 和 FP64 概率一致。

D13 已将同一同步、单进程 loop 扩展为 CPU/CUDA，接入 BF16、梯度累积、activation checkpointing 与 save/resume，CPU 验证见 `ACCELERATOR_RUNTIME.md`；GPU 实测暂停。distributed collectives 与跨进程 weight sync 尚未实现。这里的成功不能替代真实 Gemma 的 G2/G3/G4。
