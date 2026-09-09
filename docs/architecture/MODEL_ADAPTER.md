# D09：CPU 模型适配与可训练参数接入

2026-09-09 完成。D02/D04 已能从独立张量计算正确的 CE/KL，但还不能证明梯度穿过模型、到达所选参数并产生更新。D09 用本地随机初始化的 tiny causal LM 接通这条路径，并对照完整 logits 计算。没有引入新研究算法。

## 实现与取舍

`src/posttrain_lab/models/causal_lm.py` 提供 `CausalLMAdapter`、`CausalLMFeatures` 和 `TinyCausalLM`。模型是 learned positional embeddings、pre-norm causal attention、gated MLP 和 final LayerNorm 组成的文本 decoder；支持 FP32/FP64 CPU、左右 padding、全 padding 行与可选 embedding/head 权重共享。初始化使用局部 CPU RNG 状态，不下载 checkpoint，不创建 tokenizer。

```text
input IDs + Boolean attention mask
              ↓
token embedding + learned positions
              ↓
[LayerNorm → causal q/k/v/o attention → residual
 LayerNorm → gate/up/down MLP       → residual] × layers
              ↓
final LayerNorm → hidden states [B, T, H]
              ├─ dense reference: LM head → [B, T, V]
              └─ D02/D04: 只选有效 target 的前一位置 → 分块 LM head → loss

text 模式：整个文本模型可训练；共享 embedding/head 是同一个 Parameter
lora 模式：仅 attention/MLP 的 A/B 可训练；其余参数全部冻结
Teacher：全部冻结 + eval + no_grad forward
```

模型输出保留未 shift 的 hidden states 和原始 LM-head 参数引用，D02/D04 统一执行 `target[t] ← hidden[t-1]`。`src/posttrain_lab/train/model_losses.py` 的 `causal_lm_sft_loss` 与 `causal_lm_opd_loss` 检查目标及其前一位置均非 padding，传递 global token normalizer，再调用原有分块/recompute kernel。训练入口不调用完整 logits 路径；prompt 与 EOS 的 objective mask 仍由调用方通过 D01 构造。

padding key 对有效 query 不可见；左侧或全 padding query 使用一个 dummy self key，避免全 `-inf` softmax 产生 NaN，最终将 padding hidden state 清零。位置编号按有效 token 累加，因此左右 padding 不改变同一有效序列的输出。

`src/posttrain_lab/models/parameters.py` 实现 `W(x) + (alpha/r) B A x`，B 初始化为零，保持接入前后的初始输出完全一致。默认 targets 为每层 `q/k/v/o_proj` 与 `gate/up/down_proj`，拒绝 LM head 或未知 targets；不合并 BA，也不改变线性 LM-head 接口。配置需在 CPU 上、迁移设备和创建 optimizer 前完成；恢复时先构造同配置模型及 LoRA，再由 `Trainer.load_checkpoint` 恢复权重与训练状态。D13 扩展了设备与重计算路径，GPU 验证尚待执行。

零初始化 B 意味着第一步 `∂L/∂A = 0`，B 更新后第二步 A 才开始接收非零梯度。这是初始化的预期行为；测试同时检查 A/B 的两步更新，避免将第一步 A 梯度为零误判为断图。

## 可复跑的 CPU 实测

```bash
uv run --frozen python scripts/smoke_model_adapter.py --dry-run
uv run --frozen python scripts/smoke_model_adapter.py \
  --output artifacts/cpu/d09_model_adapter.json
uv run --frozen pytest -q tests/test_model_adapter.py
```

完整配置、合成 token IDs、逐步 loss、梯度范数、误差和参数记录见 [D09 JSON](../../artifacts/cpu/d09_model_adapter.json)，生成逻辑在 `src/posttrain_lab/models/smoke.py`。这是固定 prefix 的模型接入 smoke，未执行 on-policy rollout 或正式预算账本。

- 环境：Darwin arm64，Python 3.12.13，PyTorch 2.14.0，CPU 单线程，FP32。
- Student：vocab 32、hidden 32、MLP 64、2 layers、4 heads、共享 embedding/head。
- Teacher：vocab 32、hidden 48、MLP 96、2 layers、4 heads；独立随机初始化并冻结。
- batch 4、物理长度 8、有效长度 8/7/6/5、prompt 长度 2，每次更新 18 个 loss tokens。
- 每组执行 12 次 SGD 更新，lr 0.1；LoRA rank 4、alpha 8、dropout 0。Student/Teacher/adapter/data seeds 为 7/13/17/11。
- 对照：从相同参数出发，逐步比较完整 logits 上独立公式与 D02/D04 的 chunk size 3 路径。预设 FP32 容差 `rtol=1e-4, atol=2e-6`。

| Objective / 参数模式 | 更新前 loss | 12 次更新后 loss | 最大梯度绝对误差 | 最大更新后参数绝对误差 |
|---|---:|---:|---:|---:|
| SFT / text | 3.446976 | 1.913119 | 2.8313e-7 | 1.1921e-7 |
| SFT / LoRA | 3.446976 | 3.343540 | 2.2352e-8 | 7.4506e-9 |
| reverse-KL / text | 0.01707029 | 0.00654724 | 1.0478e-8 | 1.4902e-8 |
| reverse-KL / LoRA | 0.01707029 | 0.01673701 | 2.3866e-9 | 1.1642e-9 |

text 模式实际改变 26 个参数 tensor，LoRA 模式实际改变 28 个 A/B tensor；所有 frozen 参数和 Teacher 都未改变且无梯度。两组 LoRA 的第一步 A 梯度范数均为 0，第二步分别为 0.0164281（SFT）和 0.0000539690（KL）。

四组损失下降说明模型接入后能够沿当前目标更新参数。LoRA 在此固定形状、相同更新次数下下降较慢；该示例没有调参，也没有能力评测，不能据此判断正式模型的质量或两种训练信号的优劣。随机 Teacher 只用于验证分布拟合，不提供推理能力证据。

## 参数量与内存估算

| Student 模式 | 总参数 | 可训练参数 | 可训练比例 | 参数字节 | 参数 + 梯度 + AdamW 两个 moment 字节 |
|---|---:|---:|---:|---:|---:|
| text | 22,848 | 22,848 | 100% | 91,392 | 365,568 |
| LoRA | 27,200 | 4,352 | 16% | 108,800 | 161,024 |

按唯一 Parameter 计数，embedding/head 共享只计一次。FP32 参数字节为 `4 × N_total`，梯度字节为 `4 × N_trainable`，同 dtype AdamW 两个 moment 为 `8 × N_trainable`。LoRA 多存 A/B 参数，但减少梯度与 optimizer state；其 16% 比例由 tiny 模型尺寸决定，不代表 Gemma 的参数比例。

这是解析存储估算，**不是峰值 RSS、吞吐或加速结果**。表中不含 activation、attention score、分块 logits、optimizer 临时张量/step counter、运行时或 master weights。实际 smoke 用 SGD，因此 AdamW moments 是假设切换 optimizer 后的存储项；测试另执行一次 AdamW，核对估算与实际 gradient/moment tensor 字节一致。OPD 还需加上冻结 Teacher 的 49,632 参数、198,528 字节和未在表中估算的 forward activation。峰值内存和时间测量留给 D11。

## 验证范围与后续

`tests/test_model_adapter.py` 覆盖因果性、左右/全 padding、RNG 恢复、LoRA 初始等价与 target 选择、SFT/OPD × text/LoRA × tied/untied 的 FP64 loss/gradient/两步 SGD 对照、Teacher 冻结、避免模型完整 logits、不等长 microbatch 全局归一化、state_dict 恢复与参数存储核对。

D09 的接口当前仅承诺无额外 logit transform 的线性 LM head，tiny attention 为便于检查的 dense 实现。Gemma 的 hybrid attention、模型特有 scale/softcap、PLE、tokenizer 与多模态 freeze/LoRA introspection 属于 D14；不能直接把此 tiny fixture 当作 Gemma adapter。

后续 **D10：SFT/GRPO/OPD 统一训练循环** 已完成，见 `CPU_TRAINING_LOOP.md`。复用 D09 模型、D01 预算和 D02–D05 算法，接通当前 Student 采样、reward/Teacher、old-policy 快照与刷新、loss/backward/update、阶段 optimizer/scheduler reset，并输出 loss/reward/entropy/KL/clip/有效组率。D09 已完成模型参数路径；D12 的合成五臂学习已验收，见 `../experiments/CPU_LEARNING_EXPERIMENT.md`。
