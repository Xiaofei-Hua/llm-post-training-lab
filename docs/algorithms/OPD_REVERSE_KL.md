# Development Module D04 — Full-Vocabulary Reverse-KL for OPD

## 模块边界

D04 实现 same-tokenizer Student/Teacher 在 Student-generated prefix 上的 full-vocabulary reverse KL 数值核心。它负责 causal target-position 对齐、完整词表 divergence、selected-position LM-head 分块、模型 logit transform、低精度稳定计算，以及 gradient accumulation/DDP 的 global token-mean 缩放。

本模块不实现 Student rollout、Teacher/model 加载、tokenizer compatibility、optimizer、distributed collective、vLLM 或 GPU 训练。

## Deletion test：为什么 D02/D03 不足

D02 masked CE 在每个位置只读取一个 hard target `y_t`：

```text
L_CE,t = -log p_S(y_t)
```

D03 则只给实际 sampled token 乘 sequence-level reward advantage。两者都看不到 Teacher 在其余 `V-1` 个 token 上如何分配概率。存在两组 Teacher 分布，它们给 sampled token 相同概率、但在剩余词表上的质量分配不同；D02/D03 对它们产生相同信号，而 full-vocabulary reverse KL：

```text
KL(p_S || p_T) = Σ_v p_S(v)[log p_S(v)-log p_T(v)]
```

会给出不同 loss 和 gradient。因此删除 D04 会失去 OPD 的稠密 Teacher distribution signal，不能用现有 CE 或 GRPO primitive 等价替代。

## 冻结测试清单（全部通过）

1. causal target mask shift 与全 batch token mean 对齐独立公式；
2. 100 个生成 batch 的 full-vocabulary value oracle；
3. Student-logit gradient oracle与 autograd gradcheck；
4. 相同分布为零、加性 logit 常数不变、reverse/forward KL 方向不混淆；
5. Teacher 所有输入必须 stop-gradient，Student hidden/head/bias 保留梯度；
6. Student/Teacher hidden width 可不同，但 vocabulary 必须完全相同；
7. chunked linear path 与 materialized full-logit reference 的 value/gradient 对齐；
8. 每次 projection 不超过冻结 chunk size，backward 不保存 `[tokens, vocab]` logits；
9. logit scale、Gemma-style final softcap、distillation temperature 的顺序与 reference 对齐；
10. BF16 输入使用 FP32 divergence，极端有限 logits 不产生 NaN/Inf；
11. 不均匀 microbatch、DDP rank 与空 local shard 重建同一个 global token mean；
12. D01 completion-through-EOS mask 和精确末批 budget 可直接接入 D04；
13. 非法 shape/dtype/device、position 0、normalizer、transform 和 selected non-finite 值全部 hard fail；unselected sentinel 不污染 loss。

## 冻结数学合同

对 D01 mask 选中的绝对 target position `t`，Student 与 Teacher 都读取同一 prefix 对应的 prediction position `t-1`。两个模型可有不同 hidden width，但必须使用完全相同的 vocabulary/token IDs。

在可选的模型级变换后计算概率：

```text
softcap_c(z) = c tanh(z/c)
z'_S = softcap_student(scale_student z_S) / T
z'_T = softcap_teacher(scale_teacher z_T) / T
p_S = softmax(z'_S),  p_T = softmax(z'_T)
d_t = Σ_v p_S(v)[log p_S(v)-log p_T(v)]
L = (1/N_global) Σ_t M_t d_t
```

变换顺序固定为 model logit scale → model final softcap → distillation temperature；某模型没有对应变换时使用 scale 1 / softcap `None`。主实验温度冻结为 `T=1.0`，不额外乘经典 KD 的 `T²`。每个 token 先对完整 vocabulary 求和，再对全局 selected completion tokens 取均值，禁止 sequence-equal averaging。

如果调用方传入的是模型已经完成 scale/softcap 的最终 logits，必须保持 API 的 scale 1 / softcap `None`，避免二次变换；绕过模型输出层、直接从 hidden states 做分块 projection 时，adapter 必须从各自模型 config 显式传入变换参数。

## 公开 API

实现位于 `src/posttrain_lab/train/opd_reverse_kl.py`，并从 `posttrain_lab.train` 导出：

- `masked_causal_reverse_kl(...) -> MaskedReverseKLOutput`：materialized-logit reference/fallback；只 gather 被 mask 选中的 prediction rows，并按 token chunk 计算完整词表 KL；
- `chunked_masked_causal_linear_reverse_kl(...) -> MaskedReverseKLOutput`：主内存路径；分别接收 Student/Teacher hidden states、LM-head weight/bias 和模型级 logit transform；
- 输出包含 raw local `loss_sum`、detached `student_entropy_sum`、local/global token 计数、chunk size、temperature 与 DDP normalization 元数据。

production linear path 每次最多产生两块 `[max_tokens_per_chunk, vocabulary]` logits，默认 chunk size 为 128；Student chunk 使用 activation checkpoint，在 backward 时重算而不持久保存 vocabulary logits。Teacher projection 始终在 `no_grad` 中执行。该设计避免 `[batch, sequence, vocabulary]` 全量 logits 常驻，但真实 Gemma 4 峰值内存仍须由 C5 profile 验证。

## Teacher 与 on-policy 边界

- Teacher logits、hidden states、LM-head weight/bias 必须全部 stop-gradient；任何一个 `requires_grad=True` 都 hard fail；
- Student hidden/head/bias 保留完整梯度，student entropy 只作 detached 诊断；
- Student-generated token IDs、rollout freshness、Teacher `eval()`、tokenizer hash 和 vocabulary ID 对齐属于 D09/D10/C3/G4，不由这个纯数值函数伪造；
- full vocabulary 是主合同；top-k/residual-mass、cross-tokenizer KD、JSD 和 forward KL 不在 D04 API 中。

## Gradient accumulation 与 DDP

设 global logical update 有 `N_global` 个 D01-selected OPD tokens，world size 为 `W`，rank `r` 的 microbatch `m` 上 raw KL sum 为 `S_r,m`。D04 返回：

```text
L_r,m = W × S_r,m / N_global
```

直接累加 microbatch gradients，再由 default DDP 对 rank 求平均，得到精确 global token mean。调用方不得再次除以 accumulation steps。D04 不执行 collective；`N_global` 必须由所有 rank 共享。空 local shard 只有在显式提供 global count/world size 时合法，并返回连接所有 Student trainable tensors 的有限可微零值。

## 数值与失败语义

- FP16/BF16/FP32 输入默认用 FP32 projection、log-softmax、KL 与累加；任一相关输入为 FP64 时默认用 FP64；显式 override 只允许 FP32/FP64；
- selected raw/transformed logits、per-token KL 与 entropy 必须有限；unselected sentinel 不参与 projection 或检查；
- KL 不做静默 clamp，避免改变 reverse-KL 梯度；极小 Teacher probability 通过有限 `log_softmax` 保留惩罚；
- mask 必须是同 device 的 rank-2 Boolean tensor，position 0 永远不可选；
- Student/Teacher batch、sequence、vocabulary 必须一致；各自 LM head input width 必须匹配各自 hidden width；
- 非正/非有限 temperature、scale、softcap，非法 chunk size、缺一半的 distributed normalization metadata，以及 global token count 小于 local count都会失败。

## 可复现验证

```bash
uv sync --frozen --all-groups
uv run pytest -q tests/test_opd_reverse_kl.py
uv run pytest -q
```

D04 有 37 个 CPU tests，其中包含 100 个 Hypothesis 生成 batch；覆盖 full-vocabulary value/gradient oracle、autograd gradcheck、reverse/forward 方向、零点与 logit-shift 不变性、causal shift、不同 hidden width、Teacher 隔离、scale/softcap/temperature、chunk projection 与 saved-tensor 边界、BF16/FP32、不均匀 accumulation/DDP、空 rank、D01→D04 接线及 hard failures。仓库当前全量共 108 个 tests、575 个生成案例。

验证没有下载模型或数据，没有访问 MPS/CUDA。D04 通过不等于 C3 或 G4 已完成：真实 E2B/E4B tokenizer/vocabulary、model-output parity、Teacher freeze、on-policy refresh 和 full-size memory profile 尚未运行。

## 上游语义审计

- [GKD / On-Policy Distillation 论文](https://arxiv.org/abs/2306.13649)：核对 Student-generated sequence、token-level reverse KL 与 sampling stop-gradient；
- [TRL DistillationTrainer 文档](https://huggingface.co/docs/trl/main/distillation_trainer)：核对 full-distribution、selected-position chunking 与 scale/softcap/temperature 顺序；
- [TRL GKD 固定源码版本](https://github.com/huggingface/trl/blob/4506f4af8ccc5b7f8b337f589a696a19913d8a4a/trl/experimental/gkd/gkd_trainer.py)：核对 `beta=1` 的 reverse-KL 方向、causal shift 和 global valid-token normalization。

仓库不引入 TRL 运行时依赖。上述来源仅作为语义审计基线；D04 独立实现项目冻结的纯 reverse-KL API，不保留旧 GKD/JSD 兼容层。
