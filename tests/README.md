# Test suite

## 已实现：D01–D05

`test_loss_budget.py` 与 `test_torch_loss_budget.py` 验证 tensor masks 与精确 Student loss-token 预算；`test_masked_ce.py` 验证 causal shift、masked token-mean CE 与 LM-head 分块；`test_grpo_surrogate.py` 验证 exact-reward advantage、PPO clipping 与 Dr.GRPO 固定分母；`test_opd_reverse_kl.py` 验证 full-vocabulary reverse KL、双 LM-head 分块、模型 logit transforms、Teacher 隔离与 global token mean；`test_math_verifier.py` 与 `test_verifier_audit.py` 验证 D05 terminal extraction、normalization/juxtaposition cross-check、符号等价、结构类型、失败语义、provenance 与 audit schema。全量共 340 个 CPU tests、875 个 Hypothesis 生成案例，另执行 257 个冻结 adversarial verifier cases。运行方式：

```bash
uv run pytest -q
```

覆盖 first-EOS、prompt/padding/EOS 后排除、GRPO zero-variance group、末批无 overshoot、批次顺序确定性、完整 mask digest、failed step 不计数、stale/forged reservation 拒绝、checkpoint schema/round-trip、随机序列预算守恒、独立 CE/Dr.GRPO/reverse-KL value 与 gradient oracle、autograd gradcheck、sample std、signed policy gradient、双向 clipping、full-vocabulary KL 方向、不同 Teacher/Student hidden width、scale/softcap/temperature、BF16 极值、分块投影上限、禁止持久保存 vocab logits、不均匀 microbatch/rank 的全局梯度等价，以及多答案 precedence、malformed/nested-container fail-closed、trailing-math hijack、unsafe equality、unanchored prose、injection/non-finite、跨结构 false positive、invalid-reference whole-batch preflight、success-only cache、timeout/thread 边界与 Git/source/lock provenance。所有测试均在 CPU 上执行，未调用 MPS/CUDA。

## 尚待实现

实现前必须先列出测试，再进入 GPU run：

1. 至少 100 个真实模型输出的 checkpoint-blinded 人工一致率审计；
2. tokenizer/hash、VLM encoder freeze、LoRA target 与 weight-sync assertions；
3. OPD/GRPO rollout freshness 与两步 trainer integration；
4. split/dedup/sealed-field access tests；
5. loss-token、prompt exposure、FLOPs 和成本账本守恒测试；
6. paired bootstrap、sign-flip、Holm、TOST 与 pass@k synthetic tests；
7. config/result schema 和不可变 hash tests。
