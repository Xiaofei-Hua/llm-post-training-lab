# Test suite

## 已实现：D01–D02

`test_loss_budget.py` 与 `test_torch_loss_budget.py` 验证 tensor masks 与精确 Student loss-token 预算；`test_masked_ce.py` 验证 causal shift、masked token-mean CE、LM-head 分块/重计算路径、低精度、gradient accumulation 与 DDP normalization。全量共 50 个 CPU tests、375 个 Hypothesis 生成案例。运行方式：

```bash
uv run pytest -q
```

覆盖 first-EOS、prompt/padding/EOS 后排除、GRPO zero-variance group、末批无 overshoot、批次顺序确定性、完整 mask digest、failed step 不计数、stale/forged reservation 拒绝、checkpoint schema/round-trip、随机序列预算守恒、独立 CE value/gradient oracle、gradcheck、BF16 极值、autocast、分块投影上限、禁止持久保存 vocab logits，以及不均匀 microbatch/rank 的全局梯度等价。所有测试均在 CPU 上执行，未调用 MPS/CUDA。

## 尚待实现

实现前必须先列出测试，再进入 GPU run：

1. GRPO surrogate 与 reverse KL 的数值、极限及梯度 oracle；
2. tokenizer/hash、VLM encoder freeze、LoRA target 与 weight-sync assertions；
3. verifier 100–300 条 adversarial cases 与人工一致率；
4. split/dedup/sealed-field access tests；
5. loss-token、prompt exposure、FLOPs 和成本账本守恒测试；
6. paired bootstrap、sign-flip、Holm、TOST 与 pass@k synthetic tests；
7. config/result schema 和不可变 hash tests。
