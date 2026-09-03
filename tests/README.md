# Test suite

## 已实现：D01

`test_loss_budget.py` 与 `test_torch_loss_budget.py` 使用 PyTorch、pytest 和 Hypothesis 验证 tensor masks 与精确 Student loss-token 预算，共 27 个 CPU tests、275 个生成案例。运行方式：

```bash
uv run pytest -q
```

覆盖 first-EOS、prompt/padding/EOS 后排除、GRPO zero-variance group、末批无 overshoot、批次顺序确定性、完整 mask digest、failed step 不计数、stale/forged reservation 拒绝、checkpoint schema/round-trip 和随机序列预算守恒。所有 tensor 明确在 CPU 上创建，未调用 MPS/CUDA。

## 尚待实现

实现前必须先列出测试，再进入 GPU run：

1. masked CE、GRPO surrogate、reverse KL 的数值与梯度 oracle；
2. tokenizer/hash、VLM encoder freeze、LoRA target 与 weight-sync assertions；
3. verifier 100–300 条 adversarial cases 与人工一致率；
4. split/dedup/sealed-field access tests；
5. loss-token、prompt exposure、FLOPs 和成本账本守恒测试；
6. paired bootstrap、sign-flip、Holm、TOST 与 pass@k synthetic tests；
7. config/result schema 和不可变 hash tests。
