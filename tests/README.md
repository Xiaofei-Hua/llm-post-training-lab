# Planned test suite

实现前必须先列出测试，再进入 GPU run：

1. masked CE、GRPO surrogate、reverse KL 的数值与梯度 oracle；
2. tokenizer/hash、VLM encoder freeze、LoRA target 与 weight-sync assertions；
3. verifier 100–300 条 adversarial cases 与人工一致率；
4. split/dedup/sealed-field access tests；
5. loss-token、prompt exposure、FLOPs 和成本账本守恒测试；
6. paired bootstrap、sign-flip、Holm、TOST 与 pass@k synthetic tests；
7. config/result schema 和不可变 hash tests。
