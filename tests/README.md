# Test suite

## 已实现：D01–D08

`test_loss_budget.py` 与 `test_torch_loss_budget.py` 验证 tensor masks 与精确 Student loss-token 预算；`test_masked_ce.py` 验证 causal shift、masked token-mean CE 与 LM-head 分块；`test_grpo_surrogate.py` 验证 exact-reward advantage、PPO clipping 与 Dr.GRPO 固定分母；`test_opd_reverse_kl.py` 验证 full-vocabulary reverse KL、双 LM-head 分块、模型 logit transforms、Teacher 隔离与 global token mean；`test_math_verifier.py` 与 `test_verifier_audit.py` 验证 D05 verifier；D06 的 96 个 tests 验证 data trust stack；D07 的 70 个 tests 验证 sealed generation/evaluation；`test_paired_statistics.py` 与 `test_statistics_audit.py` 的 33 个 tests 验证 D08 D07-report projection、whole-vector stratified bootstrap、known exact sign-flip oracle、outcome-independent streams、Holm、effect/null/seed-instability/sequential TOST、strict result recomputation、bounded resampling memory、Git/TOCTOU 与 raw-text exclusion。全量共 539 个 CPU tests、1,575 个 Hypothesis 生成案例，另执行 257 个冻结 adversarial verifier cases。运行方式：

```bash
uv run pytest -q
```

覆盖 first-EOS、prompt/padding/EOS 后排除、GRPO zero-variance group、末批无 overshoot、批次顺序确定性、完整 mask digest、failed step 不计数、stale/forged reservation 拒绝、checkpoint schema/round-trip、随机序列预算守恒、独立 CE/Dr.GRPO/reverse-KL value 与 gradient oracle、autograd gradcheck、sample std、signed policy gradient、双向 clipping、full-vocabulary KL 方向、不同 Teacher/Student hidden width、scale/softcap/temperature、BF16 极值、分块投影上限、禁止持久保存 vocab logits、不均匀 microbatch/rank 的全局梯度等价；还覆盖 verifier 攻击面、D06 数据 trust stack，以及 sealed/public schema、checkpoint-independent paired seed、backend response bijection、持久化 stop semantics、精确 pass@k、item/sample hash 绑定、report 无原文、invalid gold/failed generation、dirty input、HEAD race 与运行中输入变化。所有测试均在 CPU 上执行，未调用 MPS/CUDA。

## 尚待实现

实现前必须先列出测试，再进入 GPU run：

1. 至少 100 个真实模型输出的 checkpoint-blinded 人工一致率审计；
2. tokenizer/hash、VLM encoder freeze、LoRA target 与 weight-sync assertions；
3. OPD/GRPO rollout freshness 与两步 trainer integration；
4. 真实 MATH/GSM8K/AIME/IFEval/MMLU-Pro adapter 与 checkpoint-blinded answer audit；
5. 真实数据规模的 family/contamination memory profile 与人工 borderline-pair audit；
6. loss-token、prompt exposure、FLOPs 和成本账本守恒测试；
7. 真实 endpoint reports 上的 paired statistics execution 与 claim audit（统计机制已由 D08 synthetic tests 完成）；
8. 跨模块 run config/result/provenance ledger 与不可变 attempt tests。
