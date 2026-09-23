# Test suite

## 已实现：D01–D14 CPU 回归与独立 GPU 验证

`test_loss_budget.py` 与 `test_torch_loss_budget.py` 验证 tensor masks 与精确 Student loss-token 预算；`test_masked_ce.py` 验证 causal shift、masked token-mean CE 与 LM-head 分块；`test_grpo_surrogate.py` 验证 exact-reward advantage、PPO clipping 与 Dr.GRPO 固定分母；`test_opd_reverse_kl.py` 验证 full-vocabulary reverse KL、双 LM-head 分块、模型 logit transforms、Teacher 隔离与 global token mean；`test_math_verifier.py` 与 `test_verifier_audit.py` 验证 D05 verifier；D06 的 96 个 tests 验证 data trust stack；D07 的 70 个 tests 验证 sealed generation/evaluation；`test_paired_statistics.py` 与 `test_statistics_audit.py` 的 33 个 tests 验证 D08 D07-report projection、whole-vector stratified bootstrap、known exact sign-flip oracle、outcome-independent streams、Holm、effect/null/seed-instability/sequential TOST、strict result recomputation、bounded resampling memory、Git/TOCTOU 与 raw-text exclusion。D09 的 `test_model_adapter.py` 验证 tiny causal LM/LoRA 的模型梯度与实际更新。全量共 633 个 CPU tests、1,575 个 Hypothesis 生成案例，另有 257 个冻结 adversarial verifier cases。运行方式：

```bash
uv run pytest -q
```

覆盖 first-EOS、prompt/padding/EOS 后排除、GRPO zero-variance group、末批无 overshoot、批次顺序确定性、完整 mask digest、failed step 不计数、stale/forged reservation 拒绝、checkpoint schema/round-trip、随机序列预算守恒、独立 CE/Dr.GRPO/reverse-KL value 与 gradient oracle、autograd gradcheck、sample std、signed policy gradient、双向 clipping、full-vocabulary KL 方向、不同 Teacher/Student hidden width、scale/softcap/temperature、BF16 极值、分块投影上限、禁止持久保存 vocab logits、不均匀 microbatch/rank 的全局梯度等价；还覆盖 verifier 攻击面、D06 数据 trust stack，以及 sealed/public schema、checkpoint-independent paired seed、backend response bijection、持久化 stop semantics、精确 pass@k、item/sample hash 绑定、report 无原文、invalid gold/failed generation、dirty input、HEAD race 与运行中输入变化。默认 pytest 均在 CPU 上执行；D13/D14 GPU 验收使用单独脚本。

D09 已验证 SFT/OPD × text/LoRA × tied/untied 的 FP64 loss/gradient/两次 SGD 更新与独立 dense reference 一致，也覆盖因果 prefix、左右/全 padding、Teacher 冻结、LoRA A/B 的首步与第二步梯度、state_dict 恢复、不等长 microbatch 全局归一化，以及参数/AdamW 存储估算。FP32 的 12 步合成示例使用 `scripts/smoke_model_adapter.py`，结果范围见 `docs/architecture/MODEL_ADAPTER.md`。

D10/D11 的 `test_cpu_training_loop.py` 覆盖三 objective × text/LoRA 的实际更新、精确末批、策略刷新、零方差组/NaN 梯度跳过、stage reset 与 last-position rollout 等价；`test_cpu_learning.py` 独立核对完整输出概率与评测无训练状态副作用。D12 另实际执行 15 条合成 run、30 个阶段，结果不替代真实模型验收。

`test_training_runtime.py` 的 27 项 CPU 测试对照完整 batch 与累积的梯度/参数更新，覆盖 text/LoRA 重计算、精确随机续训、moments/LR/预算/阶段恢复、NaN/零方差 skip 与 LoRA 配置拒绝。新增用例覆盖后一 microbatch 非有限 logits 的整批清理；旧 GRPO/OPD 路径分别复现失败，修复后三个 objective 通过。D13 CUDA/BF16、精确续训、六组数值失败恢复和 12 组显存/延迟对照已在 RTX 5070 Ti 完成；GPU 累积同时核对 loss、裁剪前梯度范数和裁剪后梯度，AdamW 更新采用独立首步公式；负对照证明会拒绝被 clipping 掩盖的两倍梯度错误。空分片 CE/GRPO/OPD 与 softcap 前溢出已有回归覆盖。见 `docs/architecture/ACCELERATOR_RUNTIME.md`。

`test_gemma4_adapter.py` 使用缩小的真实 Gemma 4 架构，验证 PLE、共享 KV、softcap、原生前向与 SFT/KL/策略梯度、重计算、BF16 基座/FP32 LoRA 更新与冻结，以及 tokenizer 边界。OPD 测试另检查 autocast 不复制整张 FP32 head。真实 E2B/E4B 权重通过独立 D14 脚本运行，范围与状态见 `docs/architecture/GEMMA4_INTEGRATION.md`。

## 尚待实现

以下真实模型/GPU 路径仍需验证：

1. 至少 100 个真实模型输出的 checkpoint-blinded 人工一致率审计；
2. 后续 rollout 引擎的 weight-sync assertions；D14 tokenizer/freeze/LoRA 接入使用现有检查；
3. 真实 Gemma OPD/GRPO 的 rollout freshness、weight sync 与两步 trainer integration；
4. 真实 MATH/GSM8K/AIME/IFEval/MMLU-Pro adapter 与 checkpoint-blinded answer audit；
5. 真实数据规模的 family/contamination memory profile 与人工 borderline-pair audit；
6. loss-token、prompt exposure、FLOPs 和成本账本守恒测试；
7. 真实 endpoint reports 上的 paired statistics execution 与 claim audit（统计机制已由 D08 synthetic tests 完成）；
8. 真实训练暴露的 checkpoint/config 恢复问题按需复用 D13 检查。
