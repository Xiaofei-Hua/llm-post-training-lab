# Package boundaries

当前已完成 D01–D12 的 CPU 算法与框架；真实模型下载、accelerator 训练栈集成和 GPU run 仍受 gate 约束：

- `data/`：registry、schema、quality funnel、family split、contamination audit；
- `rewards/`：answer parser、exact/symbolic verifier、attack corpus；
- `models/`：hidden-state/LM-head adapter、CPU tiny causal LM、text/LoRA 参数选择；
- `train/`：SFT、GRPO、OPD adapters 与 token/cost accounting；
- `evaluation/`：frozen generation/evaluator；
- `statistics/`：D07 correctness projection、paired item inference 与 audit；
- `performance/`：独立进程 CPU CE/KL/完整 step 基准与 profiler；
- `experiments/`：合成 copy-bit 五臂学习及错误 Teacher 对照；
- `analysis/`：claim tables、Pareto、error transitions。

2026-09-09，D09–D12 模型/LoRA、统一训练循环、CPU 性能与合成学习已完成。D13 的共享运行时和 CPU 测试已实现，GPU 实测因其他任务占用而暂停。上表包含规划边界，`analysis/` 等尚未实现；已有数据/评测/统计支持按需复用，不继续扩张通用校验系统。

训练器不得读取 sealed evaluator answers；Teacher 接口不得返回给数据构建流程。

## 已实现

- `train/rollout.py`：D10 当前策略 CPU 采样、首个 EOS/cap、采样时 old log probabilities；D11 支持 dense/最后位置 head 对照。
- `train/loop.py`：D10 统一三 objective 循环、零方差/非有限梯度 skip、D01 精确预算与 stage optimizer/scheduler reset。
- `performance/cpu.py`：D11 两档 CE/KL、完整 OPD step、独立 RSS 和 CPU profiler；数值 oracle 单独进程执行。
- `experiments/cpu_learning.py`：D12 copy-bit 五臂两阶段三 CPU paired seeds、held-out context 评测、错误 Teacher 对照与 Matplotlib 图表。
- `models/causal_lm.py`：D09 的 CPU tiny causal decoder、padding/causal attention、共享 embedding/head 与不物化完整 logits 的 features 接口。
- `models/parameters.py`：D09 的 text/LoRA 参数选择、A/B 投影、Teacher freeze 与去重后的参数/gradient/AdamW 存储估算。
- `models/smoke.py`：D09 四组固定合成 prefix 的 CE/KL 模型更新与 dense reference 对照；由 `scripts/smoke_model_adapter.py` 调用。
- `train/model_losses.py`：D09 的模型→D02/D04 接入，检查 padding target、冻结 Teacher 并传递全局归一化。
- `train/torch_loss_budget.py`：D01 的 batched PyTorch masks、GRPO zero-variance group 过滤、tensor prefix 截断与 selection digest；
- `train/loss_budget.py`：schema-versioned reservation/counter、失败 step 语义、checkpoint state 与 update ledger。
- `train/masked_ce.py`：D02 的 causal target shift、masked token-mean CE、有效位置 LM-head 分块投影与 global logical-update normalization。
- `train/grpo_surrogate.py`：D03 的 exact-reward group advantage、zero-variance active-group 语义、token importance ratio、PPO clipping 与 Dr.GRPO global normalization。
- `train/opd_reverse_kl.py`：D04 的 full-vocabulary reverse KL、双 LM-head selected-position 分块、模型 logit transforms 与 global token-mean normalization。
- `rewards/verifier.py`：D05 的 top-level terminal-answer extraction、Markdown/normalization/juxtaposition validation、pinned Math-Verify symbolic backend、strict structural/assignment guard、gold-first fail-fast、0/1 reward 与 structured failure semantics。
- `rewards/audit.py`：D05 的 immutable JSONL attack corpus loader、policy/corpus/source/lock/Git provenance 与 deterministic audit report。
- `data/registry.py`：D06 的 source/transform registry、strict canonical records、payload-addressed parent ledger、三维 family split 与 immutable manifest schema。
- `data/contamination.py`：D06 的全上下文字段/聚合 normalization、complete inverted n-gram candidate retrieval、exact/fuzzy/review 判定、传递 family quarantine 与内部 clean rescan。
- `data/audit.py`：D06 的 frozen expectation、单一 Git revision provenance、split/contamination/manifest 重算与 raw-text-free audit report。
- `evaluation/contracts.py`：D07 的 public/sealed snapshot、frozen greedy/sampling、paired seeds、完整 generation records 与 immutable bundle/manifest。
- `evaluation/metrics.py`：D07 的逐 sample/item score、精确 accuracy/pass@k/extraction/parse/completion-length/truncation 聚合与 evaluator/report hashes。
- `evaluation/runner.py`：D07 的 generator/evaluator capability 分离、乱序 response canonicalization、D05 exact-math 与最小 strict-label adapter。
- `evaluation/audit.py`：D07 的 frozen synthetic oracle、实际 runtime source origin、loader-consumed bytes/fixture/Git provenance、TOCTOU/HEAD-race 防护与 raw-text-free audit report。
- `statistics/contracts.py`：D08 的 exact rational/interval protocol、A0–A4 × 三 seed text-free panel，以及 D07 greedy report/public-strata 一致性投影。
- `statistics/inference.py`：D08 的 whole-vector stratified bootstrap、outcome-independent PCG64 sign-flip、exact Holm、C1/C2 decision、strict result loader 与 deterministic recomputation validation。
- `statistics/audit.py`：D08 的 frozen 10k/100k synthetic oracle、runtime/input/Git provenance 与 raw-text-free audit artifact。

当前没有保留旧标量 API、旧 CE/GRPO/KD/reward/data/evaluator/statistics 路径或占位 trainer。D01–D12 已具备 CPU 模型、目标函数、同步训练循环与合成学习验证。D13 已实现 BF16、累积/重计算与实际训练 save/resume，完成 CPU 验证；GPU 实测暂停。distributed collectives、真实 Gemma/LoRA 与 benchmark adapters 仍待后续实现。CPU 证据不代表真实数据已去污染、真实模型已评测、真实 C1/C2 已完成或 G1/G5/G6 已通过。
