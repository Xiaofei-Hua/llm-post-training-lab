# Package boundaries

当前只在 CPU 上逐模块实现算法与框架；模型下载、训练栈集成和 GPU run 仍受 gate 约束：

- `data/`：registry、schema、quality funnel、family split、contamination audit；
- `rewards/`：answer parser、exact/symbolic verifier、attack corpus；
- `train/`：SFT、GRPO、OPD adapters 与 token/cost accounting；
- `evaluation/`：frozen generation/evaluator；
- `statistics/`：D07 correctness projection、paired item inference 与 audit；
- `analysis/`：claim tables、Pareto、error transitions。

训练器不得读取 sealed evaluator answers；Teacher 接口不得返回给数据构建流程。

## 已实现

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

当前没有保留旧标量 API、旧 CE/GRPO/KD/reward/data/evaluator/statistics 路径或占位 trainer。D01–D08 是后续 runtime 可调用的 production contracts；rollout/old-policy/Teacher 生命周期、optimizer/AMP orchestration、distributed collectives、真实模型 forward、LoRA 集成与真实 benchmark adapters 仍由后续模块负责。D05–D08 只冻结 CPU algorithms/trust mechanisms，不冒充真实数据已去污染、真实模型已评测、真实 C1/C2 已完成或 G1/G5/G6 已通过。
