# Package boundaries

当前只在 CPU 上逐模块实现算法与框架；模型下载、训练栈集成和 GPU run 仍受 gate 约束：

- `data/`：registry、schema、quality funnel、family split、contamination audit；
- `rewards/`：answer parser、exact/symbolic verifier、attack corpus；
- `train/`：SFT、GRPO、OPD adapters 与 token/cost accounting；
- `evaluation/`：frozen generation/evaluator、paired statistics；
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

当前没有保留旧标量 API、旧 CE/GRPO/KD/reward/data 路径或占位 trainer。D01–D06 是后续 runtime 可调用的 production contract；rollout/old-policy/Teacher 生命周期、optimizer/AMP orchestration、distributed collectives、真实模型 forward、LoRA 集成与 sealed evaluator 仍由后续模块负责。D05 只冻结数学 reward 语义；D06 只证明 synthetic data trust mechanism，不冒充真实数据已去污染或 G1 已通过。
