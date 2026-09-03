# CPU Framework Development Modules

## 口径

核心 CPU 算法/框架开发固定为 **12 个模块（D01–D12）**。它与 `LEARNING_CURRICULUM.md` 的 12 个知识模块并行，但编号含义不同：学习模块回答“要掌握什么”，D 模块回答“仓库要交付什么”。GPU smoke、正式训练和结果分析属于后续 execution phase，不计入这 12 个开发模块。

## 冻结清单

| ID | 交付物 | 当前状态 |
|---|---|---|
| D01 | loss-position mask 与精确 Student backward token budget | COMPLETE（CPU） |
| D02 | production masked causal cross-entropy | COMPLETE（CPU） |
| D03 | exact-reward Dr.GRPO advantage 与 clipped surrogate | COMPLETE（CPU） |
| D04 | OPD full-vocabulary chunked reverse-KL | COMPLETE（CPU） |
| D05 | exact/symbolic answer parser、verifier 与 adversarial reward audit | PLANNED |
| D06 | 数据 registry、license/revision lineage、family split 与 contamination | PLANNED |
| D07 | sealed benchmark evaluator、generation/result schema 与 metric contracts | PLANNED |
| D08 | paired bootstrap、sign-flip、Holm、TOST 与 pass@k 统计核心 | PLANNED |
| D09 | tokenizer/vocabulary/hash、text-only freeze 与 LoRA target 模型合同 | PLANNED |
| D10 | SFT/GRPO/OPD objective adapters、rollout/old-policy/Teacher freshness 与 RNG 生命周期 | PLANNED |
| D11 | run/config/checkpoint provenance、双预算与成本 ledger | PLANNED |
| D12 | CPU end-to-end preflight、resume/failure semantics 与 stage-gate orchestrator | PLANNED |

当前完成 4/12；下一候选为 D05，但尚未开始。

## 边界规则

- 每轮只允许一个 `IN PROGRESS` 模块；完成、验证并 commit 后才移动到下一项。
- D01–D12 只保证算法与框架在 CPU 可验证，不替代 C0–C6、G0–G6 的真实模型和算力 gate。
- DPO、ORPO/KTO、GSPO/TIS、PRM、跨 tokenizer distillation 和 top-k KL 都是核心完成后的扩展，不计入 D01–D12。
- 若发现必须增加模块，先做 deletion test 并修改本清单；不能用临时功能悄悄扩大当前模块。
