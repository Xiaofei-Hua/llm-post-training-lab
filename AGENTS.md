# Project Instructions

## Pipeline Status

- language: zh
- phase: cpu-framework-development
- project_type: algorithm-first LLM post-training portfolio
- execution_status: CPU-only algorithm/framework development; no GPU training authorized
- core_modules: D01-D24 (5/24 complete); GPU/full-training begins at D13
- extension_modules: X01-X08 (deferred until D24; excluded from core completion)

## Problem Anchor

在有限算力和完全公开、可追溯的数据条件下，从同一前沿学生模型的 SFT checkpoint 出发，受控比较不同后训练学习信号的单独作用和顺序交互，并把收益与模型、数据、训练预算、评测及计算成本混杂区分开。

当前 Method Thesis 将该通用问题实例化为 Gemma 4 E2B Student 上的 exact-reward GRPO 与 same-lineage E4B Teacher OPD；DPO 只作核心完成后的 shadow 学习项。

## Working Rules

1. 所有实验必须对应一个预注册 claim；没有决策价值的实验不运行。
2. 首期固定模型架构、数据划分和解码设置；五臂统一两阶段，每阶段严格匹配 Student backward loss tokens。
3. 公开 benchmark 只用于评测；必须先做去污染并冻结样本哈希。
4. 所有数据必须记录来源、license、revision、处理脚本和校验和。
5. 所有结果必须记录 Git commit、配置哈希、随机种子、模型与数据 revision。
6. 不得把内部文档正文、业务数据、指标、模型或同事信息提交到仓库；内部阅读笔记放在被忽略的 `notes/private/`。
7. 不声称尚未运行的结果；规划值、预期方向和实测值必须明确区分。
8. 新增算法前先完成 deletion test：证明现有最小方案不足。
9. 每轮只开发一个已声明模块；当前用户未授权 MPS/CUDA 或 GPU 训练。
10. Python 环境只通过 `uv.lock` 复现；替换实现时删除失效 API、测试与说明，不保留废弃副本。

## Canonical Documents

- 总览：`README.md`
- 项目契约：`docs/planning/PROJECT_CHARTER.md`
- 最终研究方案：`refine-logs/FINAL_PROPOSAL.md`
- 实验计划：`refine-logs/EXPERIMENT_PLAN.md`
- 实验追踪：`refine-logs/EXPERIMENT_TRACKER.md`
- 数据计划：`docs/data/DATA_PLAN.md`
- 评测协议：`docs/evaluation/BENCHMARK_PLAN.md`
- 完整端到端模块总表：`docs/planning/DEVELOPMENT_MODULES.md`
- 当前实现 D01：`docs/algorithms/LOSS_TOKEN_BUDGET.md`
- 当前实现 D02：`docs/algorithms/MASKED_CAUSAL_CE.md`
- 当前实现 D03：`docs/algorithms/GRPO_SURROGATE.md`
- 当前实现 D04：`docs/algorithms/OPD_REVERSE_KL.md`
- 当前实现 D05：`docs/algorithms/EXACT_MATH_VERIFIER.md`

## Stage Gates

- G0：算力、时间、公开/私有发布边界确认。
- G1：数据 license、去污染、split 和 evaluator 单测通过。
- G2：E2B Student 与 E4B same-lineage Teacher 的 SFT 可复现，Teacher capability gate 通过，方可进入 RL/OPD。
- G3：GRPO reward 审计通过，方可进行正式 rollout。
- G4：Teacher/Student tokenizer 与 vocabulary 兼容，方可做 logit distillation。
- G5：A0–A4 全部完成三 paired seeds；硬件不闭合时停止，不用单 seed 结果替代主 claim。
- G6：结果经过 claim audit 后才进入简历和面试材料。
