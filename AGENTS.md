# Project Instructions

## Pipeline Status

- language: zh
- phase: cpu-framework-development
- project_type: algorithm-framework-performance LLM post-training portfolio
- development_priority: algorithms, executable training framework, performance, metrics and failure analysis
- planning_revision: 2026-09-09 technical-focus
- execution_status: CPU-only algorithm/framework development; no GPU training authorized
- core_modules: D01-D24 (8/24 complete); GPU/full-training begins at D13
- extension_modules: X01-X08 (deferred until D24; excluded from core completion)

## Problem Anchor

在有限算力和公开数据条件下，从同一学生模型的 SFT checkpoint 出发，研究稀疏 reward 与稠密 Teacher 信号如何改变学习动态、推理能力与训练效率，构建可运行的后训练框架，并比较单独作用及顺序交互。

当前 Method Thesis 将该通用问题实例化为 Gemma 4 E2B Student 上的 exact-reward GRPO 与 same-lineage E4B Teacher OPD；DPO 只作核心完成后的 shadow 学习项。

## Development Priorities

1. **算法**：目标函数、梯度、on-policy 采样、优势估计、KL 方向、稳定性与顺序交互；用推导、实现和行为曲线解释机制。
2. **框架**：模型/LoRA 接入、rollout→reward/Teacher→loss→backward→update 的完整循环、阶段切换与模块复用；交付能学习的程序。
3. **性能**：CE/KL 分块、重计算、batch/packing、rollout 与 Teacher 开销、显存和吞吐瓶颈；用同条件 baseline 与 profile 验证优化。
4. **指标**：accuracy/pass@k、有效组率、entropy/KL/clip、长度/截断、retention、tokens/s、step latency、峰值内存及 accuracy–cost 权衡。

选任务时优先考虑能否形成“技术问题→设计取舍→代码→测量→解释”的完整成果。工作量参考分配为算法 35%、框架 30%、性能 20%、指标分析 15%，不是工时验收门槛。哈希、schema、溯源、一致性和审计只作为现有基础设施复用；没有具体错误或训练/评测阻塞时，不新增此类专项模块、通用平台或重复检查，不以检查数量作为成果。

## Working Rules

1. 每轮声明一个主要技术目标及可观测的验收结果，允许为打通该目标修改必要依赖；不要求额外审批、逐模块 commit 或流程文档才能推进。
2. 确认性训练沿用预注册 C1/C2、五臂两阶段与三个 paired seeds；每阶段匹配 Student backward loss tokens，固定模型、数据和解码。CPU 数值实验、学习 smoke 和性能 microbenchmark 只需明确技术问题、baseline 与指标，不必创建新的研究 claim。
3. 新增研究算法前做简短 deletion test：说明现有方法的具体不足和新对照的决策价值。现有算法的框架接入、数值修复和性能优化不受新增算法流程限制。
4. 验收优先检查 loss/gradient、实际参数更新、学习行为、瓶颈与指标解释；只补能捕获真实问题的测试，避免围绕哈希/字段校验继续扩张。
5. 性能成果是独立技术贡献：记录 workload、硬件、精度、baseline、测量范围与波动；核函数加速、端到端加速和算法准确率收益分别表述，负优化也如实记录。
6. 公开 benchmark 只用于评测，训练/选择/test 隔离并使用已有去污染流程。来源、license、revision、配置和 seed 保留最小可复现记录；已有校验信息由工具自动生成，说明文档引用即可。
7. 不声称尚未运行的结果；区分规划、合成 CPU 实测、真实模型实测。简历可描述已实现的算法/框架能力，准确率或加速数字须有对应测量支持，不以完整五臂结束作为描述已有技术成果的前提。
8. 当前只授权 CPU 开发与验证；可用本地初始化的 tiny causal LM 和合成任务验证梯度、学习循环与 CPU 性能，不能用于选择正式 Gemma recipe 或推断其效果。当前不执行模型/真实数据下载、MPS/CUDA 或 GPU 训练。
9. Python 环境通过 `uv.lock` 复现；替换实现时移除失效 API、测试与说明，不保留废弃副本。
10. 不得把内部文档正文、业务数据、指标、模型或同事信息提交到仓库；内部阅读笔记放在被忽略的 `notes/private/`。

## Canonical Documents

- 总览：`README.md`
- 项目契约：`docs/planning/PROJECT_CHARTER.md`
- 最终研究方案：`refine-logs/FINAL_PROPOSAL.md`
- 实验计划：`refine-logs/EXPERIMENT_PLAN.md`
- 实验追踪：`refine-logs/EXPERIMENT_TRACKER.md`
- 数据计划：`docs/data/DATA_PLAN.md`
- 评测协议：`docs/evaluation/BENCHMARK_PLAN.md`
- 完整端到端模块总表：`docs/planning/DEVELOPMENT_MODULES.md`
- 性能实验与测量计划：`docs/planning/PERFORMANCE_PLAN.md`
- 近期执行顺序：`docs/planning/ROADMAP.md`
- 技术成果与简历映射：`docs/planning/PORTFOLIO_CHECKLIST.md`
- 当前实现 D01：`docs/algorithms/LOSS_TOKEN_BUDGET.md`
- 当前实现 D02：`docs/algorithms/MASKED_CAUSAL_CE.md`
- 当前实现 D03：`docs/algorithms/GRPO_SURROGATE.md`
- 当前实现 D04：`docs/algorithms/OPD_REVERSE_KL.md`
- 当前实现 D05：`docs/algorithms/EXACT_MATH_VERIFIER.md`
- 当前实现 D06：`docs/data/DATA_REGISTRY_AND_CONTAMINATION.md`
- 当前实现 D07：`docs/evaluation/SEALED_EVALUATOR.md`
- 当前实现 D08：`docs/evaluation/PAIRED_STATISTICS.md`

## Stage Gates

- G0：进入 accelerator 阶段前确认授权、算力、时间与公开边界；实际总成本在 D20/C5 校准。
- G1：真实数据可用且 train/test 隔离，evaluator 能可靠测量目标；复用 D05–D07 检查。
- G2：E2B Student 与 E4B same-lineage Teacher 的 SFT 可复现，Teacher capability gate 通过，方可进入 RL/OPD。
- G3：GRPO reward、优势估计、策略刷新和更新正确，方可进行正式 rollout。
- G4：Teacher/Student token 对齐、exact KL 与梯度正确，方可做正式 logit distillation。
- G5：A0–A4 全部完成三 paired seeds；硬件不闭合时停止，不用单 seed 结果替代主 claim。
- G6：完成算法/性能结果分析和技术表达；研究结论与量化简历表述有实际证据。按成果范围做简短核对，复用 D08 统计，不另建审计系统。

G0–G6 用于决定真实实验是否可以推进，不作为日常 CPU 开发的审批链。D01–D08 保持完成状态；下一模块 D09 为模型适配与可训练参数接入，随后 D10 统一训练循环、D11 性能剖析与优化、D12 CPU 端到端学习实验。
