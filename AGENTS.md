# Project Instructions

## Pipeline Status

- language: zh
- phase: sft-feasibility-in-progress
- project_type: research-and-algorithm-first LLM post-training portfolio
- development_priority: research questions, algorithm mechanisms, real-model learning experiments, evidence and failure analysis; framework and performance support the research
- planning_revision: 2026-10-13 research-and-algorithm-first
- execution_status: D15 complete; both D16 64-example overfit NLL checks passed, final greedy exact 0/16 for both, E4B parse 0/16 with all outputs capped; first 2k E2B LR1e-4 SFT pilot complete (500 updates and 500 D_select outputs, NLL 0.8612 to 0.6651, exact 8/500, parse 97/500, capped 457/500); complete offline diagnosis pairs all 500 improved reference NLLs with final outputs (reference-fit subset 7/211 exact and 184/211 capped), historical 304-output snapshot retained, no paired Base answer gain; second E2B LR2e-4 pilot complete (500 updates, NLL 0.8612 to 0.6631, exact 5/500, parse 68/500, capped 466/500), all 500 reference NLLs improved and reference-fit subset 2/211 exact with 194/211 capped; frozen NLL-first E2B choice is 2e-4 despite lower answer metrics, first E4B LR1e-4 complete with 500 updates/NLL/generations (0.7240 to 0.5802, exact 14/500, parse 60/500, capped 479/500); full CPU diagnosis shows all 500 reference NLLs improved, reference-fit subset 12/211 exact and 198/211 capped; final E4B LR2e-4 stopped at logged update 173 because Windows powered off on 2026-10-12, checkpoint 168 with 3833928/11484531 loss tokens and full optimizer/RNG validated and resumed on 2026-10-13, checkpoint 176 verified with 4024306/11484531 tokens, all 516 LoRA tensors changed and optimizer/scheduler advanced, replayed 169-173 logged numerical fields/data order identical, 287 targeted CPU regressions passed; joint recipe/D16 acceptance/G2 pending; real learned E2B cached/full-prefix probe reproduced all 16384 saved greedy tokens and matched next-token argmax at 24 sampled positions (max KL 0.0014865), not all-position equivalence or G3; D17 anchor/evaluator and shared GRPO/OPD pilot CPU-ready; D19 joint Base up to 2220+4096 GPU update/full-cache/resume passed with frozen Teacher decoder staging, resident 4096 backward OOM preserved; no rental needed for this bounded workload, D20 SFT projection reuse passed real E2B/E4B 2k/16k paired updates/resume (48 updates total), with E4B 16k 8.58% mean slowdown preserved and default disabled; later real E4B trace localizes attention backward bottleneck (single-window reuse speedup 9.84%, not stable adoption); synthetic E4B-shaped QKV Flex/TRITON tile16 completed four GPU cases/96 forward-backward calls with repeatable same-kernel gradients and observed cross-backend differences, default-tile shared-memory failures retained; real E4B Flex/SDPA 2k/16k paired complete updates and exact same-backend checkpoint replay measured (32 updates), mean step time reduced 30.45%/78.61% but first-step gradient cosine only 0.7646/0.7339; CPU first-Adam-moment localization reproduces the discrepancy; real E4B fixed-operand FP64 VJP diagnosis completed at 2k/16k across layers 1/5/41 and both backend origins (4 unchanged LoRA/Adam updates, 12 full captures, 72 kernel calls), BF16 local dQ relative error at most 1.08% while early-layer incoming-gradient cosine across model paths falls to 0.000891; fixed-forward whole-model intervention completed across eight real 2k/16k updates with exact identity controls and seven prior arms reproduced, nested FP32 attention checkpoint repaired 16k diagnostic OOM (13.6893/14.2168 GiB allocated/reserved), Flex/FP32 gradient cosines above 0.9997 but Adam update relative differences 11.7–12.9%, production SDPA retained; original E2B pilot checkpoint/cache resume verified; end-to-end measurement and four real-workload bindings CPU-validated, including shared post-update scoring/rollout I/O and progress snapshots; D21/D22 handoff, budget executor and D_core/measured-recipe binding CPU-validated across 30 stages with learned Teacher and exact pause/resume, fixed-order stage launcher, full evaluation cost/allocation binding and midpoint-score barrier CPU-validated with native signal/resume and Trainer scheduler-cycle fix; real C5 closure and formal GPU matrix pending; D23 n=8 sampling and frozen-checkpoint generative/closed-option evaluators CPU-validated (Student-only load, fixed-batch resume, failure/I/O costs; new classification finish separated from generation truncation), full six-public-benchmark plus 200 OpenR1 reserve diagnostic inputs prepared (14670 items, 14910 responses/checkpoint), mixed-suite execution, model release and orthogonal answer/truncation diagnostics CPU-validated; 34-position checkpoint matrix with group resume and unchanged D08 endpoint statistics CPU-validated on 30 learned tiny stages, real CLI declaration prepared (293790 responses/182 panel loads); supporting paired IFEval retention, Teacher slices and token/time learning curves CPU-validated with unchanged C1/C2; pilot Student-to-SFT-anchor full-vocabulary KL CPU-validated with cached-prefix resume and unchanged next updates; local E2B feasibility-checkpoint BF16/copy-restore comparison passed through 2220+4096 (5.825 GiB allocated, no resident optimizer/Teacher), 31-position anchor-KL matrix, cached-prefix resume, supporting analysis and conservative C5 calibration binding CPU-validated; qualified-anchor GPU diagnostics and real evaluation matrix pending; E2B/E4B cached decode memory-fill optimization independently validated on 5 paired short replays and 16384 unchanged real greedy tokens per model (full paired windows -15.19%/-13.70%), shared SFT/checkpoint generation policy and runtime cost separation enabled; real G2/G3/G4 and campaign cost pending; G1 exact-answer scope only
- core_modules: D01-D24 (15/24 complete); GPU/full-training begins at D13
- extension_modules: X01-X08 (deferred until D24; excluded from core completion)

## Project Goal

本项目服务于作者投递大模型后训练研究/算法岗位。优先通过项目建设获得可深入讲解的研究与算法经验：提出问题、阅读近邻工作、建立可检验假设、推导和实现训练目标、完成真实模型对照实验、解释结果并据此迭代。简历是这些工作完成后的提炼；用户要求完善项目时，应交付研究缺口、算法方案、实验和验收安排，而非默认把现有结果改写为简历条目。

主要成果以研究问题的清晰度、算法理解、真实学习证据、对照质量和结论解释力衡量。允许有价值的负结果，不预设必须提出新算法、获得正向准确率收益或达到 SOTA。框架与性能工作服务于算法实验的可执行性和可信度，不以功能数量、模块进度或测试数量代替研究成果。

2026-10-13 用户明确要求“以研究和算法为主”。旧规划中的工作量分配和任务排序与此冲突时，以本文件的研发优先级为准；已经冻结的实验协议、数据隔离、预算与 claim 边界继续有效。

## Problem Anchor

在有限算力和公开数据条件下，从同一学生模型的 SFT checkpoint 出发，研究稀疏 reward 与稠密 Teacher 信号如何改变学习动态、推理能力与训练效率，构建可运行的后训练框架，并比较单独作用及顺序交互。

原 Method Thesis 将该通用问题实例化为 Gemma 4 E2B Student 上的 exact-reward GRPO 与 same-lineage E4B Teacher OPD；DPO 只作核心完成后的 shadow 学习项。

近期主线采用已定稿的低预算 SFT/GRPO 专题：先比较普通 CE 与答案及真实 EOS 联合加权 SFT，再从同一父检查点比较继续 SFT 与明确规定的 Dr.GRPO recipe，研究监督分配、答案交付、稀疏奖励学习及效果–成本关系。保留单 E2B、三个 paired continuation seeds、最多 12 条正式轨迹、9M 新增正式反传 tokens 与 180 GPU 小时上限；开发诊断也计入总成本。

先完成一个能解释的真实 SFT→RL 研究闭环，再扩展合格 Teacher 上的 OPD 学习与原五臂顺序研究。低预算专题与原 C1/C2 是不同研究身份：不把前者完成称为原五臂完成，不因 Student-only 预检通过就放行原 G2/G3/G4，也不静默更换模型、Teacher 或既定 recipe。

## Development Priorities

1. **研究问题与假设**：从真实学习现象和近邻工作找具体缺口，列出竞争解释、可观测预测及最小有判别力的对照；不凭算法组合、模型新旧或框架功能宣称创新。
2. **算法机制与实现**：围绕监督分配、目标函数、梯度及实际优化器更新、on-policy 采样、优势估计、KL 方向和稳定性做推导与实现，把设计选择连接到真实学习行为。
3. **学习实验与解释**：优先完成真实模型的连续采样/更新和配对对照；联合分析 accuracy、有效组率、entropy/KL/clip、长度/截断、停止行为与完整成本，区分参考拟合、格式收益和答案能力。
4. **必要框架与性能**：复用已有 rollout→reward/Teacher→loss→backward→update、恢复和评测路径；仅为解除当前研究阻塞或回答具体效率问题补代码与优化，用同条件 baseline 验证代价和收益。

选任务时优先形成“研究问题→近邻与假设→算法设计→真实实验→机制解释→下一步决策”的成果。参考投入为研究与算法 60%、学习实验与分析 25%、必要框架与性能 15%，不是工时验收门槛；真实阻塞可以优先修复。没有明确研究决策价值时，不横向增加算法、训练后端、分布式平台或 benchmark 大矩阵。

哈希、schema、溯源、一致性和审计复用现有基础设施；没有具体错误或训练/评测阻塞时，不新增专项模块、通用平台或重复检查。已有性能成果保留其独立技术价值，后续优先测量实际学习循环的瓶颈和效果–成本取舍。

## Research Delivery

- **算法证据**：把 loss/gradient、裁剪与 Adam 后的实际更新、采样行为和独立终点联系起来。单次有限梯度、参数变化或参考 NLL 下降，只证明对应环节，不等于持续学习或推理能力提高。
- **真实学习**：尽早以有界开发实验暴露 reward 稀疏、无效组和生成预算风险；正式研究保留既定父模型预检、配对 seeds 和停止条件。开发轨迹与正式训练分开，不能继承诊断更新、挑成功题或用 tiny 结果替代真实模型学习。
- **解释与消融**：先使用现有协议、长度、分区 NLL、梯度和输出诊断；新增对照必须能够区分具体解释。答案/EOS 联合加权不识别二者单独贡献，继续 SFT/GRPO 的 recipe 比较不等于只改变 loss，也不识别两种 SFT 初始化与 RL 的交互。
- **失败与迭代**：完整训练但没有终点增益，可以形成有价值的研究结论；RL 预检失败则如实结束该研究分支，同时注明真实持续 RL 学习经验仍待补齐，并提出下一项有界研究决策。不能靠增加框架功能或改写简历宣布项目目标完成，也不能为追求正结果事后改测试、seed 或主要结论。
- **最终交付**：研究问题与近邻定位、算法推导和代码、真实学习轨迹、配对独立终点、失败解释及最小复跑入口。框架和性能成果作为实现方法与成本证据；公开评测、额外消融和优化仅在有解释价值且预算可闭合时增加。

## Working Rules

1. 每轮声明一个主要研究问题或算法目标及可观测的验收结果，允许为打通该目标修改必要依赖；不要求额外审批、逐模块 commit 或流程文档才能推进。
2. 原 C1/C2 确认性训练沿用五臂两阶段与三个 paired seeds；近期低预算专题执行其独立定稿协议。各自按既定 Student backward loss-token 预算、模型、数据和解码比较，不混用完成状态或结论。CPU 数值实验、有界真实学习开发和性能 microbenchmark 只需明确问题、baseline、指标及成本，不必创建新的确认性 claim；新建议纳入执行方案前须写清与原协议的关系。
3. 新增研究算法或改变目标前做简短近邻调研和 deletion test：说明现有方法的具体不足、可检验预测及新对照的决策价值。已有直接先例时如实定位复现、适配或受控研究。现有算法的框架接入、数值修复和性能优化不受新增算法流程限制。
4. 验收优先检查算法语义、实际更新、连续学习行为和独立结果，并解释支持或否定了什么假设；只补能捕获真实问题的测试，避免围绕哈希/字段校验继续扩张。
5. 性能工作优先解除研究瓶颈，并可形成独立技术贡献：记录 workload、硬件、精度、baseline、测量范围与波动；核函数加速、端到端加速和算法准确率收益分别表述，负优化也如实记录。不用局部加速替代算法实验的完成。
6. 公开 benchmark 只用于评测，训练/选择/test 隔离并使用已有去污染流程。来源、license、revision、配置和 seed 保留最小可复现记录；已有校验信息由工具自动生成，说明文档引用即可。
7. 不声称尚未运行的结果；区分规划、合成 CPU 实测、真实模型实测。优先完善研究和算法证据，再按用户需要提炼简历；准确率或加速数字须有测量支持。已有成果可以按实际范围描述，不以完整五臂结束作为投递前提，也不把现有成果的措辞优化当作项目完善。
8. 2026-09-09 用户提供单 GPU 服务器并授权 tiny 模型 D13 验证，随后通知 GPU 正被另一任务使用，要求依赖走镜像、先完成其余工作并 commit/push。2026-09-22 用户确认当前 GPU 可用于实验并要求继续算法开发，已恢复并完成 RTX 5070 Ti 上的 D13 tiny CUDA/BF16、续训、失败恢复和性能验收。依赖继续走镜像。不得用 tiny 结果选择正式 Gemma recipe 或推断其效果；2026-09-23 用户明确要求推进并完成 D14 后推送，已授权 D14 所需真实 Base 模型/tokenizer 下载及有界 forward/backward、LoRA 更新验证；2026-10-02 用户要求继续推进并明确使用本机 GPU；2026-10-03 已完成 D15 真实数据与有界 Base panel，用户授权以 AI 逐条盲审加独立评分器替代真人审阅，并要求持续开发、每模块测试反思、开发前调研和设计；后续实验直接使用本机 GPU，若资源无法满足则通知用户。D16 已通过双模型完整16k的本机GPU更新测量；两模型64-example overfit已完成预定NLL条件和答案诊断；两档E2B 2k pilot均已完成500更新及500题生成，按既定NLL优先规则选择2e-4但答案指标更低；2026-10-10用户要求跑完剩余评测和两档E4B、完成学习率选择及D16验收，首档E4B已完成500次更新/NLL/生成（0.7240→0.5802、正确14/500、截断479/500）及完整CPU复盘；2026-10-13用户明确要求查明退出原因、校验checkpoint并从保存点继续末档E4B 2e-4。已确认为10月12日主机关机，日志173步但最后保存点168步；校验参数/Adam/RNG/预算后恢复，第176步新保存点已验证，169-173五步重放数值与数据顺序一致，287项定向回归通过。联合D16 recipe、D16验收与D17+尚未完成。
9. Python 环境通过 `uv.lock` 复现；替换实现时移除失效 API、测试与说明，不保留废弃副本。
10. 不得把内部文档正文、业务数据、指标、模型或同事信息提交到仓库；内部阅读笔记放在被忽略的 `notes/private/`。

## Canonical Documents

- D15 当前实测与限制：`docs/data/D15_REAL_DATASET.md`
- D16 当前可行性与本机限制：`docs/architecture/SFT_FEASIBILITY.md`
- E2B/E4B 缓存解码性能与等输出验证：`docs/performance/CACHED_DECODE_RUNTIME.md`
- D17 同源 anchors 与资格统计接入（真实 G2 未运行）：`docs/architecture/SAME_LINEAGE_ANCHORS.md`
- D18 缓存采样与GRPO CPU接入（真实 G3 未运行）：`docs/architecture/CACHED_ON_POLICY_ROLLOUT.md`
- D19 OPD内存预算与padding优化（真实 G4 未运行）：`docs/performance/OPD_MEMORY.md`
- D15 历史 preflight：`docs/data/D15_REAL_DATA_PREFLIGHT.md`
- 2026-10 前沿调研与方法边界：`docs/references/FRONTIER_REVIEW_2026_10.md`

- 总览：`README.md`
- 项目契约：`docs/planning/PROJECT_CHARTER.md`
- 最终研究方案：`refine-logs/FINAL_PROPOSAL.md`
- 实验计划：`refine-logs/EXPERIMENT_PLAN.md`
- 实验追踪：`refine-logs/EXPERIMENT_TRACKER.md`
- 数据计划：`docs/data/DATA_PLAN.md`
- 评测协议：`docs/evaluation/BENCHMARK_PLAN.md`
- 完整端到端模块总表：`docs/planning/DEVELOPMENT_MODULES.md`
- 性能实验与测量计划：`docs/planning/PERFORMANCE_PLAN.md`
- D20完整尝试/评测成本、恢复计时与预算外推（CPU已验证，真实C5未闭合）：`docs/performance/END_TO_END_MEASUREMENT.md`
- D21/D22阶段交接与D_core/测量配置接入（CPU五臂/三seed与续训验证，真实矩阵未开始）：`docs/architecture/STAGE_HANDOFF.md`
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
- 当前实现 D09：`docs/architecture/MODEL_ADAPTER.md`
- 当前实现 D10：`docs/architecture/CPU_TRAINING_LOOP.md`
- 当前实现 D11：`docs/performance/CPU_BENCHMARK.md`
- 当前实现 D12：`docs/experiments/CPU_LEARNING_EXPERIMENT.md`
- 当前实现 D13（单 GPU tiny 实测完成）：`docs/architecture/ACCELERATOR_RUNTIME.md`
- 当前实现 D14（真实 E2B/E4B Base 实测完成）：`docs/architecture/GEMMA4_INTEGRATION.md`
- D01–D13 回顾与修复（五轮，最后两轮无新增问题）：`docs/experiments/D01_D13_REVIEW.md`
- D01–D14 全进度复审（三轮，最后两轮无新增问题；控制 token 与 BLAS 精度修复）：`docs/experiments/D01_D14_REVIEW.md`

## Stage Gates

- G0：进入 accelerator 阶段前确认授权、算力、时间与公开边界；实际总成本在 D20/C5 校准。
- G1：真实数据可用且 train/test 隔离，evaluator 能可靠测量目标；复用 D05–D07 检查。
- G2：E2B Student 与 E4B same-lineage Teacher 的 SFT 可复现，Teacher capability gate 通过，方可进入 RL/OPD。
- G3：GRPO reward、优势估计、策略刷新和更新正确，方可进行正式 rollout。
- G4：Teacher/Student token 对齐、exact KL 与梯度正确，方可做正式 logit distillation。
- G5：A0–A4 全部完成三 paired seeds；硬件不闭合时停止，不用单 seed 结果替代主 claim。
- G6：完成算法/性能结果分析和技术表达；研究结论与量化简历表述有实际证据。按成果范围做简短核对，复用 D08 统计，不另建审计系统。

G0–G6 用于决定真实实验是否可以推进，不作为日常开发的审批链。D01–D12 已完成 CPU 验证；D13 已完成运行时代码、CPU 回归与单 GPU tiny 实测，D14 已通过真实 E2B/E4B Base 的文本/LoRA 更新、tokenizer、logit transform、冻结与内存验收，核心进度为 15/24；D15 已完成。G1 通过范围为既有 D05 显式最终答案契约（AI/独立 scorer 一致率 100%）；语义一致率 94%，未达到语义层的 99% 目标，未做真人审阅，后续必须保留此限制。D16 正在推进。资源连接信息保留在被忽略的 `notes/private/compute/`，公开报告仅记录硬件规格和技术结果。D12 的五臂三 seed 是合成 CPU 实验，不替代 D21–D23 的真实训练或 C1/C2。
