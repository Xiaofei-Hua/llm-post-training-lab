# Experiment Plan

> 状态：2026-09-09 技术主线修订；C1/C2 方法设计保留，真实训练 EXECUTION-CONDITIONAL。
> 当前：D01–D08 CPU 完成；D09–D12 开发与性能实验待做，真实模型/GPU 实验未开始。

## 1. Research question 与 claims

研究问题：从同一 Student SFT checkpoint 出发，解释稀疏 reward 与稠密 Teacher 信号对学习动态、推理能力和顺序交互的作用；构建可运行训练框架，测量和优化训练效率。

本项目只检验以下 recipe-level claims：

| Claim | Confirmatory contrast | Endpoint | 最小支持条件 | 不能声称 |
|---|---|---|---|---|
| C1a：exact-reward GRPO 的 intervention effect | A1−A0 | MATH-500 greedy accuracy | Holm-adjusted p<.05、95% CI lower>0、点估计≥+2pp | GRPO 一般优于 SFT |
| C1b：same-lineage Teacher OPD 的 intervention effect | A2−A0 | MATH-500 greedy accuracy | 同上 | dense signal/OPD 对任意 Teacher 有效 |
| C2：OPD/GRPO 顺序效应或等效 | A3−A4 | MATH-500 greedy accuracy | superiority：95% CI 不含0且 abs(Δ)≥2pp；或 equivalence：90% CI 全在±2pp | 最优通用训练顺序 |

所有推断都条件于一个 frozen E2B anchor、一个通过独立 gate 的 frozen E4B Teacher、固定 recipe 和三个预注册 post-anchor seeds。

### 开发实验与正式研究的分工

| 类型 | 进入条件 | 主要产出 | 结论范围 |
|---|---|---|---|
| CPU 数值/学习实验 | 明确技术问题与 reference，复用现有测试 | 梯度、参数更新、学习曲线和失败解释 | 当前算法/合成 workload |
| 性能实验 | 瓶颈假设、固定 workload 与 baseline | 时间、内存、吞吐对照及波动 | 对应设备与 kernel/端到端范围 |
| 正式模型实验 | 以下 C1/C2 设计、真实数据/模型条件与资源就绪 | 五臂三 seed、能力/动态/成本分析 | 指定 Gemma recipe 的效果 |

前两类不需另建确认性 claim、审计报告或审批链。性能工作见 `docs/planning/PERFORMANCE_PLAN.md`；更改监督目标、KL 近似或策略新鲜度不能以性能优化名义混入正式 recipe。

## 2. Frozen design

### Models

- Student：`google/gemma-4-E2B` Base，经 `D_anchor` SFT 后冻结为共同 parent；
- Teacher：`google/gemma-4-E4B` Base，经同一 `D_anchor`、模板、mask、顺序、LoRA classes 和 token/epoch budget SFT 后冻结；
- LoRA：rank 32、alpha 64、dropout 0；准确 target names 在模型 introspection 后锁定；
- 只训练 text path；image/audio encoders、PLE、embedding 与 LM head 默认冻结并做 checksum/zero-grad assertion。

### Data

| Split | Size | 唯一用途 | 泄漏规则 |
|---|---:|---|---|
| D_anchor | 10,000 traces | E2B/E4B same-lineage SFT | 不与其余 split 同 family |
| D_select | 500 | 两个 SFT checkpoint selection | 不可用于 Teacher qualification |
| D_teacher_gate | 500 | 一次性 sealed Teacher gate | gate 前任何配置不可读取 |
| D_dev | 500 | objective 内最多两档 LR/smoke | 不进入 test 主表 |
| D_core | 2,000 prompts | A0–A4 formal intervention | 所有臂同 registry/distribution/cycle |
| E | sealed | MATH-500 等正式评测 | training process 无答案权限 |

所有 split 做 source/problem/template-family 分组隔离；prompt、reference solution 和 trace 均做 exact/fuzzy contamination audit。

### Symmetric arms

| Arm | Stage 1 | Stage 2 | Seeds |
|---|---|---|---|
| A0 | SFT 2M | SFT 2M | 101, 202, 303 |
| A1 | GRPO 2M | GRPO 2M | 101, 202, 303 |
| A2 | OPD 2M | OPD 2M | 101, 202, 303 |
| A3 | OPD 2M | GRPO 2M | 101, 202, 303 |
| A4 | GRPO 2M | OPD 2M | 101, 202, 303 |

每个 stage 开始时重置 optimizer/scheduler。相同 objective 在任意位置使用相同 resolved config hash，只允许 parent checkpoint、stage ID 和 seed 不同。

`U=2M` 指实际进入已执行 Student optimizer update 的有效 objective positions；prompt、padding、EOS 后、纯 forward token 与 skipped zero-variance group 不计。最后一批使用稳定 budget mask 精确填满 `U`。

### Objective contracts

**SFT**：assistant-only masked CE；采用首 assistant token 至模板 EOS（含 EOS）；logical-update 全局有效 token mean。

**GRPO**：exact/symbolic 0/1 reward；`group_size=8`、`loss_type=dr_grpo`、`epsilon=0.2`、`beta=0`、`num_iterations=1`、token importance sampling、group reward scaling、temperature 1.0；每 generation batch 刷新 old policy 并同步 weights；零方差组不重采样。默认 cap 2048，冻结 pilot truncation>5% 时所有 formal runs 统一改为4096。

**OPD**：fully on-policy，`num_iterations=1`，每 update 前由当前 Student 新生成 completion；Teacher 在相同 prefix 上 frozen forward；temperature 1.0 full-vocab chunked reverse KL；completion-through-first-EOS mask；先逐 token 对全 vocab 求 KL，再按全 batch 有效 token 数归一化。

## 3. Experiment blocks

### Block E0 — 算法框架、学习验证与性能

当前 CPU 阶段按以下顺序交付：

1. **DEV-D09**：本地 tiny causal LM 的模型适配、LoRA/text 参数接入；D02 CE/D04 KL 的真实 forward/backward/update，给出参数图与梯度解释。
2. **DEV-D10**：共用 SFT/GRPO/OPD 主循环，打通当前策略采样、reward/Teacher、loss/backward/update 与阶段切换；输出 loss、reward、entropy/KL/clip、有效组率。
3. **PERF-CPU-CE/KL/STEP**（D11）：dense reference 与 exact chunk/recompute 对照，测时间、内存和有效 tokens/s；依据 profile 做一项优化尝试并复测。
4. **DEV-D12**：用同一真实代码路径在合成任务跑通五臂两阶段，生成学习/成本曲线，解释一个退化或失败案例；不要求支持正式 C1/C2。

CPU 阶段不下载真实模型/数据、不运行 MPS/CUDA，不用 tiny 模型为 Gemma 选参。D01–D08 的 loss、verifier、数据、指标和统计实现直接复用，已有 synthetic 验证不再作为新开发周期。

授权后的真实模型准备：

- D13–D15：硬件与模型接入、真实数据分布和 Base 指标；复用已有去污染/evaluator 流程，完成必要人工抽查，避免 test 用于选参。
- D16–D19：SFT 学习验证、Teacher qualification、GRPO/OPD 更新正确性与 D_dev 稳定性/动态分析。
- D18–D20：E2B backward、E4B SFT、group-8 rollout、E2B+E4B OPD 各 100-step profile；复用 C5 队列，比较瓶颈优化前后性能。
- D20/C5：重算五臂成本并留 30% 余量，固定正式实现/recipe；预算不闭合则按既定边界同步调整 U/cap 后重新 profile。

算法数值或梯度有误时先修复再做性能结论；不以无关的新增哈希检查阻塞训练功能开发。

### Block E1 — Same-lineage anchors

目标：建立唯一 Student parent 与独立合格的 Teacher。

流程：64-example overfit → 2k sanity → 10k `D_anchor`。E2B/E4B 各自在预注册两个 LR 中只用 `D_select` 选择。锁定后只解封一次 `D_teacher_gate`。

Teacher 必须同时满足：相对 Student accuracy paired 95% CI lower>0、点估计≥+5pp、verified-solution NLL 更低、parse rate 不下降、tokenizer 完全一致。失败则 primary OPD 和 A2/A3/A4 停止；不得回到 gate 数据调参，也不得静默换 E4B-it。

### Block E2 — Single-signal interventions

目标：估计 C1a/C1b。

先在 `D_dev` 对每个 objective 运行最多两个 LR 的单 seed stability pilot；选择规则仅依据 finite loss、无 NaN/OOM、梯度/entropy/clip 等预注册健康区间，不读正式 test。随后 A0/A1/A2 全部运行三个 seeds、两个 stages，不能按初步结果删 arm。

GRPO stop gate：effective-group rate<30%、truncation>5%、weight-sync age>1 batch、independent accuracy 与 train reward 明显反向时停止扩展并记录失败。OPD stop gate：Teacher 意外更新或 token/prefix 错位、KL/gradient mismatch、非文本梯度、OOM 或 exact-kernel throughput 令 campaign 不闭合。

### Block E3 — Order intervention

目标：估计 C2。

A3/A4 使用与 E2 完全相同的 OPD/GRPO config hashes，各跑三个 paired seeds。Stage 1 与 A1/A2 中点形成机制诊断；Stage 2 后做唯一确认性 `A3−A4`。不得针对顺序臂重新调 LR、cap、reward 或 mask。

### Block E4 — Frozen evaluation、failure analysis 与小型负例

对 Base、Student anchor、15 个 A0–A4 endpoints 以及必要的 stage-1 checkpoints 使用同一 generation/evaluator revision。确认性 endpoint 只用 MATH-500；GSM8K、MathArena 06/2026、AIME 2026、IFEval、MMLU-Pro 和 sealed 200 题均标 supporting/diagnostic。

核心五臂完成后，最多运行一个小型 format-reward/宽-parser 负例以展示 reward hacking；DPO shadow、其他 KL、larger Teacher、QLoRA、PRM 和 Agent RL 都属于后续，不得挤占主矩阵。

## 4. Evaluation and inference

- 复用 D07 的 item-level correctness、pass@k、长度/截断和 paired generation，D08 负责已定义的统计；当前只有 synthetic CPU 证据，真实 benchmark 结果待测；
- Greedy 主评测：所有 checkpoint 固定模板、stop、max-new-tokens 和 evaluator；
- Sampling：每题恰好 n=8，T=0.7、top-p=.95、top-k=0，generation seeds 成对；`pass@k=1-C(n-c,k)/C(n,k)`；
- C1：对每个 item 把三个预注册 seed 的 paired correctness 差取均值；在 MATH level 内只重采样 item 10,000 次，item 携带完整 seed vector；100,000 次 item-level paired randomization，两个 p-value 做 Holm；
- C2：同一 item unit；superiority 与 TOST equivalence 不混用；
- 单列每个 training-seed effect；方向不一致必须写 seed instability；
- IFEval 以 A0 为 reference，-2pp 非劣 margin；
- 旧公开集只证明本项目 post-training data 已去污染；MathArena 是 freshness sentinel，不替代 MATH-500 主终点。

### 机制与性能诊断

除确认性 endpoint 外，固定 D_dev 诊断批次观察有效组率、reward 方差、entropy、Student-anchor KL、OPD KL/Teacher NLL、clip fraction、梯度范数、输出长度和截断。训练指标按 objective 分列，不能把 CE/GRPO/KL 的原始 loss 当成共同刻度。

端点分析 Teacher-correct/incorrect × Student-correct/incorrect 四象限、stage 1→2 的错误迁移和 retention；这些属于解释性分析，不由事后 slice 反推新主 claim。性能表列 update/端到端 tokens/s、rollout tokens/s、step p50/p95、峰值内存、分段耗时和测量波动。指标定义见 `docs/evaluation/BENCHMARK_PLAN.md`。

## 5. Budget and reporting

E1 只匹配 Student backward loss tokens，prompt exposure 与 Student FLOPs只审计。E2 记录 rollout tokens、Student/old/reference/Teacher forward、Student backward、GPU-hours、accelerator-seconds、峰值显存与能耗。

成本拆为：共享 `C_anchor`、共享 `C_teacher`、逐臂 `C_arm`。结果同时报告：

- warm-start/marginal：`C_arm`；
- cold-start：非 OPD 为 `C_anchor+C_arm`，含 OPD 为 `C_anchor+C_teacher+C_arm`；
- campaign：`C_anchor+C_teacher+ΣC_arm`，Teacher 构建只计一次。

保留 run/config/seed、模型与数据版本、原始结果入口、失败状态和成本；现有工具自动生成的 hashes 直接复用。已完成 CPU 实现/测量可提前进入 README 或简历，注明范围；真实模型效果和性能倍数须有相应实验支持。G6 核对已有结果与表述，不新增审计平台。

## 6. 正式训练的 Go/no-go checklist

- [ ] 可用硬件、总时长和 30% buffer 已确认；
- [ ] 模型许可、数据许可、公开边界与所有 immutable revisions 已锁定；
- [ ] D_select 与 D_teacher_gate 隔离，后者尚未被选择流程读取；
- [ ] evaluator、loss、mask、U counter、统计实现通过 synthetic tests；
- [ ] LoRA target/non-text freeze/tokenizer/weight-sync gates 通过；
- [ ] Teacher 独立 capability gate 通过；
- [ ] 15 个主 run 和完整评测的 campaign budget 闭合；
- [ ] 所有 objective config hashes、三个 seeds 和 formal run order 已在结果出现前冻结。

只有全部勾选，正式训练 execution status 才能从 CONDITIONAL 改为 READY；此清单不阻塞 D09–D12 的 CPU 开发与实验。

## 7. 端到端模块映射

核心仍为 **24 个模块，已完成 8/24**。实验 block 按问题分组，D 模块按实际交付推进；详细验收见 `docs/planning/DEVELOPMENT_MODULES.md`。

| 模块 | Block/Run | 核心交付 | 状态 |
|---|---|---|---|
| D01–D04 | E0 | loss-token 预算、CE、GRPO surrogate、reverse-KL 及梯度验证 | COMPLETE（CPU） |
| D05–D08 | E0 | verifier、数据隔离、评测指标、paired statistics，后续复用 | COMPLETE（CPU；真实数据/评测待做） |
| D09 | E0 / DEV-D09 | 模型适配、LoRA/text 参数与实际 forward/backward/update | PLANNED（CPU） |
| D10 | E0 / DEV-D10 | 统一训练循环与学习动态指标 | PLANNED（CPU） |
| D11 | E0 / PERF-CPU-* | 性能 baseline、瓶颈、优化与复测 | PLANNED（CPU） |
| D12 | E0 / DEV-D12 | 五臂合成学习示例、曲线与失败解释 | PLANNED（CPU） |
| D13–D15 | E0 / G0,C0,G1 | GPU/model runtime、真实任务分布与 Base 指标 | PLANNED；执行未授权 |
| D16–D17 | E1 / C1,G2,C4 | SFT 学习曲线、双 anchors 与独立 Teacher gate | PLANNED；GPU 未授权 |
| D18–D19 | E2 / G3,G4 | GRPO/OPD 学习动态、rollout 与大词表性能 | PLANNED；GPU 未授权 |
| D20 | E0 / C5,PERF-GPU-* | 端到端优化对照、正式配置与成本闭合 | PLANNED；GPU 未授权 |
| D21–D22 | E2+E3 / G5 | 五臂两阶段三个 paired seeds、完整动态与成本记录 | PLANNED；GPU 未授权 |
| D23 | E4 | C1/C2、retention、错误迁移、accuracy–cost、负结果 | PLANNED |
| D24 | G6 | 算法/框架/性能技术报告、演示与求职成果 | PLANNED |

X01–X08 继续延后到 D24 之后，不计入核心完成度。性能剖析和现有算法的实现优化属于核心工作，无需等到扩展阶段。
