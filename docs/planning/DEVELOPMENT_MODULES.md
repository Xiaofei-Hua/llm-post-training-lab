# 算法、框架、性能与指标开发模块

> 2026-09-09 更新：D01–D12 已完成 CPU 算法、模型/循环、性能测量与合成学习验收。D13 运行时代码及 CPU 验证已实现，GPU 被其他任务占用而暂停实测；现有校验设施按需维护。

## 唯一计数口径

本项目的核心端到端链路为 **24 个模块（D01–D24）**。模块用于安排技术交付，完成数量不代表研究效果或简历价值。

| 范围 | ID | 数量 | 是否计入核心完成度 |
|---|---|---:|---|
| CPU 算法、框架与性能 | D01–D12 | 12 | 是 |
| 真实模型、GPU 性能、anchors 与 pilots | D13–D20 | 8 | 是 |
| 五臂正式训练、评测与交付 | D21–D24 | 4 | 是 |
| 核心完成后的研究扩展 | X01–X08 | 8 | 否 |

因此：

- 核心项目完成分母：`24`；当前 `12/24`；
- 包含延后研究线的完整目录：`32`；当前 `12/32`；
- `LEARNING_CURRICULUM.md` 的 12 个知识章节只是学习索引，不是另一组执行模块，不进入上述分母。

当前 D01–D12 已在 CPU 完成并验证；正在开发 D13。D13 的累积、重计算与 save/resume 已通过 CPU 测试；GPU 验证按用户要求暂停。实现与剩余验收见 `docs/architecture/ACCELERATOR_RUNTIME.md`。D14 之后的真实模型/数据执行尚未授权。

## Layer 1：CPU 算法与框架（D01–D12）

| ID | 交付物 | 必须通过的 exit criteria | 状态 |
|---|---|---|---|
| D01 | loss-position mask 与精确 Student backward token budget | reservation/counter/checkpoint/DDP 边界与 property tests 通过 | COMPLETE（CPU） |
| D02 | production masked causal cross-entropy | causal shift、selected-position chunking、value/gradient/accumulation/DDP oracle 通过 | COMPLETE（CPU） |
| D03 | exact-reward Dr.GRPO advantage 与 clipped surrogate | zero-variance、ratio/clipping、active normalizer、value/gradient/DDP oracle 通过 | COMPLETE（CPU） |
| D04 | OPD full-vocabulary chunked reverse-KL | full-vocab reference、双 LM-head chunking、mask、Teacher stop-grad、value/gradient/DDP oracle 通过 | COMPLETE（CPU） |
| D05 | exact/symbolic parser、verifier 与 reward audit | 覆盖数值/分数/表达式/拒绝路径；100–300 条 adversarial cases；reward 与 evaluator 共用 canonical semantics | COMPLETE（CPU） |
| D06 | data registry、license/revision lineage、family split 与 contamination | immutable manifests/checksums；split determinism；exact/fuzzy contamination fixtures；泄漏失败闭锁 | COMPLETE（CPU） |
| D07 | sealed benchmark evaluator、generation/result schema 与 metric contracts | test-answer access boundary；greedy/sampling reproducibility；item-level raw records；evaluator version hash | COMPLETE（CPU；synthetic evidence） |
| D08 | paired statistics core | item bootstrap、paired randomization/sign-flip、Holm、TOST 与 synthetic null/effect coverage 通过；只消费 D07 correctness | COMPLETE（CPU；synthetic evidence） |
| D09 | 模型适配与可训练参数接入 | 本地初始化 tiny causal LM；接 D02/D04 的 hidden-state/LM-head 路径；text/LoRA 参数选择；真实 forward/backward/update 与 dense reference 对照；参数量/内存估算 | COMPLETE（CPU；synthetic model evidence） |
| D10 | SFT/GRPO/OPD 统一训练循环 | 三 objective 共用 sample/score/loss/backward/update 主循环；当前 Student 采样、old-policy/Teacher detach、有效 token 更新、stage reset；输出 loss/reward/entropy/KL/clip/有效组率 | COMPLETE（CPU） |
| D11 | 性能剖析与优化 | 固定 workload 下比较 dense reference、有效位置分块与重计算；测 forward/backward、端到端 step、有效 tokens/s、CPU 峰值内存；定位瓶颈并实施至少一项有依据的优化尝试，报告收益或负结果 | COMPLETE（CPU；含负结果） |
| D12 | CPU 端到端学习实验 | 用 D09–D11 实际训练路径在合成可验证任务跑通五臂两阶段；展示 loss/正确动作概率或准确率/entropy/KL/长度/成本曲线，解释一个失败或退化案例 | COMPLETE（CPU；合成五臂三 seed） |

D01–D08 保留既有实现与验证记录，不追加围绕 hash/schema 的专项开发。D09–D12 用本地 tiny 模型验证算法和框架，不下载 checkpoint；CPU 测量只解释对应 workload，不预测 Gemma 的准确率或 GPU 加速。详细性能实验见 `PERFORMANCE_PLAN.md`。

### 近期模块的可展示成果

- D09：模型结构与参数图、可训练参数比例、adapter 接入与梯度解释。
- D10：统一更新循环代码、GRPO/OPD 时序图、策略刷新和梯度归一化示例。
- D11：baseline→profile→优化→复测的对照表，解释内存与时间取舍；不预设加速倍数。
- D12：可一条命令运行的合成学习示例及行为/成本曲线，不要求出现与正式 C1/C2 相同的结论。

## Layer 2：真实模型与 GPU 训练准备（D13–D20）

| ID | 交付物 | 实际工作与 exit criteria | Gate | 状态 |
|---|---|---|---|---|
| D13 | accelerator 与分布式训练运行时 | 接入 BF16、gradient accumulation/checkpointing 与实际需要的 FSDP/ZeRO；测 update latency、显存和通信占比，完成必要 save/resume；确认硬件与资源授权 | G0 | IN_PROGRESS；代码/CPU 验证完成，GPU 实测暂停 |
| D14 | Gemma 4 E2B/E4B 模型接入 | 真实 text forward/backward、LoRA 参数更新、模型 logit transform 与 tokenizer/token 对齐；测参数/激活内存，验证冻结参数无更新 | C0 | PLANNED；GPU 未授权 |
| D15 | 真实数据、任务分布与 Base 指标 | 复用 D06/D07 materialize 各 split 和 benchmark；完成必要去污染/人工抽查；产出难度、长度、可解析率与 Base accuracy 分布 | G1 | PLANNED |
| D16 | Student/Teacher SFT feasibility | E2B/E4B 各做 64-example overfit；2k×最多两档 LR sanity；只用 D_select 选 recipe；packing parity/禁用决策留证 | C1 | PLANNED；GPU 未授权 |
| D17 | same-lineage anchors 与 Teacher qualification | 用同一 D_anchor 完成 10k E2B/E4B SFT；冻结可复现 Student anchor；一次性解封 D_teacher_gate；Teacher 同时通过 accuracy CI/+5pp/NLL/parse gate | G2/C4 | PLANNED；GPU 未授权 |
| D18 | GRPO 学习动态与 rollout 性能 | no-vLLM→vLLM 接入并验证更新生效；分析 reward/accuracy、有效组率、entropy/clip、长度/截断；分解 rollout/sync/update 时间；最多两档 LR | G3/C2 | PLANNED；GPU 未授权 |
| D19 | OPD 学习动态与大词表性能 | real full-vocab KL 数值/梯度、相同 prefix 与 Teacher freeze；测 chunk/recompute 的时间和内存，分析 KL/NLL/entropy 与 Teacher 错误迁移；最多两档 LR | G4/C3 | PLANNED；GPU 未授权 |
| D20 | 端到端性能优化与训练预算 | 四类 100-step profile；针对最大瓶颈做同条件优化对照，报告内核和端到端收益/代价；重算成本并留 30%；在主训练前固定后端、精度、batch/U/cap 和 run order | C5 / execution READY | PLANNED；GPU 未授权 |

D13–D20 是正式训练的必要前置条件。任何 gate 失败都先回到对应模块修正；不能用量化 Teacher、top-k KL、旧模型、删臂或单 seed 替代来绕过。

## Layer 3：五臂正式训练（D21–D22）

| ID | 交付物 | 规模 | 必须通过的 exit criteria | 状态 |
|---|---|---:|---|---|
| D21 | A0–A4 全部 Stage 1 | 5 arms × 3 paired seeds × 2M = 30M Student loss tokens | 15/15 midpoint checkpoints；U=2M；相同 objective recipe；学习动态和成本曲线齐备 | PLANNED；GPU 未授权 |
| D22 | 统一 reset 后全部 Stage 2 | 5 arms × 3 paired seeds × 2M = 30M Student loss tokens | 15/15 endpoints；每 stage 重置 optimizer/scheduler；累计 60M；记录顺序切换后的动态和失败，G5 通过 | PLANNED；GPU 未授权 |

正式矩阵只有 15 条 arm-seed run sequence，但每条包含两个有独立预算与 checkpoint 的 stage。D21 与 D22 分开，是为了在任何 Stage 2 结果出现前锁住中点证据与 reset 语义。

## Layer 4：指标分析、技术报告与求职交付（D23–D24）

| ID | 交付物 | 必须通过的 exit criteria | 状态 |
|---|---|---|---|
| D23 | 能力、机制与效率分析 | Base/anchor/Teacher、15 midpoints/endpoints 评测；C1/C2 与每 seed 结果；entropy/KL/有效组率、Teacher 错误迁移、retention、accuracy–cost；解释至少一个负结果 | PLANNED |
| D24 | 技术报告、框架演示与简历成果 | 公式→框架→性能→指标→结论的技术报告、可运行示例、主表/性能表、10 分钟讲稿与追问题库；量化表述核对已有测量，G6 完成 | PLANNED |

D24 完成代表核心项目交付完成。各阶段已实现的算法、框架和实测性能可以提前形成作品，按其真实验证范围表述。

## Post-core research extensions（X01–X08）

| ID | 延后研究线 | 进入条件 | 状态 |
|---|---|---|---|
| X01 | DPO offline shadow | D24 complete；新 claim、preference data contract 与独立预算 | DEFERRED_UNTIL_D24 |
| X02 | ORPO/KTO preference optimization | deletion test 证明相对 X01/core 有额外决策价值 | DEFERRED_UNTIL_D24 |
| X03 | GSPO/TIS 与 policy-mismatch variants | 主 GRPO 暴露明确限制；新预注册对照 | DEFERRED_UNTIL_D24 |
| X04 | PRM、step-level reward 与 reward modeling | 有可审计的过程标签、reward 数据与 evaluator 合同 | DEFERRED_UNTIL_D24 |
| X05 | alternative distillation | forward-KL/JS/top-k/cross-tokenizer 作为新实验族；不得替换核心 exact reverse-KL | DEFERRED_UNTIL_D24 |
| X06 | scale and transfer | larger Teacher/Student、cross-architecture transfer；重新做 matched-budget 与 compute closure | DEFERRED_UNTIL_D24 |
| X07 | multimodal post-training | 独立问题、数据 license、模态 evaluator 与算力计划 | DEFERRED_UNTIL_D24 |
| X08 | Agentic RL/tool-use | 独立 environment、安全边界、trajectory evaluator 与 credit-assignment thesis | DEFERRED_UNTIL_D24 |

扩展模块不是“迟早都必须跑”。每个 X 模块都要先做 deletion test、重新冻结 claim 和预算；若不能增强证据，不进入开发。

## 执行顺序与授权边界

```text
D01–D08 已有算法/指标 → D09 模型接入 → D10 训练循环
        → D11 CPU 性能 → D12 合成学习实验
        ↓
D13–D15 GPU runtime / real model / task baselines
        ↓
D16–D20 SFT / GRPO / OPD / learning dynamics & performance
        ↓
D21 Stage 1 → D22 Stage 2
        ↓
D23 metrics & failure analysis → D24 technical report & portfolio
        ↓
X01–X08 only by a new decision
```

- 每轮聚焦一个主要技术目标，可修改必要依赖；完成对应验证后推进，不以额外审计、审批或 commit 仪式作为前置。
- 当前只允许继续 D09–D12 的 CPU 工作；不得因为模块已列入计划而自动获得 GPU、模型下载或训练授权。
- D13 进入前必须由用户明确切换 accelerator execution 状态；D20 未通过时不得启动 D21。
- 区分代码完成、CPU 学习示例和真实研究结果；模块数量、测试数量与哈希记录都不能替代算法/性能结论。
