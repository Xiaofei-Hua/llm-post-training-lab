# End-to-End Project Modules

## 唯一计数口径

本项目的**核心端到端链路固定为 24 个模块（D01–D24）**，不是 12 个。此前的 D01–D12 只覆盖 CPU 算法/框架开发，是完整链路的第一层。

| 范围 | ID | 数量 | 是否计入核心完成度 |
|---|---|---:|---|
| CPU 算法与框架 | D01–D12 | 12 | 是 |
| 真实模型、GPU correctness、anchors 与 pilots | D13–D20 | 8 | 是 |
| 五臂正式训练、评测与交付 | D21–D24 | 4 | 是 |
| 核心完成后的研究扩展 | X01–X08 | 8 | 否 |

因此：

- 核心项目完成分母：`24`；当前 `4/24`；
- 包含延后研究线的完整目录：`32`；当前 `4/32`；
- `LEARNING_CURRICULUM.md` 的 12 个知识章节只是学习索引，不是另一组执行模块，不进入上述分母。

当前 D01–D04 已在 CPU 完成并验证；下一模块仍是 D05。D13 之后的 accelerator 动作全部只是计划，当前未获 MPS/CUDA/GPU 执行授权。

## Layer 1：CPU 算法与框架（D01–D12）

| ID | 交付物 | 必须通过的 exit criteria | 状态 |
|---|---|---|---|
| D01 | loss-position mask 与精确 Student backward token budget | reservation/counter/checkpoint/DDP 边界与 property tests 通过 | COMPLETE（CPU） |
| D02 | production masked causal cross-entropy | causal shift、selected-position chunking、value/gradient/accumulation/DDP oracle 通过 | COMPLETE（CPU） |
| D03 | exact-reward Dr.GRPO advantage 与 clipped surrogate | zero-variance、ratio/clipping、active normalizer、value/gradient/DDP oracle 通过 | COMPLETE（CPU） |
| D04 | OPD full-vocabulary chunked reverse-KL | full-vocab reference、双 LM-head chunking、mask、Teacher stop-grad、value/gradient/DDP oracle 通过 | COMPLETE（CPU） |
| D05 | exact/symbolic parser、verifier 与 reward audit | 覆盖数值/分数/表达式/拒绝路径；100–300 条 adversarial cases；reward 与 evaluator 共用 canonical semantics | PLANNED（NEXT） |
| D06 | data registry、license/revision lineage、family split 与 contamination | immutable manifests/checksums；split determinism；exact/fuzzy contamination fixtures；泄漏失败闭锁 | PLANNED |
| D07 | sealed benchmark evaluator、generation/result schema 与 metric contracts | test-answer access boundary；greedy/sampling reproducibility；item-level raw records；evaluator version hash | PLANNED |
| D08 | paired statistics core | item bootstrap、paired randomization/sign-flip、Holm、TOST、pass@k 与 synthetic null/effect coverage 通过 | PLANNED |
| D09 | model/tokenizer/parameter contracts | tokenizer/vocab/hash、LoRA target allowlist、text-only freeze/checksum、nested model-output adapters 可在 fixtures 验证 | PLANNED |
| D10 | SFT/GRPO/OPD objective runtime | shared update API；rollout/old-policy/Teacher freshness；独立 RNG；stage reset；skip/retry semantics | PLANNED |
| D11 | provenance、双预算与成本 ledger | git/config/model/data/evaluator/checkpoint hashes；E1/E2 token/FLOP/time/memory schema；不可覆盖失败 attempt | PLANNED |
| D12 | CPU end-to-end preflight 与 stage-gate orchestrator | 用 production code paths 和 reference fixtures 完成 resume/failure/gate dry-run；不得伪装成真实训练结果 | PLANNED |

D01–D12 的完成只证明实现合同在 CPU 上成立，不代表真实 Gemma 4、TRL/vLLM、显存或吞吐已经通过。

## Layer 2：真实模型与 GPU 训练准备（D13–D20）

| ID | 交付物 | 实际工作与 exit criteria | Gate | 状态 |
|---|---|---|---|---|
| D13 | accelerator/distributed runtime closure | 记录 GPU 型号/拓扑、driver、CUDA、NCCL、PyTorch/Transformers/TRL/vLLM/kernel revisions；验证 BF16、gradient accumulation/checkpointing、distributed launcher、FSDP/ZeRO 候选与 checkpoint/resume；锁定 allocation 和公开边界 | G0 | PLANNED；GPU 未授权 |
| D14 | Gemma 4 E2B/E4B real-checkpoint integration | pinned revisions 下载与 text-only forward；token IDs/vocab/hash 一致；LoRA target 命中；非文本参数 zero-grad/checksum；checkpoint save/load | C0 | PLANNED；GPU 未授权 |
| D15 | real data/evaluator freeze | materialize 公开数据 revision；冻结 D_anchor/D_select/D_teacher_gate/D_dev/D_core/E hashes；完成污染审计、blind human audit 与 Base baseline | G1 | PLANNED |
| D16 | Student/Teacher SFT feasibility | E2B/E4B 各做 64-example overfit；2k×最多两档 LR sanity；只用 D_select 选 recipe；packing parity/禁用决策留证 | C1 | PLANNED；GPU 未授权 |
| D17 | same-lineage anchors 与 Teacher qualification | 用同一 D_anchor 完成 10k E2B/E4B SFT；冻结可复现 Student anchor；一次性解封 D_teacher_gate；Teacher 同时通过 accuracy CI/+5pp/NLL/parse gate | G2/C4 | PLANNED；GPU 未授权 |
| D18 | GRPO GPU correctness 与 D_dev pilot | no-vLLM 2-step → vLLM 2-step；验证 weight sync、old-policy age、RNG、zero-variance、cap/truncation；最多两档 LR；reward audit 通过 | G3/C2 | PLANNED；GPU 未授权 |
| D19 | OPD GPU correctness 与 D_dev pilot | real E2B/E4B full-vocab KL parity；相同 prefix/token IDs；Teacher freeze；mask/gradient；最多两档 LR；exact kernel 无静默近似 | G4/C3 | PLANNED；GPU 未授权 |
| D20 | 100-step profiles 与 campaign freeze | E2B backward、E4B SFT、group-8 rollout、E2B+E4B OPD 四类 steady-state profile；审计通信、显存、吞吐、save/resume；重算总成本并留 30%；冻结 precision/sharding/U/cap/config hashes/run order | C5 / execution READY | PLANNED；GPU 未授权 |

D13–D20 是正式训练的必要前置条件。任何 gate 失败都先回到对应模块修正；不能用量化 Teacher、top-k KL、旧模型、删臂或单 seed 替代来绕过。

## Layer 3：五臂正式训练（D21–D22）

| ID | 交付物 | 规模 | 必须通过的 exit criteria | 状态 |
|---|---|---:|---|---|
| D21 | A0–A4 全部 Stage 1 | 5 arms × 3 paired seeds × 2M = 30M Student loss tokens | 15/15 midpoint checkpoints；每 run 精确闭合 U=2M；相同 objective config hash；无按结果删臂 | PLANNED；GPU 未授权 |
| D22 | 统一 reset 后全部 Stage 2 | 5 arms × 3 paired seeds × 2M = 30M Student loss tokens | 15/15 endpoints；每 stage optimizer/scheduler 重置；累计 60M Student loss tokens；所有 invariant 与 retry 记录完整，G5 通过 | PLANNED；GPU 未授权 |

正式矩阵只有 15 条 arm-seed run sequence，但每条包含两个有独立预算与 checkpoint 的 stage。D21 与 D22 分开，是为了在任何 Stage 2 结果出现前锁住中点证据与 reset 语义。

## Layer 4：证据、审计与求职交付（D23–D24）

| ID | 交付物 | 必须通过的 exit criteria | 状态 |
|---|---|---|---|
| D23 | frozen evaluation、statistics、cost 与 failure analysis | Base/anchor/Teacher、15 midpoints、15 endpoints 按冻结 revision 评测；完成 C1/C2、Holm/TOST、每 seed 结果、E1/E2 成本、error taxonomy；最多一个 reward-hacking 负例 | PLANNED |
| D24 | claim audit、reproducibility release 与 portfolio | 每个数字可回溯 immutable result；技术报告、README result cards、配置/hashes、失败记录、复现命令、10 分钟讲稿、追问题库与简历 bullet 经 G6 审计 | PLANNED |

D24 完成才算核心项目真正完成。训练跑完但没有 D23/D24，只能叫“产生 checkpoints”，不能叫“形成研究成果”。

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
D01–D12 CPU trust stack
        ↓
D13–D15 environment / real model / frozen data
        ↓
D16–D20 SFT anchors / GRPO / OPD / compute closure
        ↓
D21 Stage 1 → D22 Stage 2
        ↓
D23 evaluation & inference → D24 claim audit & release
        ↓
X01–X08 only by a new decision
```

- 每轮只允许一个核心模块处于 `IN_PROGRESS`；完成、验证并 commit 后才移动。
- 当前只允许继续 D05–D12 的 CPU 工作；不得因为模块已列入计划而自动获得 GPU、模型下载或训练授权。
- D13 进入前必须由用户明确切换 accelerator execution 状态；D20 未通过时不得启动 D21。
- 模块状态、run 状态和 gate 状态是三套不同字段：代码完成不能代替 gate evidence，run 完成也不能代替 claim audit。
