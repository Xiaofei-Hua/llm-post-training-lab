# Experiment Tracker

> 2026-09-09 修订：D01–D08 CPU 完成；D09–D12 技术开发与性能实验待做，所有真实模型/GPU gates 与 runs 均 `NOT_STARTED`。本表列出开发队列与预注册正式实验，不是结果表。
> Formal seeds：`101, 202, 303`。

## Module scope

| Layer | IDs | Count | Completion denominator | Status |
|---|---|---:|---|---|
| CPU algorithm/framework | D01–D12 | 12 | core | 8 COMPLETE / 4 PLANNED |
| accelerator readiness and pilots | D13–D20 | 8 | core | 0/8；GPU 未授权 |
| formal campaign and evidence | D21–D24 | 4 | core | 0/4；GPU 未授权 |
| post-core research extensions | X01–X08 | 8 | excluded | 0/8；DEFERRED_UNTIL_D24 |

核心进度分母为 `24`，当前 `8/24`；包含延后研究线的目录总数为 `32`。12 个学习课程章节不计入执行进度。完整定义以 `docs/planning/DEVELOPMENT_MODULES.md` 为准，当前下一模块是 D09。

## 近期 CPU 技术队列

| Run ID | Module | 技术问题/对照 | 交付与指标 | Dependency | Status |
|---|---|---|---|---|---|
| DEV-D09 | D09 | tiny causal LM/LoRA 接入 CE/KL，与 dense reference 对照 | 实际参数更新、梯度误差、参数比例与内存估算 | D02,D04 | NOT_STARTED |
| DEV-D10 | D10 | SFT/GRPO/OPD 共用当前策略采样与 update 循环 | loss/reward/KL/entropy/clip/有效组率、stage reset | DEV-D09 | NOT_STARTED |
| PERF-CPU-CE | D11 | dense CE vs selected-position chunking | forward/backward ms、有效 tokens/s、RSS、梯度误差 | DEV-D10 | NOT_STARTED |
| PERF-CPU-KL | D11 | dense exact KL vs chunk/recompute | 时间/内存取舍、chunk size、梯度误差 | DEV-D10 | NOT_STARTED |
| PERF-CPU-STEP | D11 | 完整训练 step baseline vs 瓶颈驱动优化 | 分段耗时、step p50/p95、端到端有效 tokens/s | DEV-D10 | NOT_STARTED |
| DEV-D12 | D12 | 合成可验证任务上的五臂两阶段学习 | 学习/行为/成本曲线，一个退化或失败解释 | PERF-CPU-CE/KL/STEP | NOT_STARTED |

仅用本地初始化 tiny 模型和合成输入，不下载真实模型/数据或运行 MPS/CUDA。上述开发实验只需技术问题、baseline 与测量，不新增 C1/C2 式研究 claim。性能协议见 `docs/planning/PERFORMANCE_PLAN.md`；当前无这些条目的实测结果。

### GPU/full-training coverage

| Module | Tracker coverage | Exit condition |
|---|---|---|
| D13 | G0 metadata | accelerator 授权、BF16/distributed update 可用；初始 latency/memory/communication 测量 |
| D14 | C0-001 | real E2B/E4B text/LoRA forward/backward/update 可用 |
| D15 | DATA-001/002, EVAL-001/002, EV-BASE | 数据隔离与 evaluator 可用；任务分布和 Base 指标 |
| D16 | SFT-*-OVERFIT, SFT-*-PILOT | Student/Teacher SFT recipes selected without gate leakage |
| D17 | SFT-*-ANCHOR, C4-TEACHER-GATE | G2 and Teacher capability gate pass |
| D18 | C2-001/002, PILOT-A1, PERF-GPU-ROLLOUT | GRPO 更新正确；有效组/entropy/长度动态与 rollout 开销 |
| D19 | C3-001/002, PILOT-A2, PERF-GPU-OPD | exact KL/梯度正确；Teacher 错误迁移、大词表时间/内存取舍 |
| D20 | C5-001..005, PILOT-A0, PERF-GPU-SFT/E2E | 瓶颈优化对照；campaign fits with 30% buffer；configs frozen |
| D21 | all `MAIN-*` stage 1 | 15/15 midpoint checkpoints close exactly at 2M Student loss tokens |
| D22 | all `MAIN-*` stage 2 | 15/15 endpoints close exactly at 4M cumulative tokens; G5 passes |
| D23 | EV-MID/END, STAT-C1/C2, COST-E1E2, DYNAMICS-001, PERF-REPORT, optional FAIL-001 | 能力/动态/效率结果与失败解释完成 |
| D24 | G6 plus portfolio artifacts | 算法/框架/性能报告和演示，量化表述对应实测 |

## Gate tracker

| ID | Gate | Evidence required | Status | Blocking next |
|---|---|---|---|---|
| G0 | 资源与授权 | GPU 授权、型号/拓扑、可用时长与版本；完整 profile 在 D20/C5 | NOT_STARTED | 所有 GPU 正式运行 |
| G1 | 数据与 evaluator | 来源/license、family split、去污染、≥99% 人工抽查一致率 | NOT_STARTED | Anchor 与 formal eval |
| G2 | Anchor/Teacher | E2B anchor reproducible；E4B independent gate pass | NOT_STARTED | OPD 与主矩阵 |
| G3 | GRPO correctness | loss/reward/sync/policy-age/skipped-group tests | NOT_STARTED | A1/A3/A4 |
| G4 | OPD correctness | tokenizer、exact KL、mask、gradient、freeze tests | NOT_STARTED | A2/A3/A4 |
| G5 | Main repeats | 15/15 formal arm-seed runs valid | NOT_STARTED | Confirmatory claims |
| G6 | 技术分析与作品 | 能力/动态/性能结果、失败解释、量化表述有实际支持 | NOT_STARTED | 核心作品最终交付 |

## Preflight and foundation queue

| Run ID | Action | Inputs | Pass criterion | Dependency | Status |
|---|---|---|---|---|---|
| D06-TRUST-FIXTURE | Data trust contract audit | frozen synthetic adversarial registry/records/policy/expectation | 96/96 tests；dirty→family quarantine→clean rescan；manifest/Git provenance hashes close | none | COMPLETE_CPU |
| D07-EVALUATOR-FIXTURE | Sealed evaluator contract audit | 6-item synthetic public/sealed snapshot、greedy/sampling protocols、frozen predictions/expectation | 70/70 tests；完整 item×sample grid；greedy/sampling 各 1 个 truncation；3/6、21/48、5/6 pass@8 oracle；evaluator/Git hashes close | EVAL-001,D06 contract | COMPLETE_CPU |
| D08-STATISTICS-FIXTURE | Paired statistics contract audit | 8-item synthetic correctness panel、3 seeds、2 strata、frozen protocol/expectation | 33/33 tests；10k stratified item bootstrap；100k complete-vector sign-flip；Holm、effect/null/equivalence oracle；runtime/input/Git hashes close | D07 contract | COMPLETE_CPU |
| DATA-001 | Build registry/splits | public source revisions | counts、licenses、hashes、family-disjoint | none | NOT_STARTED |
| DATA-002 | Contamination audit | all train/eval text + traces | frozen threshold；reviewed borderline pairs | DATA-001 | NOT_STARTED |
| EVAL-001 | Verifier attack suite | frozen 257-case corpus | 257/257；corpus/policy/backend/source/lock/Git hashes recorded | none | COMPLETE_CPU |
| EVAL-002 | Blind human audit | ≥100 outputs | agreement ≥99%；error matrix saved | EVAL-001 | NOT_STARTED |
| C0-001 | E2B/E4B introspection | pinned model revisions | tokenizer IDs/hash、module map、freeze assertions | G0 metadata | NOT_STARTED |
| C1-001 | Mask/CE oracle | synthetic batches | value/gradient + U counter exact | C0-001 | NOT_STARTED |
| C2-001 | GRPO no-vLLM smoke | 8–64 prompts | 2 updates、finite metrics、no hidden resample | C0-001,EVAL-001 | NOT_STARTED |
| C2-002 | GRPO vLLM sync smoke | same prompts/config | policy age=1 batch；weights actually change | C2-001 | NOT_STARTED |
| C3-001 | OPD tiny oracle | hand distributions | exact value/limit/gradient tolerance pass | C0-001 | NOT_STARTED |
| C3-002 | OPD chunked smoke | E2B+E4B tiny batch | oracle alignment、mask/freeze、no full persistent logits | C3-001 | NOT_STARTED |
| C5-001 | E2B LoRA profile | 100 steady steps | tokens/s、memory、step p50/p95；FLOPs 注明估算口径 | C1-001 | NOT_STARTED |
| C5-002 | E4B LoRA SFT profile | 100 steady steps | tokens/s、memory、forward/backward 时间 | C0-001 | NOT_STARTED |
| C5-003 | Group-8 rollout profile | 2k cap, fixed backend | throughput、P95、truncation、sync cost | C2-002 | NOT_STARTED |
| C5-004 | Exact OPD profile | E2B+E4B | throughput、peak memory、kernel evidence | C3-002 | NOT_STARTED |
| C5-005 | Campaign closure | all profiles | total budget +30% fits confirmed allocation | C5-001..004 | NOT_STARTED |

`PERF-GPU-SFT` 复用 C5-001/002，`PERF-GPU-ROLLOUT` 复用 C5-003，`PERF-GPU-OPD` 复用 C5-004；D20 的 `PERF-GPU-E2E` 在最大瓶颈处选择一项优化，与 baseline 做同条件复测，并将结果用于 C5-005。这些是同一组测量的技术分析视角，不重复计工作量。计时、内存和波动口径以 `PERFORMANCE_PLAN.md` 为准。

## Anchor and Teacher queue

| Run ID | Model | Data | Selection access | Output | Status |
|---|---|---|---|---|---|
| SFT-S-OVERFIT | E2B Base | D_anchor/64 | none | mask/data sanity | NOT_STARTED |
| SFT-T-OVERFIT | E4B Base | D_anchor/64 | none | mask/data sanity | NOT_STARTED |
| SFT-S-PILOT | E2B Base | D_anchor/2k × 2 LR | D_select only | selected Student recipe | NOT_STARTED |
| SFT-T-PILOT | E4B Base | D_anchor/2k × 2 LR | D_select only | selected Teacher recipe | NOT_STARTED |
| SFT-S-ANCHOR | E2B Base | D_anchor/10k | frozen recipe | hashed Student anchor | NOT_STARTED |
| SFT-T-ANCHOR | E4B Base | D_anchor/10k | frozen recipe | hashed Teacher candidate | NOT_STARTED |
| C4-TEACHER-GATE | Student vs Teacher | D_teacher_gate/500 | one-time sealed | pass/fail + CI/NLL/parse | NOT_STARTED |

若 `C4-TEACHER-GATE=FAIL`，A2/A3/A4 标记 `CANCELLED_BY_GATE`，不得换 E4B-it 后沿用相同实验族。

## Objective config pilots

| Run ID | Objective | Allowed search | Selection rule | Status |
|---|---|---|---|---|
| PILOT-A0 | SFT continuation | at most 2 LR | finite/monotonic loss、no test access | NOT_STARTED |
| PILOT-A1 | GRPO | at most 2 LR | health gates only；not test accuracy | NOT_STARTED |
| PILOT-A2 | OPD | at most 2 LR | health gates only；not test accuracy | NOT_STARTED |

通过后分别写出 immutable `sft_config_hash`、`grpo_config_hash`、`opd_config_hash`。这些 hash 在所有 arm/stage 中复用。

## Formal 15-run matrix

每个 run 内包含 stage 1、统一 reset 和 stage 2；中点/终点均保存 checkpoint。D21 完成全部 stage 1，D22 才执行全部 stage 2。`VALID` 需要两个 stage 的 `U=2,000,000` 精确闭合以及所有 invariant assertions 通过。

| Run ID | Arm | Seed | Stage 1 | Stage 2 | Dependency | S1 status | S2 status | Run status |
|---|---|---:|---|---|---|---|---|---|
| MAIN-A0-S101 | A0 | 101 | SFT | SFT | G0–G2,PILOT-A0 | NOT_STARTED | BLOCKED_BY_S1 | NOT_STARTED |
| MAIN-A0-S202 | A0 | 202 | SFT | SFT | G0–G2,PILOT-A0 | NOT_STARTED | BLOCKED_BY_S1 | NOT_STARTED |
| MAIN-A0-S303 | A0 | 303 | SFT | SFT | G0–G2,PILOT-A0 | NOT_STARTED | BLOCKED_BY_S1 | NOT_STARTED |
| MAIN-A1-S101 | A1 | 101 | GRPO | GRPO | G0–G3,PILOT-A1 | NOT_STARTED | BLOCKED_BY_S1 | NOT_STARTED |
| MAIN-A1-S202 | A1 | 202 | GRPO | GRPO | G0–G3,PILOT-A1 | NOT_STARTED | BLOCKED_BY_S1 | NOT_STARTED |
| MAIN-A1-S303 | A1 | 303 | GRPO | GRPO | G0–G3,PILOT-A1 | NOT_STARTED | BLOCKED_BY_S1 | NOT_STARTED |
| MAIN-A2-S101 | A2 | 101 | OPD | OPD | G0–G2,G4,PILOT-A2 | NOT_STARTED | BLOCKED_BY_S1 | NOT_STARTED |
| MAIN-A2-S202 | A2 | 202 | OPD | OPD | G0–G2,G4,PILOT-A2 | NOT_STARTED | BLOCKED_BY_S1 | NOT_STARTED |
| MAIN-A2-S303 | A2 | 303 | OPD | OPD | G0–G2,G4,PILOT-A2 | NOT_STARTED | BLOCKED_BY_S1 | NOT_STARTED |
| MAIN-A3-S101 | A3 | 101 | OPD | GRPO | G0–G4,PILOT-A1/A2 | NOT_STARTED | BLOCKED_BY_S1 | NOT_STARTED |
| MAIN-A3-S202 | A3 | 202 | OPD | GRPO | G0–G4,PILOT-A1/A2 | NOT_STARTED | BLOCKED_BY_S1 | NOT_STARTED |
| MAIN-A3-S303 | A3 | 303 | OPD | GRPO | G0–G4,PILOT-A1/A2 | NOT_STARTED | BLOCKED_BY_S1 | NOT_STARTED |
| MAIN-A4-S101 | A4 | 101 | GRPO | OPD | G0–G4,PILOT-A1/A2 | NOT_STARTED | BLOCKED_BY_S1 | NOT_STARTED |
| MAIN-A4-S202 | A4 | 202 | GRPO | OPD | G0–G4,PILOT-A1/A2 | NOT_STARTED | BLOCKED_BY_S1 | NOT_STARTED |
| MAIN-A4-S303 | A4 | 303 | GRPO | OPD | G0–G4,PILOT-A1/A2 | NOT_STARTED | BLOCKED_BY_S1 | NOT_STARTED |

## Evaluation queue

| Eval ID | Checkpoints | Suite | Output | Status |
|---|---|---|---|---|
| EV-BASE | E2B Base | all frozen suites | descriptive baseline | NOT_STARTED |
| EV-ANCHOR | E2B SFT anchor | all frozen suites | shared parent baseline | NOT_STARTED |
| EV-TEACHER | E4B candidate | D_teacher_gate once + supporting eval | qualification evidence | NOT_STARTED |
| EV-MID | 15 stage-1 checkpoints | MATH-500 + diagnostics | mechanism/order trajectory | NOT_STARTED |
| EV-END | 15 endpoints | full frozen suite | raw item×seed predictions | NOT_STARTED |
| STAT-C1 | A0/A1/A2 endpoints | preregistered item inference | Holm table + CIs | NOT_STARTED |
| STAT-C2 | A3/A4 endpoints | preregistered item inference | superiority/TOST table | NOT_STARTED |
| COST-E1E2 | all valid runs | token/FLOPs/time/memory measurements | E1、三视角 E2 与 accuracy–cost 图 | NOT_STARTED |
| DYNAMICS-001 | midpoints/endpoints + D_dev logs | entropy/KL/有效组率/长度/retention | 顺序切换和 Teacher 错误迁移分析 | NOT_STARTED |
| PERF-REPORT | D11 and D18–D20 profiles | baseline/优化对照 | CPU/GPU 分列的吞吐/内存/瓶颈分析 | NOT_STARTED |
| FAIL-001 | at most one tiny controlled run | format reward or wide parser | reward-hacking case study | DEFERRED_UNTIL_D22 |
| SHADOW-DPO | one frozen rollout bank | DPO single seed | appendix only | DEFERRED_UNTIL_D24 |

## Post-core extension queue

| ID | Research line | Admission rule | Status |
|---|---|---|---|
| X01 | DPO offline shadow | D24 complete；new preregistered claim/config | DEFERRED_UNTIL_D24 |
| X02 | ORPO/KTO preference optimization | deletion test against X01/core | DEFERRED_UNTIL_D24 |
| X03 | GSPO/TIS policy-mismatch variants | a diagnosed GRPO limitation and separate budget | DEFERRED_UNTIL_D24 |
| X04 | PRM/process supervision and reward modeling | auditable step labels/reward data contract | DEFERRED_UNTIL_D24 |
| X05 | forward-KL/JS/top-k/cross-tokenizer distillation | new experiment family；never substitute for core exact KL | DEFERRED_UNTIL_D24 |
| X06 | larger Teacher/Student and cross-architecture transfer | scaling claim, matched budget and compute closure | DEFERRED_UNTIL_D24 |
| X07 | multimodal post-training | separate data/evaluator/licensing thesis | DEFERRED_UNTIL_D24 |
| X08 | Agentic RL/tool-use | environment, safety and trajectory-evaluation contract | DEFERRED_UNTIL_D24 |

## Failure/retry policy

- Infra interruption before optimizer update：同 checkpoint/config/RNG state 恢复，记录 incident；
- NaN/OOM/kernel mismatch：该 run 失败，先回 correctness/profile gate；不得只为单臂改 precision/loss；
- 末批 token：使用预注册 budget mask，不允许 overshoot；
- formal run 重跑：保留原 run ID 为 failed，重跑使用新 attempt suffix，不能覆盖；
- seed/result 不理想：不是 retry 理由；
- budget 不闭合：所有臂同步修改 `U`/cap，产生新 protocol version 并重新审查。

## Completion counters

- Core modules complete：`8 / 24`
- Optional extension modules complete：`0 / 8`（不计入核心完成度）
- Foundation gates passed：`0 / 7`
- Teacher qualified：`0 / 1`
- Formal runs valid：`0 / 15`
- Endpoint evals complete：`0 / 15`
- Confirmatory contrasts analyzed：`0 / 3`
- Portfolio artifacts evidence-backed：`0 / 6`
