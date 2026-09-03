# Experiment Tracker

> 当前状态：D01–D05 CPU modules 已完成；所有真实模型/GPU gates 与实验 runs 均 `NOT_STARTED`。本表是预注册队列，不是结果表。
> Formal seeds：`101, 202, 303`。

## Module scope

| Layer | IDs | Count | Completion denominator | Status |
|---|---|---:|---|---|
| CPU algorithm/framework | D01–D12 | 12 | core | 5 COMPLETE / 7 PLANNED |
| accelerator readiness and pilots | D13–D20 | 8 | core | 0/8；GPU 未授权 |
| formal campaign and evidence | D21–D24 | 4 | core | 0/4；GPU 未授权 |
| post-core research extensions | X01–X08 | 8 | excluded | 0/8；DEFERRED_UNTIL_D24 |

核心进度分母为 `24`，当前 `5/24`；包含延后研究线的目录总数为 `32`。12 个学习课程章节不计入执行进度。完整定义以 `docs/planning/DEVELOPMENT_MODULES.md` 为准，当前下一模块是 D06。

### GPU/full-training coverage

| Module | Tracker coverage | Exit condition |
|---|---|---|
| D13 | G0 metadata | hardware topology、BF16/distributed/checkpoint-resume and version/allocation boundary pass |
| D14 | C0-001 | real E2B/E4B model contract passes |
| D15 | DATA-001/002, EVAL-001/002, EV-BASE | G1 data/evaluator freeze passes |
| D16 | SFT-*-OVERFIT, SFT-*-PILOT | Student/Teacher SFT recipes selected without gate leakage |
| D17 | SFT-*-ANCHOR, C4-TEACHER-GATE | G2 and Teacher capability gate pass |
| D18 | C2-001/002, PILOT-A1 | G3 GRPO correctness/reward audit passes |
| D19 | C3-001/002, PILOT-A2 | G4 OPD real-model compatibility passes |
| D20 | C5-001..005, PILOT-A0 | communication/memory/resume profiled；campaign fits with 30% buffer；configs frozen |
| D21 | all `MAIN-*` stage 1 | 15/15 midpoint checkpoints close exactly at 2M Student loss tokens |
| D22 | all `MAIN-*` stage 2 | 15/15 endpoints close exactly at 4M cumulative tokens; G5 passes |
| D23 | EV-MID/END, STAT-C1/C2, COST-E1E2, FAIL-001 | frozen evidence and failure analysis complete |
| D24 | G6 plus portfolio artifacts | every public claim resolves to immutable evidence |

## Gate tracker

| ID | Gate | Evidence required | Status | Blocking next |
|---|---|---|---|---|
| G0 | 资源与版本 | GPU inventory、框架 revisions、100-step profile | NOT_STARTED | 所有 GPU 正式运行 |
| G1 | 数据与 evaluator | license、hash、family split、contamination、≥99% audit | NOT_STARTED | Anchor 与 formal eval |
| G2 | Anchor/Teacher | E2B anchor reproducible；E4B independent gate pass | NOT_STARTED | OPD 与主矩阵 |
| G3 | GRPO correctness | loss/reward/sync/policy-age/skipped-group tests | NOT_STARTED | A1/A3/A4 |
| G4 | OPD correctness | tokenizer、exact KL、mask、gradient、freeze tests | NOT_STARTED | A2/A3/A4 |
| G5 | Main repeats | 15/15 formal arm-seed runs valid | NOT_STARTED | Confirmatory claims |
| G6 | Claim audit | immutable results、statistics、cost/claim consistency | NOT_STARTED | README/简历数字 |

## Preflight and foundation queue

| Run ID | Action | Inputs | Pass criterion | Dependency | Status |
|---|---|---|---|---|---|
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
| C5-001 | E2B LoRA profile | 100 steady steps | tokens/s、memory、FLOPs recorded | C1-001 | NOT_STARTED |
| C5-002 | E4B LoRA SFT profile | 100 steady steps | tokens/s、memory recorded | C0-001 | NOT_STARTED |
| C5-003 | Group-8 rollout profile | 2k cap, fixed backend | throughput、P95、truncation、sync cost | C2-002 | NOT_STARTED |
| C5-004 | Exact OPD profile | E2B+E4B | throughput、peak memory、kernel evidence | C3-002 | NOT_STARTED |
| C5-005 | Campaign closure | all profiles | total budget +30% fits confirmed allocation | C5-001..004 | NOT_STARTED |

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
| COST-E1E2 | all valid runs | token/FLOPs/time/memory ledgers | E1 + three-view E2 tables | NOT_STARTED |
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

- Core modules complete：`5 / 24`
- Optional extension modules complete：`0 / 8`（不计入核心完成度）
- Foundation gates passed：`0 / 7`
- Teacher qualified：`0 / 1`
- Formal runs valid：`0 / 15`
- Endpoint evals complete：`0 / 15`
- Confirmatory claims audited：`0 / 3`
- Portfolio artifacts evidence-backed：`0 / 6`
