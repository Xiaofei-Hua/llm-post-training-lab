# Experiment Plan

> 状态：METHOD-FROZEN / EXECUTION-CONDITIONAL
> 范围：前期计划，不代表模型、数据、框架或 GPU 已下载/验证。

## 1. Research question 与 claims

Immutable Problem Anchor：在有限算力和完全公开、可追溯的数据条件下，从同一前沿学生模型的 SFT checkpoint 出发，受控比较不同后训练学习信号的单独作用和顺序交互，并把收益与模型、数据、训练预算、评测及计算成本混杂区分开。

本项目只检验以下 recipe-level claims：

| Claim | Confirmatory contrast | Endpoint | 最小支持条件 | 不能声称 |
|---|---|---|---|---|
| C1a：exact-reward GRPO 的 intervention effect | A1−A0 | MATH-500 greedy accuracy | Holm-adjusted p<.05、95% CI lower>0、点估计≥+2pp | GRPO 一般优于 SFT |
| C1b：same-lineage Teacher OPD 的 intervention effect | A2−A0 | MATH-500 greedy accuracy | 同上 | dense signal/OPD 对任意 Teacher 有效 |
| C2：OPD/GRPO 顺序效应或等效 | A3−A4 | MATH-500 greedy accuracy | superiority：95% CI 不含0且 abs(Δ)≥2pp；或 equivalence：90% CI 全在±2pp | 最优通用训练顺序 |

所有推断都条件于一个 frozen E2B anchor、一个通过独立 gate 的 frozen E4B Teacher、固定 recipe 和三个预注册 post-anchor seeds。

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

### Block E0 — Trust stack 与 compute closure

目标：在任何正式训练前证明数据、evaluator、模型包装、loss 与算力可用。

必须完成：

1. 数据 license/revision/lineage、family split、污染审计；
2. 100–300 verifier adversarial tests，盲化人工抽查一致率≥99%；
3. E2B/E4B tokenizer/hash/token-ID、LoRA target、非文本 zero-grad/checksum；
4. masked CE、固定合成批次 GRPO gradient、full-vocab reverse-KL value/limit/gradient oracle；
5. no-vLLM 与 vLLM 两步 GRPO、weight-sync age、skipped-group 语义；
6. E2B backward、group-8 rollout、E4B SFT、E2B+E4B OPD 各 100-step steady-state profile；
7. 用 profile 重算 campaign cost，预留 30%。

Gate：任一 correctness test 失败则停止；若 2×80 GB/4×48 GB 候选资源不可用且预算不闭合，只能同步降低所有臂 `U`/cap 后重新 profile。

### Block E1 — Same-lineage anchors

目标：建立唯一 Student parent 与独立合格的 Teacher。

流程：64-example overfit → 2k sanity → 10k `D_anchor`。E2B/E4B 各自在预注册两个 LR 中只用 `D_select` 选择。锁定后只解封一次 `D_teacher_gate`。

Teacher 必须同时满足：相对 Student accuracy paired 95% CI lower>0、点估计≥+5pp、verified-solution NLL 更低、parse rate 不下降、tokenizer 完全一致。失败则 primary OPD 和 A2/A3/A4 停止；不得回到 gate 数据调参，也不得静默换 E4B-it。

### Block E2 — Single-signal interventions

目标：估计 C1a/C1b。

先在 `D_dev` 对每个 objective 运行最多两个 LR 的单 seed stability pilot；选择规则仅依据 finite loss、无 NaN/OOM、梯度/entropy/clip 等预注册健康区间，不读正式 test。随后 A0/A1/A2 全部运行三个 seeds、两个 stages，不能按初步结果删 arm。

GRPO stop gate：effective-group rate<30%、truncation>5%、weight-sync age>1 batch、independent accuracy 与 train reward 明显反向时停止扩展并记录失败。OPD stop gate：Teacher/Student hash 漂移、KL oracle mismatch、非文本梯度、OOM 或 exact-kernel throughput 令 campaign 不闭合。

### Block E3 — Order intervention

目标：估计 C2。

A3/A4 使用与 E2 完全相同的 OPD/GRPO config hashes，各跑三个 paired seeds。Stage 1 与 A1/A2 中点形成机制诊断；Stage 2 后做唯一确认性 `A3−A4`。不得针对顺序臂重新调 LR、cap、reward 或 mask。

### Block E4 — Frozen evaluation、failure analysis 与小型负例

对 Base、Student anchor、15 个 A0–A4 endpoints 以及必要的 stage-1 checkpoints 使用同一 generation/evaluator revision。确认性 endpoint 只用 MATH-500；GSM8K、MathArena 06/2026、AIME 2026、IFEval、MMLU-Pro 和 sealed 200 题均标 supporting/diagnostic。

核心五臂完成后，最多运行一个小型 format-reward/宽-parser 负例以展示 reward hacking；DPO shadow、其他 KL、larger Teacher、QLoRA、PRM 和 Agent RL 都属于后续，不得挤占主矩阵。

## 4. Evaluation and inference

- Greedy 主评测：所有 checkpoint 固定模板、stop、max-new-tokens 和 evaluator；
- Sampling：每题恰好 n=8，T=0.7、top-p=.95、top-k=0，generation seeds 成对；`pass@k=1-C(n-c,k)/C(n,k)`；
- C1：对每个 item 把三个预注册 seed 的 paired correctness 差取均值；在 MATH level 内只重采样 item 10,000 次，item 携带完整 seed vector；100,000 次 item-level paired randomization，两个 p-value 做 Holm；
- C2：同一 item unit；superiority 与 TOST equivalence 不混用；
- 单列每个 training-seed effect；方向不一致必须写 seed instability；
- IFEval 以 A0 为 reference，-2pp 非劣 margin；
- 旧公开集只证明本项目 post-training data 已去污染；MathArena 是 freshness sentinel，不替代 MATH-500 主终点。

## 5. Budget and reporting

E1 只匹配 Student backward loss tokens，prompt exposure 与 Student FLOPs只审计。E2 记录 rollout tokens、Student/old/reference/Teacher forward、Student backward、GPU-hours、accelerator-seconds、峰值显存与能耗。

成本拆为：共享 `C_anchor`、共享 `C_teacher`、逐臂 `C_arm`。结果同时报告：

- warm-start/marginal：`C_arm`；
- cold-start：非 OPD 为 `C_anchor+C_arm`，含 OPD 为 `C_anchor+C_teacher+C_arm`；
- campaign：`C_anchor+C_teacher+ΣC_arm`，Teacher 构建只计一次。

每个结果必须带 run/config/model/data/evaluator/git hashes、raw generation 索引、失败状态和成本。未经 claim audit 的数字不得进入 README 或简历。

## 6. Go/no-go checklist

- [ ] 可用硬件、总时长和 30% buffer 已确认；
- [ ] 模型许可、数据许可、公开边界与所有 immutable revisions 已锁定；
- [ ] D_select 与 D_teacher_gate 隔离，后者尚未被选择流程读取；
- [ ] evaluator、loss、mask、U counter、统计实现通过 synthetic tests；
- [ ] LoRA target/non-text freeze/tokenizer/weight-sync gates 通过；
- [ ] Teacher 独立 capability gate 通过；
- [ ] 15 个主 run 和完整评测的 campaign budget 闭合；
- [ ] 所有 objective config hashes、三个 seeds 和 formal run order 已在结果出现前冻结。

只有全部勾选，execution status 才能从 CONDITIONAL 改为 READY。
