# 系统与模型架构规划

## 设计原则

训练基础设施只承担“保持实验可比”的职责。算法层变量与系统层变量必须分开记录。本项目研究 Gemma 4 的 text path 后训练，不声称保持其完整图像/音频能力。

## 聚焦后的数据流

```text
public sources → license/schema → quality → family-level decontamination
                                      │
                             canonical prompt registry
                                      │
             E2B Base → frozen Student SFT anchor checkpoint
                                      │
                    two stages × 2M Student loss tokens
                                      │
               A0 SFT/SFT · A1 GRPO/GRPO · A2 OPD/OPD
                         A3 OPD/GRPO · A4 GRPO/OPD
                                      │
                 frozen evaluator + dual-budget accounting

             E4B Base → same D_anchor SFT → frozen Teacher
                                      │
                              only OPD stages read it
```

DPO 使用同一 frozen rollout bank 做单 seed offline shadow baseline，不进入主五臂和主 claim。

## 模型角色

### Main Student

- `google/gemma-4-E2B`（pretrained/base checkpoint）。
- 2026 年发布；约 2.3B effective parameters、约 5.1B including embeddings。
- 35 层 text stack，262,144 vocabulary；hybrid local/global attention，local window 512。
- smoke 与 main 均使用 E2B：smoke 只减少样本、step 和 context，避免在小模型上选出不能迁移的超参。

### Primary Teacher

- `google/gemma-4-E4B`（pretrained/base checkpoint），约 4.5B effective、约 8B including embeddings；42 层，262,144 vocabulary。
- 使用与 E2B Student **完全相同的 immutable `D_anchor`、chat template、assistant-only mask、max length、样本顺序和 SFT token/epoch budget** 独立训练；两者可在同一个预注册的两档学习率 pilot 中分别选学习率。
- LoRA module classes、rank 32、alpha 64、dropout 0 相同；层数造成的可训练参数差异属于容量差异。
- 只允许 `D_select` 选择 Student/Teacher SFT checkpoint；不得查看 `D_teacher_gate`、`D_core` gold trace、`D_dev` 或 test。选定后合并或固定 adapter，保存 config/tokenizer/checkpoint hash；Teacher 只有通过独立 gate 后才能供所有 OPD 臂只读使用。
- 该设计仍不能声称“只差参数规模”，但排除了未知厂商 instruction/post-training recipe，核心结论限定为 **same-data-lineage capacity gap 下的 dense signal**。

### Stretch Teacher

- `google/gemma-4-E4B-it`、`google/gemma-4-12B` 或 `google/gemma-4-31B-it` 都只属于核心结果完成后的 sensitivity/stretch 实验。
- same-lineage E4B Teacher 不过 gate 时，主 OPD 问题停止；不得用 E4B-it 静默替换。若仍运行 E4B-it，必须命名为 external-post-training sensitivity。
- 12B Unified 的 architecture subtype 不同，只能称 Teacher/checkpoint transfer；不能称同架构规模消融。
- 普通生成 API 不提供受控 token distribution，不能用来做本项目定义的 OPD。

## Teacher Capability Gate

same-lineage E4B Teacher 必须在从未用于 checkpoint/config 选择、与其他 split 在 family 层面隔离且始终 sealed 的 `D_teacher_gate` 上同时满足：

1. paired bootstrap 95% CI 的 answer accuracy 差值下界大于 0；
2. 绝对准确率点估计至少高于 SFT Student 5 个百分点；
3. verified reference solution tokens 的平均 NLL 更低；
4. 输出可解析率不低于 Student；
5. tokenizer hash 和 token IDs 完全一致。

Student-error coverage 只作诊断，不能替代 +5pp 门槛。不过 gate 时，先诊断模板、mask 与 decoding；仍不过则停止 primary OPD，不能悄悄换 Teacher 或测试集。

## Text-only 训练边界

- 输入不包含 image/audio tokens，禁用非文本预处理。
- 视觉与音频 encoder 全部 `requires_grad=False`；训练前后 checksum 相同且梯度始终为 `None`。
- 固定架构不等于冻结 language model：Student 的 text projection modules 通过 LoRA 更新，其他参数冻结。
- 首期 LoRA targets 限于 text attention 与 MLP projection；per-layer embeddings、token embeddings、LM head 和非文本模块默认冻结。准确模块名由 G0 模型 introspection 后锁定。
- chat template、thinking mode、special tokens、assistant loss mask 与 decoding 在所有分支一致。
- 不做多模态 retention，因此不能声称多模态能力保持。

## 五个核心训练臂

| Arm | Stage 1 | Stage 2 | Total Student budget |
|---|---|---|---:|
| A0 | SFT 2M | SFT 2M | 4M loss tokens |
| A1 | GRPO 2M | GRPO 2M | 4M loss tokens |
| A2 | OPD 2M | OPD 2M | 4M loss tokens |
| A3 | OPD 2M | GRPO 2M | 4M loss tokens |
| A4 | GRPO 2M | OPD 2M | 4M loss tokens |

所有臂均从同一个 frozen Student anchor 开始，并在 2M 边界执行相同的 optimizer/scheduler reset；阶段中点和终点均保存 checkpoint。这样 `A3−A4` 是唯一确认性顺序对照，且不会混入 stage count 或 scheduler restart。A0–A4 全部使用 3 个 paired seeds；单 seed pilot 只能选每个 objective 的学习率/实现配置，不得按结果删臂。

同一 objective 在所有位置复用完全相同的 frozen config hash：例如 A1 stage 1/2、A3 stage 2、A4 stage 1 的 GRPO 必须有相同 LR、scheduler family、batch construction、cap、mask 和 stopping semantics；OPD/SFT 同理。不同 stage 只能改变 parent checkpoint 和预注册 RNG seed。

## 双预算协议

### Estimand 1：Signal efficiency（主因果表）

唯一主匹配量为：每阶段 2M、每臂 4M **Student non-padding backward loss tokens `U`**。

精确定义：`U` 是 objective mask 为 1、实际进入一次**已执行** optimizer update 的 Student loss positions 数，每个 position 每次 update 计一次。completion 包含生成的首 token 至第一个 EOS（含 EOS）；prompt、padding、EOS 后 token、纯 rollout forward token与被跳过的 GRPO zero-variance groups均不计。SFT 只计 assistant target positions，OPD 只计有 KL 的 Student-generated completion positions，GRPO 只计实际参与 policy loss 的 completion positions。

为了精确停在 2M，最后一个 update 对有效 positions 按 `(sample_id, generation_index, token_index)` 稳定排序，只保留填满剩余 `U` 的前缀 mask；reward/advantage 可由完整 group 计算，但被 budget mask 掉的位置不进入 policy loss。三个 objective 使用同一规则，禁止用“最多多一个 batch”的近似匹配。

各臂从同一 canonical registry、同一分层 sampling distribution 与预冻结循环顺序取 prompt，但 completion length 和 group multiplicity 会令 prompt exposure 数不同，因此只报告、不声称严格匹配。Student forward/backward FLOPs 也只审计、不称 matched。E1 回答“固定 Student 接收梯度的 token 数时，不同信号如何改变行为”，不代表 prompt 数、Student FLOPs或总成本相同。

### Estimand 2：Practical efficiency（Pareto 图）

完整记录：

- Student/old/reference/Teacher forward FLOPs；
- rollout tokens 与 rejected/dropped groups；
- Student backward FLOPs；
- accelerator-seconds、GPU-hours、峰值显存与能耗（可取得时）。

它回答“实际付出多少资源得到什么结果”。两个 estimand 分表报告，禁止合称 matched-token 或 matched-compute。

### 一次性与边际成本

- `C_anchor`：共享 E2B Base→SFT anchor；
- `C_teacher`：共享 E4B Base→same-lineage SFT Teacher；
- `C_arm`：单个 post-anchor arm，包括该 arm 的 rollout、Teacher/reference forward 和 Student update。

E2 同时报 warm-start/marginal（`C_arm`）、cold-start pipeline（非 OPD 为 `C_anchor+C_arm`，含 OPD 的 arm 为 `C_anchor+C_teacher+C_arm`）和 campaign total（`C_anchor+C_teacher+ΣC_arm`）。`C_teacher` 不能隐藏，也不能在三个含 OPD 的 arm 中重复计三次。

## OPD 合同

- Student 从当前策略生成，采样结果 stop-gradient。
- `num_iterations=1`：每次 optimizer update 前都由当前 Student 新生成 completion，不复用旧 rollout。
- Teacher 在相同 Student prefix 上输出分布，Teacher 永久冻结。
- 主 loss：full-vocabulary chunked reverse KL `KL(π_student || π_teacher)`，temperature 1.0，fully on-policy。
- KL mask 只覆盖 Student-generated completion 的首 token 至第一个 EOS（含 EOS）；prompt、padding 与 EOS 后位置为 0。
- 先在每个有效 token 上对完整 vocabulary 求 reverse KL，再把全 batch 所有有效 token 的 KL 求和并除以有效 token 总数；禁止先做 sequence-equal averaging。
- 不持久化 `[batch, length, vocab]` full logits；使用 chunked/fused divergence。
- top-k 仅在 full-vocab 数值等价性小测通过后作为单独近似 variant，不进入首个主结果。
- tokenizer/vocab 不兼容时，本 arm 停止；sequence KD 必须另命名。

## GRPO 合同

- 主 reward：独立 exact/symbolic correctness，正确 1、错误/不可解析 0；无格式奖励、长度奖励或 learned reward model。
- 固定 `group_size=8`、`loss_type=dr_grpo`、`epsilon=0.2`、`beta=0.0`、`num_iterations=1`、`importance_sampling_level=token`、`scale_rewards=group`、`temperature=1.0`。
- `beta=0` 表示 reference KL 不进入优化目标；在固定诊断 prompt 上仍记录 Student-anchor KL，防止无约束漂移被遗漏。
- old policy 每个 generation batch 刷新一次；rollout/training weights 每个 generation batch 同步一次；`steps_per_generation=1` 或版本中的严格等价设置。
- prompt/data RNG 与 rollout RNG 为独立、可复现的 stream；generation backend 在 C2 后冻结，所有正式臂一致。
- 零 reward variance group 不更新、不重采样并计入日志；effective-group rate 低于 30% 时停止正式扩展并回到数据/难度 gate。
- `max_completion_length` 在 G0 按预注册规则选择：2048 tokens；若冻结 pilot 的 truncation rate >5%，所有臂统一改为 4096。正式 run 后不得再改。
- 监控 entropy、diagnostic KL、clip fraction、effective-group rate、长度、truncation、weight-sync lag 与独立 accuracy。

## 计划依赖

- PyTorch、Transformers、Datasets、Accelerate、PEFT；
- TRL：SFTTrainer、GRPOTrainer、DistillationTrainer；DPOTrainer 仅 shadow baseline；
- vLLM：rollout backend，不作为算法贡献；
- math-verify、lm-evaluation-harness、MathArena；
- W&B 或本地 JSONL：实验追踪。

实现阶段必须锁定版本与 commit，并通过 `docs/planning/COMPATIBILITY_GATES.md`。
