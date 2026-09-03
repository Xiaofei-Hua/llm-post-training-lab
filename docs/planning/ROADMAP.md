# 12 周执行路线

## Phase 0：边界与基础（第 1 周）

- 确认 GPU、总时长、公开仓库边界。
- 锁定 Python/框架/model/dataset revisions。
- 推导 CE、KL、DPO、GRPO、GKD 公式，建立算法笔记模板。
- 完成数据和 evaluator schema。
- Gate：G0 通过。

## Phase 1：数据与评测可信度（第 2 周）

- 建数据 registry、quality funnel 和 contamination audit。
- 实现 answer parser、symbolic verifier 和 100–300 条 adversarial 单测。
- 冻结 benchmark prompt/template/generation configs。
- 运行 Base 全量 baseline。
- Gate：G1 通过，evaluator 抽查一致率达到 99%。

## Phase 2：同源 Student/Teacher SFT（第 3–4 周）

- E2B/E4B 各做 64 条过拟合、2k sanity、10k `D_anchor` main。
- 两者共享 immutable data、template、mask、样本顺序、max length、LoRA module classes 和 token/epoch budget。
- 只在各自预注册两档学习率中用 `D_select` 选 checkpoint；冻结 E2B Student anchor 与 E4B Teacher。
- checkpoint/config 全部冻结后只解封一次 `D_teacher_gate`；Student SFT 稳定优于 Base，Teacher 通过 +5pp/CI/NLL/parse-rate gate。

## Phase 3：预算校准与 SFT anchor 冻结（第 5 周）

- 分别测 E2B LoRA backward、group-8 rollout、E4B SFT 与 E2B+E4B full-vocab divergence。
- 冻结 canonical `D_core` sampling cycle、GRPO/OPD 合同、objective config hashes 与精确 `U` counter；每阶段配额为 2M。
- 按 `C_anchor/C_teacher/C_arm` 重算 marginal、cold-start、campaign 成本。
- Gate：预算能覆盖 A0–A4 全部 3 paired seeds 并保留 30% 余量，否则等比例缩小 `U` 后重新 profile。

## Phase 4：Objective pilot 与 reward audit（第 6–7 周）

- 只用 `D_dev` 做每个 objective 最多两档学习率的单 seed pilot；不得淘汰结果臂。
- GRPO 固定 exact reward 与完整合同；运行 A0/A1 第一阶段 smoke。
- format/过宽 parser 只做一次小规模 reward-hacking 负例。
- 监控有效组率、entropy、KL、clip fraction、长度与独立准确率。
- Gate：G3 通过；trainer reward 与独立 evaluator 一致。

## Phase 5：OPD 正确性与两阶段主运行（第 8–9 周）

- 验证 Teacher/Student vocab 兼容。
- 主配置固定 full-vocab chunked reverse KL、temperature 1.0 与 fully on-policy。
- 先完成小型精确 full-logit 数值/梯度 oracle，再运行 A0–A4 的 stage 1，并在统一 2M 边界重置 optimizer/scheduler。
- Gate：G4 通过；loss 数值和梯度单测无误。

## Phase 6：五臂 Stage 2 与确认性评测（第 10 周）

- 完成 A0 SFT/SFT、A1 GRPO/GRPO、A2 OPD/OPD、A3 OPD/GRPO、A4 GRPO/OPD 的全部 3 paired seeds。
- 按唯一 E1 loss-token denominator 和 E2 完整成本分别比较；执行 C1 Holm 检验与 C2 superiority/equivalence 检验。
- 任一 seed 失败均按预注册 retry policy 处理，不能以预算不足为由只保留“好看”的臂。
- Gate：G5 通过。

## Phase 7：总结与求职材料（第 11–12 周）

- 生成主表、训练曲线、reward-vs-accuracy 和 error taxonomy。
- 写负结果：伪 CoT、reward gaming 或 Teacher capacity 限制中至少一个。
- 完成技术报告、README result cards、10 分钟讲稿和追问题库。
- 对简历每个数字做 claim audit。
- Gate：G6 通过。

## 延后队列

- ORPO 与 KTO；
- GSPO/TIS 与 policy mismatch；
- PRM、step-level reward 与 Agentic RL；
- DPO/ORPO offline shadow baseline（仅在 A0–A4 全部完成后）；
- Gemma 4 12B/31B Teacher、跨架构 portability 与多模态后训练；
- 4B 以上 Student scaling。

只有核心矩阵完成后才从该队列取任务。
