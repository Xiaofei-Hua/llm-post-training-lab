# Benchmark 与评测协议

## 评测目标

评测需要同时回答三件事：能力是否提高、提高是否来自正确机制、是否产生了副作用。

## Benchmark 分组

| 组 | Benchmark | 主指标 | 角色 |
|---|---|---|---|
| Confirmatory Primary | MATH-500 | greedy answer accuracy | C1/C2 唯一确认性 endpoint |
| Secondary | GSM8K test | exact/symbolic accuracy | 分布外补充，不参与多重检验主结论 |
| Freshness Sentinel | MathArena ArXivMath 06/2026 | answer accuracy、题目级结果 | 模型发布后低污染 sanity；小样本/floor 风险 |
| Hard | AIME 2026 | pass@1、pass@8、题目级结果 | 较新高难探索性 |
| Retention | IFEval | prompt-/instruction-level strict accuracy | 指令遵循保持 |
| Retention | MMLU-Pro | accuracy；开发期 1,200 条分层子集，最终跑 full | 通用能力遗忘检查 |
| Diagnostic | 自建 sealed 200 题 | accuracy + error taxonomy | reward 与 parser 诊断 |

MathArena 06/2026 只有 49 题，不能独自承载主 claim；G0 先确认 E2B/E4B 不处于全零 floor，parser 无法判定的输出做对 checkpoint 身份盲化的人工复核。数据 revision/license 必须通过 G1。MMLU-Pro 的 1,200 条开发子集按 category/difficulty 分层并在任何结果出现前冻结；最终候选跑 full set。

## 固定推理协议

### Deterministic

- greedy（不采样）；
- 固定 system prompt、chat template、max new tokens 与 stop tokens；
- 用于 pass@1 主表。

### Sampling

- `temperature=0.7`、`top_p=0.95`、`top_k=0`；
- 每题恰好生成 `n=8` 个样本，generation seeds 在 checkpoint 间成对；
- 用于 pass@8、reward 方差和探索性分析。

所有 checkpoint 共用完全相同的 generation config 和 evaluator commit。

## 指标层次

### 主能力

- answer accuracy；
- pass@1 / pass@k；
- 分难度、题型、来源的 slice accuracy。

### 训练行为

- mean/std reward；
- non-zero-advantage group rate；
- policy entropy、approx KL、clip fraction；
- completion length、truncation rate、format validity。

### 副作用

- IFEval retention；
- 相对 Base/SFT 的通用能力变化；
- 重复、过长、多个最终答案、不可解析输出比例；
- 训练 reward 与独立 evaluator accuracy 的相关性和偏差。

## 预注册统计协议

### Estimand 与重复

- A0–A4 全部运行 3 个事先写入 manifest 的 paired training seeds；相同 seed 共享 data-order stream，但每个 arm 的 rollout stream 独立且可复现。
- 唯一确认性 endpoint 是 MATH-500 greedy item-level accuracy。推断**条件于这三个预注册 training seeds**：对每个 item 先取三 seed 的 arm-pair correctness 差均值 `d_i=(1/3)Σ_s(Y_a,s,i−Y_b,s,i)`，再令 `Δ_hat=(1/500)Σ_i d_i`。它不声称推广到所有可能的 training seeds。不得把 GSM8K/MathArena/AIME pooled 进 composite。
- 不把 `D_dev` 调参结果并入 test。单 seed pilot 仅按预注册的稳定性/损失 gate 选 objective 配置，不得按 test 表现删除 A0–A4 中任何 arm。

### C1：单信号贡献

- 两个确认性 contrasts：`A1−A0`（GRPO）与 `A2−A0`（OPD）。
- bootstrap 只重采样 item：在每个 MATH level 内有放回抽取原层同数目的 item IDs，每个抽中 item 携带完整的三-seed prediction vectors，重复 10,000 次并报告 percentile 95% CI 与 bootstrap RNG seed。
- 每个 contrast 的 p-value 来自 100,000 次 item-level paired randomization：对每个 item 以 0.5 概率交换 arm/control 的完整三-seed prediction vectors，等价于对 `d_i` 翻转符号；两个 p-values 使用 Holm step-down 控制 family-wise `α=0.05`。
- 单独报告三个 paired-seed effects；若方向不一致，主表仍按预注册 estimand 报告，但结论必须标注 seed-instability。
- 统计 superiority 要求 Holm-adjusted `p<0.05` 且 95% CI 下界 >0；项目层 practical success 还要求点估计至少 +2pp。

### C2：顺序效应

- 唯一确认性 order contrast 为 `A3−A4`；使用同一个条件于三 seeds 的 item-level estimator、bootstrap 和 randomization unit。
- superiority 要求 95% CI 不含 0 且绝对点估计至少 2pp。
- 若 superiority 不成立，只在同一 item-bootstrap estimand 的 TOST 90% CI **完全落入 [-2pp,+2pp]** 时声明 practical equivalence；点估计落入区间不够。
- A3/A4 与 A0/A1/A2 的其他差异为 exploratory，明确标注且不反推主假设。

### pass@8 与小样本

- 每题固定 `n=8`；给定其中 `c` 个正确，使用标准无偏估计 `pass@k = 1 - C(n-c,k)/C(n,k)`。本项目 `k=8`，同时公布 8 个原始 correctness，禁止跨 checkpoint 改采样次数。
- AIME/MathArena 报正确题号、题目级成功率和区间，不让 30/49 题的小集单独决定 claim。
- MATH-500、GSM8K 等公开旧 benchmark 只证明本项目后训练数据已去污染，不能证明 Gemma 4 预训练从未见过它们。

## 初始成功阈值

这些是 go/no-go 标准，不是预言：

- SFT 必须稳定超过 Base，且 evaluator 人工抽查一致率 ≥99%。
- 新算法在固定 Student loss-token 预算下相对 A0 的 MATH-500 绝对提升至少 2pp，且满足上面的确认性统计条件。
- IFEval retention 以 A0 为 reference、-2pp 为非劣 margin；仅当 paired 90% CI 下界高于 -2pp 时称“未观察到实质退化”。
- 若 accuracy 无提升，但明显降低输出长度/成本，可作为 supporting result，不能替代主 claim。
- trainer reward 上涨但独立 accuracy 不涨，判定为 reward alignment failure，停止扩大训练。

## 防止 Benchmark 污染与评测投机

- test 文本哈希只能由 data audit 读取，训练器不能读取 reference solution。
- SFT、GRPO、OPD 共用唯一 canonical prompt registry，并按来源/题型/template family 分组切分。
- 题目、参考解答和 reasoning trace 都做规范化近重复检查，不只查 prompt exact match。
- 只能声称“本项目后训练数据未包含测试集”；无法证明 Gemma 4 预训练或 Teacher 从未见过公开题目。
- answer parser 先在合成 edge cases 上冻结；模型训练后不得放宽规则。
- 对多个 `boxed`、复制题面答案、矛盾答案和异常 LaTeX 设置明确规则。
- 随机抽查至少 100 个模型输出，并给 evaluator 误差矩阵。
- 每次 benchmark 运行保存原始 generations，主报告只消费不可变结果文件。

## 主表草案

| Variant | Stage 1/2 Student loss tokens | MATH-500 Δ vs A0 [95% CI] | Holm p | GSM8K | AIME pass@1/8 | IFEval Δ [90% CI] | Avg length |
|---|---:|---:|---:|---:|---:|---:|---:|
| Base | 0 | descriptive | — | TBD | TBD | TBD | TBD |
| SFT anchor | pre-stage | descriptive | — | TBD | TBD | TBD | TBD |
| A0 SFT→SFT | 2M/2M | reference | — | TBD | TBD | TBD | TBD |
| A1 GRPO→GRPO | 2M/2M | TBD | TBD | TBD | TBD | TBD | TBD |
| A2 OPD→OPD | 2M/2M | TBD | TBD | TBD | TBD | TBD | TBD |
| A3 OPD→GRPO | 2M/2M | exploratory vs A0 | — | TBD | TBD | TBD | TBD |
| A4 GRPO→OPD | 2M/2M | exploratory vs A0 | — | TBD | TBD | TBD | TBD |

另设 C2 order table 单独报告 `A3−A4` 的 95% CI 与 TOST 90% CI；另设 Practical Compute 表报告每臂总 forward/backward FLOPs、prompt exposure、rollout tokens、GPU-hours 与峰值显存，不与本表合并成“matched-token”。
