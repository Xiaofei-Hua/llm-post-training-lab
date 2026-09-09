# Benchmark 与评测协议

## 评测目标

评测围绕能力、学习机制、副作用和效率四类问题。主报告优先展示指标、曲线和错误解释；现有 D05–D08 负责必要的解析、隔离与统计支持。新增 runtime/性能指标属于 D10–D12 和后续 GPU 模块，当前没有相应模型实测。

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

MathArena 06/2026 只有 49 题，不能独自承载主 claim；真实 baseline/评测时报告是否存在全零 floor，不能据此调参。parser 无法判定的输出做对 checkpoint 身份盲化的人工复核。数据 revision/license 必须通过 G1。MMLU-Pro 的 1,200 条分层评测子集在结果出现前固定，仅作描述性评测、不用于开发选参；最终候选跑 full set。

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

## 已有指标支持

D07 已实现 item-level accuracy、组合式 pass@k、提取/解析率、长度与截断统计，并区分模型答错和运行失败；D08 提供 paired bootstrap、sign-flip、Holm 与 TOST。复用这些实现，详细接口见 `SEALED_EVALUATOR.md`、`PAIRED_STATISTICS.md`。当前只有 synthetic CPU 验证，真实 benchmark adapter、Base baseline 与人工抽查属于 D15。

## 技术指标与诊断用途

| 类别 | 指标与口径 | 用来判断什么 | 实施位置 |
|---|---|---|---|
| 主能力 | MATH-500 greedy accuracy、C1/C2 Δ 与 CI；三个 seed 单列 | 训练信号和顺序是否产生可重复效果 | D07/D08 → D23 |
| 探索性 | 固定 n=8 的 sampling pass@1/pass@8，与 greedy accuracy 分列 | 采样是否覆盖正确解，能力与解码是否混淆 | D07 → D23 |
| 稀疏反馈 | mean/std reward、有效组数/总组数、zero-variance 比例 | GRPO 的奖励是否提供足够梯度信号 | D10/D12 → D18/D21–D23 |
| 策略更新 | token entropy、ratio/clip fraction、梯度范数、Student-anchor KL | 策略坍缩、过大更新或漂移 | D10/D12 → D18/D21–D23 |
| 蒸馏 | 当前 prefix 上的 reverse-KL、Teacher verified-solution NLL、Teacher/Student 正误四象限 | Teacher 支持与错误迁移如何影响学习 | D10/D12 → D17/D19/D23 |
| 长度与格式 | completion mean/p95、truncation rate、parse rate、重复/多答案错误分类 | reward 是否主要来自格式或长度变化 | D07/D12 → D23 |
| 保持能力 | IFEval strict accuracy 相对 A0 的 Δ；MMLU-Pro Δ | 数学训练是否损害其他能力 | D23 |
| 计算效率 | update 与端到端 effective tokens/s、rollout tokens/s、step p50/p95、峰值内存 | 优化影响局部计算还是完整训练 | D11 → D18–D20 |
| 实用效率 | accuracy–GPU-hours、accuracy–生成长度；E1/E2 与三类成本 | 更好效果需要付出多少资源 | D20/D23 |

训练曲线同时按已执行 Student loss tokens 与 wall time 作横轴，标出阶段切换；dev 诊断和最终 test 结果分开。CE、GRPO surrogate 和 KL 的原始 loss 数值不能直接跨算法比较。entropy/KL 记录计算的 prefix、mask、归一方式，以及 full-distribution 或 sampled estimator；不把不同估计器混为同一指标。额外诊断 forward 开销计入总成本。

Teacher NLL 在允许读取 reference 的诊断/evaluator 流程计算，不向 OPD 训练提供 gold trace。四象限、难度、题型、长度和来源 slices 用来解释结果，报告样本数，不把事后 slice 变成确认性发现。

性能对照固定 workload、设备和精度，报告测量波动；CPU RSS、GPU allocated/reserved 与理论内存估算分别呈现。完整测量方法见 `docs/planning/PERFORMANCE_PLAN.md`。

## 预注册统计协议

### Estimand 与重复

- A0–A4 全部运行 3 个事先写入 manifest 的 paired training seeds；相同 seed 共享 data-order stream，但每个 arm 的 rollout stream 独立且可复现。
- 唯一确认性 endpoint 是 MATH-500 greedy item-level accuracy。推断**条件于这三个预注册 training seeds**：对每个 item 先取三 seed 的 arm-pair correctness 差均值 `d_i=(1/3)Σ_s(Y_a,s,i−Y_b,s,i)`，再令 `Δ_hat=(1/500)Σ_i d_i`。它不声称推广到所有可能的 training seeds。不得把 GSM8K/MathArena/AIME pooled 进 composite。
- 不把 `D_dev` 调参结果并入 test。单 seed pilot 仅按预注册的稳定性/损失 gate 选 objective 配置，不得按 test 表现删除 A0–A4 中任何 arm。

### C1：单信号贡献

- 两个确认性 contrasts：`A1−A0`（GRPO）与 `A2−A0`（OPD）。
- bootstrap 只重采样 item：在每个 MATH level 内有放回抽取原层同数目的 item IDs，每个抽中 item 携带完整的三-seed prediction vectors，重复 10,000 次并报告 percentile 95% CI 与 bootstrap RNG seed。
- 每个 contrast 的 p-value 来自 100,000 次 item-level paired randomization：对每个 item 以 0.5 概率交换 arm/control 的完整三-seed prediction vectors，等价于对 `d_i` 翻转符号；C1 的预注册方向为 improvement，使用 one-sided `greater`，两个 p-values 使用 Holm step-down 控制 family-wise `α=0.05`。
- 单独报告三个 paired-seed effects；若方向不一致，主表仍按预注册 estimand 报告，但结论必须标注 seed-instability。
- 统计 superiority 要求 Holm-adjusted `p<0.05` 且 95% CI 下界 >0；项目层 practical success 还要求点估计至少 +2pp。

### C2：顺序效应

- 唯一确认性 order contrast 为 `A3−A4`；使用同一个条件于三 seeds 的 item-level estimator、bootstrap 和 randomization unit。
- superiority 要求 95% CI 不含 0 且绝对点估计至少 2pp。
- 若 superiority 不成立，只在同一 item-bootstrap estimand 的 TOST 90% CI **完全落入 [-2pp,+2pp]** 时声明 practical equivalence；点估计落入区间不够。
- A3/A4 与 A0/A1/A2 的其他差异为 exploratory，明确标注且不反推主假设。

所有 bootstrap/randomization base seeds 在结果出现前冻结；派生流只能绑定 protocol、hypothesis、operation 与不含 correctness/report/checkpoint 的 benchmark/resampling identity，不能由观测结果或完整 result panel hash 改变。完整 panel hash单独绑定统计输出。

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

## 最小评测边界

- 正式 test 只评测；训练、配置选择与 Teacher qualification 按既有 split 隔离。
- 复用 D06 对题目、解答和 trace 的 family split 与去污染，真实数据在 D15 检查，不新增治理系统。
- 复用 D05 parser/verifier，不在看到模型结果后放宽答案规则；错误/不可解析 prediction 计 0，reference/backend 失败单独处理。
- D15 至少盲审 100 个真实输出并报告 evaluator 误差矩阵，目标一致率 ≥99%。
- 保存原始 generation、配置/版本和结果入口，现有工具自动生成的校验信息直接引用。
- 只描述本项目后训练数据的污染检查范围，不声称 Base 预训练未见过公开题目。

实现细节集中在 `docs/data/DATA_REGISTRY_AND_CONTAMINATION.md` 与 `docs/evaluation/SEALED_EVALUATOR.md`，主分析围绕上述能力、动态和效率指标。

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
