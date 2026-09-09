# 数据规划

> 当前状态：D06 数据支持已完成 CPU 合成验证；真实数据尚未下载，DATA-001/002 与 G1 为 NOT_STARTED，实际数据接入属于 D15。后续重点是任务分布、有效学习信号与质量诊断，复用已有数据处理流程。

## 数据分层

| 层 | 用途 | 首选候选 | 是否含答案/轨迹 |
|---|---|---|---|
| D0 | evaluator 单测 | 手工合成 100–300 条边界样本 | 是 |
| D_anchor | E2B Student 与 E4B Teacher 的同源 SFT | 去污染后 10k verified traces | 完整轨迹 |
| D_core | 五臂正式 intervention registry | 2k canonical prompts | A0 可见轨迹；GRPO/OPD 仅见允许字段 |
| D_select | Student/Teacher SFT checkpoint selection | 500 个独立 verified examples | 可用于选择，不能报告 Teacher gate |
| D_teacher_gate | final Teacher qualification | 500 个始终 sealed、family-disjoint examples | 只供一次 capability/NLL/parse gate |
| D_dev | objective 配置 pilot | 500 个独立 prompts | 与 D_select、D_teacher_gate、D_core、test 隔离 |
| D_shadow | 核心完成后的 DPO | D_core 的 frozen rollout bank | chosen/rejected |
| E | sealed evaluation | MATH-500、GSM8K、MathArena 06/2026、AIME 2026、IFEval、MMLU-Pro | 只允许 evaluator 访问 |

## 推荐公开来源

### OpenR1-Math-220k

- 作为 SFT 候选源；其 reasoning traces 经 verifier/LLM judge 筛选。
- 首期不直接使用全部 220k；冻结 10k `D_anchor`、2k `D_core`、500 `D_select`、500 `D_teacher_gate` 和 500 `D_dev`。
- 数据源来自 NuminaMath，可能与数学 benchmark 重叠，必须先去污染。
- 固定 Hugging Face dataset revision，记录 Apache-2.0 license。

### GSM8K / MATH-500 / MathArena / AIME

- 只作正式/支持性评测，禁止用于 dev calibration、训练或 pair 构造。
- MathArena ArXivMath 06/2026 作为模型发布后的低污染 sanity，但仍需锁 revision 和 license。
- AIME 样本少，报告题目级结果和置信区间，不把 1–2 题差异夸大成稳定提升。

### IFEval

- 只作 instruction-following retention；数学强化不应以破坏基本指令遵循为代价。

## 数据对学习信号的影响

| 分析 | 需要观察的指标 | 算法用途 |
|---|---|---|
| 难度与答案类型 | 题型占比、Base/anchor accuracy、GRPO 有效组率 | 解释全对/全错组导致的无梯度与任务难度关系 |
| 解答长度与截断 | prompt/response token 分布、p95、cap 命中率 | 解释监督 token 分配、rollout 成本和长度偏差 |
| Teacher 支持 | Teacher 与 Student 的正误交叉、verified-solution NLL | 判断蒸馏信号是否覆盖 Student 的错误区域 |
| 轨迹质量 | verifier 正确率、格式可解析率、不完整/重复推理 | 区分正确答案、有效推理监督与格式改进 |
| 来源与泛化 | 来源/题型/family 的训练分布与评测 slice | 解释分布覆盖与收益适用边界 |

D15 输出这些分布的基础描述，模型相关部分在 D16–D19/D23 补齐。dev 上的诊断可指导已允许的 recipe 选择；正式训练后只做解释性 slice，不按结果筛选 D_core 或 test。数据过滤/curriculum 新实验需先有明确算法问题，不能顺带扩大矩阵。

## 复用现有数据接口

来源、license、revision、处理脚本与固定 splits 由 D06 registry/materializer 管理；已有 hashes 和 manifests 自动生成并引用。完整 schema、lineage 和 contamination 实现集中在 `DATA_REGISTRY_AND_CONTAMINATION.md`，本计划不再重复字段或增加新的校验层。

D_anchor/D_core 等规模与监督访问边界沿用下述设计。当前 tiny 模型实验只使用本地合成数据，不把它们混入真实训练数据。

## 数据质量漏斗

```text
license gate
→ schema validation
→ unicode / latex normalization
→ exact duplicate removal
→ benchmark contamination removal
→ answer verifier
→ response completeness / truncation check
→ length and difficulty stratification
→ immutable split and manifest
```

复用处理脚本输出样本数量和拒绝原因；重点查看处理是否改变难度、长度与监督质量，不额外建设报告系统。

## 去污染协议

1. 对题面做小写、Unicode、空白、LaTeX 与数字格式归一化。
2. 先做 exact hash；再做字符/词 n-gram MinHash 或相似检索。
3. 对高相似 pair 人工复核；保存 pair ID，不把 sealed 答案暴露给训练脚本。
4. 阈值在看模型结果前冻结。
5. 训练 manifest 保存被删除条数与 benchmark 版本。

## Split 原则

- `D_anchor`、`D_core`、`D_select`、`D_teacher_gate`、`D_dev` 在 source/problem/template family 层面互斥；五个核心臂都从同一个 `D_core` registry、相同分层 sampling distribution 和预冻结循环顺序取样。
- `D_select` 可反复用于 checkpoint selection；`D_teacher_gate` 在两个 SFT checkpoint 与配置全部锁定后才解封一次，任何失败都不得据此重选 checkpoint。若重选，必须创建新 gate split/experiment family。
- 因 completion length、group multiplicity 和 objective mask 不同，不强求 prompt exposure 次数相等；唯一严格匹配量是每阶段 2M Student backward loss tokens，实际 exposure 必须逐臂报告。
- 每个 arm 只能读取目标允许的监督字段，防止 OPD/GRPO 意外读取 gold reasoning。
- 难度、答案类型和来源分层抽样，避免算法组难度不一致。
- 按来源、题型与 template family 做 group split，禁止仅按单题随机切分。
- 正式 test 永久 sealed；超参只看独立 dev。
- 只在 anchor pilot 中允许 2k ⊂ 10k 的嵌套规模检查；30k 扩展不进入 must-run。

## 数据风险

- **伪 CoT**：不根据最终答案反推长推理作为默认 SFT 数据。
- **长度偏差**：pair 的 chosen/rejected 做长度匹配或在统计中控制。
- **Verifier false positive**：抽样人工审计，并用第二 evaluator 交叉验证。
- **License 漂移**：保存抓取时的 card 与 revision；不确定时不发布衍生数据。
- **内部泄漏**：禁止加入内部 prompt、业务标签和未公开模型输出。
