# 数据规划

> 当前状态：D06 已完成可复用的数据 trust-stack 合同和 synthetic adversarial evidence；尚未下载或冻结任何真实训练/benchmark 数据，`DATA-001`、`DATA-002` 与 G1 均保持 `NOT_STARTED`，生产 materialization 属于 D15。

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

## Canonical registries 与 record schema

许可不重复写入每条样本，而由 closed-world `SourceRegistry` 统一冻结。每个 source 必须记录 portable `source_id`、公开 URI、40/64-hex Git commit 或 SHA-256 snapshot revision、SPDX-like license expression、license/card URL 与 evidence hash，以及 `train`/`evaluate`/`redistribute` 用途 allowlist。记录只引用 `source_id` 与完全一致的 revision；用途不允许、未知或 mutable 的来源直接失败。

每个处理步骤还必须出现在 `TransformRegistry`，其中 `(transform_name, transform_version, code_sha256, config_sha256)` 指向 repository-relative、Git-tracked 的代码和配置。正式 audit 会逐字节校验声明 hash。

所有样本使用严格的 `d06-data-record-v1` JSONL：

```json
{
  "schema_version": "d06-data-record-v1",
  "sample_id": "source-local:stable-id",
  "source_id": "open-r1/OpenR1-Math-220k",
  "source_revision": "<full immutable commit>",
  "split": "UNASSIGNED",
  "families": {
    "source": "<source-local family>",
    "problem": "<problem family>",
    "template": "<template family>"
  },
  "problem": "...",
  "messages": [{"role": "user", "content": "..."}],
  "reference_answer": "...",
  "response": "...",
  "quality": {"answer_verified": true, "format_valid": true},
  "strata": {"answer_type": "integer", "difficulty": "..."},
  "lineage": {
    "transform_name": "normalize-v1",
    "transform_version": "1.0.0",
    "code_sha256": "<64 hex>",
    "config_sha256": "<64 hex>",
    "parents": [{"sample_id": "...", "payload_sha256": "<64 hex>"}]
  }
}
```

内部 parent 必须解析到同 split 的完整 `payload_sha256`；外部 parent 必须由冻结的 `ParentPayloadLedger` 解析。loader 拒绝未知/缺失字段、duplicate keys、NaN/Infinity、BOM/CRLF/blank lines、非 NFC 文本、重复 ID、lineage cycle 与越界输入。未来 preference 数据必须定义独立 versioned schema；不能向该 closed schema 临时追加 `chosen`/`rejected` 字段。

完整实现合同见 `DATA_REGISTRY_AND_CONTAMINATION.md`。

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

每一步都必须输出输入数、保留数、拒绝原因分布和样本审计。

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
