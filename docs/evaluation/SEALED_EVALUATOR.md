# D07 Sealed Benchmark Evaluator

## 状态与边界

D07 完成 CPU 上的 benchmark/generation/evaluation production contracts。它解决的是“如何让同一冻结评测协议可靠地消费不同 checkpoint 的输出”，不是下载或运行真实 benchmark，也不是统计推断模块。

- 已实现：public prompt 与 sealed answer 分权、冻结 greedy/sampling 协议、paired generation seed、逐 sample generation record、逐 sample/item score、accuracy/pass@k/extraction/parse 指标、evaluator version hash、Git-bound synthetic audit。
- 未实现：MATH-500/GSM8K/AIME/IFEval/MMLU-Pro 的真实 materialization 与官方 adapter、人审、Base baseline；这些属于 D15/G1。
- 下游已实现：paired bootstrap、randomization/sign-flip、Holm 与 TOST 位于独立 D08 模块；D07 evaluator 本身仍不包含统计推断。
- 未执行：模型下载、真实生成、MPS/CUDA/GPU 训练或推理。

因此，D07 的合成 accuracy/pass@k 只是一组冻结 oracle，用来发现 evaluator 回归；它不是 Gemma 4 的实验结果，也不能通过 G1。

## Deletion test

在 D07 前，已有 D05 answer verifier 和 D06 data manifest，但仍不能安全地产生 benchmark 数字：

1. 把 prompt 与 answer 放在同一 dataset object，会让生成 backend 获得 gold capability；仅靠“不使用该字段”的约定不可审计。
2. 只保存聚合 accuracy，无法定位 item/sample、重算 pass@k、做后续 paired statistics，也无法区分模型错误与 backend failure。
3. sampling seed 若依赖遍历顺序或 checkpoint，会破坏跨 checkpoint 的 paired generation。
4. 只记录调用方填写的 evaluator 名称/commit，不能证明 parser policy 和实际 backend version 没有变化。
5. D05 verifier 只定义单个 reference/prediction 的语义；它不负责 generation 完整性、协议绑定、sealed item set 或 result provenance。

最小充分方案因此必须同时有独立 public/sealed capability、完整 generation grid、冻结协议、item-level records 和可重算的 evaluator hash。D07 没有引入统计推断或真实 benchmark adapter，因为它们对上述最小缺口不是必需的。

## 进程与权限边界

```text
public descriptor + public prompts + checkpoint + frozen protocol
                              │
                              ▼
                     generation process
                    （永远不接收 vault）
                              │
                              ▼
             immutable generation records + manifest
                              │
                              ▼
sealed references ───► evaluator process ───► sanitized item/sample report
```

核心约束：

- `LoadedPublicBenchmark` 只含 prompt、strata 与不可变身份；没有 reference/label/gold 字段。
- `prepare_generation_requests()` 和 `GenerationBackend` 的类型面只接收 public benchmark、checkpoint 与 protocol。
- `SealedAnswerVault` 不实现 mapping、getter 或 serializer，repr 会隐藏 references；它只进入 evaluator 路径。
- `EvaluationReport` 保存 sealed file SHA-256，但不保存 reference、extracted candidate、prompt、generated text 或 token IDs。
- vault 是 API/进程 capability boundary，不是 Python 安全沙箱。正式运行还必须把 generator/evaluator 分进程，并用文件权限限制 sealed 文件；拿到 evaluator 进程调试权限的人仍属于 trusted computing base。

## Benchmark snapshot contract

`BenchmarkDescriptor` 使用 `d07-benchmark-descriptor-v1`，冻结：

| 字段 | 作用 |
|---|---|
| `benchmark_id/revision_kind/benchmark_revision/split_name` | 数据集不可变身份；revision 只能是完整 40/64 位 digest |
| `task` | `exact_math` 或最小 `strict_label` 语义 |
| `item_count` | public/sealed 两侧必须精确覆盖的题数 |
| `public_items_sha256` | 原始 canonical public JSONL bytes |
| `sealed_references_sha256` | 独立 sealed JSONL bytes |
| `source_registry_sha256/data_manifest_sha256` | 与 D06/D15 数据证据链连接的 opaque binders |

public JSONL 和 sealed JSONL 使用不同 schema，二者按 immutable benchmark identity 和完全相同的 item-ID set 对齐。加载器拒绝 duplicate/unknown fields、BOM/CR、blank line、非 UTF-8、非 NFC、超限记录、错误 raw hash、错序 index、重复或缺失 item。

测试 fixture 中 `source_registry_sha256` 与 `data_manifest_sha256` 是明确的 synthetic binders，不声称来源真实性。D15 必须用真实 D06 manifest 替换它们。

## Frozen generation protocol

`GenerationProtocol` 使用整数 ppm 保存采样参数，避免 JSON 浮点表示漂移：

| 模式 | 冻结字段 |
|---|---|
| greedy | `samples_per_item=1`；temperature/top-p/top-k/seed 全部必须为 null |
| sampling | `samples_per_item=n≥2`；temperature/top-p/top-k、seed namespace 与 base seed 全部必填 |
| 两者共有 | system prompt、chat-template SHA-256、max-new-tokens、EOS、stop token IDs、stop sequences |

正式计划的 sampling 值仍是 `n=8, temperature_ppm=700000, top_p_ppm=950000, top_k=0`。

每个 sampling seed 由下面的 canonical payload 派生：

```text
SHA256(schema, protocol_sha256, benchmark_id, benchmark_revision,
       item_id, sample_index)[:8] & (2^63-1)
```

base seed 和 namespace 已包含在 protocol digest 中。checkpoint identity 不进入 seed payload，所以相同 item/sample 在不同 checkpoint 间使用同一 seed；checkpoint SHA-256 进入 request ID，因此不同 checkpoint 的输出不会被误合并。

## Generation/result schema

每个 request/response 被合并成 `d07-generation-record-v1`，至少绑定：

- benchmark/revision/item/index/sample；
- checkpoint、protocol、prompt 与 request SHA-256；
- paired seed；
- completed/failed 状态；
- generated text、完整 output token IDs、finish reason 或 portable error code；
- record self-hash。

`GenerationBatch` 要求 `item_index × sample_index` 是完整、无重、canonical 的矩形网格。backend 可以乱序返回，但 request/response 必须一一对应，随后按冻结 request 顺序 canonicalize。任何 missing/extra/duplicate/non-response 都使整 batch 失败。

终止语义不只在生成时检查，反序列化后还会再次按 protocol 验证：

- `length` 必须恰好达到 `max_new_tokens`；
- `eos` 必须以冻结 EOS token 结束；
- `stop_token` 必须以 allowlist token 结束；
- `stop_sequence` 必须以冻结字符串结束；
- failed response 不得携带伪造 output，completed response 不得携带 error code。

JSONL records 与 sibling manifest 原子写入。manifest 同时绑定 descriptor、public item set、checkpoint、完整 protocol、raw records file hash、semantic record-set hash、记录数和 failure 数。加载时重建全部 public requests，不能仅靠调用方提供的 summary/hash 通过。

## Metric contract

D07 的主指标使用确定性整数 ppm（`1_000_000 = 100%`），内部先用 `Fraction` 保持精确有理数，再做非负 half-up rounding。

对一题的 `n` 个样本中有 `c` 个正确：

```text
pass@k = 1 - C(n-c, k) / C(n, k)
```

报告包含：

- `answer_accuracy_ppm`：所有 sample correctness 的均值；greedy 时等于 item accuracy；
- `pass_at_k_ppm`：先逐 item 用精确公式计算，再跨 item 平均，最后只 round 一次；
- `extraction_rate_ppm`、`parse_rate_ppm`、逐 sample completion-token count、总 completion tokens、finish-reason counts 与 truncation rate；
- 完整 verification status counts；
- 每个 sample 的 generation-record hash、correctness 与无原文状态；
- 每个 item 的原始 sample correctness vector、correct count 与 pass@k。

`EvaluatorContract` 冻结 task、primary metric、k 集合、D05 verifier policy hash 与 score scale。`evaluator_version_sha256` 进一步绑定 contract digest 和实际 backend versions；升级 parser、ANTLR/Math-Verify 或 strict-label policy 会得到不同 evaluator identity。

本模块的 `strict_label` 只是 ASCII label 的 trim + NFC + case-sensitive whole-surface exact adapter，用来验证通用接口。它不是 IFEval 官方 prompt-/instruction-level evaluator；D15 必须实现并验证各 benchmark 的正式 adapter。

## Failure semantics

| 情况 | D07 行为 | 能否计为模型错误 |
|---|---|---|
| prediction 错误/不可提取/不可解析/超时 | 保存 item-level status，correct=false | 是 |
| sealed reference 不可解析或 verifier backend error | `EvaluatorInfrastructureError`，阻断 batch | 否 |
| generation backend failed response | 原样保存在 generation records；evaluation 阻断 | 否 |
| public/sealed item set、descriptor、protocol、policy/hash 不一致 | fail closed | 否 |
| report/manifest/record 被改动 | self-hash 或交叉绑定失败 | 否 |

这继承 D05 的关键原则：gold 或基础设施故障绝不能伪装成模型负样本。

## Git-bound synthetic audit

`scripts/audit_evaluator.py` 在一个捕获的 Git revision 上验证完整路径：

1. 断言实际执行的 `__main__` CLI 与全部已加载项目模块的 resolved source 都来自被审计 repository root，拒绝“审计 checkout A、执行 checkout B”；
2. 验证 implementation、CLI、Python/依赖配置和全部 fixture 均被 Git 跟踪；
3. 用 `git show <revision>:<path>` 比较每个工作区文件，拒绝 dirty/untracked bytes；
4. 加载 public/sealed snapshot、greedy/sampling protocol 和 synthetic predictions，并将每个 loader 实际消费的 raw digest 立即与捕获的 provenance digest 交叉绑定；
5. 故意逆序返回 backend responses，验证 canonicalization；
6. 通过 D05 canonical verifier 生成 sanitized item/sample reports；
7. 与冻结 expectation 比较 protocol、record set、contract、evaluator version、report 与全部指标 hashes；
8. audit 末尾重读所有输入并再次检查 HEAD，阻断 restore-after-read TOCTOU 与 HEAD race。

冻结 synthetic oracle 是：greedy 3/6，sampling 21/48，sampling pass@8 5/6；两个模式各有一个 `length` finish，用于把非零 truncation 路径纳入 Git-bound oracle。它们只证明实现能重现人为构造的 match/mismatch/no-extraction/truncation 情况。fixture 的 output token IDs 是有界、确定性的 synthetic stand-ins；真实 backend 必须写入真实 tokenizer token IDs，并在 D14/D15 验证。

正式 audit 输出只含 hashes、版本、计数与 metrics，不含 prompt、prediction 或 reference 原文。`artifacts/audits/D07_EVALUATOR_AUDIT.json` 已在 implementation commit `c948fe2eae50289b78513a3a9513e188bff54a98` 上生成：`passed=true`、0 failures、tracked inputs 与 Git 一致；canonical report SHA-256 为 `a6c1bed7e74fcc9bf3448fa095535237c27f183cd67938799d9d96bc40b8abf4`，pretty JSON file SHA-256 为 `4868cbf3b9572a141659f7141b3bf17fc6b9fc0b3f6bd03952e9f90d9fec8654`。

## 最小 API

```python
from posttrain_lab.evaluation import (
    EvaluatorContract,
    evaluate_generation_batch,
    load_benchmark_descriptor,
    load_public_benchmark,
    load_sealed_answer_vault,
    run_generation,
)

descriptor = load_benchmark_descriptor("descriptor.json")
public = load_public_benchmark(descriptor, "public_items.jsonl")

# generator process: backend 只收到 public GenerationRequest
batch = run_generation(
    public,
    run_id="EV-END-A1-S101",
    checkpoint=checkpoint,
    protocol=protocol,
    backend=backend,
)

# evaluator process: 只有这里加载 sealed answers
vault = load_sealed_answer_vault(public, "sealed_references.jsonl")
report = evaluate_generation_batch(
    public,
    vault,
    batch,
    evaluation_run_id="EV-END-A1-S101-EVAL",
    contract=EvaluatorContract(
        task=descriptor.task,
        primary_metric="answer_accuracy",
        pass_at_k=(1,),
        verifier_policy_sha256=verifier.policy_digest,
    ),
    verifier=verifier,
)
```

生产部署必须额外保存 generation bundle 和 evaluation report，并由 D11 provenance ledger 注册；上面的最小片段不替代运行编排。

## 验证与后续 gate

D07 定向 70 个 CPU tests 覆盖 strict schema、answer capability boundary、paired seeds、乱序/缺失/重复 backend、finish/length/truncation semantics、bundle 路径别名与 leaf-symlink/report tamper、pass@k properties、gold/infra failure、strict numeric schema、strict label、完整 runtime import closure、foreign-checkout rejection、Git dirty/untracked/root/HEAD-race/restore-after-read TOCTOU 与 raw-text exclusion。全仓 506 tests 通过；新增两个 Hypothesis properties 各执行 100 个生成案例。

D07 交付本身只把当时的代码模块进度推进到 `7/24`。以下状态保持不变：

- `DATA-001/DATA-002/EVAL-002/EV-* = NOT_STARTED`；
- `real_data_materialized=false`；
- `G1=false`；
- `accelerator_execution_authorized=false`。

D08 只联合消费 D07 report 的 item-level correctness，以及由 report 内 benchmark/public-item hashes 精确绑定的 `LoadedPublicBenchmark` snapshot 中的 strata；不会重新解析生成原文、读取 sealed answer、改变 evaluator contract，或把 statistical inference 塞回 evaluator。实现与审计见 `docs/evaluation/PAIRED_STATISTICS.md`。
