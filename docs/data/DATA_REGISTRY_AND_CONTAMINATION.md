# D06 Data Registry、Family Split 与 Contamination Gate

## 模块结论

D06 把“公开数据、固定 revision、去污染”从文字要求变成可执行的 CPU trust stack：来源/transform registry、canonical record、payload-addressed lineage、三维 family split、逐字段与聚合 exact/fuzzy contamination、family quarantine 和 immutable manifest 使用同一组冻结 schema 与 hash domain。

当前只完成框架和合成 adversarial fixture。没有下载 OpenR1、MATH-500 或其他真实数据，没有读取 sealed benchmark answer，也没有执行模型、MPS、CUDA 或 GPU。因此 DATA-001/002 的生产实现接口已经具备，但真实 materialization、license evidence 核验和 G1 仍留给 D15，不能把 D06 的合成通过写成“数据已去污染”。

## Deletion test

只保存一个 dataset 名称、seed 和随机 train/test split 不足以支持主 claim：

1. branch/tag 会移动，dataset card 与 license 也会变化；没有 revision 与 evidence hash，无法重建当时允许的用途；
2. 单题随机 split 会把同源改写、同一 problem family 或同一模板跨入 anchor/dev/gate/test；
3. 只对题面做 exact string match 会漏掉大小写、Unicode、LaTeX、空白、标点改写，以及 benchmark 题被嵌入 response/trace 的情况；
4. MinHash/LSH 是概率候选生成器，不能作为“阈值以上必不漏”的 correctness gate；
5. 删除单个命中行仍可能保留同 family 近邻；
6. manifest 若包含原文会暴露 sealed 内容，若只散列最终文件又无法定位 source/revision/lineage/split 发生了什么变化；
7. 在工作树有临时修改或 fixture 未被 Git 跟踪时生成 audit，会让同一个结果无法绑定到唯一实现。

因此 D06 保留窄而明确的 schema、complete candidate retrieval、transitive family quarantine、入口快照和逐层语义 hash。

## Source registry 与许可边界

`SourceRegistry` 是 closed-world allowlist。每个 `SourceDescriptor` 必须记录：

- portable `source_id` 与公开 `https://`/`hf://` URI；
- `git_commit` 的完整 40/64 hex revision，或完整 SHA-256 snapshot revision；
- SPDX-like `license_expression`、HTTPS license/card URL 和抓取内容的 SHA-256；
- canonical、无重复的 `train`、`evaluate`、`redistribute` usage allowlist。

记录进入任何 split 前会同时检查 source 存在、record revision 与 registry 一致，以及该 source 是否允许当前用途。未知 source、mutable revision、未知 license、evaluation-only source 被用于训练都会 fail closed。

这不是自动法律意见：D15 仍须保存真实 dataset card/license bytes、人工批准 allowlist，并校验 `license_evidence_sha256`。D06 只确保批准结果不能以缺字段或 mutable reference 进入后续 manifest。

## Canonical record 与 lineage

Schema version 为 `d06-data-record-v1`，核心结构如下：

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

Loader 拒绝 duplicate JSON keys、NaN/Infinity、UTF-8 BOM、CRLF、blank lines、未知/缺失字段、非 NFC 文本、NUL/surrogate、重复 sample ID 和显式资源上限之外的输入。

Hash domain 分离为：

- `content_sha256`：problem/messages/reference/response/quality；
- `lineage_sha256`：transform name/version/code/config/parents；
- `payload_sha256`：完整 canonical record，包括 split、families 和 lineage；
- `record_set_sha256`：按 sample ID 排序后的 `(sample_id, payload_sha256)` 序列。

内部 parent 存在时必须匹配其完整 `payload_sha256`，因此 source/revision/split/families/strata/lineage 的任何变化都会使 child link 失效；parent/child 也不得跨 split。外部 parent 必须由冻结的 `ParentPayloadLedger` 以 `(sample_id, split, payload_sha256)` 解析，缺失、hash 不符或跨 split 均失败。每个 transform 的 name/version/code/config hash 必须命中 `TransformRegistry`；正式 audit 还会把 registry 指向的 code/config 文件逐字节与捕获的 Git revision 比对。self-parent、DAG cycle 和把 derived record 重新送入 root split allocator 都直接失败。

## Family-disjoint split

`assign_family_disjoint_splits()` 先对 `source/problem/template` 三个维度建立 union-find。任何一维共享 family 的样本都会合并；这种关系继续传递，因此 `A` 与 `B` 共享 source family、`B` 与 `C` 共享 problem family 时，`A/B/C` 必须整体进入同一 split。

每个 component 的 canonical sample/family payload 与冻结 namespace 进入 SHA-256 bucket，再按 `SplitAllocation.weight` 选择 split。结果具备：

- 输入行顺序无关；
- component 不会被拆开；
- policy、assignment、counts 分别有 hash；
- policy namespace 或 weights 变化会产生新的 policy revision。

weights 是确定性比例，不承诺任意 family-size 组合都能恰好得到目标条数。D15 必须从足够大的公开候选池 materialize，并验证最终 `10k/2k/500/500/500`；若 family granularity 使目标无法闭合，必须修改预注册方案和 policy revision，不能拆 family 或手调单题。

## Exact/fuzzy contamination

默认 policy `d06-contamination-policy-v1`：

| 项 | 冻结值 |
|---|---:|
| normalization | NFKC → casefold → NFKC、digit-group/LaTeX spacing normalization、lexeme preservation |
| character n-gram | 5 |
| token n-gram | 3 |
| minimum exact length | 16 chars |
| minimum fuzzy length | 32 chars |
| fuzzy block | char/token Jaccard ≥ 0.82，或 containment ≥ 0.92 |
| review band | 距 block threshold 0.05 内 |
| score storage | integer ppm |
| scanned groups | context/system/tool、problem、user prompt、reference/response/assistant solution，以及 prompt/solution/full-record 聚合文本 |

默认扫描所有 system/tool context，并把跨 message 的完整 prompt、assistant/tool trace 和完整 record 额外拼成 aggregate fragment，避免 benchmark 内容藏在 context 或拆成多个短 message 后漏检。公共模板只有其完整 normalized SHA-256 事先写入 policy 的精确豁免表时才会排除；不能按 role 整类跳过。短答案如 `42` 不单独构成污染，避免所有共享标量答案互相命中。

实现为完整 inverted n-gram retrieval：每个 evaluation fragment 查询所有共享至少一个 char/token n-gram 的 training fragments，再精确计算 Jaccard 与 containment。任何大于零的 n-gram overlap 都会进入候选集，因此不存在 MinHash/LSH 的概率漏召回；候选 pair 只累计数量、不再额外保存全体 pair 集合。JSONL 同时采用逐行 strict UTF-8 流式加载和增量 SHA-256，避免把允许上限内的整个文件重复驻留内存；posting 数量与真实峰值仍须进入 D15 scale profile。

`exact`、`fuzzy`、`review` 都使 report fail closed。报告只保存 record/source/field IDs、normalized hashes 和 integer scores，不保存原文或 sealed answer。正式处理不是在结果出现后改阈值，而是把所有命中 training records 及其 source/problem/template 传递 component 全部 quarantine，然后用同一 policy 重新扫描；只有零命中才能冻结 manifest。

## Immutable manifest

`DataManifest` 不含 problem、answer 或 trace 原文。每条只保留 sample/split/source/revision、payload/content/lineage hashes 和三个 family hashes；顶层记录：

- source registry、transform registry、external-parent ledger、split policy、contamination policy/report hashes；
- record/split/source counts；
- ordered record-set hash；
- manifest self-hash。

公开 builder 不接收调用方给出的 `passed` 布尔值或 report hash；它先把输入 `Sequence` 冻结成唯一 tuple 快照，再从该快照导出 train/eval partitions、重跑 frozen contamination policy，只有零 match 才内部生成 report hash 和 manifest。这样既不能把 dirty report 冒充 clean，也不能用多次遍历会切换内容的自定义 `Sequence` 制造 TOCTOU。加载时会重算 counts、record-set hash 和 self-hash；任何改动、重排、截断或字段注入都会失败。JSON writer 使用同目录临时文件、flush/fsync 和 atomic replace。

Public builder 的 `split_policy_sha256` 是 tamper-evident binder，不单独证明 records 确由该 policy 分配；正式 `run_data_trust_audit()` 会在同一 Git-bound snapshot 内重跑 assignment，才是 attestation 路径。D15 前应进一步让 public builder 直接接收并核验 policy/assignment，或只允许其消费 formal-audit 产物。

## Synthetic adversarial audit

```bash
uv run python scripts/audit_data_trust.py
```

冻结 fixture 位于 `tests/fixtures/data_trust/`，专门验证机制，不冒充真实数据：

- 3 个带 immutable revision/license evidence 的 synthetic sources；
- 15 个候选记录经 12 个传递 family components 分配，并覆盖五个训练/dev split；
- dirty scan 命中 1 exact、2 fuzzy，其中一条来自 response containment；
- 3 个直接命中扩展为 6 个 family quarantine records；
- 剩余 9 train + 4 eval 重新扫描为零命中，才生成 raw-text-free manifest；
- source registry semantic SHA：`5ee29d34ac4cd242a6f36859293ccd38dcc2cc7ddd341434a94352a6d1849f43`；
- transform registry semantic SHA：`8b4a8f1eee0e1355cf2d004e86043eaf73f2c730c73e28f1a23eef6c9d8ff758`；
- empty external-parent ledger SHA：`7d0b208bf140bf80d38bdb458324439eab114bf6188b465560d7b8e160865c73`；
- split policy SHA：`e877fd58abb2a903ae31f6acc88d046b2abc0f703fd0e18a8869abe53a153959`；
- contamination policy SHA：`675338221f7b274af54592a6e5552d47355fa4219853820150a971915eb53fa7`；
- clean manifest SHA：`ae6b70c2f7667b0ee66d33ce59cd7fbb558ba4d5ba4934e69304bfd3f5abc364`。

正式 CLI 只接收 repository-relative paths，在函数内部加载 source/transform registry、candidate/evaluation records、split policy、可选 parent ledger 与 expectation，并自行收集 provenance；调用方不能注入 provenance。父包与 `data` 公共 `__init__.py`、registry/contamination/audit/CLI 源码、`uv.lock`、全部输入以及 transform code/config 都必须被 Git 跟踪、与启动时捕获的 Git revision 字节一致，且 declared transform hash 必须等于这些 Git 字节。expectation 同时冻结 registry、policy、assignment、dirty/clean report、manifest 与 record-set hashes。报告只在实现提交后生成，再单独作为 evidence commit 保存。

## 验证与未完成边界

当前全仓 `436` 个 CPU tests 通过，包含累计 `1,175` 个 Hypothesis 生成案例；D06 定向 `96` 个 tests 覆盖 strict schema、license/revision、payload lineage/cycle/external ledger、Git-bound transforms、三维 family leakage、split permutation、manifest tamper/dirty-report/TOCTOU、HEAD 移动竞态、context 与跨-message aggregate、exact/fuzzy/containment/review、短答案、raw-text exclusion、transitive quarantine、golden expectation 和不可注入的 Git provenance。

仍未完成：

- D07 sealed benchmark runner 与 evaluator result schema；
- D10/D11 训练 runtime 的字段访问控制与 run provenance；
- D15 真实数据下载、card/license bytes、实际 revisions、全量污染 pair review、目标 count materialization 与 sealed manifest；
- D15/D12 实际规模 profile 与 provenance 扩展：aggregate/超长单行、n-gram postings/match list、ledger/transform lookup 的峰值，以及 `.python-version`/`pyproject.toml` 和 Python Unicode database 版本；
- G1 只有 D07、D15 和 `EVAL-002` 全部完成后才可通过。
