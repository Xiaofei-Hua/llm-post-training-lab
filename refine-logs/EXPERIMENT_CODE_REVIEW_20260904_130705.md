# D06 Experiment Code Review

> review scope: D06 data registry、payload lineage、family split、contamination gate 与 immutable manifest
> reviewer: fresh secondary Codex reviewer
> review snapshot: 2026-09-04 13:07 Asia/Shanghai
> status: APPROVED

## 与研究合同的一致性

- closed-world source/transform registry 冻结 immutable revision、license evidence、allowed use 与处理代码/配置 hashes；
- canonical record、parent ledger、family split、contamination report 与 manifest 使用 versioned schema 和分离 hash domain；
- source/problem/template 三维传递 family 在 split 和 quarantine 两处都不允许泄漏；
- benchmark contamination 默认覆盖 system/tool、逐字段与 prompt/solution/full-record aggregate，不保存 sealed raw text；
- 本轮只使用 CPU 和 synthetic adversarial fixture，没有下载真实模型/数据，没有读取 sealed benchmark answer，也没有执行 MPS/CUDA/GPU；
- `DATA-001`、`DATA-002`、`EVAL-002` 与 G1 均保持 `NOT_STARTED`。

## 多轮审查中已修复的 blocking classes

1. 默认纳入 context/system/tool，增加 prompt、solution trace 与 full-record 聚合，覆盖跨短 message 泄漏；context 仅可按预冻结 normalized SHA-256 精确豁免。
2. 用 complete inverted n-gram candidate route 取代概率 MinHash/LSH；exact、fuzzy 和 review band 全部 fail closed。
3. 命中记录扩展到完整传递 family component，clean 集合必须使用同一 policy 重新扫描为零。
4. public manifest builder 不再接受调用方伪造的 `passed`/report hash，并先把输入冻结为单一 tuple，阻断切换内容的 Sequence TOCTOU。
5. parent 由 content hash 升级为完整 `payload_sha256`；外部 parent 必须由冻结 ledger 解析，缺失、跨 split 或 digest 不符均拒绝。
6. transform registry 将 name/version/code/config hash 与 Git-tracked code/config bytes 绑定；lineage 未登记即失败。
7. strict JSONL 改为逐行二进制流式读取与增量 SHA，并对文件、行、记录和 expectation 设定资源上限。
8. golden expectation 同时冻结 source/transform registry、split policy、parent ledger、contamination policy、assignment、dirty/clean report、manifest 与 record-set hashes。
9. formal audit 在内部收集父包/API/implementation source、`uv.lock`、全部输入与 transform artifacts 的 provenance，不允许调用方注入。
10. Git provenance 使用捕获的 `revision:path` 读取；HEAD 在 scan 中途移动的竞态已有回归并 fail closed。

## 审查快照验证

- `uv sync --frozen --all-groups`：通过；
- `uv lock --check`：通过；
- D06 定向 Ruff：通过；
- D06 定向 tests：`96 passed`；
- 全仓 tests：`436 passed`；
- Hypothesis：全仓累计 1,175 个生成案例；
- CLI help：已确认可选 `--parent-ledger`；
- fresh reviewer 未编辑工作树。

## Evidence closure

实现先独立提交为 `c4c726a9401deef9960ca041dcc59477a18d697f`。随后 strict CLI 在该 tracked/clean snapshot 上生成正式 artifact：`passed=true`、0 failures、`tracked_inputs_match_git=true`，报告内 Git revision 精确等于实现 commit；dirty scan 为 1 exact/2 fuzzy，6 records 经 family quarantine，clean rescan 为 9 train + 4 eval，manifest SHA-256 为 `ae6b70c2f7667b0ee66d33ce59cd7fbb558ba4d5ba4934e69304bfd3f5abc364`。canonical audit-report SHA-256 为 `d7c3155d8cf4a003c0e7673ad9515ec9d00d61964dab8a1ec3cf513f439c77a2`，pretty JSON file SHA-256 为 `2c3e4eaeb4aba48124495b37d34162adfd5ece8472fd926db5a7524ed49d70c2`；latest 与 timestamp artifact 逐字节一致，并由独立 evidence commit 跟踪。

## Non-blocking risks

- D15 前需做实际规模 profile：aggregate 可超过单字段字符上限；超长 JSONL 单行、n-gram postings/match list、parent ledger 和 transform lookup 尚无真实峰值证据。
- 可在 D12/D15 把 `.python-version`、`pyproject.toml` 与 Python Unicode database 版本纳入 provenance。
- public manifest 的 opaque `split_policy_sha256` 是 tamper-evident binder，不独立证明 records 由该 policy 分配；正式 audit 已重跑 assignment。后续应强化 builder 或只把 formal Git audit 作为 attest 路径。
- synthetic fixture 不能证明真实 family 标签、license/card bytes 或全量数据正确；这些仍属于 D15 materialization 和人工 allowlist/pair review。

## Reviewer verdict

**APPROVE**：无剩余 blocking issue，implementation 与双提交 formal evidence 均已闭合。D06 作为 synthetic CPU contract 可接受；不得据此提前通过 G1。
