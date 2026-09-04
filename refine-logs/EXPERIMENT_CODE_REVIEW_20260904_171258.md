# D07 Experiment Code Review

> review scope: D07 sealed benchmark evaluator、generation/result schema、item-level metric contracts 与 Git-bound synthetic audit
> reviewer: fresh secondary Codex reviewer（只读，未编辑工作树）
> review snapshot: 2026-09-04 17:09 Asia/Shanghai
> implementation status: APPROVED
> evidence closure: 2026-09-04 17:11 Asia/Shanghai
> evidence status: CLOSED

## 与研究合同的一致性

- generator 只接收 public prompts、checkpoint identity 与冻结 protocol；sealed answers 仅进入 evaluator capability；
- greedy/sampling protocol、checkpoint-independent paired seeds、完整 item×sample grid、backend response bijection 与持久化 finish semantics 均由 strict versioned schema 约束；
- evaluation report 保留逐 sample/item correctness、generation hashes、accuracy、exact pass@k、extraction/parse、completion length、finish/truncation 与 evaluator identity，不泄露 prompt、prediction、candidate 或 reference；
- D08 只能联合消费 report correctness 与由 report hashes 精确绑定的 public snapshot strata，不得重解析 raw generations 或读取 sealed answers；
- 本轮只使用 CPU 和 synthetic fixtures，没有下载模型/真实 benchmark，没有执行 MPS/CUDA/GPU；真实 adapters、blind human audit、Base baseline 与 G1 均保持未完成。

## 多轮审查中已修复的 blocking classes

1. provenance source closure 覆盖包级 import 实际执行的 data/reward/evaluation 模块、tracked CLI、Python 配置与 `uv.lock`，dirty runtime dependency 必须失败。
2. formal CLI 强制核对 `__main__` 与全部已加载项目模块的 resolved `__file__`，拒绝“审计 checkout A、执行 checkout B”。
3. descriptor、protocol、public、sealed、predictions 与 expectation 的 loader-consumed raw digest 均立即与 captured provenance 交叉绑定，读取时替换后恢复也失败。
4. provenance scan 与实际 evaluation 末尾都复核 HEAD identity；所有 tracked inputs 在返回前再次哈希，阻断 input/HEAD race。
5. generation records/manifest 先用 resolved targets 判断 sibling 与 alias，再原子写 lexical destinations；路径别名不能相互覆盖，两个不同 leaf symlink 可安全 round-trip 且不改写 target。
6. completed response 必须携带完整非空 output token IDs；EOS、stop-token、stop-sequence 与 length 的结束条件在生成和反序列化后都复核。
7. formal oracle 的 greedy/sampling 各包含一个 `length` finish，使非零 truncation 与 completion-token 路径进入 Git-bound expectation。
8. `ItemScore`/`EvaluationReport` 的全部数值 summary 和 count pairs 显式拒绝 Boolean 冒充整数，即使 `True == 1` 且攻击者重算 self-hash 也不能通过。
9. D08 strata 文档改为 report 与 hash-bound public snapshot 的显式 join，不再声称 sanitized report 内含 strata。
10. tracker 在正式 artifact 生成前保持 `IN_PROGRESS_CPU`；只有 artifact 闭合后才转为 `COMPLETE_CPU`。

## 审查快照验证

- `uv sync --frozen --all-groups`：通过；
- `uv lock --check`：通过；
- 全仓 Ruff：通过；
- D07 定向 tests：`70 passed`；
- 全仓 tests：`506 passed`；
- Hypothesis：D07 新增 200 个生成案例，全仓累计 1,375 个；
- `git diff --check`：通过；
- fresh reviewer 结论：无剩余 P1/P2 correctness、security、provenance、schema 或 anti-claim blocker。

## Evidence closure

实现已独立提交为 `c948fe2eae50289b78513a3a9513e188bff54a98`。随后从该 checkout 的 tracked `scripts/audit_evaluator.py` 生成正式 artifact：`passed=true`、0 failures、`tracked_inputs_match_git=true`，报告内 revision 精确等于 implementation commit。implementation source SHA-256 为 `e737ca0ce742b7896fbd77702a2be0115ff881b5cc723ee75df7997e82dde7ea`，canonical audit-report SHA-256 为 `a6c1bed7e74fcc9bf3448fa095535237c27f183cd67938799d9d96bc40b8abf4`，pretty JSON file SHA-256 为 `4868cbf3b9572a141659f7141b3bf17fc6b9fc0b3f6bd03952e9f90d9fec8654`。latest 与 timestamp artifact 逐字节一致，并将由独立 evidence commit 跟踪。

## Non-blocking risks

- public/sealed 分离是 API/进程 capability boundary，不是 Python 安全沙箱；D15 正式 materialization 仍需独立进程和文件权限。
- synthetic token IDs、3/6、21/48、5/6 pass@8 与 truncation oracle 只验证机制，不是 Gemma 4 或真实 benchmark 结果。
- IFEval/MMLU-Pro 等官方 adapter、真实 tokenizer token IDs、blind human audit、Base contamination baseline 和 G1 仍属于 D15。
- D08 负责 paired bootstrap、randomization/sign-flip、Holm 与 TOST；不得把统计推断回填进 D07 evaluator。

## Reviewer verdict

**APPROVE**：无剩余 P1/P2 correctness、security、provenance、schema 或 anti-claim blocker；implementation 与 formal Git-bound synthetic evidence 已闭合。D07 可作为 CPU contract 标记完成，但不得据此提前通过 G1。
