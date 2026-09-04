# D08 Experiment Code Review

> review scope: D08 paired panel、stratified item bootstrap、paired randomization、Holm、TOST decision、result contracts 与 Git-bound synthetic audit
> reviewer: fresh secondary Codex reviewer（GPT-5.6-Sol xhigh，只读，未编辑工作树）
> review snapshot: 2026-09-04 19:15 Asia/Shanghai；evidence closure: 2026-09-04 19:23 Asia/Shanghai
> implementation status: APPROVED
> evidence status: CLOSED（synthetic CPU mechanism only）

## 与研究合同的一致性

- 输入由 15 个 D07 greedy reports 与 hash-bound public snapshot 构成，只投影 item correctness、公开 `level` strata 和 immutable identities；不读取 generation、prompt 或 sealed answer；
- estimand 先在每题内平均三个固定 training-seed paired differences，再对 item 求均值；区间只对 item population 推断，不推广到 training-seed population；
- bootstrap 在 strata 内只重采样 item，sign-flip 对同一 item 的完整三-seed vector 共同翻转；
- C1a/C1b 使用 one-sided paired randomization、精确有理数 Holm、95% CI 与 +2pp practical gate；
- C2 先做 95% superiority，失败后才用严格位于 ±2pp 的 90% CI dual-TOST 判定 practical equivalence；
- 本轮只有 CPU synthetic fixtures，没有模型下载、真实 benchmark、MPS/CUDA/GPU，也没有通过 G1/G5/G6。

## Fresh review 发现并关闭的问题

1. **P1 result-to-panel binding**：初版 loader validation 只核 hash/count/decision，攻击者可重算 self-hash 后伪造 C1b。现改为从 supplied panel 和 protocol 确定性全量重算并比较完整 report；同时强制 C1a=A1−A0、C1b=A2−A0、C1/C2 alternative、point 与 per-seed mean、Holm family 和 C2 sequential semantics。新增 self-hashed forged contrast regression。
2. **P1 outcome-dependent RNG**：初版派生流包含 correctness-bound panel hash。现只用预注册 base seed、protocol、operation、hypothesis 与 outcome-independent benchmark/resampling design 派生 PCG64 stream；correctness/report/checkpoint 不进入 RNG seed，完整 panel hash仍单独绑定输出。新增不同 outcomes 保持 stream identity 的 regression。
3. **P2 peak memory**：初版固定 vector batch 与一百万题 schema 上限不闭合。现 bootstrap/sign-flip 按最大 stratum/item 数动态缩 batch，临时 resampling matrix 上限固定为 1,000,000 cells。

复核 verdict 为 **APPROVE**：上述 blockers 均已关闭，无剩余 P1/P2/P3。

## 验证快照

- `uv sync --frozen --all-groups`：通过；
- `uv lock --check`：通过；
- 全仓 Ruff：通过；
- D08 定向 tests：`33 passed`；
- 全仓 tests：`539 passed`；
- D08 新增 Hypothesis 生成案例：200；全仓累计 1,575；
- `git diff --check`：通过；
- formal 8-item audit 使用完整 10,000 bootstrap / 100,000 randomization protocol，并已从 clean tracked implementation commit 生成。

## Evidence gate closure

以下四项已全部成立，D08 可以标记为 `COMPLETE_CPU`：

1. implementation 独立提交为 `9a5cee946c617acca6d9e5a167fa725d67798eef`；
2. formal audit 从该 commit 的 tracked CLI/runtime/input bytes 运行；
3. timestamp 与 latest artifact byte-identical，均显示 `passed=true`、0 failures，Git/source/input hashes 闭合；
4. tracker/config/docs 明确区分 synthetic mechanism evidence 与尚未执行的真实 D23 analysis。

正式证据摘要：

- artifact：`artifacts/audits/D08_PAIRED_STATISTICS_AUDIT_20260904_191830.json` 与 latest copy；
- implementation source SHA-256：`2484549561c42707108f990f6cc3e1885e82e75f2c9934f6f6413929635be1cc`；
- statistics protocol SHA-256：`dc8789a4463691523867f1813d99cd3867b899673f69bfd359bf0307014f575f`；
- panel SHA-256：`3f89fb2cadeeba0137b2fa4a2b4c14a8665758c6c146e7dd45641edc68bfe455`；
- resampling design SHA-256：`16f86de6ab34b504b8a62a8bbc1cbf8c26cd34f37f859b97e16b364e7a042762`；
- canonical analysis SHA-256：`f343e8acb2c891303832c7360fe81f21dc5b5ae4a84dc841190160878e36bd82`；
- canonical audit report SHA-256：`099b4251e2056f990aa7175485334506d4c1a03a3c3b5d914928fe15f406c5d9`；
- serialized artifact SHA-256：`c2ae7263733f8a6ef647ed7c57ffbc812c97dc188e1c63a2682adb7506492830`。

这里的 artificial +100pp、0pp 与 practical-equivalence decisions 只是冻结 regression oracle；不得写成任何模型效果。

## Reviewer verdict

**APPROVE（implementation + synthetic evidence closure）**：统计算法、严格 schema、D07 projection、确定性重算、provenance 与 formal synthetic audit 均已闭合。仍不得声称任何真实模型效果、真实 C1/C2 完成或 G5/G6 通过。
