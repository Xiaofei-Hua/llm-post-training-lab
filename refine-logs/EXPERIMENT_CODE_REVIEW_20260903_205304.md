# D05 Experiment Code Review

> review scope: D05 exact/symbolic parser、verifier 与 reward audit
> reviewer: fresh secondary Codex reviewer
> review snapshot: 2026-09-03 20:53 Asia/Shanghai
> status: REQUEST_CHANGES — evidence closure pending

## 与研究合同的一致性

- reward 与未来 evaluator 共用同一 canonical semantics，并始终对 dataset reference 判定；没有把另一个模型的输出当 ground truth；
- reward 只允许 `0/1` correctness，不引入格式或长度 shaping；
- prediction 失败记 0，reference 或 backend 基础设施失败使 logical batch fail fast；
- 本轮只实现 CPU parser/verifier/audit，没有下载模型或数据，也没有执行 MPS/CUDA/GPU；
- `EVAL-002` 真实模型输出人工盲审和 G1/G3 仍保持未通过。

## 多轮审查中已修复的 blocking classes

1. 冻结 last-terminal-surface precedence；破损的最后 marker 不再回退到早先正确答案。
2. 阻止未闭合 container 内的 box/tag/text label 劫持，以及 escaped box、非法 TeX command boundary 绕过。
3. 正确跟踪 Markdown inline code 与 backtick/tilde fenced code 的开闭规则。
4. 拒绝 `$7$ because $42$`、numeric gap、narrative/logical/layout payload 和 normalization 会消隐的任意 text payload。
5. 对隐式并置做显式乘法双解析一致性检查，同时保留函数幂、函数下标、`operatorname` 等合法写法。
6. 加入 finite set、interval、tuple、relation、logical relation、matrix、scalar 的 structural-family guard。
7. 只允许安全的 `Symbol = scalar` assignment 跨 scalar/relation family，阻止共享 RHS false positive。
8. 所有 references 在 prediction parse 前预检；reference cache 只缓存成功结果。
9. 把 verifier/audit/CLI source、`uv.lock`、corpus 和 Git revision 纳入 provenance，并要求正式输入由 Git 跟踪且与 HEAD 一致。
10. 补齐 timeout、长度上限、主线程/process-worker 合同、strict corpus schema 与 atomic audit writer。

## 冻结快照验证

- `uv sync --frozen --all-groups`：通过；
- `uv lock --check`：通过；
- `uv run ruff check .`：通过；
- `uv run pytest -q`：`340 passed`；
- Hypothesis：全仓累计 875 个生成案例；
- adversarial corpus：`257/257`，17 categories；
- corpus SHA-256：`45496b679cdd13971a050f9b573b2cbc4974da56b67d54a18ee1edaaaa0d50c7`；
- policy SHA-256：`a6331e8c2c2a6a57fcbdd08a7c385404b62013fe30e1fbfaab5d701853138d13`；
- sdist/wheel build、隔离 wheel install 与 public API smoke：通过。

## 当前 blocking issue

冻结的 `artifacts/audits/D05_VERIFIER_AUDIT.json` 仍是实现提交前产生的旧临时文件：只含 `128/128`、旧 corpus SHA，且没有 provenance。因此它不得进入实现提交，也不能作为完成证据。

关闭方法：先提交 verifier、audit、CLI、fixture、tests、dependency lock 与文档；再在该 clean/tracked revision 上运行 strict CLI，确认新 artifact 为 `257/257`，并包含 corpus/policy、Git revision、source 与 lock hashes；最后单独提交证据。

## Non-blocking risks

- 自定义 extraction 不由第三方 backend 的 signal timeout 包裹；D10 必须使用 process-level watchdog 和 worker recycle 保护极端输入。
- 合成 corpus 不能替代 `EVAL-002` 的至少 100 条真实模型输出 checkpoint-blinded human audit。

## Reviewer verdict

**REQUEST_CHANGES**：实现本身可接受；仅待用两段式提交关闭 stale evidence blocker。
