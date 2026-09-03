# D05 Exact/Symbolic Math Verifier

## 模块结论

D05 提供训练 reward 与后续独立数学 evaluator 共同调用的 canonical answer semantics。它不是正则表达式 demo，也不把第三方 `verify()` 的单个 Boolean 直接当训练信号：项目层会先冻结 terminal-answer 抽取规则、输入边界、reference/prediction 失败语义、符号比较参数、结构类型 guard、审计 schema 和依赖版本。

当前只完成 CPU 实现与合成审计。未做真实模型输出人工盲审，因此 `EVAL-002` 和 G1 仍未通过；未运行模型、MPS、CUDA 或 GPU。

## Deletion test

只保留宽松正则或直接调用 Math-Verify 默认配置不足以支撑主 claim：

1. 宽松 unanchored extraction 可能从推理中间过程、复制题面或注入字符串里捞到恰好正确的数字；
2. 多个 `boxed`、后置 Final Answer、空/破损的最后 marker 若没有统一规则，会让 train reward 与 evaluator 分叉；
3. 第三方 best-effort extraction 可能从候选 `$7$ because $42$` 的右侧取出 `42`；未闭合 container、转义 command、代码围栏的伪 closing 也可能改变最终 surface，因此必须先验证唯一 top-level surface；
4. normalization 会删除任意 text-like payload、部分 layout token；parser 还会把 `1(41)` 一类常数并置解释为 `42`，因此仅靠 parse 成功不足以证明消费了完整、无歧义的答案；
5. 第三方 best-effort comparison 在某些结构上可能过宽；本轮实际发现 finite set 与 tuple/interval、以及 scalar `42` 与伪等式 `7=42` 会形成 false positive，因此加入 strict structural/assignment guard；
6. prediction 不可解析应得 0，但 reference 不可解析或 backend exception 不能伪装成负样本，必须在任何 prediction parse 前阻断 logical batch；
7. parser、normalization、ANTLR runtime、SymPy tolerance 与 timeout 若不进入 provenance，同一 generation 可能在不同环境得到不同 reward。

所以 D05 保留 pinned Math-Verify 作为符号后端，同时在外层实现更窄的项目合同。

## 冻结语义

### Prediction extraction

候选答案按文本位置选择**最后一个显式 terminal surface**：

1. `\boxed{...}` 或 `\fbox{...}`，支持嵌套 LaTeX braces；
2. `<answer>...</answer>`；
3. `Final answer: ...`、`Answer = ...`、`最终答案：...` 等显式标签；
4. 没有标签时，只接受单行、短小、词法上像数学表达式的 direct answer。

多个 top-level surface 时最后一个生效。最后一个 marker 为空、brace/tag 不闭合，或位于 inline/backtick/tilde Markdown code span 时直接失败，不回退到更早的正确答案。代码 fence 只能由同类、长度不短于 opener 且尾部仅空白的完整行关闭。未闭合 container 占有其后文本，内部 box/tag/`Answer:` 均不能把它救回；嵌套 `<answer>` 拒绝。转义 `\\boxed` 不是 surface，`\boxed42` 则因缺少 brace 作为 malformed later surface 失败。

抽出的 candidate 还必须是唯一、单行数学 surface：单个完整 `$...$`/`$$...$$`/`\(...\)`/`\[...\]` 可接受，混合或重复 math environments、narrative connectors、negated/uncertain answer labels、numeric subscript、transpose suffix、layout separators 与 code payload 拒绝。会被 unit normalization 消隐的 `text/textnormal/textbf/textit/textrm/mathrm/mathit/mathbf/mbox` 只允许出现一次、位于末尾，且完整内容只能由冻结的 ASCII unit tokens 与空格组成。这一层作用于 box、tag、text marker 和 direct answer 的全部 prediction candidates。

对 numeric/closed primary 与 group、fraction、binomial、function 的隐式并置，verifier 会再生成显式 `\cdot` 版本并调用同一 pinned backend；只有两次解析得到完全相同的类型与符号对象才接受。因此保留 `2(x+1)`、`\sin^2(x)`、`\log_2(8)` 等标准写法，但拒绝原 parser 会把乘法读成加法的 `1(41)`。同一 text/juxtaposition 预验证也作用于 reference，防止坏 gold 经 normalization 后伪装成有效 reference。

这不是额外的 format reward：没有格式分、长度分或 learned RM，最终仍只有 correctness `0/1`；direct math answer 也可得分。格式状态只作为诊断字段。

### Parse and comparison

- backend：`math-verify==0.9.0`，显式选择 `antlr4-python3-runtime==4.13.2`；
- prediction normalization：允许 basic LaTeX 与 units，关闭 malformed-operator 修补和 nits；
- reference normalization：允许较宽清洗，但 reference 无法解析会触发基础设施错误；
- 禁止 raw-string fallback；symbol variables 使用 strict matching；
- numeric comparison：`float_rounding=12`、`numeric_precision=30`；例如 `0.333333` 不等于 `1/3`，普通 double 展开的 `0.3333333333333333` 可接受；
- finite set、interval、tuple、relation 与 matrix 先做 structural-family compatibility，再进入 symbolic comparison；scalar 只允许与 `Symbol = value` 的安全 assignment 跨 family 比较，数字/compound LHS 等式不得只靠共享 RHS 命中；
- 单条 parse/verify 默认各限 5 秒；输入最多 32,768 chars，candidate 最多 4,096 chars；
- backend 使用 signal timeout，因此 bounded call 必须在各 process 的 main thread 执行。D10 应使用进程 worker，不能在线程里静默关闭 timeout。

### Reward failure semantics

| 情况 | `VerificationStatus` | Reward | 训练行为 |
|---|---|---:|---|
| 符号等价 | `match` | 1 | 可进入 GRPO group |
| 可解析但不等价 | `mismatch` | 0 | 可进入 GRPO group |
| 无 final surface / malformed | `prediction_not_extracted` | 0 | 可进入 GRPO group并记录原因 |
| prediction 无法解析或超时 | `prediction_unparseable` / `prediction_timeout` | 0 | 可进入 GRPO group并计数 |
| comparison 超时 | `verification_timeout` | 0 | 防止复杂输出通过 skip 获利 |
| reference 无效 | `reference_invalid` | — | `exact_reward`/batch fail fast |
| backend exception / 错误线程上下文 | `backend_error` | — | logical batch fail fast |

因此不会把坏 gold、依赖错误或线程错误偷偷标成“模型答错”。`verify()` 总是先验证 reference，失败时 prediction 记录为 `skipped`；`score_batch()` 在解析任一 prediction 前预检整批 references，并在第一处基础设施错误立即停止。reference LRU 只缓存成功解析，不缓存 timeout/backend failure。

## Public API

```python
from posttrain_lab.rewards import ExactMathVerifier

verifier = ExactMathVerifier()
decision = verifier.verify(
    reference=r"\frac{1}{2}",
    prediction=r"Reasoning ... \boxed{\frac{2}{4}}",
)
assert decision.reward == 1.0

batch = verifier.score_batch(
    references=["1", "2"],
    predictions=[r"\boxed{1}", r"\boxed{3}"],
)
assert batch.rewards == (1.0, 0.0)
```

`VerificationResult.to_record()` 只序列化 bounded extraction、状态、value types、policy hash 和 backend versions，不序列化任意 SymPy object。reference parser 使用 success-only bounded LRU，batch API 严格检查长度，禁止 `zip` 静默截断。

## Adversarial audit

冻结 corpus：`tests/fixtures/verifier_adversarial.jsonl`。

```bash
uv run python scripts/audit_verifier.py
```

实际 CPU audit：

- 257/257 cases passed；
- 17 个类别：integer、rational、decimal/percent/units、symbolic、sets/relations、unsafe relation、surface precedence、malformed、trailing-math hijack、ambiguous juxtaposition、text payload、unanchored prose、injection、non-finite/ambiguous、invalid reference；
- corpus SHA-256：`45496b679cdd13971a050f9b573b2cbc4974da56b67d54a18ee1edaaaa0d50c7`；
- policy SHA-256：`a6331e8c2c2a6a57fcbdd08a7c385404b62013fe30e1fbfaab5d701853138d13`；
- machine-readable evidence：`artifacts/audits/D05_VERIFIER_AUDIT.json`。

Corpus loader 拒绝 unknown/missing fields、重复 ID、非 UTF-8、非法 status/reward 组合和 100–300 之外的正式 audit 规模。报告采用原始 corpus bytes hash，失败时保留每个 structured decision；同时记录 verifier/audit/CLI 各文件 SHA、aggregate implementation SHA、`uv.lock` SHA 与 Git revision。正式 CLI 要求实现、lock 和本次输入 corpus 均已被 Git 跟踪且与该 revision 完全一致，避免“同一 policy hash、不同实现或临时修改测试集”仍伪装成同一次审计。

## 尚未完成

- D06 才负责真实数据 reference 的全量预验证、revision 与 immutable split；
- D07 才实现 sealed benchmark runner、generation/result schema 和 metric 聚合；
- D10 才把 batch reward 接入 GRPO lifecycle/process workers；
- EVAL-002 仍需对至少 100 个真实模型输出做 checkpoint-blinded human audit，并达到 ≥99% agreement；
- 训练后不得根据结果放宽 extraction/tolerance；任何变更必须产生新 evaluator revision 和新实验族。

## 上游依据

- [Hugging Face Math-Verify](https://github.com/huggingface/Math-Verify)：answer extraction → SymPy representation → gold comparison 的官方实现与推荐 reward 配置；
- [Math-Verify 0.9.0 on PyPI](https://pypi.org/project/math-verify/)：本仓库锁定的发布版本、Python 范围、依赖 extra 与发布元数据；
- [TRL reward functions](https://huggingface.co/docs/trl/rewards)：官方 `accuracy_reward` 以 dataset solution 为 gold，gold 不可解析时跳过；本项目进一步选择 fail-fast，避免无效 reference 进入正式 reward batch。
