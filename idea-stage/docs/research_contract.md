# Research Contract

> status: FROZEN
> source: `refine-logs/FINAL_PROPOSAL.md` and `refine-logs/EXPERIMENT_PLAN.md`
> note: `experiment-bridge` 的随包模板缺失；本文件是从已审查方案直接生成的 fallback，不引入新 claim。

## Immutable problem

在有限算力和完全公开、可追溯的数据条件下，从同一前沿 Student SFT checkpoint 出发，受控比较不同后训练学习信号的单独作用和顺序交互，并把收益与模型、数据、训练预算、评测及计算成本混杂区分开。

## Systems under comparison

- Student：`google/gemma-4-E2B` Base 经 immutable `D_anchor` SFT；
- Teacher：`google/gemma-4-E4B` Base 经同一 `D_anchor` SFT 后冻结；
- A0：SFT→SFT；A1：GRPO→GRPO；A2：OPD→OPD；
- A3：OPD→GRPO；A4：GRPO→OPD；
- 每个 stage 严格匹配 2M Student backward loss tokens，seeds 为 101/202/303。

## Confirmatory claims

### C1 — single-signal intervention

在同一 E2B anchor、prompt distribution、decoding 与 Student loss-token 预算下，估计 `A1−A0` 和 `A2−A0` 对 MATH-500 greedy answer accuracy 的 recipe-level intervention effect。

支持条件：Holm-adjusted `p<0.05`、item-bootstrap 95% CI lower>0、点估计至少 +2pp，并单列三个 paired-seed effects。

### C2 — order intervention

以 `A3−A4` 为唯一确认性顺序对照。superiority 要求 95% CI 不含 0 且绝对差至少 2pp；否则只有 TOST 90% CI 完全位于 `[-2,+2]pp` 才能称 practical equivalence。

## Evidence contract

- 唯一确认性 endpoint：MATH-500 greedy answer accuracy；
- verifier 只给 exact/symbolic correctness 0/1，不加入格式或长度 shaping；
- 正式 benchmark 对训练过程 sealed，评测必须对 dataset ground truth；
- 所有数据记录 source/license/revision/hash，所有结果记录 git/config/model/data/evaluator/checkpoint hashes；
- E1 只匹配 Student backward loss tokens；E2 单独报告所有 forward/backward、rollout、GPU-hour、显存与能耗；
- G0–G6 全部通过后，数字才可进入 README、报告或简历。

## Anti-claims

本项目不声称 GRPO 或 OPD 普遍更优、Teacher 差异仅来自参数量、Base 预训练无 benchmark 污染、三个 seeds 可代表所有 seed population，亦不声称达到 SOTA。

## D05 role

D05 冻结训练 reward 与独立数学 evaluator 共用的 canonical answer semantics：reference 不可解析属于数据/基础设施错误，prediction 错误或不可解析得 0，只有可审计的 exact/symbolic equivalence 得 1。任何训练后放宽 parser 的行为都违反本 contract。
