# 最终研究方案：Gemma 4 后训练信号与顺序归因

> Planning verdict：READY（独立审查 9.06/10）
> Execution verdict：CONDITIONAL（尚未下载模型、确认 GPU 或完成 profile）

## 研究定位

本仓库不是推理服务或训练框架工程，而是一项面向基模/后训练算法实习的可复现研究：在有限算力和公开、可追溯数据下，从同一个前沿 Student SFT checkpoint 出发，受控识别不同后训练学习信号的单独作用与顺序交互。

核心实例化为：

- Student：2026 `google/gemma-4-E2B` Base；
- Teacher：`google/gemma-4-E4B` Base 使用与 Student 相同的数据谱系完成 SFT 后冻结；
- sparse signal：exact/symbolic verifier 的 GRPO；
- dense signal：Student fully on-policy prefixes 上的 full-vocab reverse-KL OPD；
- 主任务：可程序验证的数学推理；
- 唯一确认性 endpoint：MATH-500 greedy answer accuracy。

不使用 Qwen3。选择 Gemma 4 的原因不是宣传“绝对最新”，而是它在 2026 前沿性、轻量 Student/Teacher 同族组合、Base checkpoint、共享 tokenizer 潜力与公开训练生态之间形成了更好的可研究平衡。

## Claim boundary

本项目只估计：**指定 same-lineage E4B Teacher 与冻结 GRPO/OPD recipe，在等 Student backward-token 预算下，对该 E2B Student 的 intervention effect。**

不声称：效果仅来自参数量、E2B/E4B 除容量外完全相同、OPD 对任意 Teacher 都有效、GRPO/稠密信号存在普遍优劣，或结果达到榜单 SOTA。

## 模型与 anchor

E2B 与 E4B 都从 Base checkpoint 开始，只使用同一份 immutable `D_anchor=10k` verified traces，保持 chat template、assistant-only mask、样本顺序、max length、SFT token/epoch budget 与 LoRA module classes 一致。LoRA 固定 rank 32、alpha 64、dropout 0；准确 module names 在 C0 introspection 后锁定。视觉/音频 encoder、PLE、token embeddings 和 LM head 默认冻结。

`D_select=500` 只允许选择两个 SFT checkpoint。选择结束后，另一个始终 sealed 且 family-disjoint 的 `D_teacher_gate=500` 只解封一次。E4B Teacher 必须相对 E2B Student 同时满足：accuracy paired 95% CI 下界>0、点估计≥+5pp、verified-solution NLL 更低、parse rate 不下降、tokenizer/hash/token IDs 兼容。失败则 primary OPD 停止；E4B-it 不得静默替代。

## 最小五臂反事实

| Arm | Stage 1 | Stage 2 | 作用 |
|---|---|---|---|
| A0 | SFT 2M | SFT 2M | continued-training control |
| A1 | GRPO 2M | GRPO 2M | sparse signal effect |
| A2 | OPD 2M | OPD 2M | dense signal effect |
| A3 | OPD 2M | GRPO 2M | dense→sparse order |
| A4 | GRPO 2M | OPD 2M | sparse→dense order |

全部从同一 E2B anchor 开始，全部在 2M 边界重置 optimizer/scheduler，全部运行预注册 seeds 101/202/303。同一 objective 跨臂/阶段必须复用相同 resolved config hash；不能针对结果重新调参。

每阶段的 `2M` 是实际进入已执行 optimizer update 的有效 Student objective positions。prompt、padding、EOS 后 token、纯 rollout/Teacher/old-policy forward token和 skipped GRPO zero-variance groups 不计。最后一批通过稳定 budget mask 精确闭合。

## 算法合同

### GRPO

- reward：exact/symbolic correctness，正确 1，其余 0；无格式/长度 reward 与 learned RM；
- `group_size=8`、`loss_type=dr_grpo`、`epsilon=0.2`、`beta=0`、`num_iterations=1`；
- token-level importance sampling、group reward scaling、temperature 1.0；
- old policy 与 rollout weights 每 generation batch 刷新/同步；prompt 与 rollout RNG 独立；
- 零方差组不重采样；effective-group rate<30% 时停止；
- completion cap 默认 2048；冻结 pilot truncation>5% 时所有正式臂统一改4096。

### OPD

- 每次 update 前由当前 Student 新生成 completion，sampling IDs stop-gradient；
- frozen Teacher 在完全相同 Student prefix 上给分布；
- temperature 1.0、full-vocabulary chunked reverse KL `KL(Student||Teacher)`；
- mask 仅覆盖生成首 token 至首个 EOS（含 EOS）；
- 逐 token 对全 vocab 求 KL，再按全 batch 有效 token 数归一；
- exact chunked/fused kernel 必须与 tiny full-tensor value/limit/gradient oracle 对齐；top-k 不能用于主结果救场。

## 数据与 benchmark

- `D_anchor=10k`：双模型同源 SFT；
- `D_select=500`：SFT checkpoint selection；
- `D_teacher_gate=500`：独立 Teacher qualification；
- `D_dev=500`：objective 内最多两档 LR/stability pilot；
- `D_core=2k prompts`：五臂 intervention；
- `E`：MATH-500、GSM8K、MathArena ArXivMath 06/2026、AIME 2026、IFEval、MMLU-Pro 与 sealed 200 题。

所有 split 在 source/problem/template family 层面互斥，并对 prompt、solution、trace 做 exact/fuzzy 去污染。只能声称本项目 post-training data 未包含 test，不能证明 Base 预训练无污染。MathArena 06/2026 作为模型发布后 freshness sentinel，但因仅 49 题与 floor 风险，不承载主 claim。

## 统计预注册

- C1：A1−A0、A2−A0；两个 item-level paired randomization p-values 用 Holm 控制 FWER=.05；95% item-bootstrap CI lower>0 且点估计≥+2pp 才达到项目成功阈值。
- C2：A3−A4 是唯一 order contrast；superiority 需 95% CI 不含0且绝对差≥2pp；否则仅当 TOST 90% CI 全部位于[-2,+2]pp才称 practical equivalence。
- 推断条件于三个预注册 training seeds：每个 item 携带完整三-seed prediction vector，bootstrap/randomization 只在 item 层进行；单列每 seed 效果，不推广到所有 training-seed population。
- sampling eval 每题固定 n=8，T=.7、top-p=.95、top-k=0；使用标准无偏 pass@k estimator。
- IFEval 相对 A0 的 non-inferiority margin 为 -2pp。

## 双预算与真实成本

E1 只严格匹配 Student backward loss tokens；prompt exposure、Student FLOPs与数据量都只报告，不称 matched。E2 记录全部 rollout tokens、Student/old/reference/Teacher forwards、Student backward、accelerator-seconds、GPU-hours、峰值显存和可得能耗。

成本拆分为：共享 Student anchor `C_anchor`、共享 Teacher SFT `C_teacher`、逐臂 `C_arm`。同时报告 marginal、cold-start pipeline 与 campaign total，Teacher 构建在 campaign 中只计一次。

主矩阵共有 5×3×4M=60M Student loss tokens，但这不是端到端 token 数。Gemma 4 E2B/E4B 约 5.1B/8B total weights，官方 BF16 静态内存估计合计约29GB；推荐候选资源为2×80GB或4×48GB。450–900 GPUh 只是 profile 前排期区间。

## 执行 gates

1. G0/C0：模型/processor/tokenizer、LoRA target、text-only freeze、版本组合；
2. G1：license、split、去污染、evaluator adversarial tests 与≥99%人工一致率；
3. G2/C4：双 SFT 可复现，独立 Teacher gate 通过；
4. G3/C2：GRPO loss/refresh/sync/skipped-group 语义通过；
5. G4/C3：OPD exact KL、mask、gradient 与 tokenizer 通过；
6. C5：四类 100-step profile 后 campaign 成本有30%余量；
7. G5：A0–A4 全部三个 seeds 完成；
8. G6：统计、结果与简历数字经过 claim audit。

当前只完成前期规划，因此 execution status 仍是 CONDITIONAL。若 C5 不闭合，只允许所有臂同步降低 `U` 或统一缩短 cap 后重新 profile；不允许换旧模型、删臂、量化 Teacher 或近似 KL 来静默改变问题。

## 12 周输出

- 第1–2周：公式、数据/evaluator、兼容性与 profile；
- 第3–4周：E2B/E4B same-lineage SFT 与 Teacher gate；
- 第5周：冻结预算、objective configs 和统计预注册；
- 第6–9周：GRPO/OPD correctness、A0–A4 stage 1/2；
- 第10周：确认性/支持性评测与成本核算；
- 第11–12周：error taxonomy、负结果、技术报告、README、10分钟讲稿与追问题库。

最终作品必须包含：可复现配置与 hashes、E1/E2 两张成本表、C1/C2 统计表、训练行为曲线、一次 reward-hacking 负例、失败实验记录，以及每个简历数字到 immutable result 的证据链。
