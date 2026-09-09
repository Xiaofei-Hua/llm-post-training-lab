# Architecture Decision Records

## ADR-011：研发聚焦算法、框架、性能与指标（2026-09-09）

- **状态**：Accepted（用户明确要求）。
- **决策**：算法机制、可执行训练框架、性能优化和指标解释共同构成主成果；性能优化作为独立技术方向，按 workload 和测量范围表述。
- **执行变化**：D09 改为模型适配与可训练参数接入，D10 为统一训练循环，D11 为性能剖析与优化，D12 为 CPU 端到端学习实验；D18–D20 加强学习动态与性能对照，D23–D24 以分析和技术作品为交付。
- **减负**：复用 D05–D08，哈希/schema/溯源/审计只修实际阻塞。CPU 数值验证、学习 smoke 和 microbenchmark 不需新建研究 claim；每轮聚焦一个主要技术目标，允许修改必要依赖。
- **保留**：D01–D08 完成状态、核心 24 模块、五臂两阶段与三 seed、数据隔离、公开边界、结果真实性、uv.lock 和 CPU 授权范围。
- **成果口径**：已实现能力与已测 CPU 性能可提前展示；Gemma 效果、GPU 吞吐和加速倍数必须等待对应实验。旧日期快照和历史评审不改写，当前无日期 canonical 文档执行本决策。

## ADR-001：选择数学推理作为首个训练域

- **状态**：Accepted
- **理由**：答案可程序验证，适合 GRPO；同时能比较 SFT、DPO 与 OPD，不依赖主观 reward model。
- **代价**：结论不能直接外推到开放式对话、Agent 或 safety alignment。

## ADR-002：采用分支实验而非固定线性配方

- **状态**：Accepted
- **理由**：不预设 `SFT→OPD→RL` 或 `SFT→RL→OPD` 的优劣，先比较各训练信号，再比较顺序。
- **代价**：run 数增加；单 seed 只能筛 objective 内配置，五个结果臂都必须跑 3 paired seeds。

## ADR-003：前沿模型是硬约束，使用 Gemma 4

- **状态**：Accepted（用户明确要求）
- **理由**：Gemma 4 E2B/E4B 是 2026 年发布、截至项目启动时更新的前沿小型模型，具备 hybrid local/global attention、p-RoPE、per-layer embeddings 和原生多模态能力；官方 TRL 已列出 Gemma 4 GRPO 支持。
- **具体选择**：E2B Base→SFT Student anchor；E4B Base 使用同一 `D_anchor` SFT 后冻结为 Teacher。E4B-it/12B/31B 仅作核心完成后的 sensitivity/stretch。
- **回退边界**：资源不足只允许所有臂同步降低 `U` 或缩短上下文并重新 profile，不退回 Qwen3。QLoRA 若研究必须新建实验族，不能作为主矩阵静默回退。

## ADR-004：使用前沿架构，但首期不比较多个架构

- **状态**：Accepted
- **理由**：固定 Gemma 4 E2B 后比较后训练算法；避免在核心表中同时混入架构变化。
- **后续**：核心结果完成后再选择另一个 2026 架构做 portability study。

## ADR-006：主矩阵收缩为 GRPO–OPD 交互

- **状态**：Accepted（Round 1 review）
- **核心五臂**：SFT continuation、GRPO、OPD、OPD→GRPO、GRPO→OPD。
- **DPO**：降为单 seed offline shadow baseline，不进入主 claim。
- **Reward**：exact verifier 是唯一主 reward；format reward 只做一次受控负例。

## ADR-007：使用双预算口径

- **状态**：Accepted（Round 1 review）
- **Signal efficiency**：唯一严格匹配量是 Student non-padding backward loss tokens；canonical prompt distribution、exposure 与 Student FLOPs只审计。
- **Practical efficiency**：另报所有 rollout/Teacher/reference forward、GPU-hours 与 accelerator-seconds。
- **禁止表述**：不再笼统声称所有算法“matched-token”。

## ADR-008：Teacher 使用同数据谱系，而非 instruction checkpoint

- **状态**：Accepted（Round 2 review）
- **理由**：直接使用 E4B-it 会把未知 instruction/post-training recipe 与 capacity gap 混入 dense signal。E4B Base 与 Student 共用 immutable `D_anchor`、模板、mask、顺序和 token/epoch budget，可显著收紧归因。
- **失败策略**：same-lineage Teacher 不过 capability gate 时停止 primary OPD；E4B-it 只能以 external-post-training sensitivity 新建实验族。
- **成本**：显式计入一次性 `C_teacher`，同时报告 marginal、cold-start 和 campaign total。

## ADR-009：五臂统一两阶段网格

- **状态**：Accepted（Round 2 review）
- **设计**：A0 SFT/SFT、A1 GRPO/GRPO、A2 OPD/OPD、A3 OPD/GRPO、A4 GRPO/OPD；每阶段 2M Student loss tokens。
- **理由**：所有臂在 2M 边界统一重置 optimizer/scheduler，排除 stage count 与 scheduler restart 混杂。

## ADR-010：确认性统计只使用 MATH-500

- **状态**：Accepted（Round 2 review）
- **C1**：A1−A0 与 A2−A0，Holm 控制 family-wise α=0.05。
- **C2**：A3−A4 是唯一 order contrast；等效性要求 TOST 90% CI 完全位于 ±2pp。
- **补充集**：GSM8K、MathArena、AIME、IFEval、MMLU-Pro 不拼接成含义不清的 composite。

## ADR-005：公开与内部材料物理隔离

- **状态**：Accepted
- **理由**：项目最终应可公开；内部阅读只用于个人学习，不进入 Git。
- **实现**：`notes/private/` 与 `*.private.md` 被 `.gitignore` 排除。
