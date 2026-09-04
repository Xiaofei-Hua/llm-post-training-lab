# Config contract

实现阶段按 `models/`、`data/`、`train/`、`eval/` 拆分 YAML。每次 run 只能引用不可变配置，启动时生成 resolved config 与 SHA-256；正式结果禁止在原 run 目录内覆盖配置。

必须显式记录：model/data revision、chat template hash、LoRA targets、objective contract、Student loss-token budget、seed/RNG streams、generation config、evaluator commit 和硬件拓扑。

D07 已定义 generation protocol 的 JSON contract：greedy 禁止含糊的 sampling 字段，sampling 必须显式给出 n/temperature/top-p/top-k/base-seed/namespace；system prompt、chat-template SHA、max tokens、EOS/stop 全部参与 protocol hash。

D08 固定三个 training seeds、MATH `level` 分层、10,000 bootstrap、100,000 sign-flip、PCG64 base seeds、exact type-7、Holm family 和 ±2pp sequential TOST。实际随机流只绑定 outcome-independent resampling design；完整 result panel hash单独进入 analysis identity。`configs/project.yaml` 记录绑定 `9a5cee9…` 的 synthetic audit revision/source/analysis/audit/file hashes；这些不是 D23 真实 endpoint config，D15/D20/D22 后仍须用真实冻结 identities 构造 panel。
