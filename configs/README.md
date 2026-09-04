# Config contract

实现阶段按 `models/`、`data/`、`train/`、`eval/` 拆分 YAML。每次 run 只能引用不可变配置，启动时生成 resolved config 与 SHA-256；正式结果禁止在原 run 目录内覆盖配置。

必须显式记录：model/data revision、chat template hash、LoRA targets、objective contract、Student loss-token budget、seed/RNG streams、generation config、evaluator commit 和硬件拓扑。

D07 已定义 generation protocol 的 JSON contract：greedy 禁止含糊的 sampling 字段，sampling 必须显式给出 n/temperature/top-p/top-k/base-seed/namespace；system prompt、chat-template SHA、max tokens、EOS/stop 全部参与 protocol hash。`configs/project.yaml` 记录了绑定 `c948fe2e…` 的 formal synthetic audit revision、implementation、canonical report 与 pretty-file hashes；这些不可直接作为真实 benchmark run config，D15/D20 必须重新冻结正式配置。
