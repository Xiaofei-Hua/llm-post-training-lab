# Config contract

实现阶段按 `models/`、`data/`、`train/`、`eval/` 拆分 YAML。每次 run 只能引用不可变配置，启动时生成 resolved config 与 SHA-256；正式结果禁止在原 run 目录内覆盖配置。

必须显式记录：model/data revision、chat template hash、LoRA targets、objective contract、Student loss-token budget、seed/RNG streams、generation config、evaluator commit 和硬件拓扑。
