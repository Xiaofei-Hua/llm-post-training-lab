# Future package boundaries

前期规划阶段不实现训练代码。进入 G0 后按以下边界落地：

- `data/`：registry、schema、quality funnel、family split、contamination audit；
- `rewards/`：answer parser、exact/symbolic verifier、attack corpus；
- `train/`：SFT、GRPO、OPD adapters 与 token/cost accounting；
- `evaluation/`：frozen generation/evaluator、paired statistics；
- `analysis/`：claim tables、Pareto、error transitions。

训练器不得读取 sealed evaluator answers；Teacher 接口不得返回给数据构建流程。
