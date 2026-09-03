# Package boundaries

当前只在 CPU 上逐模块实现算法与框架；模型下载、训练栈集成和 GPU run 仍受 gate 约束：

- `data/`：registry、schema、quality funnel、family split、contamination audit；
- `rewards/`：answer parser、exact/symbolic verifier、attack corpus；
- `train/`：SFT、GRPO、OPD adapters 与 token/cost accounting；
- `evaluation/`：frozen generation/evaluator、paired statistics；
- `analysis/`：claim tables、Pareto、error transitions。

训练器不得读取 sealed evaluator answers；Teacher 接口不得返回给数据构建流程。

## 已实现

- `train/torch_loss_budget.py`：D01 的 batched PyTorch masks、GRPO zero-variance group 过滤、tensor prefix 截断与 selection digest；
- `train/loss_budget.py`：schema-versioned reservation/counter、失败 step 语义、checkpoint state 与 update ledger。

当前没有保留旧标量 API 或占位 trainer。D01 是训练器可调用的真实张量层，但 optimizer/AMP、gradient accumulation 和 distributed collect/scatter 仍由后续 trainer integration 模块负责。
