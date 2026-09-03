# Artifact policy

这里仅存放可公开、体积可控的图表、数据卡、reward card、model/result cards 和报告。模型权重、原始数据与敏感笔记不得提交；它们的地址、revision 和 checksum 只写入 manifest。

## 已实现

- `audits/D05_VERIFIER_AUDIT.json`：对公开合成 attack corpus 的确定性 CPU audit；只包含分类计数、依赖版本、policy/corpus/source/lock/Git hashes 和失败记录，不包含模型或业务数据。正式生成要求 verifier、audit、CLI、`uv.lock` 与本次 corpus 均已被 Git 跟踪且和记录 revision 一致。
