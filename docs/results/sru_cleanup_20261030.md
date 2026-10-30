# SRU 代码清理

本轮按 `codebase-design` 和 `code-simplification` 检查 SRU 适配提交。
原功能提交 `fe9684a` 及训练记录 `0ba961b` 已在 main。
清理在独立分支 `refactor/sru-cleanup-20261030` 完成验证，再快进合并至 main。

## 修改

- `ZhangPolicy.forward` 删除冗余的 `output_hidden` 别名，两种模型均明确使用 `h` 生成输出。
- 补全 Interface 说明：GRU 返回 `h`，SRU 返回打包的 `[h, c]`，返回宽度由 `recurrent_state_size` 指定。
- 术语表只保留 Zhang 的定义，删除本次任务的范围说明。范围仍记录在 ADR 和适配说明中。

`ZhangPolicy` 是两种循环模型共用的 Module。
状态打包集中在该 Module 内，调用方通过同一 Interface 处理重置、截断和诊断。
当前 Seam 已支持两个实际选择，不需增加另一层 Adapter。
参数名称、检查点键名、初始化、计算顺序、错误校验及训练配置保持原行为。

## 验证

- 99 项测试、325 个子测试通过，测试文件未改动。
- 上游数值对照显式使用 `SRU_REFERENCE_ROOT=/tmp/sru-pytorch-spatial-learning-audit`，未跳过。
- GRU 和 SRU 均对比清理前后的 3 步输出、最终记忆和梯度。
  显式初态及隐式零初态共 4 组结果完全一致。
- Ruff、格式检查、`git diff --check` 通过。
- `uv build --wheel` 通过。

本轮使用 CPU 验证，未重新启动 GPU 仿真。
既有长期训练继续使用冻结提交 `fe9684a`；训练快照和历史结果不随 main 更新。
