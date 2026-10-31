# 实验数据索引

这组证据包含一个被收紧的初始候选，读取时须区分版本。

| 文件 | 含义 |
|---|---|
| `provenance.json` | 初始候选 e911e96：射线/共享缓存、mesh 表、删除 DelayBuffer clone 三项 |
| `final_provenance.json` | 最终候选：撤回 clone 删除，仅保留两项相机优化 |
| `training_abba.json`、`training_summary.json` | 初始候选与基线的四次完整训练；这组参数偏移促使撤回 clone 清理 |
| `ablations.json` | 逐文件单次 update 定位；不能作为稳定性能基准 |
| `final_training_validation.json` | 最终两项组合的独立四次 update 验证 |
| `training_with_final_summary.json` | 初始四次运行加最终验证，包含最终 checkpoint 对两个基线的 CPU 比较；吞吐分组不混合不同候选 |
| `components.json` | DelayBuffer 所有权/数值、缓存 storage、unproject_depth 直接替换负例、observe 微计时 |
| `ray_baseline_diagnostic.json` | 调用实际旧/新射线方法，包含训练中的 32 环境及尾批 |
| `camera_parity.json`、`camera_parity_physx.json` | Newton/PhysX 原生相机逐像素对照，各 8 阶段 |
| `shared_geometry_parity.json` | 共享 Newton 动态几何与全局移动 |
| `warp_environment_contract.json` | 真正 Warp 环境的延迟、扰动、reset、终止合同 |
| `unit_tests.txt` | 12 项定向 pytest 的原始输出，未隐藏 warnings |
| `verification_commands.json` | 验证命令索引 |
| `component_inventory.json` | 最终生产代码全部 55 文件、顶层符号、行数 |

完整训练日志、源码快照、编译 cache 和 checkpoint 位于同一工作树的 `logs/official_components_20261031/`。本目录只归档体积较小、足以复核结论的数据。报告中的 MiB 对应 2²⁰ bytes；原始日志个别字段沿用 `mb` 名称。

详细解释和保留/撤回决定见 [主报告](../../migration/OFFICIAL_COMPONENTS_AUDIT_20261031.md)。
