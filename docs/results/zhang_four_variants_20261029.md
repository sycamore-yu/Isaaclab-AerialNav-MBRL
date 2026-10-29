# Zhang 四方案正式训练记录（2026-10-29）

实现提交：`d471060`。改动前基线：`242b494`，标签 `baseline/zhang-window-loss-d160-20261029`。新训练不读取历史模型。PPO 已按用户要求停止。

## 共同配置

1024 环境、随机种子 0、H=150 策略步、100 Hz 物理与 10 Hz 策略、160×120 原始深度、12×16 网络深度、相机 batch=16。四组使用相同的边界损失（系数 1、margin=0.5 m、scale=0.5 m）、AdamW 学习率 0.001、400 更新余弦周期、梯度裁剪 5 和其余原损失系数。窗口方案 W=20，方案 1 直接按策略步计算速度项。

所有组依次完成 300/60、400/120、300/60、400/100 四阶段。前两阶段关闭额外扰动，后两阶段开启。每阶段按至少 1024 个自然结束回合的训练首次到达率达到 80% 晋级。训练不设更新次数上限；阶段间仅加载同一实验的前序检查点。

## 运行目录

以下路径相对仓库根目录。

| 方案 | 配置 | 目录 | 启动状态 |
|---|---|---|---|
| 1 | `step` | `logs/curriculum/zhang_four_step_s0_20261029` | 正式训练已启动，未验收 |
| 2 | `continuous` | `logs/curriculum/zhang_four_continuous_s0_20261029` | 正式训练已启动，未验收 |
| 3 | `dcol` | `logs/curriculum/zhang_four_dcol_s0_20261029` | 正式训练已启动，未验收 |
| 4 | `truncated_window` | `logs/curriculum/zhang_four_truncated_window_s0_20261029` | 正式训练已启动，未验收 |

启动参数、提交号、时间及进程组记录在每组同名 `.launch.json` 和 `logs/curriculum/zhang_four_variants_20261029.study.json`。四组在同一 RTX 4090 上并行；当前墙钟耗时受共享 GPU 调度影响。

汇总文件为 `logs/curriculum/zhang_four_variants_20261029_comparison.json` 与 `.md`。单组真实统计来自其 `zhang/<stage>/training_statistics.jsonl`；优化指标来自 `updates.jsonl`；最终通过状态必须由完整四阶段 `training_result.json` 支持。TensorBoard 使用 `http://100.92.244.71:6012/`。

## 实现验证

真实证据汇总见 [zhang_four_variant_checks.json](zhang_four_variant_checks.json)：75 项测试、30 个子测试通过；四方案分别完成 64 环境、两次 H=150 的真实 PhysX 采样和反传。均无非有限状态、非有限梯度或执行动作不一致。这些是有限实现检查，不计入正式成功率。

两环境的 160×120 原始深度和 12×16 池化结果均已实测。方案 2 读取 16 架无人机全部 80 个 USD collision 属性，均为关闭；注入碰撞后到达仍保持首次碰撞结果。方案 3 实测首次碰撞子步为 1，其位置与策略步末位置相差约 0.0675 m，保存了同一子步动态障碍位置并保留重置前快照。

两个初次短检查在 Kit 启动阶段失败，栈包含 OpenBLAS 线程清理和 fork；设 `OPENBLAS_NUM_THREADS=1`、`OMP_NUM_THREADS=1`、`MKL_NUM_THREADS=1` 后重试通过，正式训练子进程沿用这些设置。失败原始日志保留，不算作训练失败或速度测量。

方案 3 明确采用 `hybrid_dynamic_dcol`：凸动态盒体/有限圆柱使用 DCOL，静态非凸地形保留实际 mesh 原始 unsigned clearance。曾检查的小三角柱分解会产生人工内部面梯度，因此未用于正式训练；不能把本组解释为已获得整个静态非凸世界的穿透梯度。完整语义见 [四方案比较](../ZHANG_FOUR_VARIANTS.md)。
