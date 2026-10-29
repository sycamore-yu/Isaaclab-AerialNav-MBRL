# NavRL 与 Zhang · Isaac Lab 3.0

本项目在正式 Isaac Lab 3.0.0 / PhysX 中提供两个独立单机导航任务。并行无人机共用静态地形和动态物体，不相互碰撞。

| 方法 | Gym 任务 | 训练方式 |
|---|---|---|
| NavRL | `Isaac-NavRL-Hummingbird-Direct-v0` | 官方 RSL-RL PPO、共享编码器、Beta 速度动作 |
| Zhang | `Isaac-NavRL-Zhang-Hummingbird-Direct-v0` | CNN + GRU192、PhysX 前向、可微质点模型反向和状态对齐 |

## 训练与验收

当前工作已统一到 `main`，旧训练停止并保留检查点。新九组对照从干净主分支随机初始化，优先 20 Hz，见 [main 整合与对照](docs/MAIN_COMPARISON_20261029.md)。五阶段依次完成预训练 S0、S1、S2 和微调 S1、S2，各阶段自然首次到达率达到 80% 后晋级。

| 阶段 | 静态障碍 | 动态障碍 | 额外扰动 |
|---|---:|---:|---|
| 预训练 S0 | 300 | 0 | 关闭 |
| 预训练 S1 | 300 | 60 | 关闭 |
| 预训练 S2 | 400 | 120 | 关闭 |
| 微调 S1 | 300 | 60 | 开启 |
| 微调 S2 | 400 | 100 | 开启 |

预训练从随机初始化开始。后续阶段加载本次实验的前序检查点，保留网络和优化器。每阶段重建地图并清空回合历史。正式验收默认使用1024个并行环境且至少1024，无更新次数上限。小规模比较须显式使用`--screening`，即使五阶段门槛均通过也不标记正式验收。输出目录必须不存在。

```bash
cd /home/tong/tongworkspace/free/isaaclab-navrl
.venv/bin/python scripts/run_acceptance.py \
  --method zhang --output logs/curriculum/sim2real_zhang_seed0 --train_seed 0 \
  --train_args --variant dcol --timesteps 150 --max_grad_norm 5 --camera_batch_size 64
```

NavRL/PPO 使用独立的 `--method navrl` 入口；Zhang 超参数只传给 Zhang 入口。PPO 的观测、频率和动作与 Zhang 不同，属于方法对照，不能解释为只换优化器。对照规范与有时间标记的历史状态见 [truncated 与 PPO 对照](docs/ZHANG_TRUNCATED_PPO_CONTROLS_20261029.md)。有限步调试使用各训练脚本的 `--max_iterations`，不计入正式验收。

运行环境由 `.venv`、`pyproject.toml` 和 `uv.lock` 固定：Python 3.12、Isaac Lab 3.0.0、Isaac Sim 6.2、RSL-RL 5.5.1。

## 任务配置

| 配置 | NavRL | Zhang |
|---|---|---|
| 物理频率 | 62.5 Hz | 100 Hz |
| 策略频率 | 62.5 Hz | 10 Hz |
| 每次更新 | 32 个策略步 | 默认 150 个策略步 |
| 单回合超时 | 2200 个策略步 | 352 个策略步 |
| 观测 | 8 维状态、1×36×4 静态 LiDAR、5×10 动态物体真值 | 深度图池化为 1×12×16，加 10 维状态 |
| 动作 | 每轴 [-2,2] m/s 目标速度 | 六维网络输出解码为三维世界加速度，无每轴速度限制 |

新增 Zhang `--timing unified_20hz` 将物理、策略动作和传感器统一到20 Hz，完整2秒速度窗口对应40步。默认时钟保持上表设置。

两任务保留 Hummingbird USD、Lee 控制律和 RotorGroup 电机响应。两环境的深度相机默认采样 160×120 像素，内参随分辨率缩放以保留原视场，再完成逆深度变换、噪声和 12×16 池化。NavRL 策略仍使用原 LiDAR 特征。动态物体同时进入相机、碰撞检测和损失。

Zhang保留速度匹配、避障、碰撞、加速度、Jerk和速度预测辅助项。高度项默认`height_loss_mode=target_exp`，可选目标SmoothL1（`target_huber`）或旧边界SmoothL1（`boundary_huber`）；目标使用当前回合导航目标z。默认step逐步速度监督；continuous/dcol默认20步窗口加合格终止短窗，显式`velocity_loss_mode=full_window`保留原完整窗口。continuous的碰撞和高度越界不重置，首次失败仍锁存。新接口及历史设置的区别见[新机制](docs/ZHANG_NEW_MECHANISMS_20261029.md)。

actor默认AdamW、weight_decay=.01，可显式选Adam；预训练lr默认.001，微调.00005。cosine默认周期400次更新；Plateau默认patience2，可独立配置。AHAC共同H默认初始32、范围16–300，BPTT默认H150。全局梯度裁剪为5，激活检查点控制显存。可选`height_priority`梯度投影仅用于BPTT，需用实际参数位移验证，不能把原始梯度投影当作成功率改善。

## 扰动与日志

两任务共用 `SimToRealCfg`。位置、速度、姿态、LiDAR、深度噪声，观测和动作延迟，控制响应，质量、惯量、推力系数均可独立配置。观测扰动仅进入策略输入。延迟由官方 `DelayBuffer` 按策略步推进，部分回合重置清空对应历史。

训练日志记录首次到达率、碰撞率、超时率、成功到达时间、return 和每步 reward。Zhang 另记原损失、梯度范数和 `executed_action_max_error`。动作差分在 Zhang 中沿用已有 Jerk 损失。参数监控另记 Optim/param_norm、Optim/update_norm、Optim/update_ratio，以及每层和六维输出行的变化率；稳定模型按5%起每5个百分点保存历史；性能退化先降学习率，观察后再恢复同阶段最高稳定档，见[分档协议](docs/migration/RECOVERY_LADDER_20261029.md)。归一化与 Plateau 在取得训练对照证据前均不替换默认。

真实成功率来自 `training_statistics.jsonl`；阶段状态来自 `training_result.json`。TensorBoard 服务使用已有地址 `http://100.92.244.71:6012/`，日志根目录为 `logs/curriculum/`。

## 决策与证据

- [训练课程与验收](docs/TRAINING_CURRICULUM.md)
- [共享 Sim-to-Real 扰动](docs/SIM_TO_REAL_TRAINING.md)
- [NavRL 迁移决策](docs/migration/NAVRL_ISAACLAB_3_MIGRATION.md)
- [Zhang 迁移决策](docs/migration/ZHANG_MIGRATION_DECISIONS_1_4.md)
- [实现与验证状态](docs/IMPLEMENTATION_STATUS.md)
- [SHAC/AHAC 适配与边界](docs/migration/SHAC_AHAC_ADAPTATION.md)
- [DCOL 速度监督与 PPO 方法对照](docs/ZHANG_TRUNCATED_PPO_CONTROLS_20261029.md)
- [相机批量位姿读取经验](docs/migration/CAMERA_POSE_AGGREGATION_EXPERIENCE.md)

最初迁移基线为 `2456a5b`；新机制前冻结基线为 `f6d5fd4`，归档与相机经验提交为 `1597ffe`。历史 NavRL 在 350 静态、60 动态场景的最高训练首次到达率为 49.28%，未通过旧验收。正式结果以本轮真实训练记录为准。
