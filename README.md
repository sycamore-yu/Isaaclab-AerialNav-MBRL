# NavRL 与 Zhang · Isaac Lab 3.0

本项目在正式 Isaac Lab 3.0 / PhysX 中提供两个独立单机导航任务。1024 架并行无人机共用静态地形和动态物体，不相互碰撞。

| 方法 | Gym 任务 | 训练方式 |
|---|---|---|
| NavRL | `Isaac-NavRL-Hummingbird-Direct-v0` | 官方 RSL-RL PPO、共享编码器、Beta 速度动作 |
| Zhang | `Isaac-NavRL-Zhang-Hummingbird-Direct-v0` | CNN + GRU192、PhysX 前向、可微质点模型反向和状态对齐 |

## 训练与验收

两个任务默认使用 **60→80→100→120 动态障碍课程**。静态障碍数为 350。当前阶段的训练成功率超过 80% 后晋级。验收目标是 **100 动态障碍阶段的训练成功率约 80%**。保存 `checkpoint_best_100.pt`，随后继续训练 120 阶段。正式验收入口在 100 阶段达标且完成首次 120 阶段训练统计后结束；120 阶段不增加成功率门槛。

正式训练从随机初始化开始，**没有训练步数上限，不加载预训练权重，不续训，不运行独立评测环境**。统计来自自然结束的训练回合。沿用原 NavRL `EpisodeStats` 的汇总频率，累计至少 `num_envs` 个完成回合后汇总一次。成功指碰撞前首次到达目标 0.5 m 范围；到达不改变原任务终止条件。

按顺序完成两个任务：

```bash
cd /home/tong/tongworkspace/free/isaaclab-navrl
.venv/bin/python scripts/run_acceptance.py \
  --method both --output logs/curriculum/fresh_seed0 --train_seed 0 \
  --train_args --timesteps 150 --velocity_window_s 6.0 --max_grad_norm 5
```

入口先运行 NavRL。NavRL 正常通过后释放其仿真进程，再启动 Zhang。两者分别写入 `navrl/` 和 `zhang/`，各自从随机初始化开始。任一任务出错或未通过就退出时，入口停止并报告错误。单独运行其中一个任务：

```bash
cd /home/tong/tongworkspace/free/isaaclab-navrl
.venv/bin/python scripts/run_acceptance.py \
  --method navrl --output logs/curriculum/navrl_seed0 --train_seed 0
```

```bash
.venv/bin/python scripts/run_acceptance.py \
  --method zhang --output logs/curriculum/zhang_seed0 --train_seed 0 \
  --train_args --timesteps 150 --velocity_window_s 6.0 --max_grad_norm 5
```

输出目录必须不存在。入口固定 1024 个并行环境和 350 个静态障碍，并设置本机启动需要的 `OPENBLAS_NUM_THREADS=1`。运行环境由 `.venv`、`pyproject.toml` 和 `uv.lock` 固定：Python 3.12、Isaac Lab 3.0.0、Isaac Sim 6.2、RSL-RL 5.5.1。

### 日志与策略

`training_result.json` 持续记录已完成更新数、当前难度、各阶段最佳训练结果和 `acceptance_passed`。`training_statistics.jsonl` 保存每次训练回合汇总，包括样本数、成功率、碰撞率、超时率、越界率、无效状态率和成功到达时间。相同回合还统计平均 return、平均回合长度及每策略步平均 reward。平均 return 按完成回合数加权，每步 reward 为这些回合的总 return 除以总策略步数。`curriculum.json` 保存最近一次课程统计。

TensorBoard 标签为 `Train/60/*`、`Train/80/*`、`Train/100/*`、`Train/120/*` 和 `Curriculum/dynamic_obstacles`。NavRL 的 TensorBoard 位于运行目录，Zhang 位于运行目录的 `tensorboard/`。原训练回报和即时 `reach_goal` 记录继续保留。

每个阶段有独立的 `checkpoint_best_<数量>.pt`。周期检查点使用 `checkpoint_latest.pt`，避免无限训练持续增加文件数。训练状态中的成功标记只有在 1024 环境、350 静态障碍和 100 动态障碍阶段满足目标时才置为真。

## 保留的算法配置

| 配置 | NavRL | Zhang |
|---|---|---|
| 物理频率 | 62.5 Hz | 100 Hz |
| 策略频率 | 62.5 Hz | 10 Hz |
| 每次更新 | 32 个策略步 | 默认 150 个策略步 |
| 单回合超时 | 2200 个策略步 | 352 个策略步 |
| 观测 | 8 维状态、1×36×4 静态 LiDAR、5×10 动态物体真值 | 深度图池化为 1×12×16，加 10 维状态 |
| 动作 | 每轴 [-2,2] m/s 目标速度 | 六维网络输出解码为三维世界加速度，无每轴速度限制 |

两任务保留原 Hummingbird USD、Lee 控制律和 RotorGroup 电机响应。原生 Multirotor/Thruster 接收实际旋翼推力并施加合力矩。相机追踪动态物体；NavRL 的 LiDAR 仅查询静态地形。

Zhang 保留速度匹配、避障、碰撞、加速度、jerk 和速度预测辅助损失，不加入高度损失。速度窗口为 60 个策略步；`velocity_window_s=6.0` 按 10 Hz 策略频率换算。学习率默认 0.001，余弦周期为 400 次更新，对应初始 2000 次更新内 5 个周期。2000 次后每增加 500 次检查进度，训练和调度持续执行。全局梯度范数裁剪默认为 5。`grad_decay=0.4` 按物理步使用 `grad_decay ** dt`。

Zhang 训练通过 PyTorch 激活检查点减少网络中间张量的存储。反向传播重算 CNN 与 GRU，不重复执行 PhysX，也不改变展开长度或损失。

`velocity_window_valid_fraction` 记录同一回合内有效速度窗口的比例。它用于识别回合过短导致的速度训练信号不足，不参与损失。

场景预先分配 120 个动态物体。每阶段按原尺寸类别选取正确数量，并使用该难度的原始布局采样。未激活物体移出传感器范围，不参与动态观测、碰撞和损失。晋级发生在优化更新边界，所有无人机开始新回合，清空跨回合历史，网络和优化器连续保留。

## 验证

```bash
OPENBLAS_NUM_THREADS=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  .venv/bin/python -m pytest -q tests/test_training_curriculum.py \
  tests/test_continuous_runner.py tests/test_zhang_contract.py \
  tests/test_policy_checkpoint_conversion.py tests/test_policy_contract.py
```

```bash
OPENBLAS_NUM_THREADS=1 OMNI_KIT_ACCEPT_EULA=YES \
  .venv/bin/python tests/curriculum_scene.py \
  --output logs/checks/curriculum_scene.json
```

场景检查使用合成成功结果触发晋级，只验证共享物体、深度和梯度，不用于策略验收。`train_navrl.py` 和 `train_zhang.py` 的 `--max_iterations` 仅供有限步调试；正式入口不传该参数。

## 决策与证据

- [训练课程与验收](docs/TRAINING_CURRICULUM.md)
- [NavRL 迁移决策](docs/NAVRL_ISAACLAB_3_MIGRATION.md)与 [ADR](docs/adr/0001-port-navrl-with-original-task-semantics.md)
- [Zhang 迁移决策](docs/ZHANG_MIGRATION_DECISIONS_1_4.md)
- [实现与验证状态](docs/IMPLEMENTATION_STATUS.md)

旧固定难度训练和独立评测记录保留在历史日志中，不作为本轮课程验收。原版检查点转换只用于源码对照，正式训练不读取这些权重。
