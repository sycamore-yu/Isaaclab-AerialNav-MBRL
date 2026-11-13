# 训练课程与恢复

完整课程由 5 个阶段组成。
每阶段使用自然结束的训练回合判断是否达到 80% 成功率。

| 阶段 | 配置名 | 静态障碍 | 动态障碍 | 时限 | 额外扰动 |
|---|---|---:|---:|---:|---|
| 预训练 S0 | `pretrain_s0` | 240 | 0 | 30 s | 关闭 |
| 预训练 S1 | `pretrain_s1` | 240 | 120 | 35 s | 关闭 |
| 预训练 S2 | `pretrain_s2` | 180 | 180 | 40 s | 关闭 |
| 微调 S1 | `finetune_s1` | 320 | 160 | 40 s | 开启 |
| 微调 S2 | `finetune_s2` | 240 | 240 | 45 s | 开启 |

地图为 120 × 40 m。
Zhang 每回合目标速度为 1–6 m/s。
起终点水平距离满足 `d_xy = 18 s × v_target`。
路线相对正、负 X 轴的偏角为 −8° 至 8°。
X 两端各留 6 m 无障碍区域。

## 完整课程

在项目根目录使用已有 `.venv` 启动课程：

```bash
export PYTHONPATH="$PWD/src"
ARTIFACT_ROOT=/home/tong/tongworkspace/projects/aerial/.worktrees/isaaclab-navrl/archive-newton-cutover-20261113/artifacts
.venv/bin/python scripts/run_acceptance.py \
  --method zhang --num_envs 1024 --train_seed 0 \
  --output "$ARTIFACT_ROOT/curriculum-$(date +%Y%m%d-%H%M%S)"
```

NavRL 使用 `--method navrl`。
输出目录必须不存在，启动器从已提交源码创建独立快照。
预训练 S0 从随机权重开始。
后续阶段只加载本次课程上一阶段的检查点。
阶段切换会重建地图、环境和时序历史。

正式验收使用至少 1024 个环境。
每份报告至少包含 1024 个完整回合。
课程没有默认更新次数上限。
五阶段按顺序达标后，才记录完整课程通过。

小规模检查使用 `--screening`。
直接从某个阶段开始还需指定 `--start_stage`。
这些运行保留跳过阶段的记录，并记录阶段阈值结果。
正式课程通过条件仍要求完整的 5 阶段。

## 回合如何计成功

距目标严格小于 0.5 m 时，记录首次到达及其时间。
首次到达超过 5 s，且无人机当前仍在目标区时，触发到达截断。
离开或重新进入目标区不会重启计时。
到达等待期间，碰撞、越界或无效状态仍计失败。

每个完整回合只计一个最终结果。
成功要求回合结束时仍在目标区，且没有发生失败。
同一步事件优先级为：无效状态、碰撞、越界、最终成功、超时。
首次到达率与最终成功率分别记录。

`Train/success_rate`、return 和失败率按自然完成的报告窗口计算。
平均 return 按完成回合数加权。
每步平均 reward 为同批回合总 return 除以总策略步数。
`Evaluation/` 用于独立检查点评估。

## 恢复与阶段交接

Zhang 和 PointNet 的同阶段继续训练使用新输出目录：

```bash
.venv/bin/python scripts/train_zhang.py \
  --stage finetune_s1 --resume_checkpoint /absolute/checkpoint_stopped.pt \
  --log_dir /absolute/new-run
```

继续训练恢复以下学习状态：

- 网络与 AdamW/Adam 状态。
- 归一化与调度器。
- 随机状态与累计计数。

环境、循环记忆、动作与观测历史、未完成的报告窗口重新初始化。
跨恢复边界的窗口分别解释。
PointNet 继续训练需同时指定 `presets=pointnet`。

阶段交接使用 `--checkpoint`，载入本次课程上一阶段的检查点。
微调交接学习率默认 `5e-5`。
新初始化训练的学习率默认 `0.001`。
PointNet 配方的学习率为 `1e-4`。

同阶段恢复保留检查点保存的终点参考公式。
Zhang 新训练默认使用 `smooth` 终点速度和 2 s 接近时间。
平滑引导使用已知地图净空查询。
首次到达后的截断等待仍为 5 s。

## 修改阶段配置

`apply_training_stage()` 集中设置阶段名、障碍数量、扰动和时限：

```python
from isaaclab_navrl.tasks.navigation.config.hummingbird.env_cfg import ZhangNavRLEnvCfg

cfg = ZhangNavRLEnvCfg()
cfg.apply_training_stage("pretrain_s1")
cfg.scene.num_envs = 64
cfg.configure_scene()
```

显式覆盖使用 `stage_config`，并复制阶段表条目后修改。
后续不带覆盖调用 `apply_training_stage()` 时，会清除上次覆盖。
默认值以 `isaaclab_navrl.training.STAGE_CONFIG` 为准。
