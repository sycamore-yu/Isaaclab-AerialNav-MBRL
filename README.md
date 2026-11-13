# 无人机导航 · Newton/Warp

阅读并接受 [NVIDIA Omniverse 许可协议](https://docs.omniverse.nvidia.com/platform/latest/common/NVIDIA_Omniverse_License_Agreement.html)后，使用项目已有的 Python 环境检查 Newton 仿真：

```bash
cd /home/tong/tongworkspace/projects/aerial/isaaclab-navrl
export OMNI_KIT_ACCEPT_EULA=YES
export PYTHONPATH="$PWD/src"
RUN_ROOT="$(mktemp -d /tmp/navrl.XXXXXX)"
.venv/bin/python scripts/smoke.py \
  --method zhang --num_envs 4 --steps 16 --dynamic_obstacles 8 \
  --output "$RUN_ROOT/smoke.json"
```

本仓库使用 Isaac Lab 3.0.0、Newton MJWarp 和 Warp 环境。
并行无人机共用静态地图和动态障碍运动，不相互碰撞。
PyTorch 承担策略网络、训练和张量计算。

## 选择导航方法

| 方法 | 输入与动作 | 训练入口 |
|---|---|---|
| NavRL | LiDAR、8 维状态、动态物体状态；目标速度 | `train_navrl.py`，RSL-RL PPO |
| Zhang | 深度图、10 维状态；世界坐标系加速度 | `train_zhang.py`，BPTT、SHAC 或 AHAC |
| PointNet | 机体坐标系 LiDAR 点云、10 维状态；机体加速度 | `train_zhang.py presets=pointnet`，BPTT |

以下命令分别完成短程训练，输出写入仓库外的新目录：

```bash
.venv/bin/python scripts/train_navrl.py \
  --stage pretrain_s0 --num_envs 4 --max_iterations 2 \
  --log_dir "$RUN_ROOT/navrl"

.venv/bin/python scripts/train_zhang.py \
  --stage pretrain_s0 --num_envs 4 --max_iterations 2 --timesteps 4 \
  --velocity_window_s 0.1 --log_dir "$RUN_ROOT/zhang"

.venv/bin/python scripts/train_zhang.py presets=pointnet \
  --stage pretrain_s0 --num_envs 4 --max_iterations 2 --timesteps 4 \
  --velocity_window_s 0.1 --log_dir "$RUN_ROOT/pointnet"
```

短跑检查环境步进、训练更新和检查点保存。正式课程通过条件见[训练课程](docs/TRAINING_CURRICULUM.md)。
长期微调使用现有 `tune/optim-loop-fs1-20261031` 工作树。

## 评估与诊断

用 Zhang 检查点评估 100 个完整回合：

```bash
.venv/bin/python scripts/evaluate.py \
  --method zhang --checkpoint "$RUN_ROOT/zhang/checkpoint_final.pt" \
  --episodes 100 --stage pretrain_s0 --output "$RUN_ROOT/evaluation.json"
```

NavRL 检查点使用 `--method navrl`。
PointNet 检查点使用 `--method zhang presets=pointnet`。
评估输出记录场景、实际环境类型和每个回合的最终结果。

现役诊断入口为 `benchmark_camera.py`、`profile_step.py` 和 `diagnose_zhang.py`。
这些入口与训练共用运行时 module，并记录实际 frontend 和环境类型。
`record_pointcloud.py` 录制 PointNet 的实际轨迹与 LiDAR 命中点，见[点云导航](docs/pointnet.md)。

## 日志与恢复

`Train/success_rate`、return 和失败率来自自然结束的训练回合。
`Evaluation/` 用于独立检查点评估。
课程间沿用累计更新步数和 TensorBoard 事件目录。
短程更新次数、损失下降和独立评估结果不能替代训练课程通过条件。

Zhang 和 PointNet 的同阶段继续训练使用 `--resume_checkpoint`，并指定新的输出目录。
阶段交接使用 `--checkpoint`。
恢复的学习状态和重新初始化的环境状态见[训练课程](docs/TRAINING_CURRICULUM.md#恢复与阶段交接)。

## 环境与结构

依赖由 `pyproject.toml` 和 `uv.lock` 固定，使用 Python 3.12。
全新 checkout 可执行 `uv sync --frozen` 创建环境。
已有训练使用共享 `.venv` 时，保留该环境及冻结源码。

- [导航 module、interface 与状态所有权](docs/NAVIGATION_ARCHITECTURE.md)
- [课程、完整回合统计与恢复](docs/TRAINING_CURRICULUM.md)
- [PointNet 输入、训练和录制](docs/pointnet.md)

`main` 只维护现役 Newton/Warp 源码、测试和当前使用说明；运行输出使用仓库外目录。
历史命令、实验配置、结果与复现材料归 `archive/newton-cutover-20261113`。
