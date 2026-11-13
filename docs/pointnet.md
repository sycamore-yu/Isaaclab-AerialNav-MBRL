# PointNet 点云导航

通过现有 Zhang 入口选择 PointNet 配方：

```bash
export PYTHONPATH="$PWD/src"
.venv/bin/python scripts/train_zhang.py presets=pointnet \
  --num_envs 128 --log_dir /absolute/new-pointnet-run
```

PointNet 默认从预训练 S0 开始。
配置使用 240 个静态障碍、0 个动态障碍，并关闭额外扰动。
达到训练阶段阈值后结束，不自动进入 S1。
回合规则与报告窗口见[训练课程](TRAINING_CURRICULUM.md)。

## 输入与策略

机体 LiDAR 的射线方向随无人机的滚转、俯仰和偏航变化。
LiDAR 共 5400 条射线，量程 20 m。
水平覆盖 360°，垂直覆盖 60°，分辨率为 2°。
命中点由世界坐标转换为机体 XYZ。
未命中射线的 XYZ 为 0，validity 为 0。
validity 用于排除无效点，不作为可学习的第 4 个输入通道。

PointNet 逐点特征维度为 3→64→128→1024。
LeakyReLU 的负斜率为 0.05。
全局最大池化后投影为 192 维特征，再进入 GRU192。
策略结合点云和 10 维自身、目标状态，输出 3 维机体加速度。

训练默认使用 20 Hz、H160、完整 2 s 因果速度窗口。
有效点打包保留全部测量点。
编码器默认以 64 个环境为一组计算，使用 BF16 仿射运算与 `torch.compile`。
动力学、损失、GRU 和动作保持 FP32。
执行选项为 `--point_microbatch_envs`、`--point_precision` 和 `--point_encoder_backend`。

## 训练目标与检查点

速度目标组合速度模长和各轴 Huber 误差，系数分别为 0.8 和 0.6。
避障损失用非负、停止梯度的接近速度加权距离惩罚。
其他目标为加速度平方范数、按回合掩码的 Jerk 正则和目标高度区间约束。
外层损失权重、H160 和 2 s 速度窗口均为项目参数。
AdamW 的默认学习率为 `1e-4`，weight decay 为 `0.01`。

检查点记录 `method="pointnet"`、观测约定、传感器配置和动作维度。
加载时先核对这些字段，再加载权重。
同阶段继续训练使用 `presets=pointnet --resume_checkpoint` 和新输出目录。

## 录制真实轨迹和命中点

录制需安装 `recording` 可选依赖。
全新环境可执行 `uv sync --frozen --extra recording`。
使用已有策略检查点录制一个回合：

```bash
.venv/bin/python scripts/record_pointcloud.py presets=pointnet \
  --checkpoint /absolute/pointnet-checkpoint.pt \
  --output /absolute/new-pointnet-recording --seed 0 --terrain_seed 0
.venv/bin/rerun /absolute/new-pointnet-recording/navigation.rrd
```

输出目录必须不存在。
`navigation.rrd` 包含实际地形、无人机位姿、飞行轨迹和有限命中点。
`trace.npz` 保留相同的测量点、位姿和有效性索引。
`result.json` 记录策略、场景和回合结果。
轨迹在终止重置前结束。

独立评估使用 `evaluate.py --method zhang presets=pointnet`。
