# Newton 共享几何与 Warp 前端吞吐优化（2026-10-30）

## 结论

在 RTX 4090 上，保留其他同卡任务运行，使用 **32 环境、H=150、20 Hz、300 静态障碍、60 动态障碍** 比较相同的 Zhang DCOL＋BPTT＋GRU192 训练。每个新进程完成 5 次更新；前 2 次作为预热，统计剩余 3 次完整更新的中位时间。一次更新包含 4,800 环境步，计时边界执行 CUDA 同步。

| 组合 | 预热后中位更新耗时 | 吞吐 | 相对同轮 PhysX 吞吐 |
|---|---:|---:|---:|
| PhysX＋Torch | 22.1593 s | 216.61 环境步/s | 1.000 倍 |
| Newton＋Torch | 17.4912 s | 274.42 环境步/s | 1.267 倍 |
| **Newton＋Warp** | **16.5762 s** | **289.57 环境步/s** | **1.337 倍** |

结果来源：`logs/shared_geometry_optimization/benchmark_32env/benchmark.json`。三组运行的 `valid` 为 `true`，每组累计 24,000 环境步，生成可加载的训练检查点，报告状态为 `completed`，非有限状态和梯度事件均为 0。源码摘要检查 `source_unchanged=true`。分组原始更新、训练配置、GPU 使用情况与子进程输出在同一目录。

历史对照（旧代码、不同同卡负载）分别为 146.26、100.08、105.88 环境步/s。该跨时段数据只能辅助判断改动幅度；**不能**将其中的加速全归因于代码改动。新的同轮 PhysX 作为首要对照。

## 机制与代码

旧 Newton 为每个物理世界复制 60 个动态障碍物刚体。32 环境共额外导入 1,920 个运动学刚体。它们已关闭物理碰撞，但仍增加 MJWarp 模型规模、状态处理和位姿写入开销。

现已改为：

1. `make_shared_dynamic_geometry` 在全局 `/World/Dynamic/` 创建一次 USD 几何，不附加 `RigidBodyAPI`、质量或碰撞实体。
2. `DynamicObstacleMotion` 继续以一份 GPU 张量更新位置、速度、尺寸和目标。它独立于任意无人机的回合重置。
3. `NewtonBatchedDepthCamera` 直接读取共享运动张量。固定形状 Warp kernel 把位置广播到各相机的共享网格变换表。静态网格和包围体层次结构（BVH）继续复用；不创建 Newton site 和逐环境目标刚体。
4. 任务对动态障碍物的碰撞检测和代价仍使用原解析距离。之前的动态刚体已禁用物理接触，因此移除它们不改变此项目的动态碰撞响应机制。
5. PhysX 继续使用原全局运动学 `RigidObjectCollection`，不改动其物理路径。Newton 的无人机和静态地形仍通过官方克隆配置工作。

相关实现：`config/hummingbird/env_cfg.py`、`navrl_env.py`、`newton_sensors.py`。Newton＋Warp 的 Warp 任务计算仍复用 `navrl_warp_env.py` 和 `warp_task_kernels.py`，训练反向仍使用 Zhang 的可微质点模型，**没有**通过 Newton MJWarp 求解器反向传播。

## 正确性门槛

`tests/shared_geometry_camera_parity.py` 已通过真实 Newton 场景。测试中 4 个环境、8 个移动障碍，Newton 动态障碍刚体数为 **0**。与官方原生多网格相机对比，初态和障碍物移动后各 76,800 个深度像素的最大绝对误差均为 **0.0 m**，移动会改变相机深度。

`tests/camera_parity.py` 的 4 环境、60 个动态障碍测试覆盖初态、机体姿态、全部动态物体移动、局部重置、像素噪声、局部相机内参、恢复内参及共享运动路径。所有 8 组原始 160×120 深度及 12×16 池化结果均与原生相机一致，最大误差 **0.0**。

`tests/warp_environment_contract.py` 在 4 环境的 Zhang DCOL 场景中通过：逐步输入／动作延迟、质量惯量随机化、间隔扰动、局部重置、自动重置、超时与人工重置叠加、边界事件及无效状态检测。对应记录在 `logs/shared_geometry_optimization/warp_dcol_4env_v2.json`。

新版本还应运行单元、环境隔离与 Sim-to-Real 测试。只有回归通过后才能合并主分支。不会为提高吞吐而关闭实际任务中已有的传感器、避障距离、运动扰动或训练反向过程。

## 运行方式

项目训练、评测和验收脚本的默认组合为 `physics=newton_mjwarp --frontend warp`。只指定 `physics=isaacsim_physx` 时自动选择其唯一受支持的 Torch 前端；显式请求 PhysX＋Warp 会报错。

Isaac Lab 官方 `.venv/bin/isaaclab train` 的 `--frontend` 默认值不受项目控制，因此最快组合仍需明确指定：

```bash
.venv/bin/isaaclab train --task Isaac-NavRL-Hummingbird-Direct-v0 \
  --num_envs 32 --max_iterations 1 --frontend warp physics=newton_mjwarp

PYTHONPATH=src .venv/bin/python scripts/train_zhang.py \
  --stage finetune_s1 --fresh_stage --num_envs 32 \
  --timesteps 150 --max_iterations 1 \
  --log_dir logs/zhang_warp_default_smoke

PYTHONPATH=src .venv/bin/python scripts/benchmark_backends.py \
  --num-envs 32 --updates 5 --warmup 2 --rounds 1 \
  --output logs/benchmarks/recheck_shared_geometry
```

## 边界与后续

- 吞吐来自一个按顺序运行的三配置矩阵；未暂停其他 GPU 训练。由于其他训练会调整负载，结果不能当成独占 GPU 或多轮稳定性结论。
- 动态障碍物的**物理碰撞原本就被禁用**。如果未来需要障碍物与无人机发生真实刚体接触，必须重新设计接触几何，不能沿用纯传感器网格。
- USD 全局几何保留初始静态位姿。当前移动位置只更新用于观测的 Warp 相机网格和策略状态。需要在 USD/Kit 查看器中实时显示障碍运动时，应单独增加仅用于可视化的位姿桥接，不应把渲染同步混入训练热路径。
- Isaac Lab 的实验性 Warp 前端只能与 Newton 组合；它尚未把全部资产写入、执行器和传感器操作迁移为 Warp kernel。详情见 [Isaac Lab 3.0 Warp Environments](https://docs.nvidia.com/isaaclab/3.0/experimental-features/warp-environments)。

Newton 相关官方依据：[Isaac Lab Physics Backends](https://docs.nvidia.com/isaaclab/3.0/concepts/physics-backends)、[Newton Simulation Tuning](https://newton-physics.github.io/newton/latest/concepts/simulation_tuning.html)、[Newton Collisions](https://newton-physics.github.io/newton/latest/concepts/collisions.html)。
