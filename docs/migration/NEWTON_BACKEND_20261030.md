# PhysX / Newton 后端适配（2026-10-30）

## 范围与接口

两个注册任务共用 `NavigationPhysicsCfg(PresetCfg)`：

| 任务 | 动作与训练器 | 物理预设 |
|---|---|---|
| `Isaac-NavRL-Hummingbird-Direct-v0` | 三维目标速度，官方 RSL-RL PPO | `isaacsim_physx` / `newton_mjwarp` |
| `Isaac-NavRL-Zhang-Hummingbird-Direct-v0` | 三维世界加速度，Zhang BPTT / SHAC / AHAC | `isaacsim_physx` / `newton_mjwarp` |

本节记录首次 Newton 迁移时的配置。当时默认预设为 `isaacsim_physx`。后续共享几何优化已将项目脚本默认配置改为 Newton＋Warp。最新机制和性能见 [共享几何优化报告](NEWTON_SHARED_GEOMETRY_PERFORMANCE_20261030.md)。

```bash
.venv/bin/isaaclab train \
  --task Isaac-NavRL-Hummingbird-Direct-v0 \
  --num_envs 256 --max_iterations 1 physics=isaacsim_physx

.venv/bin/isaaclab train \
  --task Isaac-NavRL-Hummingbird-Direct-v0 \
  --num_envs 256 --max_iterations 1 physics=newton_mjwarp
```

运行依赖沿用现有锁文件：Isaac Lab 3.0.0、Isaac Sim 6.2.0.0、Newton 1.6.1、MuJoCo / MuJoCo Warp 3.12.0、Warp 1.17.0。无 site-packages 补丁。

## 控制与机械资产

Isaac Lab 3.0.0 的 contrib `Multirotor` 直接继承 PhysX Articulation。两任务现在通过核心 `ArticulationCfg` 工厂加载原 Hummingbird USD，并用公共 `permanent_wrench_composer` 施加合力矩。每步仍采用原 Lee 控制律、原 `RotorGroup`、原旋翼分配矩阵和按关节名写入的转速。没有增加第二份电机响应，也没有更改控制增益。

合力矩以机体坐标作用在 `base_link` 质心。Zhang 的世界系扰动加速度乘以当前总质量后变换到机体系，与旋翼合力一同提交；因此质量随机化后的扰动力也使用实际质量。质量、惯量的 reset 随机化继续走 Isaac Lab Event Manager 的公共 API。

源 USD 的总质量为约 0.716 kg：机身 0.680 kg，加四个 0.009 kg 旋翼。机身主惯量为 `(0.007, 0.007, 0.012)` kg·m²。旋翼保留独立刚体和关节，均按名称索引，不假定后端实体排列相同。

## 接触、共享世界和传感器

Newton 预设使用 `MJWarpSolverCfg(integrator="implicitfast", use_mujoco_contacts=False, disable_sensors=True)`。关闭的是 MuJoCo 自己的接触生成和内部传感器；Isaac Lab 的激光与深度传感器继续工作。接触由 Newton 的碰撞管线生成，约束与积分仍由 MuJoCo Warp 执行。

该选择是为保留原森林三角网格：安装版 Newton `solver_mujoco.py` 将 MuJoCo mesh geom 转为凸包，不能直接用整片森林的凸包替代原有障碍与通道。任务没有换成近似高度场，没有对原始地形网格做简化。

首次迁移时，Newton 为每个世界建立运动学副本；它们由同一份 `DynamicObstacleMotion` 驱动。**该实现已被取代**。现在 Newton 保留全局纯几何 USD 物体和共享 GPU 运动状态，不再创建任何动态障碍刚体。PhysX 仍保留原全局运动学集合。详见 [共享几何优化报告](NEWTON_SHARED_GEOMETRY_PERFORMANCE_20261030.md)。

首次迁移为动态副本显式设置 `MassCfg(mass=1.0)`。共享几何版本已删除动态副本和这部分物理质量设置。无人机仍使用 `replicate_physics=True` 和官方 clone context。

PhysX 和 Newton 仍复用 Warp 射线、mesh/BVH、深度噪声与池化逻辑。共享几何版本的 Newton 不再使用动态障碍 site 索引；它直接将共享运动位置写入相机网格变换缓冲区。此实现保留 160×120 原始深度、12×16 策略深度和 36×4 LiDAR。

项目训练与评测脚本接受相同的 `physics=...` 参数。解析器保存为 `physics_preset`，先消解任务预设，再交给启动器，避免启动器的保留参数 `physics` 用通用默认配置覆盖上述任务设置。运行配置、结果及检查点均记录后端。

## Zhang 训练边界

`newton_mjwarp` 改变实际前向状态转移。BPTT / SHAC / AHAC 仍按现有实现，使用与真实状态对齐的可微质点模型计算梯度。此适配不宣称通过 Newton / MuJoCo Warp 求解器本身反向传播，也不宣称两种求解器产生逐位相同的轨迹或成功率。

## 验证与运行记录

本次代码在 `feat/newton-backend-20261030` 隔离工作树完成，原主目录已有的结果文档修改保留。

验证在 RTX 4090 上执行，日志汇总到主目录 `logs/newton_migration_20261030/`。原始工作树日志位于 `logs/newton_migration/`。

| 验证 | 结果 | 日志或结果文件 |
|---|---|---|
| 用户指定 PhysX CLI，256 环境、1 次 PPO 迭代 | 退出 0，8192 环境步，生成 `model_0.pt` | `navrl_physx_256.log` |
| 用户指定 Newton CLI，256 环境、1 次 PPO 迭代 | 退出 0，8192 环境步，生成 `model_0.pt` | `navrl_newton_256.log` |
| Zhang Newton BPTT，baseline 同配置，256 环境、H150、20 Hz | 完成 2 次更新、76800 环境步；非有限状态和梯度事件均为 0，生成最终检查点 | `zhang_train_256/training_result.json`、`checkpoint_final.pt` |
| 双后端共享场景与环境隔离 | 2 环境、64 步，同初态同控制的状态最大差为 0 | `scene_isolation_{physx,newton}.json` |
| Zhang DCOL，双后端扰动与部分重置 | 16 环境、140 物理步，动作执行最大误差 0 | `zhang_dcol_{physx,newton}.json` |
| Zhang Newton step / continuous | 分别 140 / 180 物理步；第一碰撞、终止与部分重置检查通过 | `zhang_{step,continuous}_newton.json` |
| 双后端真实森林接触与通道 | 2 环境、240 物理步；通道正常落地，障碍上方产生空中接触冲量 | `contact_{physx,newton}.json` |
| 双后端深度相机与官方原生实现对照 | 各 4 环境、8 组检查，160×120 原图与 12×16 池化结果最大误差均为 0；覆盖尾批次、动态物体移动、部分重置、噪声和内参变化 | `camera_parity_{physx,newton}.json` |
| 控制律、电机响应、地形、任务契约、脚本参数与传感器单元检查 | 共 138 项及 401 个子测试通过；重置修复后相关 36 项再次通过 | `controller_parity.log`、`shared_task_unit.log`、`entrypoint_unit.log`、`reset_contract_unit.log` |

continuous 测试主动注入一次 `invalid` 检测事件来检验终止语义，因此该测试的 `nonfinite_events=1` 是预期计数；实际刚体状态逐步通过有限值检查。它也暴露并回归验证了连续手动部分重置的问题：`_stepping` 现在在 `step()` 的 `finally` 中复原，避免读取前次重置已清空的 `_terms`，并保留自动重置的完成回合日志。

真实接触测试清除全部外力矩，不调用会自动重置的 `env.step()`。Newton 在障碍上方约 4.79 m 处出现接触冲量；无人机会从窄障碍翻落，测试不要求它在顶部保持平衡。通道中离地阶段的非重力速度变化最大约 `2.49e-7 m/s`，最终高度约 0.0697 m；PhysX 对照也通过。

训练吞吐受同一 GPU 上其他既有实验影响，不作为公平的后端性能比较。短程训练验证启动、更新与落盘，不代表已经完成成功率验收。

## 无预算上限的 Zhang baseline

沿用 `configs/experiments/zhang_finetune_s1.json` 的 baseline 参数，仅选择 Newton：随机初始化、seed 0、GRU 192、256 环境、固定 `finetune_s1`、300 个静态和 60 个动态障碍、扰动开启。完整启动命令如下；从合并后的主目录执行，输出目录必须不存在。

```bash
PYTHONPATH=src .venv/bin/python scripts/run_acceptance.py \
  --method zhang \
  --output logs/curriculum/2026-10-30_newton_mjwarp_baseline_seed0 \
  --screening --num_envs 256 --train_seed 0 --device cuda:0 \
  --report_episodes 1024 --start_stage finetune_s1 --fixed_stage \
  physics=newton_mjwarp \
  --train_args --variant dcol --algorithm bptt --recurrent_model gru \
  --timesteps 150 --velocity_loss_mode full_window --velocity_window_s 2.0 \
  --height_loss_mode navrl_range --lr 0.001 \
  --lr_schedule cosine --cosine_period_updates 400 \
  --recovery_enabled 1 --timing unified_20hz \
  --gradient_diagnostic_updates 1,50,250,1000
```

没有 `--max_iterations`、`--max_updates` 或 `--stop_after_acceptance`。固定阶段不自动升级，`report_episodes=1024` 仅是自然完成回合的报告窗口。管线冻结已合并源码并核对实际导入路径；进度查看运行目录下 `zhang/finetune_s1/updates.jsonl`、`training_result.json` 和 `zhang/tensorboard/`。启动进程及提交核验写入运行目录的 `launch_verification.json`。

## 依据

- [Isaac Lab 3.0 迁移指南](https://docs.nvidia.com/isaaclab/3.0/isaac-lab/migrating-to-3-0)
- [Physics backends](https://docs.nvidia.com/isaaclab/3.0/concepts/physics-backends)
- [Backends and presets](https://docs.nvidia.com/isaaclab/3.0/concepts/backends-and-presets)
- [Physics schemas](https://docs.nvidia.com/isaaclab/3.0/concepts/physics-schemas)
- 实际安装版本中的 `isaaclab_tasks.utils.hydra.resolve_presets`、`isaaclab_newton` 资产与 Legacy ray-caster、`newton/_src/solvers/mujoco/solver_mujoco.py`。
