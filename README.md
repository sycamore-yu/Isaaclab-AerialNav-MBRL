# NavRL · Isaac Lab 3.0

本项目将 NavRL 的共享导航场景迁入正式 Isaac Lab 3.0 / PhysX，并提供独立的 Zhang 可微物理训练任务。两种方法共用 Hummingbird 资产、静态地形和全局动态障碍物，分别使用官方 RSL-RL PPO 与 Zhang BPTT 训练器。

**当前状态：环境、训练入口和评测协议已实现；正式训练与约 80% 成功率验收仍需完成实测。** 环境启动和单元测试通过不等于策略达到验收要求。已验证内容及结果文件见 [实现与验证状态](docs/IMPLEMENTATION_STATUS.md)。

## 环境

使用项目独立的 `.venv` 和提交的 `uv.lock`。运行目标为 Linux x86_64、Python 3.12、Isaac Lab 3.0.0、Isaac Sim 6.2 和 PhysX；RSL-RL 采用 3.0 正式版集成的 5.5.1。依赖和软件源见 [pyproject.toml](pyproject.toml)。

```bash
cd /home/tong/tongworkspace/free/isaaclab-navrl
/home/tong/.local/bin/uv sync --frozen
```

以下仿真命令均设置 `OPENBLAS_NUM_THREADS=1`。本机曾在 Kit 启动阶段遇到 OpenBLAS 的 atfork 段错误，该进程级线程设置用于消除已观察到的启动问题。它不修改其他项目的环境或运行进程。

原 Hummingbird USD 已随包保存在 `src/isaaclab_navrl/assets/`。训练和评测不依赖兄弟目录的旧 NavRL 运行环境；原源码差分测试及历史闭环对照脚本仍只读 `../NavRL`。

## 两个任务

| 方法 | Gym 任务 ID | 训练方式 |
|---|---|---|
| NavRL | `Isaac-NavRL-Hummingbird-Direct-v0` | 官方 `OnPolicyRunner`、`PPO`、`RslRlVecEnvWrapper` |
| Zhang | `Isaac-NavRL-Zhang-Hummingbird-Direct-v0` | CNN + GRU192，PhysX 前向、可微质点模型反向和状态对齐 |

| 设置 | NavRL | Zhang |
|---|---|---|
| 物理步长 | 0.016 s，62.5 Hz | 0.01 s，100 Hz |
| 策略步长 | 0.016 s，62.5 Hz | 0.1 s，10 Hz |
| 每次更新采样/展开 | 32 个策略步 | 默认 150 个策略步，即 15 s |
| 任务超时 | 35.2 s，2200 个策略步 | 35.2 s，352 个策略步 |
| 策略观测 | 状态 8 维、LiDAR `1×36×4`、最近 5 个动态物体各 10 维 | 深度图经变换、池化为 `1×12×16`，另加 10 维状态 |
| 控制输出 | Beta 分布映射为每轴 `[-2, 2]` m/s 的目标速度 | 原始六维交错输出解码为世界坐标系三维加速度，无每轴 ±2 m/s 限制 |

NavRL 保留原 Lee 控制律、RotorGroup 电机响应与旋翼关节转速。原生 Multirotor 通过零延迟 Thruster 接收实际推力并施加合力矩。原版地形采样和 LiDAR 射线排列、偏航旋转在任务代码中保留；动态障碍物不混入 NavRL 静态 LiDAR。

Zhang 深度相机采用 NavRL 的安装参数和 `640×480` 分辨率，完整像素分批求交后再执行原深度变换及池化。状态对齐只替换可微模型的前向数值；不会把预测位置写回 PhysX。跨回合重置会清除对应动力学、GRU 和速度窗口历史。

## 完整验收入口

每种方法进行一次全新训练，然后在独立进程中评测五个难度。训练固定为 350 静态 + 100 动态障碍，1024 架无人机、2200 次优化更新。`2200` 次更新与回合超时的 `2200` 个 NavRL 策略步是两个不同参数。

```bash
OPENBLAS_NUM_THREADS=1 OMNI_KIT_ACCEPT_EULA=YES \
  .venv/bin/python scripts/run_acceptance.py \
  --method navrl --output logs/acceptance/navrl_seed0 \
  --num_envs 1024 --max_iterations 2200 --episodes 100 \
  --train_seed 0 --eval_seed 10000 --device cuda:0
```

```bash
OPENBLAS_NUM_THREADS=1 OMNI_KIT_ACCEPT_EULA=YES \
  .venv/bin/python scripts/run_acceptance.py \
  --method zhang --output logs/acceptance/zhang_seed0 \
  --num_envs 1024 --max_iterations 2200 --episodes 100 \
  --train_seed 0 --eval_seed 10000 --device cuda:0 \
  --train_args --timesteps 150 --velocity_window_s 6.0 --max_grad_norm 5
```

输出目录必须不存在。训练入口不接受续训或预训练权重；每次调参使用新目录并从随机初始化开始。只有训练结果确认完成指定更新数，验收入口才开始评测。

五个难度固定 350 个静态障碍，动态障碍数分别为 **0、60、80、100、120**，每个难度至少 100 回合。评测采用独立于训练的种子，锁定每架无人机首个回合的首次结果。记录成功、碰撞、超时、高度越界、无效状态及成功到达时间。同一时刻发生碰撞与到达时计为碰撞；Zhang 在 100 Hz 物理子步记录首次事件。

主验收场景是 **350 静态 + 100 动态**，首次到达率目标约 80%。脚本的 `main_success_target_reached` 按 `success_rate >= 0.8` 判断。

验收目录包含：

- `acceptance.json`：预算、种子、状态和各难度汇总。
- `train.log`、`train/training_result.json`：训练输出与实际完成的更新数。
- `train/checkpoint_final.pt`：该次全新训练的最终策略。
- `evaluation_350_0.json` 至 `evaluation_350_120.json`：各难度逐回合记录和统计；每个难度另有 `.log`。
- `summary.csv`：五个难度的成功率、碰撞率、超时率及到达时间汇总。

## 分步训练与评测

需要分别运行时，使用以下入口。示例目录与完整验收入口分开。

```bash
OPENBLAS_NUM_THREADS=1 OMNI_KIT_ACCEPT_EULA=YES \
  .venv/bin/python scripts/train_navrl.py \
  --log_dir logs/train/navrl_seed0 --num_envs 1024 --max_iterations 2200 \
  --seed 0 --terrain_seed 0 --static_obstacles 350 --dynamic_obstacles 100 \
  --device cuda:0
```

```bash
OPENBLAS_NUM_THREADS=1 OMNI_KIT_ACCEPT_EULA=YES \
  .venv/bin/python scripts/train_zhang.py \
  --log_dir logs/train/zhang_seed0 --num_envs 1024 --max_iterations 2200 \
  --seed 0 --terrain_seed 0 --static_obstacles 350 --dynamic_obstacles 100 \
  --timesteps 150 --velocity_window_s 6.0 --lr 0.001 --max_grad_norm 5 \
  --grad_decay 0.4 --coef_obj_avoidance 3.0 --coef_collide 7.5 \
  --coef_d_acc 0.01 --coef_d_jerk 0.001 --device cuda:0
```

对一个已完成的策略单独评测主难度：

```bash
OPENBLAS_NUM_THREADS=1 OMNI_KIT_ACCEPT_EULA=YES \
  .venv/bin/python scripts/evaluate.py --method navrl \
  --checkpoint logs/train/navrl_seed0/checkpoint_final.pt \
  --output logs/evaluation/navrl_350_100.json --episodes 100 \
  --seed 10000 --terrain_seed 10000 --static_obstacles 350 --dynamic_obstacles 100 \
  --device cuda:0
```

Zhang 使用同一评测入口，将 `--method`、检查点与输出路径改为对应训练结果。正式验收必须完成全部五个难度。

## Zhang 参数与半径

默认速度损失窗口为 60 个策略步，即 `velocity_window_s=6.0`。`timesteps` 候选为 64、96、150，梯度裁剪候选为 1、3、5。AdamW 默认学习率为 `0.001`；在 2200 次更新预算内运行 5 个余弦周期，每周期 440 次更新。`grad_decay=0.4` 按秒使用 `grad_decay ** dt`，不是每个策略步直接乘 0.4。

无人机碰撞几何的保守球半径为 0.271 m。为同时满足原 NavRL 的 0.3 m 判碰规则，Zhang 基础半径取 `max(0.271, 0.3) = 0.3 m`；训练再叠加 `[0, 0.2]` m 安全余量，评测使用基础半径。测量记录见 [Hummingbird 碰撞几何](docs/results/hummingbird_collision_geometry.json)。

损失保留速度匹配、避障、碰撞、加速度、jerk 和速度预测项，移除决策中排除的高度损失。推力误差、响应滞后、阻力、风及相机扰动由 Event Manager 管理。候选参数应先用独立开发实验评估；最终验收仍重新训练，正式评测不用于选择候选。

## 验证入口

CPU 差分与行为测试：

```bash
OPENBLAS_NUM_THREADS=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  .venv/bin/python -m pytest -q \
  tests/test_task_parity.py tests/test_lidar_contract.py \
  tests/test_navigation_contract.py tests/test_navrl_reward_reference.py \
  tests/test_geometry_contract.py tests/test_policy_contract.py \
  tests/test_policy_checkpoint_conversion.py tests/test_zhang_contract.py
```

真实 PhysX 环境 smoke：

```bash
OPENBLAS_NUM_THREADS=1 OMNI_KIT_ACCEPT_EULA=YES \
  .venv/bin/python scripts/smoke.py --method navrl --num_envs 16 --steps 128 \
  --output logs/smoke/navrl_16.json
```

Zhang 使用 `--method zhang` 和独立的输出文件。各项检查的当前状态见 [实现与验证状态](docs/IMPLEMENTATION_STATUS.md)。

## 代码与决策

- `src/isaaclab_navrl/tasks/navigation/`：共享场景、观测、奖励、事件、LiDAR 和深度相机。
- `src/isaaclab_navrl/control.py`、`terrain.py`：原控制器、电机和静态地形适配。
- `src/isaaclab_navrl/agents/`：官方 RSL-RL 模型接口、共享编码器和旧检查点转换。
- `src/isaaclab_navrl/zhang/`：策略、可微动力学、损失、Event Manager 回调和 BPTT 训练器。
- [NavRL 迁移决策与验收](docs/NAVRL_ISAACLAB_3_MIGRATION.md)、[ADR 0001](docs/adr/0001-port-navrl-with-original-task-semantics.md)、[Zhang 迁移决策](docs/ZHANG_MIGRATION_DECISIONS_1_4.md)。
- [受控 PhysX 动力学对照](docs/PHYSX_CLOSED_LOOP_COMPARISON.md)。

旧 NavRL 权重转换使用 `scripts/convert_checkpoint.py SOURCE DESTINATION`，只用于对照和推理，不进入本轮正式新训练。转换不包含旧优化器状态。NavRL、OmniDrones 和 Isaac Lab 的许可证保留在仓库的 `LICENSE-*` 文件中；Zhang 源实现的引用和版本记录见其迁移决策文件。
