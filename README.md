# NavRL 与 Zhang · Isaac Lab 3.0

本项目基于 Isaac Lab 3.0.0 提供 NavRL 与 Zhang 两个导航任务。支持 PhysX＋Torch、Newton＋Torch 和 Newton＋Warp。项目脚本默认采用 **Newton＋Warp**，也可显式选择其他组合。并行无人机共用静态地图，不相互碰撞。

三个组合使用相同的动态障碍运动模型。PhysX 保留全局、无物理碰撞的运动学障碍；Newton 不再为动态障碍创建刚体。Newton 相机从全局运动张量更新共享 Warp 网格；Zhang 默认用 cuRoboV2 查询几何距离和碰撞，NavRL/PPO 保留解析几何。

| 方法 | Gym 任务 | 训练方式 |
|---|---|---|
| NavRL | `Isaac-NavRL-Hummingbird-Direct-v0` | 官方 RSL-RL PPO、共享编码器、Beta 速度动作 |
| Zhang | `Isaac-NavRL-Zhang-Hummingbird-Direct-v0` | CNN + GRU192、所选物理后端前向、可微质点模型反向 |

Zhang 默认场地为 **X=120 m、Y=40 m**，X 两端各留 6 m 无障碍边界。每回合目标速度为 **Uniform(1,6) m/s**，XY 起终点距离严格为 **18v=18–108 m**，路线相对正／负 X 轴偏转 **Uniform(-8°,8°)**。起终点在可行域内采样且检查球体净空；目标速度不会因拒绝重采样而改变。默认几何后端为 **cuRoboV2**，使用 SDF＋梯形积分、pre_reset 损失。当前推荐起始超参为 **H150、避障权重 3.0、lr=0.001、余弦周期 400 updates**；旧任务 C12 在双种子复验中未通过晋升条件，因此保留基线超参。[新任务与四组对照](docs/RECTANGULAR_GRADIENT_2X2_20261031.md)。

Zhang BPTT 可通过 `--recurrent_model sru_lstm_gated` 选择 SRU-LSTM-Gated，默认仍为 `gru`。
SRU 沿用 CNN、10 维状态及 6 维输出，完整维护 192 维 `h` 和 192 维 `c`。
检查点记录模型类型，推理按记录自动加载；GRU 与 SRU 检查点不能互换。
SRU 暂不支持 SHAC/AHAC，参数验证会在创建环境前报错。
来源与适配范围见 [SRU 适配说明](docs/SRU_LSTM_GATED_ADAPTATION.md)。

单条 SRU 长期训练使用 256 环境、seed 0、固定 `finetune_s1` 和随机初始化。
配置复用下述基线，仅改变循环模型；每份报告至少 1024 个完整回合，达到 80% 后继续训练。
运行前需提交源码，以便管线生成可核验的独立快照：

```bash
PYTHONPATH=src .venv/bin/python scripts/run_comparison_queue.py \
  --plan configs/experiments/zhang_sru_lstm_gated.json \
  --commit "$(git rev-parse HEAD)"
```

## 训练与验收

Zhang 新训练默认使用平滑终点速度引导（`goal_speed_profile=smooth`、`goal_approach_time_s=2`）：最终直达段从 `2v` 米开始减速，以二阶连续曲线接入巡航速度，接近中心时速度渐近为零；0.5 m 成功区域不作硬停。旧检查点默认保留原参考公式，显式传入上述参数才迁移。公式、已知地图通行门控与验证见[平滑终点引导](docs/SMOOTH_GOAL_APPROACH_20261101.md)。

当前默认采用 20 Hz 同步 SDF+BPTT、完整 2 秒速度窗口、重置前逐步计算损失、高度越界重置，以及 `navrl_range` 高度损失。展开结束后统一反传；SHAC/AHAC 与其他速度监督模式保留原损失计算路径。当前碰撞损失仅使用已有几何 SDF；`variant=dcol` 只保留终端采样时序兼容，DCOL 数学实现位于显式历史对照包。动态障碍前视位置已按查询时刻预测。旧训练已停止并保留检查点。此前 P0/P1 诊断见 [实施报告](docs/P0_P1_20261030.md)。六组固定微调 S1 对照见 [实验说明](docs/TIME_ALIGNED_AVOIDANCE_20261030.md)。

默认对照为：原梯度／论文侧向梯度重塑 × 直接 fS1／完整课程，共四组，均从随机权重开始，seed 0。每份报告至少 1024 个自然完成回合，没有训练更新上限；课程在一个完整报告达到 80% 时晋级，fS2 达标后继续训练以观察保持情况。

训练与评测默认在首次进入目标 0.5 m 区域后记录 `arrived/arrive_time`，等待 `wait_before_truncate=10 s`，首次进入已超过 10 s 且当前仍在目标区时截断，或由原回合条件触发重置；成功要求回合结束时仍在目标区且未发生失败。离开或重新进入不重启首次到达计时；截止时刻已离开则继续运行，之后重入且已超过 10 s 才触发到达截断。旧记录保留，新机制继续训练使用独立目录与 `--resume_checkpoint`，详见[训练课程](docs/TRAINING_CURRICULUM.md)。

```bash
PYTHONPATH=src .venv/bin/python scripts/launch_rectangular_comparison.py \
  --output "logs/curriculum/$(date +%Y-%m-%d_%H-%M-%S)_rectangular_2x2"
```

启动器先用最重的 fS2／重塑梯度／H150 跑两次更新测量显存，从 N1024 起自动减半，选择能让四组同时运行的共同 N；容量试运行的权重不会用于正式训练。报告窗口固定至少 1024 回合。实际规模、显存峰值、源码提交和失败试运行均保留。TensorBoard 沿用端口 6012 与 `Train/`、`Loss/` 等原有标签。

默认计划为 `configs/experiments/zhang_rectangular_gradient_2x2.json`。单组入口 `scripts/train_zhang.py` 仍默认直接 fS1。已有冻结实验、checkpoint 和历史结果不随默认值修改；旧比较计划保留其显式控制值。

新管线从已提交源码生成独立快照，并在阶段切换及实际导入时核验。队列支持固定 commit 的干净功能分支。正式五阶段依次为预训练 S0、S1、S2 和微调 S1、S2；各阶段完整回合成功率达到 80% 后晋级。

| 阶段 | 静态障碍 | 密度（个/m²） | 动态障碍 | 回合时限 | 额外扰动 |
|---|---:|---:|---:|---:|---|
| 预训练 S0 | 240 | 0.05 | 0 | 30 s | 关闭 |
| 预训练 S1 | 360 | 0.075 | 16 | 35 s | 关闭 |
| 预训练 S2 | 480 | 0.1 | 32 | 40 s | 关闭 |
| 微调 S1 | 480 | 0.1 | 32 | 40 s | 开启 |
| 微调 S2 | 600 | 0.125 | 48 | 45 s | 开启 |

预训练从随机初始化开始。后续阶段加载本次实验的前序检查点，保留网络和优化器。每阶段重建地图并清空回合历史。正式验收默认使用 1024 个并行环境。每份报告至少包含 1024 个完整回合，无更新次数上限。小规模比较须显式使用 `--screening`，即使五阶段门槛均通过也不标记正式验收。输出目录必须不存在。

```bash
cd /home/tong/tongworkspace/free/isaaclab-navrl
.venv/bin/python scripts/run_acceptance.py \
  --method zhang --output "logs/curriculum/$(date +%Y-%m-%d_%H-%M-%S)_sim2real_zhang_seed0" --train_seed 0 \
  --train_args --variant dcol --timesteps 150 --max_grad_norm 5 --camera_batch_size 64
```

NavRL/PPO 使用独立的 `--method navrl` 入口；Zhang 超参数只传给 Zhang 入口。PPO 的观测、频率和动作与 Zhang 不同，属于方法对照，不能解释为只换优化器。对照规范与有时间标记的历史状态见 [truncated 与 PPO 对照](docs/ZHANG_TRUNCATED_PPO_CONTROLS_20261029.md)。有限步调试使用各训练脚本的 `--max_iterations`，不计入正式验收。

公共 PPO 入口支持任务自动选择 RSL-RL、物理后端预设和 Hydra 覆盖。公共训练需显式给出有限的 `--max_iterations`；正式课程继续使用上面的验收入口。

```bash
# PhysX
.venv/bin/isaaclab train --task Isaac-NavRL-Hummingbird-Direct-v0 \
  --num_envs 256 --max_iterations 1 physics=isaacsim_physx
# Newton / MuJoCo Warp
.venv/bin/isaaclab train --task Isaac-NavRL-Hummingbird-Direct-v0 \
  --num_envs 256 --max_iterations 1 physics=newton_mjwarp
# Newton＋Warp 前端（本项目脚本的默认组合）
.venv/bin/isaaclab train --task Isaac-NavRL-Hummingbird-Direct-v0 \
  --num_envs 256 --max_iterations 1 --frontend warp physics=newton_mjwarp
.venv/bin/isaaclab play --task Isaac-NavRL-Hummingbird-Direct-v0 \
  --checkpoint /absolute/model.pt --num_envs 2 --device cuda:0 --visualizer none \
  physics=newton_mjwarp
```

Zhang 使用独立的 `scripts/train_zhang.py` 或 `scripts/run_acceptance.py --method zhang`。Newton 切换前向仿真，反向仍使用现有可微质点模型，不对 Newton 求解器本身求导。后端配置、接触处理和验证记录见 [Newton 适配说明](docs/migration/NEWTON_BACKEND_20261030.md)。

`scripts/train_zhang.py`、`scripts/train_navrl.py`、`scripts/run_acceptance.py` 等项目入口不指定后端时，默认解析为 `physics=newton_mjwarp --frontend warp`。**Isaac Lab 自带的 `.venv/bin/isaaclab train` 仍默认使用 Torch 前端**，该入口必须显式加 `--frontend warp` 才能采用最快的组合。PhysX 仅支持 Torch 前端。

本项目使用固定 32 环境、H150、20 Hz、300 静态／60 动态障碍进行三组完整训练计时。当前同卡实测吞吐分别为 PhysX＋Torch **216.6**、Newton＋Torch **274.4**、Newton＋Warp **289.6 环境步/秒**。详细机制、有效性检查、结果限制与复测命令见 [Newton 共享几何和 Warp 性能报告](docs/migration/NEWTON_SHARED_GEOMETRY_PERFORMANCE_20261030.md)。

运行环境由 `.venv`、`pyproject.toml` 和 `uv.lock` 固定：Python 3.12、Isaac Lab 3.0.0、Isaac Sim 6.2、Newton 1.6.1、MuJoCo Warp 3.12.0、RSL-RL 5.5.1。

## 任务配置

| 配置 | NavRL | Zhang |
|---|---|---|
| 物理频率 | 62.5 Hz | 20 Hz |
| 策略频率 | 62.5 Hz | 20 Hz |
| 每次更新 | 32 个策略步 | 默认 150 个策略步 |
| 单回合超时 | 2200 个策略步 | 704 个策略步（35.2秒） |
| 观测 | 8 维状态、1×36×4 静态 LiDAR、5×10 动态物体真值 | 深度图池化为 1×12×16，加 10 维状态 |
| 动作 | 每轴 [-2,2] m/s 目标速度 | 六维网络输出解码为三维世界加速度，无每轴速度限制 |

Zhang 默认 `--timing unified_20hz`；可选 `unified_50hz` 或旧 `baseline`。2秒完整速度窗口分别为40、100、20步。

两任务保留 Hummingbird USD、Lee 控制律和 RotorGroup 电机响应。两环境的深度相机默认采样 160×120 像素，内参随分辨率缩放以保留原视场，再完成逆深度变换、噪声和 12×16 池化。NavRL 策略仍使用原 LiDAR 特征。动态物体同时进入相机、碰撞检测和损失。

Zhang 保留速度匹配、避障、碰撞、加速度、Jerk 和速度预测辅助项。
高度项默认 `height_loss_mode=navrl_range`，权重为 1。
NavRL 高度区间由当前回合起点和目标确定，并向两侧各扩展 0.2 m。
区间外采用平方损失，区间内为 0。
默认 BPTT 使用完整 2 秒速度窗口，20 Hz 下为 40 步。
SHAC/AHAC 默认逐步速度监督。
continuous 的碰撞和高度越界不重置，首次失败仍锁存。
当前实验结论与直接阶段对照见 [EXP](docs/EXP/20261030_20HZ_CONTROLS.md)。

actor默认AdamW、weight_decay=.01，可显式选Adam；预训练lr默认.001，微调.00005。cosine默认周期400次更新；Plateau默认patience2，可独立配置。AHAC共同H默认初始32、范围16–300，BPTT默认H150。全局梯度裁剪为5，激活检查点控制显存。可选`height_priority`梯度投影仅用于BPTT，需用实际参数位移验证，不能把原始梯度投影当作成功率改善。

## 扰动与日志

两任务共用 `SimToRealCfg`。位置、速度、姿态、LiDAR、深度噪声，观测和动作延迟，控制响应，质量、惯量、推力系数均可独立配置。观测扰动仅进入策略输入。延迟由官方 `DelayBuffer` 按策略步推进，部分回合重置清空对应历史。

TensorBoard 使用统一的 `Train/success_rate`、`Train/episodes` 等标签。同一实验跨课程共用事件目录，课程编号单独记在 `Curriculum/stage`。Zhang 指标按 Loss、Optimization、Timing、Motion、Failures、ActorCritic 和 Memory 分类。逐层优化器指标仍保存在 `updates.jsonl`，不再显示在仪表盘。

稳定模型从 5% 起每 5 个百分点保存一次。性能退化先降低学习率，观察后再恢复同阶段的最高稳定档。详见 [分档协议](docs/migration/RECOVERY_LADDER_20261029.md)。归一化与 Plateau 保持可选。

真实成功率来自 `training_statistics.jsonl`；阶段状态来自 `training_result.json`。TensorBoard 服务使用已有地址 `http://100.92.244.71:6012/`，日志根目录为 `logs/curriculum/`。

## 决策与证据

- [训练课程与验收](docs/TRAINING_CURRICULUM.md)
- [共享 Sim-to-Real 扰动](docs/SIM_TO_REAL_TRAINING.md)
- [NavRL 迁移决策](docs/migration/NAVRL_ISAACLAB_3_MIGRATION.md)
- [PhysX / Newton 后端适配](docs/migration/NEWTON_BACKEND_20261030.md)
- [Newton 共享几何和 Warp 性能报告](docs/migration/NEWTON_SHARED_GEOMETRY_PERFORMANCE_20261030.md)
- [官方组件逐项审计与 demo 实验](docs/migration/OFFICIAL_COMPONENTS_AUDIT_20261031.md)
- [Zhang 迁移决策](docs/migration/ZHANG_MIGRATION_DECISIONS_1_4.md)
- [实现与验证状态](docs/IMPLEMENTATION_STATUS.md)
- [SHAC/AHAC 适配与边界](docs/migration/SHAC_AHAC_ADAPTATION.md)
- [DCOL 速度监督与 PPO 方法对照](docs/ZHANG_TRUNCATED_PPO_CONTROLS_20261029.md)
- [相机批量位姿读取经验](docs/migration/CAMERA_POSE_AGGREGATION_EXPERIENCE.md)

最初迁移基线为 `2456a5b`；新机制前冻结基线为 `f6d5fd4`，归档与相机经验提交为 `1597ffe`。历史 NavRL 在 350 静态、60 动态场景的最高训练首次到达率为 49.28%，未通过旧验收。正式结果以本轮真实训练记录为准。
