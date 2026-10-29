# 实现与验证状态

2026-10-29：两个任务已接入训练内课程。验收采用 100 动态障碍阶段的训练成功率，目标约 80%。正式训练没有更新上限，也不运行独立评测。口径见 [训练课程与验收](TRAINING_CURRICULUM.md)。当前训练达标状态以对应运行目录的 `training_result.json` 为准。

## 当前实现

环境使用正式 Isaac Lab 3.0.0、Isaac Sim 6.2 和 PhysX。NavRL 使用原 Hummingbird 资产、Lee 控制律、RotorGroup、静态地形、观测、奖励和终止条件，训练使用官方 RSL-RL 5.5.1 PPO。

两任务默认预分配 120 个全局动态物体，从 60 个开始。成功率超过 80% 后，依次增加到 80、100、120 个。阶段切换使用同一个场景，保留网络和优化器。未激活物体不参与动态观测、碰撞或 Zhang 损失。所有无人机在晋级时开始新回合，清空跨阶段历史。

统计按原 NavRL `EpisodeStats` 的频率汇总已完成的训练回合。每回合锁定一个首次结果，在自然终止时计数。Zhang 按 100 Hz 物理子步记录首次事件。训练回报、原即时 `reach_goal` 与首次到达率分别记录。

Zhang 保留 CNN + GRU192、10 维状态输入、六维交错输出和现有损失。控制为世界坐标系三维加速度。PhysX 前向结果逐策略步对齐可微质点模型。物理频率 100 Hz，策略频率 10 Hz，展开 150 个策略步，速度窗口 60 个策略步。深度相机完整查询 640×480 像素，再变换并池化为 12×16。

## 本轮验证

| 检查 | 结果 | 原始记录 |
|---|---|---|
| CPU 回归测试 | 56 项通过 | `logs/curriculum_checks/full_tests.log`、`full_tests.xml` |
| 课程统计与晋级 | 首次结果、自然终止、按样本数汇总及四阶段通过 | `tests/test_training_curriculum.py` |
| 官方 PPO 连续更新 | 参数、优化器、Adam 步计数及 TensorBoard writer 连续保留 | `tests/test_continuous_runner.py` |
| Zhang 跨阶段更新 | 状态、GRU 和速度历史清空，优化器保持连续 | `tests/test_zhang_contract.py` |
| 真实 PhysX 四阶段切换 | 60/80/100/120 数量、共享物体、深度及位置梯度检查通过 | `logs/curriculum_checks/scene_v2.json` |
| NavRL 1024 环境短训练 | 从随机初始化完成 3 次 PPO 更新，非有限状态事件 0 | `logs/curriculum_checks/navrl_1024/training_result.json` |
| Zhang 激活检查点 | 递归输出、参数梯度、状态输入梯度逐元素相等 | `logs/curriculum_checks/activation_checkpoint_tests.log` |
| 回报统计与正式入口回归 | 31 项通过；覆盖终止步奖励、不同回合数加权、PPO 连续更新、Zhang 重置和顺序验收入口 | `tests/test_training_curriculum.py`、`test_continuous_runner.py`、`test_acceptance_protocol.py`、`test_zhang_contract.py` |
| Zhang 1024 环境连续更新 | H=150、速度窗口=60 策略步，随机初始化连续完成 2 次更新；非有限梯度和仿真状态事件均为 0 | `logs/curriculum_checks/zhang_current_1024_h150/training_result.json`、`updates.jsonl` |

场景切换检查使用合成结果触发晋级，未作为策略成功率。有限步训练用于检查完整执行链；正式运行仍从随机初始化开始。

### 已修正的特征顺序

原版 TorchRL `CatTensors` 默认排序输入键。实际拼接顺序为 CNN 特征 128 维、动态物体特征 64 维、状态 8 维。旧迁移将状态放在动态特征之前。本轮执行原版 `CatTensors` 构造与前向后复现差异，并修正迁移顺序。原版检查点动作及反归一化价值对照已通过。旧迁移检查点保留为历史结果，本轮不加载。

### 显存检查

首个 1024 环境、150 策略步 Zhang 检查因显存不足退出，记录保留在 `logs/curriculum_checks/zhang_1024_h150/`。当时其他两个进程合计占用约 14.4 GiB，该检查进程约占用 9 GiB。未停止其他项目的进程。

已加入非重入激活检查点，重算网络中间量以减少存储。完整 1024 环境复测位于 `logs/curriculum_checks/zhang_1024_h150_checkpointed/`。随后使用当前代码完成两次连续更新，记录在 `logs/curriculum_checks/zhang_current_1024_h150/`：总训练时间 250.77 秒，两次更新分别约 126.99、123.73 秒，PyTorch 峰值 allocated 显存 1305.48 MiB。该数字不包含 Kit、Warp 和其他进程的显存，不代表整个进程峰值。

两次更新的裁剪前梯度范数为 28.66、24.71，均按配置裁剪到上限 5；总损失为 8.2579、8.2097。满足同一回合内连续 60 策略步条件的速度窗口比例为 39.83%、40.36%。短检查的 60 障碍首次到达率最高为 0.51%，仅证明执行链完成，不代表验收通过。

### 相机性能测量

`scripts/profile_step.py` 只测零指令下的环境开销，不包含网络、反向传播或优化器。1024 环境、100 动态障碍、完整 640×480 深度图下，每批 16 个相机的未插桩基线约 726.60 ms/策略步；插桩测得深度采样约 547.55 ms/策略步。将批量增加到 64 后，未插桩基线约 864.04 ms，深度采样约 560.40 ms，未测得提速，因此正式配置保留 16。

原始记录为 `logs/profile/zhang_1024_current.json` 和 `logs/profile/zhang_1024_batch64.json`。测量时有其他训练任务共享 GPU，且插桩会改变 CUDA 调度；这些数字用于定位当前瓶颈，不能视为独占 GPU 的训练吞吐量。相机分辨率、采样频率、噪声与池化语义均未更改。

## 已有迁移检查

原地形、Lee 控制与电机响应、LiDAR 排列与偏航、NavRL 奖励与终止、共享场景碰撞隔离等检查保留。相机加速采用原生局部三角网格与顶层包围盒筛选；`logs/smoke/camera_tlas_parity.json` 中，17 个环境的初始、倾斜、动态物体移动、局部重置和噪声处理对照均为深度误差 0。控制器 CUDA Graph 的 1024 环境输出、部分重置和随机数状态对照见 `logs/profile/controller_cuda_graph_1024.json`。

## 官方 PPO 的差异

本项目采用已批准的官方 PPO。它使用单一优化器、平方价值损失及自身梯度裁剪，不复刻原自定义训练循环的三个优化器、`ValueNorm`、Huber 价值损失及策略损失乘动作维数。网络结构与 Beta 动作语义保留。差异在 `run_config.json` 中记录。

## 验收结果

正式结果来自 `training_statistics.jsonl`，按难度分别记录。`checkpoint_best_100.pt` 保存 100 障碍阶段的最佳策略。`training_result.json` 中的 `acceptance_passed` 要求 1024 并行环境、350 静态障碍和 100 动态障碍阶段的训练成功率至少 0.8。旧固定难度训练及独立评测文件不用于本轮验收。

已启动 `run_acceptance.py --method both`，输出根目录为 `logs/curriculum/fresh_seed0_20261029/`，启动代码提交为 `e50ba1f`。NavRL 从随机初始化持续训练，未设置更新上限。100 阶段通过并完成首次 120 阶段报告后，入口将启动同样从随机初始化开始的 Zhang 正式训练。两次更新的 Zhang 短检查不会作为该训练的初始权重。

截至 2026-10-29 03:12 UTC，正式 NavRL 最新统计在第 319 次更新：60 动态障碍，1045 个完成回合，首次到达率 0.287%、碰撞率 67.368%、高度越界率 1.818%、超时率 30.526%、无效状态率 0，平均 return 3772.03。尚未进入 80/100/120 阶段，验收未通过；正式 Zhang 尚未启动。此处是一次进度快照，后续状态以运行目录实时 JSON 为准。

TensorBoard 服务地址为 `http://100.92.244.71:6012/`，日志根目录 `logs/curriculum/`。两任务均按阶段记录 `Train/<动态障碍数>/success_rate`、`collision_rate`、`bounds_rate`、`timeout_rate`、`invalid_rate`、`success_time_mean_s`、`episode_return_mean`、`episode_length_mean` 和 `reward_per_step_mean`。
