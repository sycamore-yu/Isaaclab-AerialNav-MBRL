# 实现与验证状态

本页区分代码已经实现、行为检查已经通过和正式策略验收三个层次。**当前尚无完成 1024 架无人机、2200 次全新训练更新及五难度评测的正式结果，不能据此宣称达到约 80% 成功率。** 实测最终指标由正式运行结束后补入本页，并保留原始结果文件。

任务约束以 [NavRL 迁移决策](NAVRL_ISAACLAB_3_MIGRATION.md)、[ADR 0001](adr/0001-port-navrl-with-original-task-semantics.md) 和 [Zhang 迁移决策](ZHANG_MIGRATION_DECISIONS_1_4.md) 为准。运行命令见 [README](../README.md)。

## 已实现

| 范围 | 实现与约束 |
|---|---|
| 运行环境 | 项目独立 `.venv`；正式 `isaaclab[isaacsim,rsl-rl]==3.0.0`；`uv.lock` 固定依赖；PhysX 后端 |
| 任务与场景 | 两个独立 Gym 任务；`DirectRLEnv`、`InteractiveScene`；共享静态地图和一组全局动态物体；单机重置不重置全局世界 |
| 无人机 | 原 Hummingbird USD 随包保存；原 Lee、RotorGroup 和旋翼关节速度；零延迟 Thruster 转发实际推力；Multirotor 施加合力矩 |
| 静态地形 | 复刻原高度概率、离散网格、特殊间距筛选、覆盖及舍入；重放原 MT19937 的随机数顺序；保留三角网格 |
| NavRL 观测 | 8 维状态、`1×36×4` 静态 LiDAR 和 `1×5×10` 动态物体真值；LiDAR 起点和方向均仅随偏航旋转 |
| NavRL 训练 | 官方 RSL-RL PPO / Runner；共享 CNN、动态编码器和融合网络；Beta 三维目标速度；共享参数单一优化器所有权 |
| Zhang 训练 | 独立 BPTT；原 CNN + GRU192；交错六维输出；世界加速度执行；PhysX 前向、可微质点反向及逐策略步状态对齐 |
| Zhang 相机 | NavRL 安装参数及 `640×480` 深度像素；分批完整射线查询后执行深度变换和 `12×16` 池化；动态障碍可见 |
| Zhang 损失与随机化 | 保留已批准的六类损失并移除高度项；Event Manager 管理动力学与相机随机化；跨回合切断图和历史 |
| 评测 | 单一首次结果，保留逐回合起点、目标、事件和时间；同一步碰撞优先于到达；Zhang 按物理子步记录首次事件 |
| 训练完整性 | 新目录、随机初始化、不续训、不加载预训练权重；训练结果记录预算及实际完成更新数；正式评测使用不同种子 |

NavRL 保留原任务的奖励和终止条件；原 `reach_goal` 即时记录与首次到达统计分别保留。成功本身不是原训练任务的终止条件，正式评测会锁定首次结果。

### 官方 PPO 的已知差异

迁移使用官方 PPO 的单一优化器与价值损失，不复刻原自定义训练循环的两个优化器、`ValueNorm`、Huber 价值损失或额外按动作维数缩放的策略损失。网络共享和 Beta 动作语义保留，但不能声称两套优化过程逐项相同。运行配置在 `train/run_config.json` 中记录这些差异。

### 频率、展开与半径

NavRL 保留 0.016 s 物理/策略步。Zhang 使用 0.01 s 物理步和 0.1 s 策略步，默认展开 150 个策略步，速度窗口 60 个策略步，即 6 s。两任务的回合超时均为 35.2 s。正式预算中的 2200 指优化更新次数。

Zhang 的六维输出按源排列 `[ax, vx, ay, vy, az, vz]` 解码，保留一步策略命令延迟；不施加每轴 ±2 m/s 速度限制。目标速度尺度采用 `(0.75 + 2.5U) × 3`。梯度衰减采用 `grad_decay ** dt`；默认 5 个余弦周期、梯度范数裁剪 5，候选展开长度为 64、96、150，候选裁剪值为 1、3、5。

[实测碰撞几何](results/hummingbird_collision_geometry.json) 包含机身方体及四个旋翼碰撞圆柱，保守球半径向上取为 0.271 m。NavRL 任务使用 0.3 m 中心判碰阈值，因此 Zhang 的基础半径为：

```text
max(物理几何保守半径 0.271 m, 原 NavRL 判碰半径 0.3 m) = 0.3 m
```

策略半径输入和避障损失共同使用该值；训练再加 `[0, 0.2]` m 随机安全余量。这样损失不会允许进入任务已经判为碰撞的距离区间。

## 已通过的检查

以下状态来自实际执行记录，不表示完整训练已经通过。

| 检查 | 状态 | 证据与范围 |
|---|---|---|
| 原 Lee 与电机迁移 | 通过 | `tests/test_task_parity.py`：随机状态与原源码电机命令比较，8 环境 ×250 电机步及单环境重置 |
| 原地形迁移 | 通过 | 同一测试：两种子高度场逐元素比较，seed 0 网格顶点、面索引及原点比较 |
| LiDAR 排列与偏航 | 通过 | `tests/test_lidar_contract.py`：原 Bpearl 144 射线逐元素比较；有倾斜和偏航时的起点、方向及环境掩码 |
| 观测、奖励、事件 | 通过 | `tests/test_navrl_reward_reference.py` 与 `tests/test_navigation_contract.py`：原源码差分、首次事件、精确动态数量及坐标变换 |
| 几何查询及位置梯度 | 通过 | `tests/test_geometry_contract.py`：实际网格最近点、动态圆柱与查询位置梯度 |
| Zhang 算法行为 | 通过 | `tests/test_zhang_contract.py`：已运行的策略、动力学、损失、历史和状态对齐行为测试 |
| 正式运行环境中的受控闭环 | 通过 | `logs/smoke/force_new_runtime.json`、`.log`：480 个物理步的原旋翼施力/合力矩对照 |
| NavRL 1 与 16 环境 | 通过 | `logs/smoke/navrl_1.json`、`navrl_16.json` 及同名 `.log`：reset/step、有限状态、悬停/速度指令及共享障碍重置隔离 |
| 克隆之间物理隔离 | 通过 | `logs/smoke/scene_isolation.json`、`.log`：两机同位置飞行，保持相同状态/观测；全局动态集合未复制 |
| 完整 CPU 测试集 | 通过 | 40 项通过，14.70 秒；`docs/results/unit_tests.xml` 保存逐项记录 |
| NavRL 1024 环境 | 通过 | `logs/smoke/navrl_1024.json`：128 步、350+100 障碍，有限状态、悬停、速度与重置检查通过；此计时包含检查开销 |
| 原生相机逐像素对照 | 通过 | `logs/smoke/camera_parity.json`：640×480 深度最大误差 0；移动障碍可见且不进入静态 LiDAR |
| NavRL 短训练及独立评估流程 | 通过 | `logs/smoke/navrl_pipeline_v2/acceptance.json`：16 环境、2 更新、两档各 4 回合；预算标记为 false，不作为成功率验收 |

历史受控施力对照仍保存在 [PHYSX_CLOSED_LOOP_COMPARISON.md](PHYSX_CLOSED_LOOP_COMPARISON.md) 及 `docs/results/physx_*.json`。这些对照验证控制和施力适配，不能代替导航策略成功率评测。

## 尚需实测完成

| 项目 | 当前结论 |
|---|---|
| Zhang 完整 PhysX rollout、反向传播及优化更新 | 需完整闭环实测，CPU 行为测试不能代替 |
| Zhang 1024 环境吞吐与显存 | 需在实际共享场景中测量；小批量 smoke 不外推数值 |
| NavRL 2200 次全新训练更新 | 待正式完成记录 |
| Zhang 2200 次全新训练更新 | 待正式完成记录 |
| 每方法五难度、每难度至少 100 回合 | 待正式逐回合结果 |
| 350 静态 +100 动态约 80% 首次到达率 | 未有可用于判断的正式结果 |

最终结果应记录训练种子、最终检查点、实际更新数、非有限状态/梯度事件、训练时间、吞吐与显存；每难度记录首次到达率、碰撞率、超时率、高度越界率、无效状态率及成功到达时间。正式成功率表只引用完成的运行结果。

## 正式结果的位置与判断

执行 `scripts/run_acceptance.py` 后，方法对应的输出目录包含 `acceptance.json`、`train/training_result.json`、最终检查点、五个 `evaluation_350_<动态数>.json` 和 `summary.csv`。

`acceptance.json` 的 `status="completed"` 表示该次训练和评测命令均完成；`acceptance_passed` 还要求正式预算、完整五难度、每档至少 100 回合、主难度成功率至少 0.8，以及无非法状态或非有限梯度。短 smoke 或开发实验不能替代正式预算。每次参数试验必须使用独立目录，最终选定参数重新开始训练，保持单次策略来源清楚。

需要继续优化时，先使用开发种子比较训练速度、稳定性和成功率，再确定正式参数。正式评测不用于选择参数，也不从已训练检查点续跑补足 2200 次更新。
