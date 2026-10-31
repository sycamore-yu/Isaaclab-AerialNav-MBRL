# 自定义部件逐项清单 · 2026-10-31

范围：基线 `7f415ba` 的生产源码；初始实验 `e911e96`，撤回 clone 清理后为 `165c6cf`。覆盖 `src/isaaclab_navrl/` 下全部 **55 个 Python 文件**，最终候选版本合计 **9,659 行**。机器可读文件和顶层符号清单见 [component_inventory.json](../results/official_components_20261031/component_inventory.json)。

这里的“保留”指保留任务定义或必要适配，不表示永远不能优化。“官方候选”表示已找到对应官方能力，尚未证明该项目中的直接替换等价或更快。只有主报告实验节明确标注的项目做了本轮动态验证。所有路径均相对于 `src/isaaclab_navrl/`。

## 1. 包、公共入口与 NavRL

| 文件 | 自定义职责 | 官方对应和决定 | 必须保持的行为 |
|---|---|---|---|
| `__init__.py` | 包入口 | 保留标准包结构 | 无运行时优化对象 |
| `agents/__init__.py` | 策略包入口 | 保留标准包结构 | 同上 |
| `agents/checkpoint_conversion.py` | 原 NavRL 权重名称/形状转换 | 保留；官方 RSL checkpoint 不自动理解原仓库映射 | 转换后 logits、Beta 参数、动作分布一致 |
| `agents/continuous_runner.py` | ContinuousLogger、ContinuousNavRLRunner | 已继承官方 OnPolicyRunner；保留 42 行生命周期适配 | 连续 learn(1) 不关闭 writer；阶段/最终 checkpoint 由外层统一保存 |
| `agents/inference.py` | 加载 NavRL 策略 | 保留薄适配，可继续沿用官方 runner/export 接口 | 模型类型、观测布局、动作缩放 |
| `agents/models.py` | NavRLEncoder、BetaHeads、Actor/Critic、Export | 网络是方法定义；PPO 分布使用官方能力 | 共享编码器、源观测嵌入和 Beta 动作参数化不能借优化改变 |
| `agents/rsl_rl_cfg.py` | 模型和 PPO 配置 | **已官方化**：RSL-RL PPO、BetaDistribution、runner 配置 | 不再实现一份 PPO；换 Gaussian 会改变方法 |
| `control.py` | LeeRotorController、RotorGroup、无人机资产配置、混控 | 官方 `isaaclab_contrib.actuators.Thruster` 是有条件候选；资产/关节 actuator 已用官方 | 原 0.43 每步响应、rad/s 与 rps 换算、饱和、初值、偏航力矩符号、XYZW/WXYZ 边界 |
| `evaluation.py` | first_outcome、EpisodeEvaluation | 保留任务评估协议 | 第一次到达/碰撞/出界、自然完成回合分母；不能用任意平均 reward 替代成功率 |
| `physics.py` | 运行时检查、CLI 后端选择、环境工厂 | **已官方化**：PresetCfg/resolve_presets/create_isaaclab_env；保留薄 CLI 适配 | 后端和前端明确记录；不改变 dt、接触或求解器质量换速度 |
| `source.py` | 冻结源码、导入验证 | 保留实验可追溯性 | 训练真实导入指定提交的快照，不能只记录当前工作目录的 HEAD |
| `terrain.py` | 自定义 heightfield 障碍采样 | **已复用** TerrainGenerator/TerrainImporter/height_field_to_mesh；官方缓存是启动阶段候选 | NumPy 随机序列、栅格、舍入、重叠和非凸三角形集合保持一致；缓存不是训练步提速证据 |
| `training.py` | 日志目录、原生日志标签、优雅停止、课程写盘 | 保留实验入口；可复用官方 logger 的输出能力 | 原始曲线和 checkpoint 不覆盖，阶段连续日志，Train 与 Evaluation 含义分明 |
| `warp_control.py` | 融合 Lee + 电机更新 Warp kernel | 已基于官方 Warp；没有确认可直接替代整段的官方同语义融合控制器 | 原公式、响应、重置和保留零权重随机抽样的 RNG 推进；不能换原生 thruster 后遗漏控制链 |
| `warp_utils.py` | Torch/Warp stream 所有权与寿命 | 官方 stream interop 之上的 26 行版本适配，保留 | Warp 1.17 中借用 handle 不提前注销；capture 与 replay 使用有效同一 stream |

## 2. 场景、传感器、扰动和任务循环

| 文件 | 自定义职责 | 官方对应和决定 | 必须保持的行为 |
|---|---|---|---|
| `tasks/__init__.py` | 注册导航任务 | 标准 Gym/Isaac Lab 扩展入口 | 任务名与入口类 |
| `tasks/navigation/__init__.py` | 导航包 | 标准结构 | 无运行时优化对象 |
| `tasks/navigation/config/__init__.py` | 配置包 | 标准结构 | 同上 |
| `tasks/navigation/config/hummingbird/__init__.py` | Hummingbird Gym 注册 | 已用标准注册协议 | NavRL 与 Zhang 分开注册，不混用训练器 |
| `tasks/navigation/config/hummingbird/env_cfg.py` | Physics/Scene/Events/Direct env 配置，动态几何 | **已复用** DirectRLEnvCfg、InteractiveSceneCfg、EventTermCfg、官方质量/惯量事件 | 共享全局障碍、阶段参数、静态非凸接触；Newton 的 use_mujoco_contacts=False 不随意切换 |
| `tasks/navigation/curriculum.py` | TrainingCurriculum、动态障碍激活索引 | 可托管到 CurriculumManager，保留本项目规则 | 五阶段、自然 1024 回合统计、80% 门槛、阶段重建与 checkpoint 承接 |
| `tasks/navigation/depth_camera.py` | PhysX 批量深度相机适配 | 官方 MultiMeshRayCasterCamera 扩展；本轮通过 PhysX 原生相机对照 | 选中的 USD 三角形、相机位姿、局部 reset、不重置其他环境帧 |
| `tasks/navigation/depth_camera_cfg.py` | 批量相机配置扩展 | 继承官方配置，保留 batch_size/noise 扩展 | 分块大小只影响资源，不能影响图像 |
| `tasks/navigation/depth_history.py` | reproject_depth、历史补偿、深度事件 | 官方 unproject_depth 可作为部分原语，**直接替换已否决**；普通 frame stack 也不等价 | +0.5 像素中心、光学坐标、无梯度重投影、前向 splat/min-z、未知区域掩码与年龄 |
| `tasks/navigation/depth_processing.py` | depth_features | 保留 12 行任务观测变换 | 3/depth.clamp(0.3,24)-0.6，先像素噪声后 12×16 max-pool；不能用 resize/标准 normalize_depth 改写 |
| `tasks/navigation/geometry.py` | TerrainSignedQuery、TerrainDistance、动态 primitive signed distance 和出口梯度 | **已复用** wp.Mesh、BVH、mesh query；保留专门的符号/出口语义层 | heightfield 占用定义、可验证退出方向、选中三角形 float64 精修；普通 mesh SDF/winding 对折叠高墙不等价 |
| `tasks/navigation/lidar.py` | NavRLRayCaster 类组合 | 官方 RayCaster 的薄适配 | LiDAR 只看静态地图；与深度相机的动态物可见性不同 |
| `tasks/navigation/lidar_kernels.py` | rotate_directions_with_yaw | Warp 自定义小 kernel，保留 | 源 LiDAR 的 yaw 方向约定；不能换完整四元数旋转 |
| `tasks/navigation/navrl_env.py` | NavRLEnv/ZhangNavRLEnv、观测/动作/终止/重置 | 已建于 DirectRLEnv；ManagerBased 是组织方式候选，非免费加速 | 各方法观测、频率、reward/loss、partial reset、终态快照和事件先后 |
| `tasks/navigation/navrl_warp_env.py` | Warp 场景/循环、控制图、reset 合并 | **已复用** 官方 DirectRLEnvWarp；保留共享世界与任务适配 | 计步、interval 事件、动作延迟、自动 reset、UI reset 各执行一次；本轮真实环境合同通过 |
| `tasks/navigation/newton_sensors.py` | Legacy 目标筛选相机、共享动态位姿 | **本轮优化**：目标表 N 行缩为拥有独立存储的 1 行，只更新全局一次 | 当前所有查询使用第 0 行；已验证 targets 共享且无物理 sites；不是对任意多世界场景做广播 |
| `tasks/navigation/randomization.py` | PolicyPerturbation、DifferentiableDelay、reset_sim_to_real | **已用官方 DelayBuffer**；尝试删除返回值 clone 后因无峰值收益且首轮训练偏移而撤回；可微延迟保留 | 独立 RNG、每策略 tick 只采一次、reset 零保持、年龄、SO(3) 噪声；BPTT 历史梯度不能用可变环形缓存替掉 |
| `tasks/navigation/scene.py` | SharedNavigationScene.reset | 10 行官方 Scene 子类适配，保留 | 单个无人机重置不重置全局动态障碍运动 |
| `tasks/navigation/sensor_mixins.py` | 分块射线、共享 TLAS、局部 rays/cache、帧/位姿管理 | **本轮替换**重复 pinhole 公式为官方 pattern；**本轮优化**共享 rays 为单份 storage + 零 stride 视图；整套 TLAS 后续评估 SensorTiledCamera | 原分块数值、内参 setter、mask、三角形、深度定义、噪声和池化次序；不对可变内参视图误共享 |
| `tasks/navigation/task_logic.py` | DynamicObstacleMotion、布局分类、navigation_terms | 保留任务层；随机数与数学用官方 Torch/NumPy | 全局运动、8 类几何、独立随机序列、2 秒重采速度、goal 仅对必要 ID 抽样，与无人机 reset 解耦 |
| `tasks/navigation/warp_task_kernels.py` | WarpNavigationTerms、WarpFirstEvents 和融合 kernel | 已用 Warp；与 Torch 任务公式对应的自定义语义保留 | 阈值相等时的比较方向、首次失败锁存、失败后到达不算成功、所有参数与 Torch 路径一致 |

## 3. Zhang 学习方法

这些文件主要表达研究方法，官方仿真/RL 包提供的同名工具不自动具有相同数学含义。

| 文件 | 自定义职责 | 官方对应和决定 | 必须保持的行为 |
|---|---|---|---|
| `zhang/__init__.py` | 方法包 | 标准结构 | 无运行时优化对象 |
| `zhang/actor_critic.py` | TD-lambda、SHAC/AHAC、CostValueNetwork、AdaptiveHorizonController | 保留；官方 PPO 不替代 SHAC/AHAC | cost 符号、bootstrap、终止/截断、折扣、critic 切分和 horizon 调整 |
| `zhang/cli.py` | 方法参数解析与组合验证 | 可用 Hydra 管理配置，当前 argparse 薄适配可保留 | 非法算法/循环模型组合提前报错；默认值不能因迁移漂移 |
| `zhang/config.py` | ZhangTrainCfg | 官方 configclass/Hydra 可承载，配置内容保留 | 学习率、窗口、损失权重、阶段默认值、冻结快照 |
| `zhang/controller.py` | GraphedLeeRotorController | 已用官方 CUDA graph 能力；保留生命周期适配 | graph 的状态更新、reset、输入 tensor 地址与 stream |
| `zhang/critic_state.py` | CriticHistory、privileged observation | 保留方法定义 | 观测与特权量边界、延迟特征、终态历史 |
| `zhang/dcol.py` | sphere/box/cylinder alpha、hybrid gap、height bounded gap | 保留；Newton 接触距离和标准 SDF 不是 scaled-geometry DCOL 的替代品 | 尺度、margin、内部点分段、导数和时间对齐的动态几何 |
| `zhang/diagnostics.py` | GradientDiagnostic、collision 记录、指标比较 | 保留必要诊断，记录开销 | 对照必须使用相同诊断计划，不能关闭诊断后声称训练加速 |
| `zhang/dynamics.py` | 可微质点 surrogate、真实状态对齐、梯度衰减 | 保留；换 Newton/Warp Tape 的完整物理反传会改变算法 | 真实仿真前向、surrogate 反向、对齐/扰动/梯度衰减、dt |
| `zhang/events.py` | Zhang 参数和 interval 扰动 | 已挂在官方 EventManager/EventTermCfg；保留随机分布 | dedicated generator、阶段条件、相机参数、质量/惯量/推力尺度的关系 |
| `zhang/inference.py` | ZhangInference 与 checkpoint 加载 | 保留小型模型适配 | 记录的循环模型、归一化状态、动作解码和历史重置 |
| `zhang/losses.py` | velocity/window、avoidance、height、acceleration/jerk 组合 | 保留方法；Manager 可组织调用但不能替换数学 | 完整 2 秒窗口、短终止窗口、lookahead 查询时刻、区间高度项与掩码 |
| `zhang/model.py` | ZhangPolicy、policy_state、decode_action | 保留 CNN/GRU 和动作定义 | 10 维状态、6 维网络输出、世界加速度解码、四元数约定 |
| `zhang/normalization.py` | RunningMoments、ObservationSnapshot、PolicyObservationNormalizer | RSL empirical normalization 是候选原语，**不能直接替换** | rollout 冻结统计、float64 矩、非有限 batch 跳过、variance floor、无 clipping、输入梯度和 checkpoint 重算 |
| `zhang/optimization.py` | OptimizerStepMonitor、Plateau、RecoveryController | **已复用** Torch AdamW/ReduceLROnPlateau；自定义恢复协议保留 | 5%–80% 阶梯、退化先降 LR 再恢复、阶段边界、实际参数步长 |
| `zhang/projection.py` | height_priority_gradient、ActionUpdateProbe | 保留可选研究机制 | 按真实参数位移评估投影，不把原始梯度投影等同于优化器更新 |
| `zhang/runner.py` | ZhangTrainer、rollout、checkpoint、BPTT | 不换成 OnPolicyRunner；该 runner 本身就是 BPTT/SHAC/AHAC 协议 | 梯度截断、checkpoint 重算、final observation、rollout 冻结统计、周期保存与自然回合统计 |
| `zhang/sru.py` | LSTMSRUGateCell | 保留 SRU-LSTM-Gated 精确定义 | h/c 各 192 维、门控公式、checkpoint 类型；不是任意 LSTM 替换 |
| `zhang/termination.py` | termination_mask、first_events、首次事件锁存 | 可注册官方 termination term；逻辑保留 | terminated/truncated 分离、continuous 不立即 reset、首次失败不能被后来成功覆盖 |

## 4. 脚本、资产、配置和验证基础设施

| 文件/组 | 决定及原因 |
|---|---|
| `scripts/train_navrl.py` | 训练已经委托官方 RSL；保留课程、checkpoint 和日志接入，不再实现第二套 PPO。 |
| `scripts/train_zhang.py` | 方法专用入口，不能用 PPO train 命令替换 BPTT/SHAC/AHAC。 |
| `scripts/run_acceptance.py` | 保留正式五阶段/自然 1024 回合协议；基准短跑不替代验收。 |
| `scripts/run_comparison_queue.py` | 保留隔离输出、提交快照、串行调度；可与外部作业系统集成，但本轮没有重建管理平台。 |
| `scripts/evaluate.py` | 保留 checkpoint 独立评估；不能混入训练自然回合曲线。 |
| `scripts/convert_checkpoint.py` | 保留原权重迁移入口和一致性验证。 |
| `scripts/smoke.py`、`scripts/demo_depth_events.py` | 保留有限步交互与诊断入口。 |
| `scripts/benchmark_backends.py` | 复用已有同步计时和真实训练子进程；本轮基准依赖此脚本。 |
| `scripts/benchmark_camera.py`、`scripts/benchmark_controller.py` | 保留组件测量；不得把组件收益直接当端到端收益。 |
| `scripts/benchmark_zhang_compile.py`、`scripts/profile_step.py` | 后续 profiler/compile 实验入口；保留正确预热、同步与同配置比较。 |
| `scripts/analyze_zhang_gradient_projection.py`、`scripts/diagnose_zhang.py` | 方法诊断，保留。 |
| `scripts/summarize_zhang_comparison.py`、`scripts/summarize_zhang_mechanisms.py` | 实验汇总，保留标签/分母/阶段语义。 |
| 本轮新增 `scripts/benchmark_official_components.py`、`scripts/summarize_official_components.py`、`scripts/ablate_official_components.py` | 复现官方替换、负例、ABBA 与逐项消融；CPU 比较最终 checkpoint，明确初始候选与撤回后的最终候选。 |
| `configs/experiments/` | 保留实际研究配置；不是重复实现仿真后端。 |
| USD/机器人参数与原仓库引用 | 资产几何、惯量和混控参数属于实验输入；官方 Hummingbird/Multirotor 资产不能仅凭名称视为相同。 |
| `tests/` | 以原公式、原生传感器和真实环境合同为回归基准；本轮只执行受改动影响的检查。新增实际旧/新射线方法对照。 |
| `pyproject.toml`、`uv.lock` | 保留版本锁定；不为使用新文档中的 API 顺便升级 Newton/Warp。 |

## 5. 官方来源与使用原则

- [Isaac Lab 3.0 迁移与后端/配置接口](https://docs.nvidia.com/isaaclab/3.0/isaac-lab/migrating-to-3-0)
- [Isaac Lab 3.0 utils / DelayBuffer](https://isaac-sim.github.io/IsaacLab/v3.0.0/source/api/lab/isaaclab.utils.html)
- [Isaac Lab 3.0 Ray Caster](https://docs.nvidia.com/isaaclab/3.0/concepts/sensors/ray-caster)
- [Isaac Lab 3.0 Camera](https://docs.nvidia.com/isaaclab/3.0/concepts/sensors/camera)
- [Isaac Lab 3.0 Warp environments](https://docs.nvidia.com/isaaclab/3.0/experimental-features/warp-environments)
- [Isaac Lab 3.0 Thruster](https://isaac-sim.github.io/IsaacLab/v3.0.0/source/api/lab_contrib/isaaclab_contrib.actuators.html)
- [Newton 1.6.1 SensorTiledCamera](https://newton-physics.github.io/newton/1.6.1/api/_generated/newton.sensors.SensorTiledCamera.html)
- [Warp Torch interop](https://nvidia.github.io/warp/latest/user_guide/interoperability/pytorch.html)

Warp 链接显示的是 latest 文档，不能据此断言已安装 1.17 支持所有新功能。本轮实际使用的零 stride `wp.from_torch`、相机 kernel 和 CUDA stream 行为在本地 1.17 环境运行验证。Isaac Lab pattern、DelayBuffer 及 Newton legacy sensor 以已安装版本源码核对；最新原生传感器与 legacy 对目标集合的处理不同。
