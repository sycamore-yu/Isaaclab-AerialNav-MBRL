# 小环境、多实验与共享场景：设计审计和排队建议

日期：2026-10-29。范围：读取当前隔离镜像及 Isaac Lab 官方 main 文档，分析资源与实验语义；**本审计没有启动 GPU 任务、测量新的吞吐或修改训练代码**。以下并发数量是待容量测量验证的起点，不是已证实的最佳配置。

## 1. 当前建议

本轮采用 **每组 256 环境、独立进程、独立 fresh 初始化** 做筛选，按信息价值逐批运行十二组。正式验收另以至少 1024 环境从随机初始化执行完整五阶段；筛选权重不冒作该次正式 fresh 验收的起点。每次实验内部的阶段检查点继承保持原协议。

当前单 run 已共享静态地形、地面和动态障碍。把十二个实验进一步合成一个 `InteractiveScene` 是新的执行架构，需要单独证明采样、重置、课程和梯度没有串扰，不能视为修改 `num_envs` 或换成 `DirectMARLEnv` 就能完成的优化。当前几何正确性检查仍须先通过；并发安排不能替代它。

## 2. 三种“共享”对应不同边界

| 形式 | 共享内容 | 必须独立的内容 | 当前状态 |
| --- | --- | --- | --- |
| 单 run 的并行无人机 | 同一地图、全局动态物体、一次仿真时钟、同一策略 | 各无人机的状态、回合、目标、延迟和首次事件 | 已实现 |
| 多个独立进程共用一张 GPU | GPU 计算与显存带宽、同版本资产文件 | 各进程的 Kit/场景、策略、优化器、缓冲、RNG、课程与日志 | 本轮建议 |
| 一个进程/场景内放多个学习器组 | 可能共享 Kit、不可变几何及部分查询结构 | 全部学习状态、分组环境状态与课程，还需统一步进协调器 | 独立后续架构候选，未实现、未验证 |

### 已有单 run 共享的具体证据

当前 [env_cfg.py](../../src/isaaclab_navrl/tasks/navigation/config/hummingbird/env_cfg.py) 将地形放在 `/World/ground`，地面放在 `/World/defaultGroundPlane`，动态物体放在全局 `/World/Dynamic/obstacle_...`；无人机使用 `{ENV_REGEX_NS}/Hummingbird`。`NavRLEnv._setup_scene()` 显式调用 `scene.filter_collisions(global_prim_paths=[...])`。[SharedNavigationScene.reset](../../src/isaaclab_navrl/tasks/navigation/scene.py) 只重置指定无人机和传感器；[navrl_env.py](../../src/isaaclab_navrl/tasks/navigation/navrl_env.py) 的局部回合重置不重置共享动态轨迹。

官方 `InteractiveScene.filter_collisions` 的职责是隔离声明的环境间碰撞，同时允许与指定全局 prim 碰撞；它不隔离优化器、随机数、训练报告或课程。[官方 scene API](https://isaac-sim.github.io/IsaacLab/main/source/api/lab/isaaclab.scene.html#isaaclab.scene.InteractiveScene.filter_collisions) 与 [cloning 生命周期](https://isaac-sim.github.io/IsaacLab/main/source/how-to/cloning.html) 均将它作为场景构建的一部分。传感器的目标选择也需单独正确配置，不能把物理碰撞过滤自动当成射线可见性过滤。

版本注意：2026-10-29 检索到的官方 main 已将 `InteractiveSceneCfg.replicate_physics` 标为 deprecated/ignored，并介绍新的 `clone_cfg`。本仓库冻结安装接口仍有 `replicate_physics=False`。本审计不据滚动 main 页面直接改动已验证的安装版本；真正实现新的共享架构前须记录安装包版本、源码和实际可调用 API。

## 3. 共享多学习器所需的隔离

每个实验必须有自己的 policy、actor optimizer、critic/target/critic optimizer、归一化统计、LR scheduler、RecoveryController、AHAC 的 H/dual，以及梯度诊断缓存。RMS 组不能向其他组提供统计；不同 critic 不能混合 TD targets；完整窗口与短窗模式也不能共用带状态的历史。

环境侧同样需要按组拥有 episode ID、长度、目标、首次事件锁存、回报累计、GRU、动作和观测延迟、速度窗口、物理随机参数及历史。一个组重置某些 lane，不能清空其他组状态，也不能重置共享动态物体的时间。模型初始化、动作采样、reset、观测噪声和 critic shuffle 的 RNG 需按组管理；只调用一次全局 `manual_seed` 不会消除互相消耗随机数序列的影响。

**课程不能被同一全局地图强迫同步。** 一组达到 S1 时另一组可能仍在 S0；300/0、300/60、400/120 等场景不同。若等待最慢组才能晋级，或让尚未晋级组看到已升级障碍，便改变了实验协议。解决方案需为不同阶段提供独立场景实体、明确的传感器/碰撞选择，或者将晋级组迁往另一个进程；这些都不是当前接口的免费能力。

学习器的展开边界也不同：BPTT H150、SHAC 固定 H32、AHAC H16～300。共享协调器必须允许每组按本算法边界反传、拟合 critic 和更新策略；不能为了凑整齐而提前截断 BPTT、冻结某组无人机、补造 transition，或混合其他组经验。活跃计算图和 critic 内存仍按各组保留，节省一个 Kit 并不会消除这些训练开销。

这些要求是由当前实验状态所有权推导出的设计约束，不是对尚未实现共享执行器的性能承诺。相关实现来源为 [runner.py](../../src/isaaclab_navrl/zhang/runner.py)、[normalization.py](../../src/isaaclab_navrl/zhang/normalization.py)、[optimization.py](../../src/isaaclab_navrl/zhang/optimization.py) 和 [actor_critic.py](../../src/isaaclab_navrl/zhang/actor_critic.py)。

### 为什么 DirectMARLEnv 不是多实验封装

官方 `DirectMARLEnv` 提供按 agent ID 组织的动作、观测、奖励和终止输出，属于一个多智能体任务的向量化接口；它没有自动创建十二个独立实验的学习器和课程。官方也规定环境自身管理 simulation context，已有 context 时再次构造环境会报错。因此不能把现有两个 `DirectRLEnv` 实例简单塞进同一进程后交替调用 `.step()`，就认为实现了独立实验。[官方 DirectMARLEnv 与环境初始化/步进接口](https://isaac-sim.github.io/IsaacLab/main/source/api/lab/isaaclab.envs.html#isaaclab.envs.DirectMARLEnv)

## 4. PPO 与 Zhang 的时钟不能悄悄统一

官方把 `physics_dt` 与 `step_dt` 分开，环境策略步为物理步长乘 `decimation`。[官方步进接口](https://isaac-sim.github.io/IsaacLab/main/source/api/lab/isaaclab.envs.html#isaaclab.envs.DirectRLEnv.step)

| 当前项目设置 | NavRL/PPO | Zhang |
| --- | --- | --- |
| 物理步长 | 0.016 s，62.5 Hz | 0.01 s，100 Hz |
| decimation | 1 | 10 |
| 策略步长 | 0.016 s，62.5 Hz | 0.1 s，10 Hz |
| 观测/动作 | 202维 LiDAR/状态/动态物体特征；目标方向坐标系的每轴速度动作 | 深度与10维状态；六维网络输出解码为世界加速度 |
| 优化目标 | 原 NavRL reward | 七项成本及配置的 AC bootstrap/约束 |

这是 [当前环境配置](../../src/isaaclab_navrl/tasks/navigation/config/hummingbird/env_cfg.py) 的真实数值。一个现有场景的单一物理时钟不能同时等同于 0.016 s 和 0.01 s。改用共同 0.002 s 虽可整除两者，却改变两边的物理积分步长与接触/执行器前向；仅保持动作持有时间相同不能证明等价。本轮维持两个独立进程的原时钟。

两个任务都声明了160×120深度相机，但 PPO 策略仍使用 LiDAR。传感器 lazy update 与数据访问会影响实际开销，因此不能直接把 Zhang 的相机耗时当成 PPO 耗时。PPO 属于方法对照，不是只换优化器的消融。

## 5. 小 N 不会把每进程内存变成四分之一

当前 `NavRLEnvCfg.sim.physics=PhysxCfg(...)` 明确覆盖了下列容量；Zhang 继承这些值，配置中没有随 `num_envs` 自动缩放的公式。

| 字段 | 当前显式值 | 官方接口描述的内存位置 |
| --- | ---: | --- |
| `gpu_max_rigid_contact_count` | 524,288 | 页锁定主机内存中的接触流 |
| `gpu_max_rigid_patch_count` | 163,840 | 页锁定主机内存中的接触片流 |
| `gpu_found_lost_pairs_capacity` | 4,194,304 | GPU全局内存中的新增/丢失配对 |
| `gpu_found_lost_aggregate_pairs_capacity` | 33,554,432 | GPU全局内存中的聚合体新增/丢失配对 |
| `gpu_total_aggregate_pairs_capacity` | 4,194,304 | GPU全局内存中的聚合体配对 |

数值来自项目配置；内存类别来自 [官方 PhysxCfg API](https://isaac-sim.github.io/IsaacLab/main/source/api/lab_physx/isaaclab_physx.physics.html#isaaclab_physx.physics.PhysxCfg)。这些容量值不能直接当作字节相加，也不能全部计入显存。heap、collision stack、temp buffer 等未在项目此处覆盖的实际 resolved 值，需从已安装配置另外导出；这里不以滚动 main 的默认值冒充本机实测。PhysX 通过 Kit 运行，其 launcher 也是 [官方 PhysxCfg 的组成](https://isaac-sim.github.io/IsaacLab/main/source/api/lab_physx/isaaclab_physx.physics.html#isaaclab_physx.physics.PhysxCfg.launcher_type)。

从1024降至256，主要减少与 lane 数有关的物理状态、射线和训练张量。单张float32的160×120原始深度，仅像素存储可由计算得出为75 MiB降至18.75 MiB；这不包括射线、命中、几何、传感器缓存和反向图，不能拿它估算总进程显存。多进程仍分别创建 runtime/context、几何、缓冲和分配器；同一路径的USD或相同terrain seed不会自动合并这些GPU资源。

因此本轮**保留现有 buffers**。若以后有必要缩小，先保存 resolved PhysxCfg、进程/全卡峰值与 contact/aggregate overflow 日志，再针对300/0、400/120和扰动阶段做容量及行为对照。只看到S0短检查没报错，不能推出微调高难度也足够；更不能为了十二组同时驻留而无测量下调容量。

## 6. RTX 4090 上的十二组优先级

RTX 4090 标准显存为24 GB，见 [NVIDIA官方规格](https://www.nvidia.com/en-us/geforce/graphics-cards/40-series/rtx-4090/)。本项目已有五组并发下的相机测量，条件和原始记录见 [相机位姿聚合经验](CAMERA_POSE_AGGREGATION_EXPERIENCE.md)；它证明过往特定配置能并发运行，不能证明当前256环境的新十二组可以同时装入或更快。

建议先启动两组核心控制，完成首次完整更新并记录峰值后，再逐个加到三组，测量允许时最多先尝试四组。Kit初始化顺序进行，避免同时启动造成瞬时峰值。若同类Zhang的总实际样本吞吐已不再增加，就维持较少并发、继续排队；不能仅凭显存未满或GPU utilization数字决定增加进程。

下表是容量审计时的队列优先级建议，不是实际启动顺序。随后形成的 [十二组计划矩阵](../results/zhang_restart_experiment_matrix_20261029.md) 与 [参数计划](../../configs/experiments/zhang_restart_controls_20261029.json) 明确了各组配置；最终启动 manifest 决定实际队列和是否执行 fullwindowref。所有组均保持自身已批准的损失、H、学习率和训练预算，不因等待资源增加新的自动停止上限，也不把排队等待计入某组的训练墙钟时间。

| 顺序/波次 | 组 | 关键比较与注意事项 |
| --- | --- | --- |
| A1 | base truncated | 新共同基线，优先取得可比较的实际样本和成功/失败统计 |
| A2 | fullwindowref | 显式`full_window`；同几何、同高度项、同优化器，检验终止补窗影响 |
| A3 | PPO | 保持62.5 Hz及原观测/动作/reward；作为独立方法锚点 |
| A4 | step | 本轮已明确为DCOL下`per_step`，只改变监督时序；实际计划名为`dcol_per_step` |
| B1 | Huber | `target_huber`对比默认`target_exp`，导航目标与高度系数保持一致 |
| B2 | projection | 仅BPTT开启`height_priority`；同时看实际Adam位移，不能只看投影后原始梯度 |
| B3 | plateau2 | 显式Plateau、patience2；Recovery规则保持一致，记录每次实际降率原因 |
| B4 | Adam | 本轮显式Adam、weight_decay=.01；与AdamW保持数值相同，但耦合L2和去耦衰减的机制不同 |
| C1 | RMS | 只切换运行归一化；旧seed0窗口结果不作为新目标高度配置的淘汰证据 |
| C2 | continuous | 保持其碰撞响应与终止语义；属于方案对照，不能只解释为速度模式差异 |
| C3 | SHAC | 固定H32、critic/TDλ及既定value head；先单独量到critic拟合峰值 |
| D1 | AHAC300 | 初始H32、允许增长至300；峰值必须按更长H和critic拟合预留，不只看U1 |

A/B是最多四槽的候选波次，实际先从两槽逐步增加。C初次运行时不同时加入AHAC；确认AC峰值后再决定能否重叠。前一组自然完成或经同预算分析获准结束筛选后，空槽才接收排队组；本表不要求缩短任何现有算法的展开或训练预算。正式1024组另做容量测量，不能直接沿用256组的并发上限。

最终启动回执仍须保存上述实际配置；简短组名不能代替可核对的参数。当前计划没有把十二组同时驻留作为要求，也没有为增加并发而缩短算法H或训练预算。

## 7. 并发测量与公平结论的口径

按相同stage、seed、256环境及实际`environment_steps`截取比较；只能使用完整update及其已经产生的自然报告，不插值、不以某组最新峰值对比另一组早期值。报告最好值、最新值、最近五份按episodes加权的成功率和失败构成；缺报告不能填0。改变并发只影响资源安排，不能改变课程80%门槛或损失归约。

每组每update的真实采样数是`num_envs × rollout_steps`。256环境下BPTT H150是38,400转移，SHAC H32是8,192，AHAC必须逐update累加实际H；PPO H32虽也是8,192转移，但每转移仿真时长只有0.016 s。跨PPO/Zhang同时记录策略转移数与累计仿真秒数，不能声称相同update或相同transition就等于相同物理时间。

256环境的单报告更小，梯度批量及首次80%出现概率都可能与1024不同。十二组也增加了看到偶然尖峰的机会；单seed筛选只确定值得复核的配置，不能宣布普遍优胜。某项损失获得更多有效样本，不等于到达率已经提高。

每次容量记录至少注明：源码和manifest、环境/相机批次、当前stage、H used/next、并发进程数和组名、启动/首次更新/稳态耗时、全卡占用与各进程峰值、Torch allocated/reserved及PhysX警告。Torch峰值不包含全部Kit/PhysX分配，需结合全卡记录。共享负载的wall clock带有竞争，不能冒作独占速度；AHAC后续长H和400/120阶段的峰值必须重新确认。

当前工具的screening标记和正式门槛见 [五阶段课程](../TRAINING_CURRICULUM.md)；稳定档、恢复和自然报告计数见 [Recovery协议](RECOVERY_LADDER_20261029.md)。

## 8. 若以后实现同进程共享，最小可信范围

先限定为**两组同物理频率、同阶段、同几何的Zhang BPTT**，一个owner驱动仿真，两个独立learner只访问自己的lane与状态。实现前先记录两独立进程的固定动作、局部reset和事件序列作为对照，再核对原始深度、执行动作、首次事件、窗口成本和梯度；随后验证某组reset、降率或保存不会影响另一组。噪声关闭时先比较确定性路径，噪声开启后再核对各组RNG独立性和分布。这里只定义验证范围，不承诺跨布局/并发浮点计算逐位相同。

只有这些检查和完整更新的实际吞吐/峰值都支持收益，才考虑AHAC异步更新边界和不同课程阶段。PPO时钟保持独立，除非另立明确改变物理步长的实验。本轮十二组筛选不等待这个架构研究完成。
