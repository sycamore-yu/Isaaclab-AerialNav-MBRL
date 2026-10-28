---
status: accepted
date: 2026-10-28
---

# 迁移 NavRL，同时保留原始导航任务的行为

NavRL 基于 Isaac Sim 2023.1.0-hotfix.1、修改过的 Orbit 和 OmniDrones。
迁移至 Isaac Lab v3.0.0/PhysX 时，使用 `DirectRLEnv` 和
`InteractiveScene` 管理多个并行无人机。所有无人机共用静态地形
和全局动态障碍物。不同并行无人机之间过滤物理碰撞，
而它们仍接收同一全局地形与动态物体的观测。需验证克隆模式、世界坐标和传感器路径。
继续使用 OmniDrones 原版 Hummingbird USD，保留机体质量、惯量、旋翼位置和关节结构。

无人机资产采用 `isaaclab_contrib.assets.Multirotor`。
导航策略保留原版 3 维目标速度、固定的回合目标方向坐标系和
原 OmniDrones `LeePositionController` 速度控制模式。
**2026-10-28 更新决策**：保留原版 OmniDrones `RotorGroup`，
直接执行其归一化指令、转速状态和电机动态计算。
此前“用原生 `Thruster` 近似原电机动态”的选项不再采用。
为接入原生 `Multirotor` 的牛顿推力接口，将原版 `RotorGroup`
计算出的实际旋翼推力交给设置为零延迟的 `Thruster`。
通过 `allocation_matrix` 计算总推力、滚转、俯仰和偏航力矩，
不得再次单独施加 `RotorGroup` 输出的反扭矩。
`ThrusterCfg.torque_to_thrust_ratio` 在 3.0.0 执行路径中未被直接使用。
实际原版 `RotorGroup` 经过 `torch.func.functional_call` /
`torch.vmap` 兼容执行测试通过。
使用原版 Hummingbird USD 和原版 Lee 控制器完成受控 PhysX 闭环对照：
当两方案均按原版更新 4 个旋翼关节速度时，合力矩方案与逐旋翼施力方案
在 480 步内最大位置差为 0.0156 mm。因此采用原生合力矩；
同时保留每物理步的原版旋翼关节速度更新。
省略该更新会使最大位置差增至 12.01 mm。
暂不把控制器替换为 ARL 的 `LeeVelController` 或
`NavigationAction`，因为控制律和动作含义均不同。

保留 8 维导航状态、1×36×4 静态 LiDAR 与最近 5 个动态物体的
5×10 真实状态，保留动态障碍物的运动规律、原始奖励、
原 `reach_goal` 记录、碰撞和终止条件。
原修改版 Orbit 地形算法应作为任务内代码迁移。
该算法包含按概率生成高度及特定间距范围筛选，
标准 3.0 地形配置没有相同的采样规则。

强化学习训练器改为 Isaac Lab 3.0 官方集成的 RSL-RL PPO。
使用原生 Runner、PPO、Beta 动作分布与检查点格式；
允许通过标准模型接口实现 NavRL 的特征网络。
Actor/Critic 必须继续共享原版 CNN、动态障碍编码器和融合网络。
Isaac Lab v3.0.0 使用的 RSL-RL 5.5.1 已包含 `BetaDistribution`；
若要保留原网络结构，需要实现相应模型适配。
原始旧权重应有独立的推理加载及新模型初始化路径，
不把旧的 `policy.state_dict()` 视为可直接恢复训练的 RSL-RL 检查点。
RSL-RL 的优化器、共享编码器、价值损失和归一化差异须保留在实验报告中。

以上选择优先保证环境与控制行为可核对，同时使用标准维护中的训练器。
旋翼施力方式已完成受控 PhysX 闭环验证，见
[`../PHYSX_CLOSED_LOOP_COMPARISON.md`](../PHYSX_CLOSED_LOOP_COMPARISON.md)。
网络共享方式、Hummingbird 原版 USD 和跨环境碰撞隔离现已确定。
原始代码基线不再作为需要用户决定的事项。
导航环境和 RSL-RL 训练入口已实现；小规模共享场景与碰撞隔离检查已通过。1024 环境正式训练和最终成功率验收尚待完成，状态见 [实现与验证状态](../IMPLEMENTATION_STATUS.md)。
详细来源、数值配置、已通过的电机单元测试和迁移验收见
[`../NAVRL_ISAACLAB_3_MIGRATION.md`](../NAVRL_ISAACLAB_3_MIGRATION.md)。
