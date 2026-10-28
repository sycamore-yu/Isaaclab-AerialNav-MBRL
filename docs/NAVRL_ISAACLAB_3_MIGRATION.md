# NavRL → Isaac Lab 3.0：迁移决策与验收

状态：**动力学闭环对照已通过；完整 NavRL 任务及 RSL-RL 训练待实现**。修订：2026-10-29。
采用以下固定控制链：

```text
NavRL 3 维速度策略 → OmniDrones LeePositionController
→ OmniDrones RotorGroup → Isaac Lab Multirotor 合力矩 → PhysX
```

每个物理步同时执行
`rotor_joint_velocity[i] = RotorGroup.throttle[i] * directions[i] * max_rotation_velocities[i]`。

目标版本为
[Isaac Lab v3.0.0](https://github.com/isaac-sim/IsaacLab/tree/v3.0.0)、
Isaac Sim 6.2、PhysX。原始基线为
[Zhefan-Xu/NavRL](https://github.com/Zhefan-Xu/NavRL)


## 一、已接受的决策

| 范围 | 决策 | 理由 |
|---|---|---|
| 环境 | 独立 NavRL 任务，采用 `DirectRLEnv` | 保持原版步进、奖励和重置逻辑 |
| 并行仿真 | 使用 `InteractiveScene` 管理多架无人机；共用一份静态地形和一组动态障碍物 | 保留原版共享世界训练 |
| 碰撞过滤 | 不同并行无人机之间过滤物理碰撞；它们仍共同使用静态地形与动态障碍物 | 保持原版跨环境碰撞隔离，不改变共享地图 |
| 动力学 | 使用 `isaaclab_contrib.assets.Multirotor` 的机体合力矩；原版 `RotorGroup` 更新电机状态和旋翼关节速度，`Thruster` 仅作为零延迟推力接口 | 保留原版电机和旋翼转动动力学；已通过受控 PhysX 闭环对照 |
| 无人机 USD | 继续使用 OmniDrones 原版 Hummingbird USD，保留质量、惯量、旋翼位置与关节结构 | 维持原版物理资产 |
| 高层动作 | 保留 NavRL 的 3 维目标速度与目标坐标系变换 | 原版策略动作的物理意义不变 |
| 底层控制 | 直接复用 OmniDrones `LeePositionController` 的速度控制模式、参数及控制分配 | 保留原策略控制律 |
| 观测 | 保留 8 维状态、1×36×4 LiDAR、最近 5 个动态障碍物各 10 维真实状态 | 保留原策略输入；动态物体真值允许输入策略 |
| 动态障碍物 | 共用全局动态物体；维持原始运动、尺寸、速度采样和几何碰撞判定 | 保留同一动态场景和奖励输入 |
| 静态障碍物 | 保留原始高度场生成算法、种子与几何分布 | NavRL 的 Orbit 地形生成器曾经过定制 |
| 强化学习 | 改为 Isaac Lab 官方集成的 **RSL-RL PPO**，不再运行旧版自定义 PPO 训练循环 | 使用标准训练器、记录及检查点格式 |
| 网络与检查点 | Actor/Critic 必须继续共享原版 CNN、动态障碍编码器及融合特征；保持 Beta 分布，转换旧检查点 | 保留原始网络参数共享语义 |
| 奖励与成功率 | 保留 NavRL 原始奖励、终止条件及 `reach_goal` 记录方法 | 保持任务目标和原始评估口径 |
| 验证 | 已完成受控 PhysX 施力对照；后续验证传感器、奖励、检查点、共享场景与训练 | 按第六节验收 |

本次迁移保留原版动态物体真值输入、奖励及任务定义；使用 PhysX，

## 二、动作与控制器：保持 NavRL 原样

### 2.1 原版实际控制器

证据：
[`train.py`](https://github.com/Zhefan-Xu/NavRL/blob/main/isaac-training/training/scripts/train.py)、
[`transforms.py`](https://github.com/Zhefan-Xu/NavRL/blob/main/isaac-training/third_party/OmniDrones/omni_drones/utils/torchrl/transforms.py)、
[`lee_position_controller.py`](https://github.com/Zhefan-Xu/NavRL/blob/main/isaac-training/third_party/OmniDrones/omni_drones/controllers/lee_position_controller.py)。

原版明确使用 `LeePositionController(9.81, env.drone.params)` 和
`VelController(controller, yaw_control=False)`。虽然名字含有 Position，
但在这一配置下**实际按目标速度进行反馈控制**：

1. PPO 的三个 Beta 样本处于 `[0, 1]`，分别仿射映射至 `[-2, 2]` m/s。
2. `vec_to_world` 将起点至目标形成的局部水平坐标系速度变换为世界坐标系速度 `v_cmd^w`。
3. `VelController` 将 `v_cmd^w` 传入 `LeePositionController(target_vel=...)`。
4. `target_pos` 采用**当前**位置，故该次调用的位置误差为 0；`target_acc=0`。
5. `target_yaw=None` 时**每步读取当前偏航角**作为参考，而非固定起飞时的偏航角。
6. 控制器以世界速度误差生成期望加速度，用 SO(3) 姿态误差和角速度反馈生成转矩，再经 4 旋翼控制分配生成 `[-1, 1]` 格式的电机命令。

速度环在原始调用下可写成：

```text
e_v = v_world - v_command_world
a_aux = K_v * e_v - g * e_z
F_des = -m * a_aux
期望姿态由 F_des 的方向与当前偏航参考构造
姿态几何反馈 + 角速度反馈 → 3 轴角加速度控制量
旋翼混合矩阵 → 4 旋翼推力 → 归一化电机命令
```

Hummingbird 原版配置：

| 参数 | 原版值 |
|---|---|
| 质量 | 0.716 kg |
| 惯量对角线 | [0.007, 0.007, 0.012] kg·m² |
| 旋翼臂长 | 0.17 m，4 个旋翼 |
| 最大转速 | 838 rad/s |
| 速度增益 | [2.2, 2.2, 2.2] |
| 位置增益 | [4, 4, 4]；本速度调用的位置误差为 0 |
| 姿态增益 | [0.7, 0.7, 0.035]，源码内还与惯量逆矩阵相乘 |
| 角速度增益 | [0.1, 0.1, 0.025]，源码内还与惯量逆矩阵相乘 |
| 最大单旋翼推力 | 约 6.00319 N |
| 初始悬停每旋翼推力 | 约 1.75599 N |

原版姿态控制中有 `torch.cross(ang_vel, ang_vel)`，结果恒为零。
因此迁移时应保留这个实际控制律，不要自行加入完整的
`ω × (Jω)` 反馈后再声称与原版等价。

**4 个归一化旋翼指令**：控制器输出 `u = [u1,u2,u3,u4]`，
每个分量对应一台电机，以 `[-1,1]` 为名义范围。
旧控制器的输出未在返回前裁剪；原版 `RotorGroup` 在换算时裁剪到有效范围，按
`q_target = sqrt(clamp((u+1)/2,0,1))` 设置归一化目标转速，
再按 `q_next = q + 0.43*(q_target-q)` 更新实际转速；
实际推力为 `KF*q_next²`。因此稳态时：

| 电机指令 | 目标推力占单旋翼最大推力的比例 |
|---|---:|
| `-1` | 0% |
| `0` | 50% |
| `1` | 100% |

负指令不代表反转旋翼。原始 Hummingbird 悬停时单旋翼推力为
`0.716*9.81/4 = 1.75599 N`，对应约 `-0.415` 的稳态归一化指令。

**容易忽视的坐标约定**：`target_dir` 在 `_reset_idx` 中由起点和目标计算，
在同一回合内不逐步更新。策略使用的是**回合开始时固定的目标方向坐标系**。
`direction` 是动作变换所需的元数据，不应新增至 CNN/MLP 策略输入。

### 2.2 原版 Lee 与 Isaac Lab 3.0 ARL Lee 的关系

两者都属于 Lee 风格的 SE(3) 几何控制器，但实现并不相同。
Isaac Lab 的 ARL 默认使用
[`LeeVelController`](https://github.com/isaac-sim/IsaacLab/blob/v3.0.0/source/isaaclab_contrib/isaaclab_contrib/controllers/lee_velocity_control.py)。

| 比较项 | NavRL `LeePositionController` 速度调用 | 3.0 `LeeVelController` |
|---|---|---|
| 速度命令输入 | 世界坐标系 3 维速度 | 航向旋转坐标系 3 维速度加偏航角速度，共 4 维 |
| 偏航 | 每步采用当前偏航角作为参考 | 额外输入偏航角速度 |
| 增益 | 固定来自 `lee_controller_hummingbird.yaml` | 可在每次重置时从增益区间采样 |
| 几何反馈 | 由旋转矩阵构造姿态误差，角速度阻尼 | 由相对四元数构造姿态误差，角速度阻尼 |
| 角动力学补偿 | 交叉乘相同角速度，恒为零 | 包含 `ω × (Jω)` 项 |
| 结果 | 4 个归一化旋翼指令，内部完成分配 | 6 维机体力与力矩，外部完成分配 |

ARL 自带的
[`NavigationAction`](https://github.com/isaac-sim/IsaacLab/blob/v3.0.0/source/isaaclab_contrib/isaaclab_contrib/mdp/actions/thrust_actions.py)
虽然也是 3 维动作，但三个分量表示**速度幅值、俯仰倾角、偏航指令**，
并将侧向速度设为 0。**不能替代 NavRL 的三轴笛卡尔速度动作**。

已接受的适配方式：将原版 `LeePositionController` 作为任务内的控制器实现；
实现小型动作适配层，读取 Isaac Lab 的根状态，把位置、四元数和速度整理成
旧控制器所需的 13 维状态；把 4 个旧电机命令转换为牛顿单位的目标推力，
传给 `Multirotor.set_thrust_target()`。不要修改 RSL-RL PPO 来完成此转换。

**四元数顺序**：NavRL/OmniDrones 使用 `(w,x,y,z)`；Isaac Lab 3.0 的
`isaaclab.utils.math` 和资产数据使用 `(x,y,z,w)`。
必须在控制器适配层显式调整顺序，同时区分世界角速度和机体角速度；
原版 `LeePositionController(body_rate=False)` 期望世界角速度并自行转换。
3.0 的 `lee_controller_utils.compute_desired_orientation` 文档字符串仍写着
`wxyz`，但它调用的 `isaaclab.utils.math.quat_from_matrix` 实际采用
`xyzw`；以正式版数学函数实现与迁移指南为准。

Isaac Lab 3.0 另有 `LeePosController`（目标位置和偏航角）、
`LeeAccController`（加速度和偏航角速度）及 `LeeAttController`
（推力、滚转、俯仰和偏航角速度），均不替换本次选定的原版控制器。
`LeePosController` 不接收原版 `LeePositionController` 支持的目标速度或加速度前馈。
`LeePosController` 的位置误差直接由 `command[:,:3] - root_pos_w` 计算，
实际按世界坐标系解释位置指令；其文档中的“body frame”与源码不一致。

## 三、保留原版 `RotorGroup`

保留原版 `RotorGroup` 的电机公式、归一化指令、`throttle` 状态及重置规则。
原生 `Thruster` 只转发推力，不负责电机动态。

使用 `isaaclab_contrib.assets.Multirotor`。
其 `_apply_actuator_model()` 只处理 `isinstance(actuator, Thruster)` 的对象，
因此不能直接将 `RotorGroup` 填入 `MultirotorCfg.actuators`。
原生 `Thruster` 采用零延迟配置作为接口层：

1. 原版 `LeePositionController` 产生 4 个归一化电机指令。
2. 原版 `RotorGroup` 每个物理步执行一次，更新 `throttle` 并输出实际推力和反扭矩。
3. 调用 `Multirotor.set_thrust_target()`，传入 `RotorGroup` **已经计算好的实际推力（N）**。
4. 接口层的 `Thruster` 使用 Euler、离散近似及 `tau_inc=tau_dec=0`，
   不再增加电机动态。推力变化率上限须足够大，避免额外限幅。
5. `Multirotor` 使用 `allocation_matrix` 由每个旋翼的实际推力计算机体合力与力矩，交给 PhysX。
6. 每个物理步按原版代码写入 4 个旋翼关节速度：
   `omega_i = throttle_i * direction_i * max_rotation_velocity_i`。

该方案保持 `Multirotor` 的牛顿推力接口，无须修改 Isaac Lab 或 RSL-RL PPO。
原版 `RotorGroup` 的反扭矩由分配矩阵统一生成，**不得重复施加**。
完整 `6×4` 分配矩阵的行顺序为 `[Fx,Fy,Fz,τx,τy,τz]`，
每个旋翼 `i` 的列由以下关系给出：

```text
x_i = l_i cos(theta_i), y_i = l_i sin(theta_i)
Fz = Σ T_i, τx = Σ y_i*T_i, τy = -Σ x_i*T_i
τz = -Σ directions_i*(KM_i/KF_i)*T_i
```

`ThrusterCfg.torque_to_thrust_ratio` 在 3.0.0 的执行路径中不参与偏航力矩计算，
必须正确填写 `allocation_matrix` 的偏航行。
Hummingbird 使用 `directions=[-1,1,-1,1]`、
`KM/KF≈0.016 m`；按原版 `k_f=8.54858e-6 N/(rad/s)²` 和
`ω_max=838 rad/s` 配置推力范围。原生 `Thruster` 的 RPS 系数为
`k_f(rps)=k_f(rad/s)·(2π)²`。
`Multirotor.reset()` 会将默认转速 RPS 写入 `thrust_target`；
应在首个物理步前将其覆盖为实际推力（N），同时恢复 `RotorGroup.throttle`。

原 OmniDrones `MultirotorBase` 依赖
`tensordict.nn.make_functional`。在当前 Isaac Lab 3.0 环境中直接执行会报
`RuntimeError: make_functional has been removed from tensordict.`。
可以在不修改 `RotorGroup` 类本身的前提下，使用
`torch.func.functional_call` 与 `torch.vmap(randomness="different")`
批量执行并保留每个环境独立的 `throttle` 状态。
原控制器仅迁移所需实现和数学函数，避免引入旧 OmniDrones 的仿真依赖。

**已通过的 CPU 对照**：`tests/test_original_rotor_integration.py`
使用真正的原版 `RotorGroup` 类，在 8 个环境、4 个旋翼、250 个电机更新步内：

| 比较 | 最大绝对误差 |
|---|---:|
| 新 `vmap` 调用与原版逐个调用（推力、反扭矩） | 0 |
| 零延迟 `Thruster` 的推力透传 | `9.54e-7 N` |
| 分配矩阵偏航力矩与原始 `RotorGroup` 输出 | `2.98e-8 N·m` |

同一 Python 环境中，直接加载原版 `LeePositionController` 源码完成了
3 台 Hummingbird 的静止悬停指令计算，得到约 `-0.415` 的每旋翼命令。
随后已使用原控制器和原 `RotorGroup` 完成 PhysX 闭环对照，
见 [`PHYSX_CLOSED_LOOP_COMPARISON.md`](PHYSX_CLOSED_LOOP_COMPARISON.md)。
当两方案均更新旋翼关节速度时，480 步最大轨迹位置差为
`0.0156 mm`；若只有原版逐旋翼施力方案更新关节速度，
最大位置差为 `12.01 mm`。因此采用原生合力矩，同时保留
原版旋翼关节速度更新。

## 四、保留共享训练世界

原版 `NavigationEnv._design_scene` 使用：

- 全局 `/World/ground` 静态地形，默认 350 个障碍物；
- 全局位置数组管理的 80 个动态障碍物；
- `GridCloner` 只复制无人机实例；多架无人机面对同一份地形和运动障碍；
- `replicate_physics=false`，且显式调用 `filter_collisions`；
- 原配置 `cfg.env.env_spacing=8.0`；不要把这个环境克隆间距误认为地形间距。

迁移实现：`InteractiveScene` 管理每架无人机的独立资产和
`env_ids`，通过单例对象或场景级资源管理共享地形及动态障碍物。
静态地形、动态障碍物的真实位置使用**世界坐标系**。
在应用任何 `scene.env_origins` 偏移前，确认写入和观测都是哪种坐标。

首先固定 `replicate_physics=False`，验证所有无人机都对同一份全局
地形求射线。保留原版克隆间碰撞过滤意图；不能把全局地形过滤掉。
每步应保证：更新无人机控制 → PhysX 步 → 更新动态障碍物根状态
→ 更新 LiDAR → 构造下一步观测和奖励。
若 Isaac Lab 场景生命周期要求调整顺序，应通过步进测试检查物体可见性与滞后。

## 五、还原原版时必须额外处理的冲突

这些是 NavRL/OmniDrones 特有的行为，不能只依赖官方
[3.0 迁移指南源码](https://github.com/isaac-sim/IsaacLab/blob/v3.0.0/docs/source/isaaclab/migrating.rst)。

1. **定制地形生成器缺失**：旧 Orbit
   `HfDiscreteObstaclesTerrainCfg` 修改版支持
   `obstacle_height_mode="range"`、
   `obstacle_height_range=[1.0,1.5,2.0,4.0,6.0]`、
   `obstacle_height_probability=[0.1,0.15,0.20,0.55]`
   和障碍物间距约束。3.0 标准生成器只处理 `choice/fixed`，
   不含上述概率和间距逻辑。需要把原函数迁至任务专用的地形生成器，
   不能直接使用同名原生配置来声称场景一致。原版 `good_distance`
   排除特定网格间距范围内的候选布局，且部分重叠情况会被直接放行；
   它不是通常意义上的统一最小间距。应复用实际采样逻辑，
   不要自行换成常规 Poisson-disk 最小距离采样。
2. **动态物体不参与原版 LiDAR**：`RayCasterCfg.mesh_prim_paths`
   原版只有 `/World/ground`。动态障碍物的真值经单独的
   5×10 输入传给策略。迁移保留两条通道，且动态刚体原版设置
   `collision_enabled=False`。
3. **动态物体的碰撞是几何判定**：原版根据最近 5 个物体的
   距离、尺寸和 0.3 m 安全余量计算碰撞。物理接触传感器不能替代。
   保留静态 LiDAR 的 0.3 m 碰撞阈值。
4. **动态更新每策略步运行一次**：原版 `substeps=1`，
   `sim.dt=0.016`，每隔 `int(2.0/dt)=125` 个动态更新步
   重新采样速度。不能直接改成 ARL 的 `sim.dt=0.01`、
   `decimation=10`。
5. **目标坐标系在回合内固定**：`target_dir` 只在重置时更新。
   目标方向速度奖励使用当前目标相对位置，控制动作的坐标变换
   则使用固定的初始目标方向。这两个计算不能合并成同一个更新方向。
6. **原始成功事件不是终止条件**：`distance < 0.5 m`
   设置即时 `reach_goal`；`terminated` 由高度越界或碰撞触发，
   超时由 `progress_buf >= 2200` 触发。迁移保留此定义，
   并可额外记录“首次到达率”，不能替换原日志口径。
7. **旧奖励完整保留**：目标方向速度奖励 + 常数 1 +
   静态安全对数奖励 +（动态场景下）动态安全对数奖励
   − 0.1×速度变化惩罚 − 8×高度惩罚。
   原代码的碰撞终端 `-50` 已被注释，不能擅自启用。
8. **旋翼施力方式已验证**：同一 Hummingbird USD、原版控制器和电机模型
   在 480 步 PhysX 闭环中，合力矩与逐旋翼施力的最大位置差为
   `0.0156 mm`（两组均同步更新旋翼关节速度）。保留原版关节速度更新。
   原版 `MultirotorBase` 默认 `drag_coef=0`，
   `self.n > 1` 才执行下洗计算；不要因为 YAML 中的
   `drag_coef: 0.2` 就直接启用气动力。
9. **重置和状态缓冲变化**：旧 `set_world_poses`、RotorGroup
   内部电机状态与旧 `TensorDict` 缓冲，均须按
   `Multirotor` 的状态写入与重置生命周期重建。
   应验证同一步重置后电机推力、速度、偏航与观测是否一致。
10. **观测顺序与随机样本**：LiDAR 的 `[1,36,4]`
    布局、未命中值、深度取值为 `4.0-distance`、
    起点终点分布、初始机头朝向、动态障碍物排序与截断逻辑
    必须保持一致。旋转顺序不能只靠数据形状判断。
11. **策略输入和动作变换不同层**：策略 CNN/MLP 实际读取
    `state`、`lidar`、`dynamic_obstacle`；
    `direction` 供动作变换使用。RSL-RL 的
    `obs_groups` 不应悄悄把 `direction` 加入策略网络。
12. **不同版本会改变随机性和物理结果**：即使采用相同种子，
    旧 Orbit 的 NumPy 采样、网格生成、PhysX 接触和
    新 RSL-RL 的采样顺序也未必逐样本一致。
    应比较统计分布、动力学响应与成功率，而非要求随机轨迹逐点相同。

### 5.1 RSL-RL PPO 的兼容范围

Isaac Lab v3.0.0 锁定 `rsl-rl-lib==5.5.1`。该版本已有
[`BetaDistribution`](https://github.com/leggedrobotics/rsl_rl/blob/main/rsl_rl/modules/distribution.py)
与 CNN 模型接口；其 Beta 分布支持将 `[0,1]` 样本映射到
`[-2,2]`。因此可以保留 3 维 Beta 动作分布。
对照脚本 `tests/test_rsl_beta_contract.py` 已验证：
32 组随机分布参数下，旧 Beta 均值经速度缩放后与 RSL-RL 均值
的最大差为 `4.77e-7 m/s`；计入尺度变换的对数概率差
最大为 `1.43e-6`。这验证分布语义，不代表网络权重已成功转换。

使用标准 `OnPolicyRunner`、`PPO`、
`RslRlVecEnvWrapper` 和训练入口。原版 PPO 训练参数的迁移起点：

| 原版参数 | 值 | RSL-RL 对应 |
|---|---:|---|
| 每个环境每次采样步数 | 32 | `num_steps_per_env=32` |
| 更新轮数 | 4 | `num_learning_epochs=4` |
| 小批量数 | 16 | `num_mini_batches=16` |
| 折扣系数 | 0.99 | `gamma=0.99` |
| GAE λ | 0.95 | `lam=0.95` |
| 策略裁剪 | 0.1 | `clip_param=0.1` |
| 熵系数 | 0.001 | `entropy_coef=0.001` |
| 学习率 | 0.0005 | `learning_rate=0.0005` |
| 梯度裁剪 | 旧 actor/critic 各 5 | `max_grad_norm=5` 起点；实际裁剪语义有差异 |

必须记录**无法保持逐项相同的训练行为**：

- **已决定保留原版共享编码器**：Actor/Critic 使用同一个
  LiDAR CNN、动态障碍特征网络及融合 MLP 对象，共享参数和梯度。
  RSL-RL 标准 PPO 构造优化器时依次收集 actor/critic 参数；
  模型适配时应确保共享参数只在优化器中注册一次，
  否则会重复更新。应在不修改 PPO 算法更新规则的前提下，
  使用单一共享模块所有权和模型引用，并添加参数 ID 唯一性、
  双分支梯度累积及保存/加载测试。
- 原版 Critic 使用 `HuberLoss(delta=10)` 和 `ValueNorm`，
  RSL-RL 标准 PPO 使用其内建价值损失与优势计算。
  默认不新增观测归一化，记录价值函数归一化差异。
- 原版在 PPO 更新里给 actor、critic 分别设置优化器，
  还按 `action_dim` 缩放 actor 损失。
  RSL-RL 的优化器、损失缩放和学习率调整不同。


实现路线：使用官方 PPO 与 Runner；通过 RSL-RL 标准模型接口添加
兼容原始 NavRL 特征网络的 actor/critic 模型类，而不是复制或修改 PPO 算法本体。


### 5.2 原版检查点与 RSL-RL 检查点

旧 `train.py` 保存的是 `torch.save(policy.state_dict(), path)`；
RSL-RL Runner 保存的字典还包含 actor、critic、
优化器和训练迭代信息。两者**不能直接互相 `load`**。

迁移时提供两个独立操作：

1. **旧模型推理**：读取旧 `checkpoint_*.pt`，
   用原始输入布局和 Beta 分布恢复旧策略；在固定观测批次上
   比对新适配模型的输出参数 `α,β`、期望速度及最终电机命令。
2. **RSL-RL 初始化**：将可匹配的旧 CNN、动态特征网络、
   共享 MLP、actor/critic 头权重分别映射到新模型；
   初始化后用标准 RSL-RL 训练并保存其正式检查点。

旧检查点没有 RSL-RL 需要的优化器状态，
因此转换仅提供推理兼容和权重初始化，**不能称为精确续训**。
转换脚本必须列出每个已映射键、形状不符键与未加载键，
并检查 deterministic 动作输出。所有旧检查点保持只读。

## 六、迁移验收

| 阶段 | 验收内容 | 通过条件 |
|---|---|---|
| A：环境启动 | 加载 Hummingbird、地形、动态物体 | 1 与 16 个无人机可 reset/step；全部状态有限 |
| B：动力学 | 原版 Lee、RotorGroup、悬停、速度阶跃及偏航 | **受控 PhysX 施力对照已通过**；正式导航场景中的闭环仍待验收 |
| C：LiDAR | 36×4 射线、方向、距离、未命中及更新时序 | 帧形状一致，静态地形可检测，动态障碍不混入静态 LiDAR |
| D：动态场景 | 80 个全局动态障碍物，速度采样和位姿更新 | 位置/速度连续、125 更新步重采样、所有无人机观测同一组物体 |
| E：奖励与终止 | 所有奖励项和 NavRL 碰撞、高度、超时判定 | 同样输入状态的逐项奖励与判定一致 |
| F：控制动作 | Beta 3 维、目标坐标系、Lee 电机命令 | 原版与新适配器在固定状态/动作下输出可比较 |
| G：检查点 | 旧权重映射、RSL-RL 保存/加载 | 推理输出比对通过；新检查点可独立恢复 |
| H：并行训练 | 16 → 1024 架无人机共享世界 | 不复制全局障碍物；不错误过滤静态网格；统计吞吐和显存 |
|学习效果使用并行无人机数量1024	最大训练步数2200，单次训练得到的（保持纯净不要续训）策略，每个难度独立评测至少 100 回合。记录首次到达成功率、碰撞率、超时率及成功到达时间。在 350 静态 + 100 动态障碍物下，平均成功率达到 80% 左右 

通过 A 至 G 才进入正式长时训练。未测得旧基线前不指定
轨迹误差和最终成功率的数值阈值。首次到达率、回合末到达率、
碰撞率、超时率与到达时间可作为**附加诊断**，
原 `reach_goal` 和奖励口径保持独立。

### 已完成验证

```bash
cd /home/tong/tongworkspace/free/isaaclab-navrl
/home/tong/tongworkspace/reference_repos/IsaacLab-v3.0.0/.venv/bin/python \
  tests/test_original_rotor_integration.py
/home/tong/tongworkspace/reference_repos/IsaacLab-v3.0.0/.venv/bin/python \
  tests/test_rsl_beta_contract.py
```

CPU 已验证原版 `RotorGroup` 批量执行、零延迟接口透传及 Beta 分布。
独立 Hummingbird PhysX 闭环复现命令：

```bash
OMNI_KIT_ACCEPT_EULA=YES \
  /home/tong/tongworkspace/free/issaclab-ARL/.venv/bin/python \
  tests/physx_closed_loop_compare.py --visualizer none --steps 480 \
  --spin-rotors both \
  --output docs/results/physx_force_placement_spinning_both_480.json
```

三组施力对照结果见
[`PHYSX_CLOSED_LOOP_COMPARISON.md`](PHYSX_CLOSED_LOOP_COMPARISON.md)。
完整 NavRL 共享场景、RSL-RL 训练和正式成功率评估尚未完成。

## 七、来源

- NavRL 基线：[训练环境](https://github.com/Zhefan-Xu/NavRL/blob/main/isaac-training/training/scripts/env.py)、[训练入口](https://github.com/Zhefan-Xu/NavRL/blob/main/isaac-training/training/scripts/train.py)、[PPO](https://github.com/Zhefan-Xu/NavRL/blob/main/isaac-training/training/scripts/ppo.py)、[Hummingbird 配置](https://github.com/Zhefan-Xu/NavRL/blob/main/isaac-training/third_party/OmniDrones/omni_drones/robots/assets/usd/hummingbird.yaml)。
- Isaac Lab 3.0：[迁移指南源码](https://github.com/isaac-sim/IsaacLab/blob/v3.0.0/docs/source/isaaclab/migrating.rst)、[多旋翼与 Thruster](https://github.com/isaac-sim/IsaacLab/tree/v3.0.0/source/isaaclab_contrib/isaaclab_contrib)、[ARL 导航环境](https://github.com/isaac-sim/IsaacLab/blob/v3.0.0/source/isaaclab_tasks/isaaclab_tasks/contrib/drone_arl/navigation/config/arl_robot_1/navigation_env_cfg.py)。
- 原始理论：[T. Lee 等，Geometric Tracking Control of a Quadrotor UAV on SE(3)](https://arxiv.org/abs/1003.2005)。控制律与具体软件实现的差异以源码为准。
