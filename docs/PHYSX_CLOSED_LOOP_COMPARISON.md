# Hummingbird 旋翼施力方式：PhysX 闭环对照

结果（2026-10-28）：在 Isaac Lab 3.0.0 / Isaac Sim 6.2 PhysX 中，
**等效机体合力矩与逐旋翼施力的闭环轨迹接近一致**。
在两架无人机均执行原版旋翼关节转速更新时，480 个仿真步的最大位置差为
0.0156 mm。若只让原版逐旋翼施力方案更新旋翼关节转速，
最大位置差增至 12.01 mm。因此，迁移时应保留旋翼关节速度更新，
物理推力则可交给原生 `Multirotor` 的合力矩接口。

## 实验设计

源码：[`tests/physx_closed_loop_compare.py`](../tests/physx_closed_loop_compare.py)。

在同一个 Isaac Sim 6.2 PhysX 世界中加载两份**同一原版 OmniDrones
Hummingbird USD**，分别放在不同位置。两架无人机使用相同的：

- 原版 `LeePositionController`，参数取自 `lee_controller_hummingbird.yaml`；
- 原版 `RotorGroup`，逐物理步更新电机 `throttle`；
- 时间步 `dt=0.016`，控制器、旋翼更新与物理模拟每步执行一次；
- 质量、惯量、4 个刚体旋翼及 4 个旋翼关节；实测基座质量
  0.680 kg，4 个旋翼各 0.009 kg，总质量 0.716 kg；
- 速度控制命令：悬停 120 步、世界 X 方向 1 m/s 120 步、
  世界 Z 方向 0.5 m/s 120 步；最后 120 步执行 60° 偏航参考。

最后一个偏航参考用于测试姿态及偏航力矩通道。
原版 NavRL 默认没有独立偏航动作，此处额外激励原控制器已有的
`target_yaw` 参数，不改变控制律。

唯一的施力方式差别如下：

| 方案 | 施力过程 |
|---|---|
| 合力矩 | 原版 `RotorGroup` 输出实际旋翼推力，原生 `Multirotor` 通过零延迟 `Thruster` 和 `allocation_matrix` 将其合成为 6 维机体力矩，施加于机体质心 |
| 逐旋翼 | 原版 `RotorGroup` 的 4 个推力分别施加在对应旋翼刚体，反扭矩施加于基座；与 OmniDrones 原始执行方法一致 |

两种方案均使用一致的机体参考姿态和世界速度状态。
测试读取根节点位置、世界线速度、世界角速度和偏航角，
比较扣除初始平移后的世界轨迹。
不在场景中添加地形、LiDAR 或动态障碍，避免其他因素干扰施力方式比较。

## 实测结果

每种设置独立运行 480 步。

| 旋翼关节速度更新 | 位置最大差异 | 位置 RMS 差异 | 速度 RMS 差异 | 偏航最大差异 |
|---|---:|---:|---:|---:|
| 两架均不更新 | 0.0080 mm | 0.0041 mm | 9.74×10⁻⁷ m/s | 0.000041° |
| 两架均按 `RotorGroup.throttle` 更新 | 0.0156 mm | 0.0113 mm | 1.41×10⁻⁶ m/s | 0.000102° |
| **只有逐旋翼施力组更新** | **12.01 mm** | **4.92 mm** | **0.00822 m/s** | **0.01898°** |

在两架均更新旋翼关节速度的设置下，各阶段最大位置差：

| 阶段 | 最大差异 |
|---|---:|
| 悬停 | 0.00444 mm |
| 世界 X 方向速度 | 0.01554 mm |
| 世界 Z 方向速度 | 0.01563 mm |
| 偏航参考 | 0.01542 mm |

在这一闭环中，悬停阶段结束时两方案的高度均比起点低约
0.0712 m；水平速度阶段末，合力矩方案的世界 X 速度约为
0.9998 m/s；垂直速度阶段末，世界 Z 速度约为 0.4932 m/s。
偏航阶段末约为 62.47°。本对照评估的是两种施力方式之间的差异，
这些绝对响应数据留作后续控制器整体验收参考。

**结论**：旋翼施力位置这一迁移冲突已完成受控 PhysX 闭环验证。
对于原版 Hummingbird 刚体、旋翼布局和当前测试动作，
在同步更新旋翼关节速度的条件下，采用原生合力矩没有产生有意义的轨迹差异。
如果省略原版旋翼关节转速更新，水平与垂直速度段会产生厘米级差异。

## 迁移实现要求

1. 读取原版 `LeePositionController` 的 4 个归一化电机命令。
2. 用原版 `RotorGroup` 更新每个无人机独立的 `throttle`。
3. 将 `RotorGroup` 输出的**实际**旋翼推力作为
   `Multirotor.set_thrust_target()` 的输入，单位为 N。
4. 将原生 `Thruster` 设置为 Euler、离散近似、零延迟，
   避免再次引入电机惯性。
5. 为 `allocation_matrix` 配置完整的 6 维力矩映射；
   其偏航行使用 `-direction_i*KM_i/KF_i`。
6. 同一物理步按原版代码设置 4 个旋翼关节速度：
   `omega_i=throttle_i*direction_i*omega_max_i`。
7. 控制器状态输入从 Isaac Lab `XYZW` 四元数转换为原版 `WXYZ`，
   保持世界角速度和速度的含义不变。

测试直接使用原版控制器和旋翼代码，但尚未实现
NavRL 任务注册、共享地形、全局动态障碍物、跨环境碰撞过滤和
RSL-RL PPO 训练。因此，上述结论适用于本次**受控施力对照**，
不是整个 NavRL 导航任务的成功率结果。

## 复现实验

```bash
cd /home/tong/tongworkspace/free/isaaclab-navrl
PY=/home/tong/tongworkspace/free/issaclab-ARL/.venv/bin/python
OMNI_KIT_ACCEPT_EULA=YES "$PY" tests/physx_closed_loop_compare.py \
  --visualizer none --steps 480 --spin-rotors none \
  --output docs/results/physx_force_placement_no_spin_480.json
OMNI_KIT_ACCEPT_EULA=YES "$PY" tests/physx_closed_loop_compare.py \
  --visualizer none --steps 480 --spin-rotors both \
  --output docs/results/physx_force_placement_spinning_both_480.json
OMNI_KIT_ACCEPT_EULA=YES "$PY" tests/physx_closed_loop_compare.py \
  --visualizer none --steps 480 --spin-rotors individual \
  --output docs/results/physx_original_rotor_spin_only_480.json
```

完整逐步状态和误差分别保存于
[`docs/results/physx_force_placement_no_spin_480.json`](results/physx_force_placement_no_spin_480.json)、
[`docs/results/physx_force_placement_spinning_both_480.json`](results/physx_force_placement_spinning_both_480.json)
和
[`docs/results/physx_original_rotor_spin_only_480.json`](results/physx_original_rotor_spin_only_480.json)。
