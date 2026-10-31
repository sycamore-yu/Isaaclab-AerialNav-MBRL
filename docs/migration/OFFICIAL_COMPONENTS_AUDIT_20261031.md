# 官方组件替换与资源优化实验 · 2026-10-31

## 结论

已在独立 demo 分支完成源码清点、3 项候选实验、组件与真实环境验证、完整训练 ABBA、逐项消融和最终版本复测。根据实验撤回 DelayBuffer clone 清理，最终只保留两项相机优化。

**已证明的收益是显存下降；本轮没有证明稳定训练提速。** 最终两项版本在 32 环境 fS1/40、H150、20 Hz、Newton+Warp 下，PyTorch 峰值分配从基线 **182.018 MiB 降到 166.361 MiB**，减少 **15.657 MiB（8.60%）**；进程显存采样峰值从 **1,436 MiB 降到 1,408 MiB**，减少 **28 MiB（1.95%）**。初始三项版本的 ABBA 吞吐为 **178.876 vs 178.851 环境步/秒**，差 -0.014%。最终版本单次复测为 186.282 环境步/秒，不能以一次后续运行证明稳定提速。

组件行为合同通过：延迟/RNG/持有输出、原生相机逐像素深度、池化、局部 reset、动态障碍、内参改变、Warp 环境延迟/扰动/终止均按所测场景保持一致。**完整训练不是逐位相同，也尚不能签署“训练级数值等价”结论**：撤回 clone 清理后，最终模型跨实现相对 L2 差约 **0.0588%–0.0593%**，仍高于本轮基线自身重复差约 **0.01185%**；最大单参数差约 **0.00270**。因此本分支是已测量的显存优化 demo，不作为可以直接合入的严格训练等价证明。

### 工作位置

- 基线：`7f415bac229e2419a859146703b0e38a21d03092`
- 分支：`demo/official-components-20261031`
- 初始三项实现：`e911e964dc47b1a8d096bca952e52a07976a6696`
- 撤回 clone 清理：`165c6cfeb85205852a5385d43014d05637ac222b`
- 最终训练源码/工具提交：`cca3aec76b80772dbefc6d92387c39e9ca431679`（后续提交只补报告、证据和 CPU 汇总工具）
- 工作树：`/home/tong/tongworkspace/free/.worktrees/isaaclab-navrl/official-components-20261031`
- [55 个源码文件逐项清单](OFFICIAL_COMPONENTS_INVENTORY_20261031.md)
- [最终版本和生产代码哈希](../results/official_components_20261031/final_provenance.json)
- [含最终复测的训练汇总与 checkpoint 比较](../results/official_components_20261031/training_with_final_summary.json)
- [完整 ABBA 原始记录](../results/official_components_20261031/training_abba.json)

主工作树未切分支、未合并。本轮没有停止已有训练，也未修改依赖版本、训练默认配置、物理步长、求解器或目标函数。

## 1. 审计边界和当前基础

覆盖全部 55 个生产 Python 文件、顶层类/函数索引和项目脚本职责。逐项清单区分已经使用官方能力、可清理的重复实现、必须保留的任务/学习方法、需要另行验证的官方替换；不把每一个自定义文件都视为需要删除的重复轮子。

实际环境：

| 项目 | 本机版本/情况 |
|---|---|
| Python | 3.12.14 |
| Isaac Lab | 3.0.0 |
| Isaac Sim | 6.2.0.0 |
| Newton | 1.6.1 |
| Warp | 1.17.0 |
| Torch | 2.11.0+cu130 |
| MuJoCo / MuJoCo Warp | 3.12.0 / 3.12.0 |
| RSL-RL | 5.5.1 |
| GPU | RTX 4090，24,564 MiB，driver 595.84 |
| 测量背景 | 同卡已有 5 个训练进程，开始时约 19,870 MiB、99–100% 利用率；保持运行 |

`isaaclab-rl`、`isaaclab-experimental` 未以独立 distribution 暴露版本号，不表示对应模块缺失。Warp latest 文档已到 1.20 dev，本轮只使用在本地 1.17 实际核对、运行的能力。

以下能力在基线中就已有，不能算本轮成果：

- 官方物理 presets、环境工厂、DirectRLEnv/DirectRLEnvWarp。
- 官方 EventManager/EventTermCfg 及质量/惯量随机化接口。
- 官方 DelayBuffer 和原生 RSL-RL PPO/BetaDistribution。
- 官方 TerrainGenerator/Importer 与 heightfield 到 mesh 的转换。
- Newton+Warp 默认路径、共享动态障碍运动、Newton 中不再为每个动态障碍创建刚体。
- Warp mesh/BVH、Torch/Warp 零复制接口、现有 controller/task CUDA graphs。

Isaac Lab 3.0 本来就允许后端与前端的独立组合，并提供官方扩展点；把现有 Direct task 改成 ManagerBased task 不是自动提速的保证。[官方迁移文档](https://docs.nvidia.com/isaaclab/3.0/isaac-lab/migrating-to-3-0)、[Warp environment 文档](https://docs.nvidia.com/isaaclab/3.0/experimental-features/warp-environments)。

## 2. 本轮实际替换和优化

### 2.1 官方 pinhole pattern + 共享只读射线

位置：`src/isaaclab_navrl/tasks/navigation/sensor_mixins.py`。

删除了重复维护的像素网格、坐标轴符号和 pinhole 计算，调用配置中的官方 `pattern_cfg.func`，保留官方半像素中心、相机坐标变换和归一化。

对于所有相机内参相同的情况，仍以原有完整分块形状计算射线，逐值确认每行完全相同，然后只 clone 一行、用 `expand_as` 形成视图。传给 `wp.from_torch` 时不再调用 `contiguous()`，避免把共享视图重新物化成 N 份。缓存持有拥有存储的 Torch tensor，Warp 读取它的零 stride 行；不会持有已释放的临时张量。

- 固定单个 64 相机分块、每相机 19,200 条射线，starts+directions storage：**29,491,200 → 460,800 bytes**，即 **28.125 → 0.439 MiB**，减少 98.4375%。
- 这是该射线缓存的降幅，不是总训练显存降低 98%。
- 32 环境训练的实际分块是 32，因此缓存节省不能直接套用 64 相机数字。
- 同一模型里不同内参仍走逐分块原生 pattern；没有错误共享。该路径会重新创建原生网格并计算 inverse，其性能尚未测量，可能比旧的缓存常量路径慢；本轮端到端性能结论只适用于固定共享内参的 fS1 配置。
- 对照真正的旧/新 `_compute_local_rays` 方法，在 4、32、65 环境及 3/64 分块、尾批情况下，数值全部逐位相同。[证据](../results/official_components_20261031/ray_baseline_diagnostic.json)

[Warp 官方 interop](https://nvidia.github.io/warp/latest/user_guide/interoperability/pytorch.html)解释了无复制 array/tensor 转换；本轮另外在已安装 1.17 上验证了零 stride `(0, 12)` 和真实相机 kernel。

### 2.2 Newton 共享网格位姿表只保留一行

位置：`src/isaaclab_navrl/tasks/navigation/newton_sensors.py`。

在确认目标属于全局共享 mesh、每目标只有一份 mesh、没有物理 sites 后，把 mesh IDs、position、orientation 表从 N 环境行改成拥有独立存储的单行，更新 launch 从 `(N, M)` 改成 `(1, M)`。既有世界包围盒和 raycast 代码本来就读取第 0 行，其他行属于无用复制。

使用 `wp.clone(array[:1])` 真正释放旧 N 行 allocation，而不是保留旧大 allocation 的切片。此优化只适用于当前共享全局障碍场景，不推广到每个 world 拥有不同障碍位姿的场景。

原生相机对照、动态目标移动和 partial reset 验证已通过。该变化消除了重复更新，但对本轮完整训练没有可单独归因的提速证据。

### 2.3 已撤回：去掉官方 DelayBuffer 返回值上的重复 clone

位置：`src/isaaclab_navrl/tasks/navigation/randomization.py`。

官方 `DelayBuffer.compute` 的返回值具有独立存储，后续 buffer 写入不会修改已有返回值；已安装源码用 advanced indexing 返回结果。因此去掉其后多余的 `.clone()`。保留 `sample().clone()`，因为采样回调可能返回后续会复用/修改的缓冲区。

混合延迟、16 次推进、两次局部 reset、重复读取同 tick、输入被修改、历史返回值保留、RNG 状态全部验证。公式、延迟索引、观测年龄和推进时机未改。[DelayBuffer 官方契约](https://isaac-sim.github.io/IsaacLab/v3.0.0/source/api/lab/isaaclab.utils.html)。

256 环境、19,405 观测值的完整 observe 微基准中，两版增量峰值均为 **39,748,608 bytes**，ABBA 耗时交错，**没有证明它单独降低峰值显存或稳定提速**。随后单项训练消融出现首轮 loss/梯度微小偏移；因此在 `165c6cf` 撤回此改动，最终分支的 `randomization.py` 与基线完全相同。独立存储的 API 契约虽然成立，也不足以保证后续 GPU 计算在改变分配后逐位相同；本轮没有把其内部数值原因当成已查明事实。

`DifferentiableDelay` 保留：它服务 BPTT 图、跨步梯度和逐环境 reset；不能以官方普通可变环形缓存直接替换。

## 3. 行为验证

| 验证 | 规模和覆盖 | 结果 |
|---|---|---|
| 定向 pytest | test_sim_to_real、test_depth_camera_contract、test_depth_history | **12 passed** |
| 新旧 ray 方法 | 4、32、65 环境；完整批/尾批 | starts/directions 最大绝对误差 **0** |
| 共享几何相机 | Newton、4 环境、8 动态障碍，初始与移动后 | 深度最大误差 **0**，动态障碍物理刚体数 **0** |
| Newton 原生相机对照 | 4 环境、60 动态障碍、3+1 分块，8 个阶段 | 120×160 原始图与 12×16 池化图最大误差均 **0** |
| PhysX 原生相机对照 | 同样 4 环境、60 动态障碍、8 个阶段 | 原始图与池化图最大误差均 **0** |
| DelayBuffer 对照 | 8 环境、16 tick，混合 lag、partial reset、返回值所有权 | **逐值一致**，RNG 一致 |
| Warp 环境合同 | Zhang/DCOL，4 环境、10 个策略步 | 延迟、interval 扰动、局部 reset、timeout、UI reset、真实越界、输出持有、质量/惯量事件均通过 |
| 完整训练 | 初始 ABBA 4×4 updates，加最终 4 updates；32 环境、H150 | 全部完成，共 **96,000 环境步**；另有消融 **14,400 环境步**。非有限状态/梯度 0，执行动作误差 0 |

相机阶段包括：初始、roll/pitch/yaw、所有动态物体移动、partial reset、0.02 像素噪声后池化、部分内参更新、恢复共享内参。Newton 额外检查共享全局运动，PhysX 额外检查 tracked view 请求顺序变化。其他环境的图像/帧计数在局部重置时保持。

Warp 环境合同测试使用该测试本身的配置，训练性能比较使用 fS1/40、20 Hz；二者作用不同，不能把短环境合同测试解释为正式训练验收。

原始结果：
[组件](../results/official_components_20261031/components.json) ·
[Newton 相机](../results/official_components_20261031/camera_parity.json) ·
[PhysX 相机](../results/official_components_20261031/camera_parity_physx.json) ·
[共享几何](../results/official_components_20261031/shared_geometry_parity.json) ·
[Warp 环境](../results/official_components_20261031/warp_environment_contract.json) ·
[pytest 输出](../results/official_components_20261031/unit_tests.txt)。

### 已否决的直接替换：unproject_depth

对常量 10 m 深度，原实现使用像素中心 +0.5，官方 `unproject_depth` 默认使用整数像素坐标，直接替换的点坐标最大差 **0.036085 m**。因此本轮没有替换深度历史重投影。

这不代表官方函数永远不能用：可以设计经过验证的内参/像素中心适配，但它也只替代 unprojection 原语，不能替代当前的 ego-motion、forward splat/min-z、mask、age 和 no-grad 全流程。直接套用普通 frame stack 会改变策略观测。

## 4. 初始三项改动的 ABBA 与数值边界

全部运行使用相同 config.json（逐键比较无差异）：seed 0、fS1/40、300 静态/40 动态障碍、扰动开启、Newton+Warp、Zhang DCOL+BPTT/GRU、H150、20 Hz、完整 2 秒速度窗口、navrl_range、原损失权重/优化器/诊断计划。

每次从相同随机初始化启动，4 次更新；排除第一次初始化、capture 和梯度诊断预热，比较其后 3 次完整更新。计时使用现有训练子进程内的 CUDA synchronize 边界。ABBA 降低单向时间漂移的影响，但不能消除共享 GPU 干扰。

| 次序 | 实现 | 后三次 update 中位数/s | 后三次实际吞吐/环境步每秒 | Torch 峰值分配/MiB | Torch 峰值保留/MiB | 进程采样峰值/MiB |
|---|---|---:|---:|---:|---:|---:|
| A1 | 基线 | 27.038 | 177.127 | 182.018 | 248 | 1,436 |
| B1 | 初始三项候选 | 27.583 | 175.216 | 166.962 | 222 | 1,410 |
| B2 | 初始三项候选 | 26.146 | 182.640 | 166.962 | 222 | 1,410 |
| A2 | 基线 | 26.678 | 180.659 | 182.018 | 248 | 1,436 |

各组吞吐由总环境步数/总耗时计算，不是对瞬时 FPS 取平均。合并两次重复后基线 178.876、demo 178.851，名义差 -0.014%。不能拿此前其他工作负载、60 动态障碍报告中的吞吐来解释本轮变化。

Torch allocated 只包含 Torch 分配，不能当全部显存；Torch reserved 还包含缓存。进程显存由 nvidia-smi 每秒采样，包含更多 CUDA/Warp/框架资源，但它是采样峰值，不是毫秒级准确峰值。所有内存数字均使用 MiB；项目原始字段名称沿用 `_mb`。

### 初始候选的训练级数值一致性未通过

- 四次运行每个 update 的完成回合数均为 **70、71、69、77**。
- 最终 Torch CPU/CUDA RNG 状态相同。
- 所有运行状态/梯度有限，执行动作与目标动作合同误差为 0。
- 基线 A1 与 demo B1 的最大逐 update loss 差 **6.3896e-5**。
- 但最终模型 514,496 个参数的最大绝对差 **0.00217838**、RMSE **4.5036e-5**、相对 L2 **0.00137684**。
- 基线 A1/A2 自身重复：参数最大差 **0.00052979**、相对 L2 **0.00011850**。
- demo B1/B2 重复：参数最大差 **1.2620e-5**、相对 L2 **2.4488e-6**。

跨实现差异大于这两组自身重复，不能全归因于背景负载，也不能用 loss 很接近就掩盖参数漂移。相机原生逐像素对照和实际旧/新射线方法都为 0；已排除所测固定内参射线计算本身的差异，尚未定位全部训练数值差异。

额外按文件做单次 update 消融，记录在 [ablations.json](../results/official_components_20261031/ablations.json)。这是数值定位，不是新增的性能基准。首轮结果如下：

| 版本 | 首轮 loss | 首轮 grad_norm | Torch 峰值/MiB | 决定 |
|---|---:|---:|---:|---|
| 原基线 | 6.8198628426 | 4.7783155441 | 182.0181 | 比较锚点 |
| 只删除 DelayBuffer clone | 6.8198657036 | 4.7783231735 | 182.0181 | 无峰值收益且出现偏移，撤回 |
| 只改原生射线/共享缓存 | 6.8198628426 | 4.7783155441 | 166.3613 | 保留，继续组合验证 |
| 只压缩共享 mesh 表 | 6.8198628426 | 4.7783155441 | 182.0181 | 保留，继续组合验证 |

这些结果支持撤回 clone 清理，但单次 update 不能替代完整学习轨迹与收敛等价。最终两项组合另外重跑 4 次更新，没有沿用初始候选的数值结论。

**初始三项版本未满足严格训练等价要求，已根据消融收紧实现。** 没有完成五阶段 80%/1024 自然回合正式验收，也没有训练收敛证明。

### 最终两项版本独立复测

最终版本再次完成 4 updates，配置逐键比较与基线完全一致。后 3 次 update 分别为 24.8300、25.4215、27.0506 秒；中位数 25.4215 秒，吞吐 186.282 环境步/秒。Torch allocated 峰值 **166.361 MiB**、reserved 峰值 **220 MiB**、进程采样峰值 **1,408 MiB**。仅射线一项的消融也得到同样的 Torch allocated 峰值，支持主要收益来自射线存储。

最终版每 update 完成回合仍为 70、71、69、77，CPU/CUDA RNG 终态一致。相对 A1/A2 基线的最大 loss 差分别为 **2.4796e-5 / 2.5272e-5**；最终模型相对 L2 差 **0.00058845 / 0.00059340**，最大单参数差 **0.0026990 / 0.0027026**。相对 L2 偏移比初始三项版本小，但最大单参数偏移并未改善，且仍超出本轮基线自身重复差。因此没有把初始数值偏移简单宣布为已解决。

[最终原始运行](../results/official_components_20261031/final_training_validation.json)与[完整比较](../results/official_components_20261031/training_with_final_summary.json)保留了全部值。后续应先完成 P0 的固定输入/单步梯度定位；当前证据足以确认存储收益和覆盖到的任务合同，尚不足以保证长期学习曲线不变。

## 5. 更大幅度的后续方案

### P0：先把数值差异定位到输入、前向或反向

使用固定的状态、动作、深度和观测样本，分别比较原生/候选中间量，再在相同参数与优化器状态下执行单次 forward/backward/update。对同一实现重复测量，明确分清结构性偏移和浮点/非确定性波动。

合入门槛应事先指定：离散事件和 RNG 序列精确相同；观测/动力学按既有合同阈值；loss/梯度/参数步长在约定数值容差内；随后再做长跑成功率与学习曲线对照。本轮没有事后选择宽松阈值把结果写成通过。

### P1：独立 sensor-only Newton Model + 官方 SensorTiledCamera

这是最值得继续尝试的结构性替换：将相机可见的静态三角网格和全局动态障碍放入**独立、只服务传感器的 Newton Model**，不向它运行动力学求解器；相机作为多个视图。复用官方 Model/BVH/refit 和 SensorTiledCamera，争取删除自维护 TLAS/ray traversal 与 legacy sensor 继承层。

理由和边界：

1. Newton 1.6.1 的 SensorTiledCamera 接受 Model，update 接受 State、相机变换、rays 和 output，且支持 forward_depth；移动几何前需要 refit BVH。因此 sensor-only model 在 API 结构上可行，**本轮尚未实现或证明性能**。[1.6.1 API](https://newton-physics.github.io/newton/1.6.1/api/_generated/newton.sensors.SensorTiledCamera.html)
2. 不能直接换 Isaac Lab 新 Newton RayCaster：它查询 live physics BVH，忽略 mesh_prim_paths。本项目动态障碍已经不是 Newton 物理刚体，NavRL LiDAR 又只看静态地图，直接切换会改变可见集合。[官方 Ray Caster 说明](https://docs.nvidia.com/isaaclab/3.0/concepts/sensors/ray-caster)
3. 新 sensor-only model 必须使用当前相同三角形与变换，不把三角网格圆柱悄悄换成解析圆柱；正确处理 VISIBLE/world、无命中值、24 m clipping、forward depth、像素中心、相机轴、背面剔除和 fast-math。
4. 先对同一相机位姿与场景做全像素对照，再测 32/256/1024 环境、batch 16/32/64。只分配 depth，关闭未用 color/normal/texture/shadow 等输出。额外 model/state/BVH 本身也占显存，必须测净收益。
5. 继续执行当前 partial reset、内参改变、噪声/池化、深度延迟/重投影和真实训练 gate，不能只用静态平面的平均深度验证。

### P2：有条件换官方 Thruster，而不是整段换控制方法

官方 Thruster 已提供上下行时间常数、推力限幅、Euler/RK4 和离散响应。现有仓库已有 `tests/test_thruster_parity.py`，包含 rad/s→rps、0.43 每步响应映射和混控检验，可作为起点；本轮未重新执行该历史测试，不能把它算新验证。

建议先评估 Torch 参考路径的可维护性，逐项覆盖初值、饱和、重置、扰动、响应时常、yaw moment、body wrench，再测真实闭环。当前 Warp 路径已融合 controller+motor 并 capture；官方 Torch Thruster 可能增加 kernel 数与缓冲区，不能未经测量替掉更快的融合路径。[官方 Thruster](https://isaac-sim.github.io/IsaacLab/v3.0.0/source/api/lab_contrib/isaaclab_contrib.actuators.html)

### P3：用官方管理接口收敛配置，保留方法语义

后端 preset、EventManager、DelayBuffer、RSL PPO、TerrainGenerator 继续作为标准边界。可以把重复的配置/事件注册整理到这些扩展点，考虑官方地形 cache 降低启动开销。不要为“管理统一”同时改 curriculum 规则、reset 顺序、noise/delay 次序或 checkpoint 协议。

以下不作为同语义替换：

| 拟议变化 | 会改变的内容 |
|---|---|
| Zhang runner → RSL PPO runner | BPTT/SHAC/AHAC 目标和梯度 |
| surrogate dynamics → Newton/Warp 完整物理反传 | 反向模型及控制梯度 |
| 自定义 DCOL/heightfield sign → 通用 contact/SDF | 距离定义、内部点方向与导数 |
| 历史深度补偿 → 普通 frame stack | ego-motion 对齐、未知区域、观测年龄 |
| 缩小深度分辨率、降低传感器更新率 | 观测采样与时间语义 |
| 修改 dt、decimation、接触近似、solver iterations | 动力学与训练任务 |
| 关闭诊断、缩短 H150、缩小网络来刷速度 | 测量工作负载或学习方法 |

## 6. 复现

在新输出目录复测最终两项版本，保留原结果。初始三项版本位于 e911e96，不能把当前 HEAD 的复测当成初始三项版本的复现。示例使用此 demo 工作树及已经建立的运行环境；EULA 环境配置沿用当前已有训练的设置。

```bash
cd /home/tong/tongworkspace/free/.worktrees/isaaclab-navrl/official-components-20261031
NAVRL_PYTHON=/home/tong/tongworkspace/free/isaaclab-navrl/.venv/bin/python
export PYTHONPATH=src
export OMNI_KIT_ACCEPT_EULA=YES

# 新目录名应未存在。
mkdir -p logs/official_components_recheck/baseline
git archive 7f415bac229e2419a859146703b0e38a21d03092 | tar -x -C logs/official_components_recheck/baseline
export WARP_CACHE_PATH="$PWD/logs/official_components_recheck/warp_cache"

"$NAVRL_PYTHON" scripts/benchmark_official_components.py --mode components \
  --baseline logs/official_components_recheck/baseline \
  --num-envs 256 --output logs/official_components_recheck/components.json

"$NAVRL_PYTHON" scripts/benchmark_official_components.py --mode training \
  --baseline logs/official_components_recheck/baseline \
  --num-envs 32 --updates 4 --warmup 1 \
  --output logs/official_components_recheck/training_abba.json

"$NAVRL_PYTHON" scripts/summarize_official_components.py \
  logs/official_components_recheck/training_abba.json \
  --output logs/official_components_recheck/training_summary.json

"$NAVRL_PYTHON" tests/official_component_ray_parity.py \
  --baseline logs/official_components_recheck/baseline \
  --output logs/official_components_recheck/ray_baseline_diagnostic.json
```

真实相机/环境命令见 [verification_commands.json](../results/official_components_20261031/verification_commands.json)。本轮大 checkpoint、完整 console log、冻结基线和编译 cache 保留在工作树 `logs/official_components_20261031/`；审计所需 JSON 与定向测试输出归档在 `docs/results/official_components_20261031/`。生产文件 SHA256 可与 provenance.json 核对。
