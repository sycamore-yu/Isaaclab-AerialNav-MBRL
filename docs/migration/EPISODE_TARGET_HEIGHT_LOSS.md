# Zhang 每回合目标高度损失

2026-10-29 的新实验按用户要求，把默认高度项改为追踪每回合重新采样的目标高度。本变更只影响 Zhang；NavRL PPO 仍使用其原奖励和训练算法。此文件说明损失接口与数学性质，不代表训练已经达到验收阈值。

## 三种明确的模式

令误差 `e = z_after_step - z_target_before_step`，位置与目标都以米为单位。

| `height_loss_mode` | 每个转移上的高度代价 | 用途 |
| --- | --- | --- |
| `target_exp`（新默认） | `-expm1(-abs(e))`，即 `1 - exp(-abs(e))` | 用户指定的精确公式，指数尺度为 1 m |
| `target_huber` | `0.5 * e²`（`abs(e) < 1`），否则 `abs(e) - 0.5` | 同一目标高度的 SmoothL1 对照，转折尺度为 1 m |
| `boundary_huber` | 原上下边界安全带的 SmoothL1 | 保留旧模式供显式对照 |

`target_exp` 不叠加边界项、线性远场项或其他惩罚。在非零误差处，其高度导数为 `sign(e) * exp(-abs(e))`；代价趋于 1，恢复梯度随距离衰减。这是该公式本身的性质。它在目标处有尖点，不是 C1 函数；PyTorch 的 `abs` 在零误差处选取零次梯度。使用 `expm1` 可避免极小误差处直接计算 `1-exp(...)` 的相消。

`target_huber` 的 1 m 转折尺度独立于旧 `boundary_scale`。旧边界模式仍使用 `height_bounds`、`boundary_margin` 和 `boundary_scale`：先求上下安全带的违反量，再除以 `boundary_scale`，最后计算原 SmoothL1。新目标模式不读取这些边界参数。

## 目标、重置与梯度契约

目标高度采用本回合导航目标的 z 分量，默认采样区间仍为 `[0.5, 2.5]` m，由环境的 `height_target_range` 控制。目标在该回合内固定，在 reset 时重新采样。既有策略目标速度已含目标 z 的信息，不新增策略输入维度。

环境提供 `observations['height_target']`。Runner 必须在执行动作前将它 `detach().clone()`，写入该转移的 `history['height_target']`；堆叠后形状必须为 `[T, N]`。高度位置采用对齐后的实际 post-step 状态，形状为 `[T, N, 3]`。发生 autoreset 时，终止位置仍与旧回合目标配对，不能改用 reset 后的位置或新目标。

`height_target_loss` 也会 detach 目标，以保证采样目标不是可优化变量。它保留位置至动作的既有可微动力学路径。高度项在每一个转移上计算，包括终止转移；不会因为速度窗口无有效样本而被屏蔽，也不会应用速度窗口的 `T*N/valid_count` 缩放。三种速度采样模式和 BPTT/SHAC/AHAC 都接收相同的逐转移高度代价。

目标模式缺失 `height_target`、目标形状不匹配或模式未知时直接报错；即使高度系数为零也不默默改用另一个公式。`boundary_huber` 不要求目标字段。

## 日志及几何输入

为保持既有 TensorBoard、分项梯度诊断和训练曲线的 key 稳定，标量 `height_boundary` 与权重 `coef_height_boundary` 沿用原名。它们现在代表**选中的高度项**，不能仅凭这个历史名称认定使用了边界模式。配置与结果必须同时记录真实 `height_loss_mode`。

同批修复要求所有 Zhang variant 的障碍项接收 `history['obstacle_signed_distance']`，形状为 `[T, N, 10]`，表示尚未减无人机 margin 的十个原始 look-ahead 查询距离。几何层合并静态有符号距离、地面有符号距离与动态体距离；损失层再减 margin。损失不再从 `obstacle_vector.norm()` 重建无符号距离，也不再使用 continuous 专有的 `ground_signed_distance` 替换分支。缺字段或形状错误直接失败。

DCOL variant 的 collision 项仍使用单独的 `dcol_gap`，因此其几何生产端也必须提供一致的静态/地面穿透语义；上述新字段同时用于各 variant 的 avoidance 和 approach。此处不改变 DCOL 的动态凸体定义，也不宣称获得 PhysX 接触动力学导数。

当前静态实现标识为 `signed_heightfield_occupancy_exit_proxy`。它在保守的最高投影占据内部使用到原三角网格合法出口的负距，在占据外保留原无符号距离；折叠和自交地形不能据此称为精确或全局连续的 SDF。障碍侧面的出口验证、2 m 有界局部搜索和原网格屋顶回退的范围见[静态几何审计](SIGNED_STATIC_GEOMETRY_AUDIT_20261029.md)。

## 针对性验证

新增 `tests/test_zhang_height_loss.py` 检查：精确指数值及梯度（含极小误差、零误差、远场饱和）、1 m Huber 值及梯度、目标 detach、终止状态保留旧目标和动作梯度、旧边界模式的数值、三种速度模式下逐步高度代价、其他六项及其梯度保持不变，以及三 variant 对 signed 距离的严格消费。原有精确边界回归测试应显式选择 `boundary_huber`。

真实主环境已完成S0/DCOL、S0/continuous及60动态扰动三项集成检查：目标高度在reset时重采样、非reset lane不变、终止状态保留旧目标；原始深度为160×120，扰动检查中PhysX与可微路径的执行动作最大误差为0。具体回执和Torch回归见[验证索引](../results/zhang_restart_validation_20261029.md)。这些接口检查不证明公式更快收敛；是否更稳定以及是否达到80%首次到达率，以新随机初始化实验的自然报告为准。
