# 高度优先梯度与完整优化器动作诊断

本机制用于区分三件事：各损失项的原始参数梯度是否冲突、真实 Adam/AdamW 更新是否降低高度项，以及固定观测下的策略命令如何变化。它不修改高度公式、障碍几何、终止条件或成功率口径。默认 `gradient_projection='none'`；诊断只在显式选择的全局 update 上执行。

## 原始梯度投影的准确含义

记七项加权总损失的梯度为 `g_total`，当前选中高度项的**未加权**梯度为 `g_h`，其系数为 `lambda_h = coef_height_boundary`。首先计算：

```text
g_other = g_total - lambda_h * g_h
```

若 `dot(g_other, g_h) < 0`，从 `g_other` 去掉反向平行分量，然后加回原高度梯度：

```text
g_other_projected = g_other - dot(g_other, g_h) / dot(g_h, g_h) * g_h
g_projected = g_other_projected + lambda_h * g_h
```

其余情况保持原总梯度；`g_h` 全零时也严格保持原值。实现先在 float64 中缩放、归一化高度方向，再计算投影，避免指数高度项极小梯度造成平方下溢或比例系数溢出。输入张量不修改，返回值为相同 dtype/device 的 detached 副本。

当系数非负时，精确算术下有 `dot(g_projected, g_h) >= lambda_h * ||g_h||²`。因此，对固定当前状态上的局部线性近似，沿 `-g_projected` 做普通梯度下降不会增加高度项。全局范数裁剪只乘正标量，也不改变这一原始方向关系；实际浮点结果会有舍入误差。

**这不保证 Adam/AdamW 更新后的高度项下降。** 旧一阶动量、各坐标的二阶矩缩放，以及 AdamW 的独立权重衰减都可改变最终参数位移。即使原始梯度投影满足上述不等式，也可能出现 `dot(g_h, delta_theta) > 0`。测试包含一个旧负动量保留的标量 AdamW 例子，明确验证这一限制。若 `target_exp` 在远场已提供几乎为零的高度梯度，投影也不会凭空产生恢复信号。

当前真实投影只允许未含 critic bootstrap 的 BPTT 目标。SHAC/AHAC 的 actor 目标包含折扣、端点价值或约束项，配置会拒绝把它们直接解释成当前七项分解。未来若支持它们，必须先定义并验证相应的完整 actor 梯度分解。

## 一次完整 Adam/AdamW 更新的反事实 probe

在指定的诊断 update，runner 选择该 rollout 最后一个策略调用前、十个 look-ahead 点中最小 signed clearance 最小的至多 16 个环境。保存输入包括：已完成当前归一化的 depth/features、**调用前的 GRU hidden**、heading 与 thrust error。样本立即 `detach().clone()`，环境后续复用 buffer 或 reset 不会改变它们。环境编号与 clearance 写入 `optimizer_action_probe.probe_selection`。

真实训练的顺序为：

1. 在现有 rollout 图上计算分项梯度；执行原始总目标的 `backward`。
2. 在修改 `.grad` 之前检查加权分项和与原始总梯度的一致性。
3. 创建 `ActionUpdateProbe`。它分别 `deepcopy((model, optimizer))` 两次，保留每份 model 参数与相应 optimizer 参数的引用关系。
4. 两份副本从同一旧状态出发，分别使用原始总梯度和高度优先梯度，运行与真实训练相同的范数裁剪及**完整的一次 Adam/AdamW step**。学习率、动量、二阶矩、step、epsilon、权重衰减和参数组均来自旧 optimizer 的副本。
5. 若真实配置启用 `height_priority`，把投影写入真实 `.grad`；随后进行原裁剪和真实 optimizer step。
6. `finish` 核对真实更新是否匹配对应的反事实分支，并记录实际参数位移与固定输入输出。

Probe 不改真实模型、optimizer、`.grad`、训练标志或 RNG，不推进环境，也不重复一个新的 rollout。它在隔离副本上执行普通前向，恢复 Torch CPU 及相关 CUDA RNG 状态，不对正式随机流重新设种子。输入与保存的梯度、参数快照均 detached；JSON 中不保存计算图或 Tensor。

输出使用现有 `decode_action`，因此是世界坐标系下的 acceleration command，已包含 heading 和 thrust-error 变换。它尚未经过动作延迟队列、控制器或 PhysX；不能称作新的已执行动作或未来轨迹。Probe 固定了 observation 和 hidden，衡量的是一次参数更新引起的策略输出变化。

## 日志字段与判读

`gradient_diagnostics.jsonl` 保留原 Gram、分项范数和实际位移记录，并增加 `applied_gradient_projection` 与 `optimizer_action_probe`。

| 字段 | 含义 |
| --- | --- |
| `total_gradient_check`、分项 Gram | 投影前的原始目标梯度；不是投影后的七项重新加权 |
| 顶层 `pre_clip_grad_norm` | 真实裁剪前的范数；启用投影时来自投影后的梯度 |
| `optimizer_action_probe.projection` | 原始投影的冲突、范数、方向与修正量 |
| `branches.original` / `branches.height_priority` | 各自完整 optimizer step 的 clip scale、参数位移和固定输入命令变化 |
| `height_gradient_dot_parameter_delta` | 当前 rollout 上高度项的一阶预测变化；负值表示局部改善，正值表示局部恶化 |
| `command_before_xyz_mean` / `command_after_xyz_mean` | 固定样本更新前后命令的三个轴均值 |
| `command_delta.xy_rms` | `sqrt(mean(delta_x² + delta_y²))`，表示水平向量变化的 RMS |
| `command_delta.z_rms` | 竖直分量变化的 RMS |
| `height_priority_minus_original` | 两种完整更新的参数与命令差异 |
| `actual_step` | 真实参数与命令是否匹配对应分支，以及真实位移的高度投影 |
| `probe_selection` | 所选环境的编号、排序准则与 pre-step 最小 clearance |

先检查 `actual_step` 匹配误差和原始梯度闭合，再解释方向差。若原始投影改善但 Adam 后的 `g_h·delta_theta` 仍为正，证据指向优化器变换与其他坐标的相互作用；不能据原始 Gram 宣称高度已受保护。若两分支的命令 z 改变不同，也只能说明固定输入的策略输出变化，不能直接推断成功率或碰撞率。

## 对照约束与验证

默认关闭真实投影；通过 `gradient_diagnostic_updates` 选择有限次数的诊断，编号为累计全局 update。比较原始与高度优先训练时，应从随机初始化分别启动，保持相同 seed、课程、环境数量、展开长度、损失/几何配置、optimizer 和学习率，仅改变 `gradient_projection`。若还要比较 Adam 与 AdamW 或权重衰减，另设明确的独立配置，不能把多个同时变化的因素归因于投影。

`tests/test_zhang_gradient_projection.py` 验证投影性质、极小梯度、零梯度、动量反例、分项梯度副本独立性，以及 CPU/CUDA 上 Adam/AdamW 的非空状态、裁剪、真实状态/RNG 不变和完整更新匹配。`tests/test_zhang_projection_runner.py` 使用原 `AnalyticEnv` 运行两次真实 trainer 更新，核对诊断开/关的模型、optimizer、normalizer、RNG 一致性，全局诊断编号、调用前 hidden、最近样本元数据及真实执行顺序。

这些验证建立实现与记录的可信度，不证明新机制必然提升任务收敛。正式验收仍使用训练自然回合的首次到达率和相同阶段阈值。
