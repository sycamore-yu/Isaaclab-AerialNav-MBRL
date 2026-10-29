# DCOL 独立逐步速度监督：诊断与对照协议

记录日期：2026-10-29。本文记录窗口覆盖诊断及已接入的独立开关；下述两组 fresh 对照在本文编写时尚未启动，没有新训练结果，也不预言胜负。课程、自然回合首次事件统计及 80% 晋级门槛沿用现有决策。

## 已观测到的窗口覆盖问题

证据来自 `active_controls_snapshot_1040.json`（导出时间 2026-10-29 10:40:08 UTC）。原始记录位于 `logs/curriculum/zhang_new_dcol_baseline_seed0_20261029/zhang/pretrain_s0/` 和 `logs/curriculum/zhang_new_dcol_shac_seed0_20261029/zhang/pretrain_s0/` 的 `updates.jsonl`、`training_statistics.jsonl`。两组均为 seed 0、1024 环境、300 静态 / 0 动态障碍、无额外扰动。

| 同环境步预算附近的记录 | DCOL + BPTT | DCOL + SHAC |
| --- | ---: | ---: |
| 更新 / 环境步数 | U40 / 6,144,000 | U188 / 6,160,384 |
| H / W（策略步） | 150 / 20 | 32 / 20 |
| 有效速度窗口 endpoint 比例 | 90.20% | 1.071% |
| 标量速度损失 | 9.800 | 4.387 |
| 完成回合平均长度（策略步） | 185.70 | 17.51 |
| bounds 首事件率 | 33.36% | 99.30% |

SHAC U181–188 每批 32,768 个 transition 中只有 225–356 个有效窗口 endpoint，占 0.687%–1.086%；标量速度损失仍为 4.247–4.608。不能把标量仍有量级理解为所有短回合都有速度监督，也不能把 endpoint 比例直接当成获得梯度的回合比例。

原窗口仅统计同一 episode 内的完整 W 步均速。回合长度小于 W 时，该回合没有速度 tracking 项；terminal velocity 虽然保留，也不能单独组成完整窗口。缓存允许窗口跨优化更新，但 episode ID 禁止窗口跨 reset。源实现省略整段轨迹最初一个速度样本的边界行为也保持不变。

令 K 为当前批次有效 endpoint 数，原标量及逐 transition 速度成本为：

```text
L_velocity = sum(valid * window_error) / max(K, 1)
velocity_cost[t, n] = H * N / max(K, 1) * valid[t, n] * window_error[t, n]
```

因此 K 很小时，速度标量仍是少量有效窗口的均值；K=0 时速度项才为零，其他成本仍存在。U188 的 K=351，endpoint 系数为 93.36，有效 endpoint 平均速度成本约 409.55，全批均值仍是 4.387。该整批系数随窗口覆盖变化，使 critic 的逐 transition 成本尺度具有额外非平稳性。它是具体的估计压力，现有日志不足以证明它独自造成 critic 或策略退化。

非饱和高度项在 terminal 状态仍有向内梯度，但 DCOL 在 bounds 后立即 reset，不能获得连续越界后的多步恢复监督。Continuing-reset bootstrap 保留实际 reset 状态的未来价值，整体 detach 并切断 TD trace；它不会补回短回合中未定义的窗口速度误差。负 bootstrap 的标量也不等于推动 actor 的同方向梯度，需要区分 reset 与 live endpoint 部分。

## 原 SHAC window 控制的停止与归档

原 SHAC window 运行已在 **U230 / 7,536,640 环境步**完整停止并保存检查点。停止是资源分配与诊断筛选，不是算法最终排序或收敛失败结论。

- 最佳自然报告：**1.3385%，U227**。
- 最新自然报告：**0.5405%，U230**。
- 最新 `critic_target_mean`：**+301.2626**；此前负值已经恢复，不能描述为持续负 critic。
- nonfinite 计数：**0**；没有达到 80% 晋级要求。
- 冻结归档：`logs/archive/20261029_shac_window_diagnostic/frozen_run.tar.gz`。
- 归档 SHA256：`cbf5425bed83414e4c72533cbba305797b913a86e01d0cadc188cd58c0da9866`。
- 对应检查点 SHA256：`cdd03f77b126ed57393a914d5077f159ebc8cea3da79420db6afcfe06c510353`。

## 独立开关的实现合同

配置及 CLI 新增 `velocity_loss_mode` / `--velocity_loss_mode`，取值为 `variant_default` 或 `per_step`。`resolved_velocity_loss_mode` 记录实际解析出的 `window` 或 `per_step`。

- **默认不变：** `variant_default` 保留 step 的逐步速度损失及 continuous/DCOL 的原窗口数值、梯度、缓存和样本缩放行为。
- **逐步监督：** `per_step` 使用每个 transition 的 post-step terminal velocity，对比同一 transition 的 pre-step target velocity，逐步计算 `SmoothL1(norm(error))`。首末步骤、碰撞和越界 terminal 步均计入。
- **重置隔离：** terminal 状态先与真实 PhysX 测量对齐，再为下一状态截断 reset 图。per_step 不读取旧速度窗口缓存，返回缓存为 `None`；reset 后速度不能替代 terminal velocity。
- **逐步成本：** 每步速度成本直接进入 `cost[T,N]`，其均值等于标量项；有效样本为 `T*N`、比例为 1，不使用窗口的 `HN/K` 放大。SHAC 仍使用原有折扣和 bootstrap。
- **其余行为不变：** DCOL 几何、碰撞时刻、终止规则、另外六项成本与系数保持原样；critic 架构、输入维度和窗口历史保持原样，`velocity_window_s=2.0` 仍决定其原 W20 历史尺寸。
- 用户取消 `truncated_window` 的决定继续有效。本次没有恢复短碰撞窗口补项，也没有恢复该训练变体；历史日志仅用于追溯。

实现位于 `src/isaaclab_navrl/zhang/losses.py`；专门测试位于 `tests/test_zhang_losses.py`。测试覆盖短回合逐步梯度、terminal/reset 隔离、默认窗口固定数值与梯度，以及 DCOL 其他成本及梯度不变。单元测试通过不构成真实训练验收。

## 两组 fresh 对照协议

| 新计划组 | 匹配的原 window 控制 | 唯一训练因素变化 |
| --- | --- | --- |
| DCOL + BPTT，H150 | DCOL baseline + BPTT，H150 | `velocity_loss_mode: variant_default → per_step` |
| DCOL + SHAC，H32 | 原 DCOL + SHAC，H32 | `velocity_loss_mode: variant_default → per_step` |

两组均随机初始化、新输出目录、不加载旧控制检查点；从 `pretrain_s0` 开始，按既有课程在同一 run 内交接阶段。分别固定各自控制的 seed 0、1024 环境、深度相机与批大小、物理/策略频率、初始 lr=0.001、原学习率调度、grad_decay=0.4、clip=5、全部七项系数与其余配置。SHAC 保留原控制的 critic 配置，不同时切换 linear/softplus，也不改变 H 或网络。正式命令由启动记录保存；两组的区分参数为：

```text
--variant dcol --algorithm bptt --timesteps 150 --velocity_loss_mode per_step
--variant dcol --algorithm shac --timesteps 32 --velocity_loss_mode per_step
```

比较时先匹配阶段、场景与累计环境步数，再附更新数；分别与各自 window 控制比较，不能用 BPTT 与 SHAC 的相同更新数当作相同采样预算。保持无预设训练上限的正式流程，诊断采样预算仅用于比较，不代替晋级和最终验收门槛。

主要观察自然首次成功率、碰撞/上下界/超时率、成功耗时、回合长度和 return；结合有效速度样本、垂直速度、七项损失、梯度、执行动作误差及 critic 指标解释变化。per_step 纳入原先被排除的短回合，速度损失可能先升高，不能据此单独判退化。只有一个 seed，不作统计显著性或跨 seed 稳定性声明；同 GPU 并发耗时只反映本次调度，不作为独占 GPU 速度基准。
