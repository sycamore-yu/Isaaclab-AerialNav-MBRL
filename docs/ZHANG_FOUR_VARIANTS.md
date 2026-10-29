# Zhang 四方案比较

本轮按用户追加要求比较速度损失、碰撞重置与碰撞几何的四种处理方式。正式训练只运行 Zhang。PPO 停止；NavRL 仍接收共用的 160×120 相机默认配置和此前的共享扰动实现。

改动前的窗口基线为 `242b494`，标签 `baseline/zhang-window-loss-d160-20261029`。历史日志完整保留。基线未通过验收，不作为新方案的预训练权重。

## 共同条件

四组均从随机初始化开始，首轮固定 seed=0、1024 环境、100 Hz PhysX、10 Hz 策略、H=150 策略步、160×120 原始深度和 12×16 策略图像。CNN+GRU192、六维输出解码、Lee/RotorGroup 控制链、状态对齐与同一执行动作保持不变。AdamW 预训练学习率 0.001、微调 0.00005；余弦周期 400 次更新、梯度裁剪 5，其他原损失系数保持一致。

预训练 300/60→400/120，微调 300/60→400/100。预训练关闭额外扰动，微调开启。每阶段累计至少 1024 个自然结束回合后，训练首次到达率达到 80% 才晋级。每组只加载自身前一阶段的检查点。训练无更新次数上限。

相同种子不会保证四组遇到完全相同的后续轨迹：动作和碰撞重置时间不同，会改变随机数消耗和回合分布。这是算法行为造成的差别。单种子结果只用于首轮筛选，不作为跨种子稳定性结论。

## 四种配置

| 用户方案 | `--variant` | 速度损失 | 碰撞响应与重置 | 碰撞损失 |
|---|---|---|---|---|
| 1，默认 | `step` | 每个策略步直接匹配目标速度，H 步均值 | 保留无人机接触；逻辑碰撞后重置 | 原 Zhang 几何项 |
| 2 | `continuous` | 原窗口均值，默认 W=20 | 禁用无人机 PhysX 接触；碰撞不中断展开 | 原 Zhang 几何项 |
| 3 | `dcol` | 原窗口均值，默认 W=20 | 保留无人机接触；逻辑碰撞后重置 | 动态凸体 DCOL + 原静态网格距离 |
| 4 | `truncated_window` | 正常完整窗口；碰撞终止时补当前短窗口 | 保留无人机接触；逻辑碰撞后重置 | 原 Zhang 几何项 |

所有方案都在收集 150 个策略步后统一反向传播和更新一次网络。中途重置只切断下一回合的历史，已保存的终止步图继续参与本次反传。方案 2 仍保留高度越界、非有限状态和原 35.2 秒超时重置，不在每次更新开始时强制重置全部环境。碰撞后再到达目标仍按首次碰撞计失败。

原任务的动态障碍一直禁用 PhysX 接触，以逻辑碰撞规则判定失败。表中的“保留接触”指无人机保留原资产接触属性，不能解读为给动态物体新增了接触响应。

## 速度项

方案 1 使用用户指定的逐步目标：

\[
L_v=\frac1{HN}\sum_{t=1}^{H}\sum_{i=1}^{N}
\operatorname{SmoothL1}(\|v_{t,i}-v^{target}_{t,i}\|,0).
\]

`v_t` 是本次物理转移后的对齐速度，`v_target_t` 是执行该转移前生成的目标速度。首步、末步和终止步均参与，不因回合短于 W 而丢失速度训练信号。APG 提供的是逐步目标聚合与固定展开后反传的结构；这个 SmoothL1 速度目标是本项目按用户要求选择的。

窗口方案保留原前向窗口定义和目标索引。W 默认 20 策略步，`--velocity_window_s 6.0` 可设为原 60 步。方案 4 的补项只作用于 `done & collision`，用当前回合实际样本数作分母，以该窗口起点的目标速度匹配平均速度；不把重置后样本混进窗口。首回合恰好 W 步而被原窗口索引省略的碰撞也补计，已经计分的完整窗口不重复计分。完整项与短项先求和再除以总有效样本数。

## 边界高度项

物理终止边界保持 z∈[0.2,4.0] m。设 margin=0.5 m、scale=0.5 m：

\[
d_-=\max(0,0.7-z),\quad d_+=\max(0,z-3.5),
\qquad L_z=\mathbb E[1-e^{-(d_-/0.5)^2}+1-e^{-(d_+/0.5)^2}].
\]

它在 [0.7,3.5] m 内为零；边界激活处连续可微，梯度方向朝向高度范围内部。损失在远离范围时饱和，因此仍保留越界重置。它使用对齐后的本步实际位置，包含终止步。默认系数 `coef_height_boundary=1.0`；设为 0 可独立关闭。`boundary_margin` 与 `boundary_scale` 分别控制激活区域和增长尺度。

## DCOL 适配范围

DCOL 求两个凸体围绕各自原点共同缩放至刚好相交所需的最小比例 α。α>1 表示分离，α=1 表示接触，α<1 表示相交。球对轴对齐盒的方程为：

\[
\|\max(|p-c|-\alpha e,0)\|=\alpha r.
\]

盒半尺寸为 e，无人机球半径为 r。有限 Z 轴圆柱使用径向与垂直的相同方程。实现按活动集合解析求根并用隐式一阶导数反传，所有动态凸体都参与最小 α 选择；仅对胜出物体保存图，几何参数使用不可变快照。

动态碰撞代理为 `r*(α-1)`。它具有米制尺度，但不是精确有符号距离。该项与实际静态网格原有的表面距离减半径、地面高度差减半径取最小，再进入原 `softplus(-32*gap)` 碰撞项。静态分支只查询实际 terrain mesh，不混入旧动态距离。

静态高度场是非凸体。本轮不把每个小网格柱冒充完整障碍：这会制造内部表面，使穿入后的梯度朝相邻内部单元移动。方案 3 明确标记为 `hybrid_dynamic_dcol`；静态仍使用原 unsigned mesh clearance，未声称解决静态深度穿透梯度。动态盒和圆柱才使用完整的凸体 DCOL。该方法也不提供 PhysX 接触力、接触时刻或离散重置的导数。

每步保留原 lookahead 第 1–8 个位置。最后一个碰撞样本使用本策略步首次检测碰撞的实际物理子步位置、该子步的动态障碍位置及同一子步的可微预测；未碰撞时使用正常策略步终态。这样不会因为策略步末已经穿过障碍而丢掉首次碰撞点，也不把重置后位置用于碰撞损失。物理重置仍可先执行，已保存的图在 H 步末反传。

## 日志与比较

各组保存独立的 `config.json`、`updates.jsonl`、`training_statistics.jsonl`、TensorBoard 和阶段检查点。方案 3 额外保存 `dcol_geometry.json`，说明几何范围。训练器及阶段加载都核对 `variant`，防止跨方案接续。

主要比较首次达到每阶段 80% 的更新次数、环境策略步数和训练循环累计耗时；同时检查最新/最佳首次到达率、碰撞/越界/超时、成功时间、return、非有限梯度、动作一致性误差、近期报告波动与更新耗时。缺失值保持缺失，不记成零或改善。

```bash
.venv/bin/python scripts/run_acceptance.py \
  --method zhang --num_envs 1024 --train_seed 0 \
  --output logs/curriculum/zhang_step_seed0 \
  --train_args --variant step --timesteps 150 --velocity_window_s 2.0

.venv/bin/python scripts/summarize_zhang_comparison.py \
  RUN_STEP RUN_CONTINUOUS RUN_DCOL RUN_TRUNCATED \
  --output logs/zhang_four_variants_comparison
```

其他组只替换 `variant` 和全新输出目录。比较工具不会修改原训练文件。近期最多五份报告的均值/标准差只是描述波动，不改变晋级统计。共享 GPU 下的墙钟时间反映实际并发调度；训练循环计时不含场景初始化，不能当作独占 GPU 的纯算法速度基准。

## 来源

- [Brax APG](https://github.com/google/brax/blob/main/brax/training/agents/apg/train.py)：固定展开、逐步目标聚合。
- [DCOL 论文](https://rexlab.ri.cmu.edu/papers/dcol.pdf) 与 [作者实现](https://github.com/kevin-tracy/DifferentiableCollisions.jl)：凸体共同缩放及隐式微分。
- [Isaac Lab 3.0 迁移规范](https://isaac-sim.github.io/IsaacLab/main/source/isaaclab/migrating.html)：使用现有 DirectRLEnv、事件和传感器接口。
- 本项目的 [Zhang 迁移决策](migration/ZHANG_MIGRATION_DECISIONS_1_4.md) 与 [四阶段验收](TRAINING_CURRICULUM.md)。
