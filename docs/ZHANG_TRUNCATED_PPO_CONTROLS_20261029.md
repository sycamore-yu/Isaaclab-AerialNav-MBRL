# DCOL 速度监督与 NavRL/PPO 方法对照

日期：2026-10-29。本文最初记录重新检验短回合窗口速度监督及fresh NavRL/PPO的协议，编写时两组尚未启动。后续源码775dc6已冻结，PPO launcher于11:40 UTC启动；最新用户又要求替换当前实验并全部fresh重跑，主代理已请求六个训练子进程在完整更新后停止。本文当前接口按这次重启更新，历史运行的参数与结果不回写。新轮次启动和验收必须另有真实回执，功能接入与合成测试不能代替。

## 1. 旧 variant 与新速度模式的边界

旧 `variant=truncated_window` 已取消，其公式、几何条件和实验结果保留在 [四方案历史记录](ZHANG_FOUR_VARIANTS.md) 与 [旧实验归档](results/zhang_pre_new_mechanisms_archive_20261029.md)。旧50更新中的单报告最好值不能作为新组结果，也不证明短窗方法已被充分否定。

最新恢复的是 **`variant=dcol` 下独立的 `velocity_loss_mode=truncated_window`**。几何 variant 仍只有 `step / continuous / dcol`；不重新注册旧同名 variant。比较速度模式时必须固定同一版DCOL终止、首次碰撞位置及几何计算。最新重启另有静态signed距离与高度项变化，因此不能把新旧代码的整体差异归因于短窗一个因素。

配置来源：[config.py](../src/isaaclab_navrl/zhang/config.py)、[losses.py](../src/isaaclab_navrl/zhang/losses.py)、[runner.py](../src/isaaclab_navrl/zhang/runner.py)。

## 2. 三种速度监督的公平因素

| 对照名称 | 显式参数 | DCOL 下 resolved mode | 速度监督 |
| --- | --- | --- | --- |
| 原 window 控制 | `velocity_loss_mode=full_window` | `window` | 保留原完整窗口及源实现的初始样本偏移 |
| per-step | `velocity_loss_mode=per_step` | `per_step` | 每个 transition 的 post-step 速度对比同一步 pre-step 目标速度 |
| 新 truncated | `velocity_loss_mode=truncated_window` | `truncated_window` | 原完整窗口加符合条件、尚未被完整窗口计分的终止短窗 |

直接比较时，三组固定 `variant=dcol`、相同算法、seed、1024 环境、阶段与地图采样规则。保持100 Hz PhysX、10 Hz策略、160×120原始深度、1×12×16策略深度、CNN+GRU192、动作解码、控制链、随机化、全部七项系数及学习率设置一致。正式相机批次可显式固定64，配置默认仍为16。

BPTT 控制固定 H150；若另做 SHAC/AHAC 对照，必须分别固定其匹配控制的 critic、value transform、bootstrap 与 H 规则。不能同时改变速度模式、critic head、展开长度或高度系数后仍称单因素。`per_step` 不需要速度损失缓存，但 critic 的历史输入维度保持原样，不能借此同步简化 critic。

最新默认仍以`velocity_loss_mode=variant_default`表示，但映射已按用户重启要求改变：step为`per_step`，continuous/dcol为`truncated_window`。显式`full_window`解析为旧`window`；`window`本身仍不是CLI取值。此前775dc6及更早实验的默认完整窗口语义保持历史记录，不能用新映射重解释旧日志。合法BPTT/SHAC/AHAC均可与DCOL及显式速度模式组合。

## 3. 短窗、共同分母与终止图

### 真实长度与可补样本

默认 `velocity_window_s=2.0`，策略步长0.1 s，因此 W=20个策略步。短窗使用真实的 `episode_step`（本回合已经执行的策略转移数），不能使用剩余缓存的可见后缀长度。

每个候选 terminal endpoint 必须同时满足：

```text
done & (collision | bounds | timeout) & ~invalid
1 <= episode_step <= W
该 endpoint 尚未被原完整窗口计分
缓存与当前 rollout 可见样本覆盖该回合全部真实步数
```

这里采用实际终止及原始事件标志，不改变首次事件锁存，也不把逻辑到达成功变成新终止条件。无效状态不补项；普通非终止 rollout 尾部不补项；真实长度大于W的长回合不能因为缓存只剩短后缀而补项。长度等于W时，如果源实现的初始偏移使该 endpoint 没有完整窗口分数，则补一次；已经计分的 endpoint 不重复。

对于长度n的合格短回合，计算本回合全部n个 post-step 速度的均值（包含 terminal velocity），与回合第一步的 pre-step 目标速度比较，使用原 `SmoothL1(norm(error), 0)`。完整窗口部分不改原偏移、目标对齐和episode mask。

### 统一归约

令F为本次展开中有效完整窗口 endpoint 集合，S为补充的终止 endpoint 集合，K=|F|+|S|。速度标量及逐transition成本为：

```text
L_velocity = (sum(full_window_errors) + sum(terminal_short_errors)) / max(K, 1)
velocity_cost[t,n] = H*N/max(K,1) * 当前endpoint上的有效速度误差
```

完整窗口与短窗使用**同一个分母**；不是先对两组分别取均值再相加。成本仍记在各自 endpoint 上，`velocity_cost.mean()` 等于标量速度项。K=0时速度项为零，其他六项成本仍按原定义计算。按H*N/K缩放的逐transition成本仍随窗口覆盖变化，不能把补项称为已经消除 critic 的非平稳估计压力。

记录 `velocity_valid_samples`、`velocity_sample_fraction`、`velocity_short_terminal_samples`，truncated另记 `velocity_full_window_samples`。样本比例是 endpoint 覆盖，不是有梯度回合比例；有效样本增加也不自动证明首次到达率改善。

### 图与几何保持

terminal 位置、速度先与真实 PhysX 测量对齐并保留旧回合的可微图；下一回合状态、GRU及延迟历史再按reset规则隔离。窗口不得跨episode；跨优化更新的旧速度窗口前缀detach，仅为当前endpoint提供历史数值。所需真实长度或可见样本不一致时应报错，不能静默平均后缀。

DCOL碰撞项使用首次碰撞物理子步的实际位置与同一时刻动态障碍位置。最新重启将静态项改为实际高度场的signed距离；动态分支仍是凸体DCOL代理，不是可微接触动力学。旧unsigned结果保持原样。单独比较速度模式时必须固定静态实现、所选高度项与另外各项系数，不改变碰撞、bounds、invalid、timeout终止规则。

## 4. NavRL/PPO 是方法对照

两任务共用地图构建、五阶段课程、共享扰动配置及首次到达验收口径，但保留各自的策略接口：

| 项目 | NavRL/PPO | Zhang 三种速度模式 |
| --- | --- | --- |
| 策略输入 | 202维：8维状态、1×36×4静态LiDAR、5×10动态物体特征 | 1×12×16深度与10维状态，CNN+GRU192 |
| 深度传感器 | 场景同样有160×120相机，当前策略不读取它 | 160×120完整射线，经原处理与池化进入策略 |
| 物理步长 | 0.016 s，62.5 Hz | 0.01 s，100 Hz |
| 策略步长 | 0.016 s，62.5 Hz | 0.1 s，10 Hz |
| 默认更新采样 | 每环境32策略步 | BPTT每环境150策略步；AC按实际H记录 |
| 动作 | Beta分布输出目标方向坐标系各轴[-2,2] m/s目标速度，转换到世界系后交给速度控制链 | 六维交错输出解码为三维世界加速度；不增加每轴±2 m/s速度限制 |
| 训练目标 | 原NavRL奖励与官方RSL-RL PPO更新 | Zhang七项成本及所选BPTT/AC目标 |
| 回合超时 | 35.2 s，即2200策略步 | 35.2 s，即352策略步 |

真实定义见 [环境配置](../src/isaaclab_navrl/tasks/navigation/config/hummingbird/env_cfg.py)、[NavRL策略](../src/isaaclab_navrl/agents/models.py)、[PPO配置](../src/isaaclab_navrl/agents/rsl_rl_cfg.py) 和 [Zhang动作解码](../src/isaaclab_navrl/zhang/model.py)。PPO保留原动态物体特征，其输入信息与纯深度策略不同；不能把跨方法结果归因于优化器一个因素。

同为0–5步输入延迟、0–2步动作延迟时，因策略频率不同，对应秒数也不同；共享配置不意味着相同的物理时延。跨方法比较同时记录真实环境转移数、策略与物理步长、累计模拟时间和更新数。相同环境转移数不等于相同模拟时长，不能用更新数或共享GPU墙钟时间单独宣布样本效率/训练速度胜负。

PPO 使用独立 `--method navrl` 运行，不接收 Zhang 的 `--train_args`；新 truncated 使用 `--method zhang --train_args --variant dcol --velocity_loss_mode truncated_window`。这些是接口说明，不是已执行的启动命令。

## 5. 五阶段与正式验收

| 阶段 | 静态障碍 | 动态障碍 | 额外扰动 |
| --- | ---: | ---: | --- |
| 预训练S0 | 300 | 0 | 关闭 |
| 预训练S1 | 300 | 60 | 关闭 |
| 预训练S2 | 400 | 120 | 关闭 |
| 微调S1 | 300 | 60 | 开启 |
| 微调S2 | 400 | 100 | 开启 |

每个新实验使用独立输出目录、fresh随机初始化、1024并行环境，从S0开始，无最大优化更新次数。不得加载旧实验、已停止对照或有限步检查的权重。后续四阶段只继承本run的直接前序阶段检查点及算法状态；微调从本run预训练权重继续，重建场景、清空回合和历史。

同一Zhang实验的 raw/resolved速度模式必须在所有阶段一致；checkpoint严格比较配置，父协议核对真实结果中的两种模式字段。达到阶段要求不能成为偷偷切换模式或几何的理由。

每阶段累计至少num_envs个自然结束回合后按原首次事件统计汇总，训练首次到达率达到80%才晋级；不新增平滑晋级门槛或用独立评测替代。最终必须通过全部五阶段，保存400静态/100动态、扰动开启场景的策略。记录首次到达、碰撞、bounds、invalid、timeout、成功时间、return/reward，以及Zhang的成本和梯度指标。详见 [课程与验收](TRAINING_CURRICULUM.md)。

授权停止应向训练子进程发送信号，使其在完整更新后保存 `checkpoint_stopped.pt`；父管线保留该引用并记 `stopped`，不推进课程、不宣布验收。信号不直接发给尚未实现转发的父编排进程。停止与通过同一更新发生时，原子结果保留真实curriculum状态，但父协议仍按停止处理，不启动下一阶段。

## 6. 已完成的资源归档与尚待回执的运行

11:13 的同预算比较见 [机制比较](results/zhang_mechanism_comparison_1113_20261029.md)。之后RMS与Plateau均在 **U81／12441600个环境转移** 完整停止，检查点和实际已保存日志已冻结归档；这不改变原快照的U76指标，也不代表这两种机制最终无效。

- 归档：`logs/archive/20261029_rms_plateau_control/frozen_runs.tar.gz`，35227801 bytes。
- 归档SHA-256：`813ff24d4ee0031b53cf73228b78b5889fa2761886857b7f60acbf6b1d4e4658`。
- RMS停止检查点SHA-256：`232616621b43fbfdfd066eac92e57fb291d31d24b5b2f9ec3aa7dce89fa8a1f7`。
- Plateau停止检查点SHA-256：`3a61edd5dd19047641af3fc92518cf50a23733353b045f1cfc21e9eadcfd6a33`。

SHAC per-step按本轮决定继续观察更长预算，已经观察到的负critic问题如实保留。PPO/truncated的正式启动、实际配置、任何首达率或验收结论均待真实运行回执；此文不生成或推测它们。

## 7. 小规模筛选与正式验收的标记

`run_acceptance.py --screening --num_envs N`允许小规模fresh比较，仍要求五阶段同一障碍课程、自然报告80%门槛和本实验内部检查点继承。全部完成时标记`purpose=training_comparison`与`stage_thresholds_passed=true`，父级`acceptance_passed`始终为false；1024环境显式screening也不能冒作正式验收。正式运行必须不设screening且至少1024环境。新高度/优化器/投影接口见[新机制](ZHANG_NEW_MECHANISMS_20261029.md)，稳定检查点与stage80区分见[恢复协议](migration/RECOVERY_LADDER_20261029.md)。

## 7. 提交前验证

实际验证见 [结构化验证记录](results/zhang_truncated_ppo_validation_20261029.json)：140项测试与288个subtests通过；PPO真实64环境S0→S1加载及完整更新停止保存通过；DCOL短窗真实64环境2×H150展开通过。自然展开中未出现≤20步终止，因此另以0.5秒超时的测试环境验证短窗分支，两次更新各384个短窗样本，执行动作误差为0且反传有限。该测试环境不参与正式训练或验收。

当前PPO策略没有四元数输入，观测构造也不使用扰动后的四元数；共享配置中的1°姿态输入噪声在PPO中没有等同于Zhang的直接作用。位置、速度、LiDAR距离噪声和延迟按现有PPO接口生效。方法对照保留该输入定义，不能声称两方法获得完全相同的有效观测扰动。
