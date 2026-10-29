# 非负 cost-value 输出：独立 fresh 对照

日期：2026-10-29。这是针对已观察到的 value 外推问题增加的可选实验。默认仍为 `linear`，已经运行的六组实验保持其原始代码和配置。该设置不是原版 SHAC/AHAC 的复现声明，也不表示导航验收已经通过。

## 1. 实际观察及其含义

冻结基线提交为 `50766ecfc9b7971ee2a7e64d9f5ed45fd3cd43f5`。依据 2026-10-29 10:09:08 UTC 采集的真实日志快照，`dcol_ahac` 在第一阶段的部分指标为：

| Update | 实际 H | TD target 均值 | Critic 最后一轮 pass MSE | Actor bootstrap |
| --- | ---: | ---: | ---: | ---: |
| 2 | 36 | 131.57 | 37.71 | 1.659 |
| 4 | 44 | -10.58 | 1029.13 | 0.215 |
| 10 | 68 | -651.76 | 5923.36 | -19.556 |
| 18 | 100 | -784.95 | 6690.42 | -21.260 |

U18 的原始平均 cost 为 5.4754，折扣直接成本为 4.1996，actor objective 为 -17.0603。该轮 critic loss 从第一轮 pass 的 18657.91 降到 6690.42，训练了 12 个 pass；target 标准差为 155.23。此时自然报告为 2170 回合，成功 3、bounds 2167、collision 0。这里的数字来自训练日志，不是单元测试 fixture。

U1 附近的微小负预测可以由未约束输出的随机初始化产生。但上述持续、大幅负 target 表明问题已进入 bootstrap 反馈：critic 在拟合负的 TD targets，不能简单解释为“再增加拟合次数就会好”。U2 已将正 target 拟合到较小 MSE，后续新状态上的负预测仍然出现，也支持检查外推与 actor/critic 耦合。

静态审查没有发现以下具体实现错误：

- 七项原始 cost 及其当前系数均非负；TD(λ) 中折扣与混合系数非负。
- cost 与 next value 的 `[T,N]` 形状相同；critic 拟合的 `[B,2]` 与 target `[B,1]` 广播符合双 critic MSE。
- AHAC 使用 max-cost 双 critic；没有把 reward-min 的符号错误搬入 cost。
- actor backward 完成后才拟合并更新 frozen critic；reset value detached、普通末端 value 保留输入梯度。

这不能证明所有运行行为都没有缺陷，但目前没有依据把负值归为 cost 符号或 Tensor 维度 bug。需要区分近似 value 的外推、优化幅度和反馈失稳。负 bootstrap 的日志还含 detached reset 项，单凭其标量大小不能证明 live bootstrap 梯度支配 actor。

## 2. 单一变更

`ActorCriticSettings.value_transform` 支持：

- `linear`：直接返回原最后一层标量输出；默认值，保留原始前向与梯度。
- `softplus`：仅在最后一层标量输出后计算 `F.softplus(raw, beta=1, threshold=20)`。

训练配置/CLI 对应字段为 `critic_value_transform`，由 runner 显式传入算法设置。推荐新建 fresh AHAC 运行，只打开此选项。保持初始化种子、网络宽度、权重与偏置初始化、critic/actor 优化器、学习率、TD(λ)、七项成本系数、归约、bootstrap 时序、reset 语义、dual 与 H 反馈规则不变。该变换不引入参数或 RNG 调用，两个 critic 仍有各自独立参数。

这是对 value 输出施加成本非负先验，不是对 loss 或 TD target 做截断。actor 仍能通过 frozen value 的输入反向传播。

## 3. 为什么该先验适用于这里

对非负成本及非负折扣，真实成本回报为：

\[
V^\pi(s)=\mathbb E\left[\sum_{t\ge0}\gamma^t c_t\right]\ge0.
\]

continuing reset 继续累积未来成本，不改变此结论。softplus 单 critic 输出非负；两个非负输出的 maximum 也非负。因此，当 `c_t >= 0`、`gamma` 和 `lambda` 在 `[0,1]` 内时，当前 TD(λ) 的逆序递推保持 target 非负。最后一步与 reset trace cut 都只使用非负 next value，所以此性质也覆盖它们。

这一保证不能替代训练验证。网络仍可能把真实的正未来成本低估到接近零，也仍可能给出不合适的状态梯度。负 target 不再出现是构造性质，不能单独作为算法改善的证据。

## 4. 饱和与 warm-start 限制

通常区间内，softplus 对 raw 的导数为 `sigmoid(raw)`。raw 接近零时导数约 0.5；raw 大于 20 时实现使用线性分支，正的大 cost value 不发生上端饱和。raw 很负时梯度可能接近零：例如 raw=-20 的导数约为 `2.06e-9`，更负时有限精度会下溢。

因此，本对照从随机初始化开始。不能把已产生约 -785 输出的 linear checkpoint 直接套上 softplus；这可能立即进入下端饱和，而且已经同时改变了正在进行的实验。

Checkpoint 保存 `value_transform`，恢复时要求一致。旧 checkpoint 没有该字段时仅视为 `linear`；不能静默加载到 softplus 模型。相同 transform 的 optimizer、frozen value、私有 RNG 与计数恢复逻辑保持不变。

## 5. 比较与后续诊断

主要比较仍是同 seed、同 stage、共同**实际环境步数**预算下的自然首次到达率、最近五份报告的 episodes 加权到达率、碰撞/越界/超时率、成功到达时间，以及原始七项成本。不能用 update 数比较可变 H，也不能仅因一次峰值改变默认。

已有日志可继续观察 target mean/std、critic 第一轮与最后一轮 pass loss、critic 参数更新比例、pre-clip gradient norm、actor direct/bootstrap、H 与风险。MSE 应结合 target 尺度解释；记录的 pass loss 并非对最后固定权重重新计算的独立评估误差。

本隔离版本另外增加不参与 objective 的 detached 观测量：

- `bootstrap_live_endpoint`：最后一步未 reset 的 bootstrap 贡献；
- `bootstrap_reset`：原 bootstrap 总和扣除上述贡献，包含最后一步发生 reset 的项；
- `critic_next_value_min/mean/negative_fraction`：该 rollout 传入的 frozen next values 统计，描述拟合前的 value snapshot，不是拟合后网络重新预测的统计。

原 objective 与 bootstrap 累加计算保持不变。分解统计仅用于解释日志，不会额外加入目标或改变梯度；分量和可能有普通浮点舍入差异。next-value 三项统计在设备端归约后一起传回主机。

若需解释原因，后续独立诊断优先：

1. 将 bootstrap 分成 reset 的 detached 部分与普通末端的 live 部分；分别测量 direct 与 live-bootstrap 对 actor 的梯度范数和夹角。
2. 检查 live bootstrap 对物理状态、GRU hidden、窗口历史及延迟命令输入的梯度，避免仅凭负标量推断梯度来源。
3. 在真实固定 probe states 上记录 raw value 分布，特别是 `<-10`、`<-20` 的比例，以及 reset/ordinary/endpoint 各类别预测。若 softplus 大面积下端饱和，即使 target 非负也不能声称问题已解决。
4. 核对相邻 transition 的 `next_critic_observation[t]` 与下一步 `critic_observation[t+1]` 对齐，独立重算少量 TD targets，完成运行时的形状与时间契约验证。

本轮不同时改变另外两个已知适配边界：当前 H 控制器只看几何风险，上界失败不会自动提高这种风险；窗口速度成本还含 `H*N/valid_samples` 的全批次归约因子，使局部 value 拟合面对随 rollout 改变的成本尺度。保留这些因素，才能辨认输出变换这个单独改动的效果。

## 6. 验证状态

扩展的 actor–critic 测试覆盖：默认线性输出仍可为负且梯度不变、两种 transform 的初始权重/RNG/optimizer 配置一致、独立双 critic、非负 TD target、frozen value 的位置/记忆梯度、极端输入与下端饱和、旧 linear checkpoint 兼容和跨 transform 拒绝恢复，以及 bootstrap 分解和 next-value 统计不引入额外 actor 梯度。

执行端CPU集成检查已通过124项测试与143个子测试（JUnit共267项）；本次改动文件Ruff通过。
真实Isaac Lab 64环境AHAC+softplus完成两次完整更新，实际H32/36、采样4352步，
执行动作对齐最大误差0、非有限事件0。两轮冻结next-value最小值分别约0.6229和56.0741，
负值比例均为0；TD目标均值约72.63和132.02。
这些短检查确认实现与非负性质，不能证明收敛优于linear。
完整证据见[隔离修正验证索引](../results/zhang_followup_validation_20261029.json)。
