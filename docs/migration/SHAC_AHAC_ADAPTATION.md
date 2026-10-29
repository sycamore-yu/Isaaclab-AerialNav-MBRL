# Zhang SHAC 与同步 AHAC 风格 DCOL 适配

日期：2026-10-29。状态：算法模块与解析测试已实现；真实 PyTorch / Isaac Lab 验证由训练端执行并记录结果。本文不表示训练已经收敛或通过任务验收。

## 1. 范围与命名

`actor_critic.py` 独立实现 cost-form critic、TD(λ)、可微短时域 bootstrap 和同步时域控制器。它不创建 Isaac Lab 环境，不改变 Zhang 六维动作解码、控制器、仿真步长或七项原始损失定义。runner 提供原始逐步 `cost[T,N]`，要求其未折扣均值等于对应 Zhang total loss。

两种算法分别为：

- **SHAC 适配**：固定共同 rollout 时域，单 critic，EMA target，TD(λ) 拟合，普通末端 bootstrap 保留状态梯度。
- **AHAC-derived DCOL synchronous adaptation**：共同 rollout 时域，两个独立 critic、TD(λ)、收敛判据拟合、投影 dual 和 DCOL 几何风险约束。下一轮时域由明确的风险反馈规则改变。它不是作者代码的逐行复现，也不是 AHAC-1 的逐环境接触截断。

DCOL 风险是本任务授权的几何代理。动态凸体使用当前 DCOL 距离代理，静态几何保留当前真实 mesh 距离分支；不能将这些量描述为接触力、接触刚性、动力学 Jacobian 或整场景有符号距离。算法不新增接触动力学，也不能凭本适配宣称满足原论文的接触梯度误差理论条件。

## 2. 来源与源码审查结论

参考来源均为作者原始资料：

1. [SHAC 官方仓库，固定提交 a4c0dd1696d3c3b885ce85a3cb64370b580cb913](https://github.com/NVlabs/DiffRL/blob/a4c0dd1696d3c3b885ce85a3cb64370b580cb913/algorithms/shac.py)。
2. [AHAC 官方仓库，固定提交 bb59db5cf65e63740787bf22f91bae3103b30d19](https://github.com/imgeorgiev/DiffRL/blob/bb59db5cf65e63740787bf22f91bae3103b30d19/src/shac/algorithms/ahac.py)。同提交的 `ahac1.py`、`src/shac/models/critic.py`、`scripts/cfg/alg/ahac.yaml` 用于交叉核对。
3. [Georgiev 等，ICML 2024，论文页面](https://proceedings.mlr.press/v235/georgiev24a.html)，[作者论文 PDF](https://arxiv.org/pdf/2405.17784)。

作者资料区分了单环境接触截断与同步批次时域；同步方法包含约束 actor 目标、双 critic 和迭代 critic 拟合。reward-form 双 critic 的取值为 minimum，因此 cost-form 的 actor bootstrap 和 TD bootstrap 都取 maximum。

源码存在需要明确记录的疑点。固定 AHAC 提交中，dual 更新在风险超预算时增大，但随后 `H` 也随 dual 总和增大；低风险时 dual 可以为负且没有非负投影。actor loss 没有实际加入该 dual 约束项，重新分配缓冲时还把第一个 dual 值复制到所有时刻。论文 Algorithm 1 的时域更新也有相同方向疑点。另一个源码问题是双 critic 复用了同一模块列表，未形成两个独立参数集合。本适配不复制这些实现问题，也不据此推断作者已发表实验结果的真实性或具体运行版本。

下面的风险反馈是经本项目确认的适配定义。代码为独立实现，没有复制作者受限制文件的大段内容。

## 3. 共同 cost-form actor–critic

对每个实际 reset 或共同时域末端形成一个 segment，actor 最小化该 segment 的折扣成本与末端 value 之和。所有 segment 的总和除以本轮收集的 `H × N`。折扣幂在 reset 后重置。设一个 segment 长度为 `L`，其贡献为：

\[
\sum_{t=0}^{L-1}\gamma^t c_t + \gamma^L V(s_L,h_L).
\]

这使原始 `cost[T,N]` 的语义保持不变，但 actor objective 不再等于原 BPTT 的未折扣 total；新增的折扣和 bootstrap 本来就是算法差异，日志必须分别记录。

TD(λ) 使用递推：

\[
G_t=c_t+\gamma\big((1-\lambda)V_{t+1}+\lambda G_{t+1}\big).
\]

在 trace cut 处，递推里的 `G_{t+1}` 换成该步实际 supplied `V_{t+1}`，因此不会混入后续新回合成本。最后一个样本也从自己的 next value bootstrap。整个 TD target 和 critic 拟合输入均 detach。

### Continuing-reset 语义

本任务成本通常为非负。失败 reset 若强行 `Vnext=0`，会引入“尽早失败即可免去未来成本”的激励。因此本项目明确采用：

- reset 切断物理状态、GRU、动作历史及 TD trace；
- 旧 segment 保留真实 terminal collision loss 的梯度；
- reset bootstrap 来自实际 reset 后状态，数值保留，整体 detach；
- 普通非 reset 末端的状态、GRU、历史窗口输入保持梯度；
- 仅明确声明为 absorbing 的终点才可以传零 next value。

这与作者示例把部分 failure bootstrap 设为零的处理不同，是本项目的任务语义决策。

### Critic 输入与训练顺序

critic 需要增强状态，而非只使用含 detach 目标方向的十维 policy state。runner 至少考虑位置/目标差、速度、加速度/执行器、旋转、上一命令、剩余时间、实际深度特征和 GRU hidden。窗口速度损失还需要历史速度、目标和有效 mask；动作延迟存在时，尚未执行的命令队列也是相关状态。随机参数与 delay lag 可作为 detached 特征。

`critic_obs[t]` 对齐 action 前的 `h_t`；next observation 对齐 action 后的 `h_{t+1}`。普通末端 hidden 与有效历史保留输入梯度，rollout 起点 detach，reset 后清零。若省略完整观测延迟队列，应把 critic 描述为近似的增强状态 value function。

训练顺序固定为：收集 rollout → actor backward / optimizer step → critic fit → 更新 frozen critic → 选择下一轮 H。禁止在 actor graph 使用完毕前修改 frozen critic 参数。frozen 参数 `requires_grad=False`；普通 actor bootstrap 不能放进 `no_grad()`，否则会失去状态导数。

### 默认 value 配置

| 项目 | 默认值 / 规则 |
| --- | --- |
| 折扣 γ / TD λ | 0.99 / 0.95 |
| critic 网络 | 两层 256 ELU，输出单个 cost value；AHAC 为独立双网络 |
| critic 优化器 | AdamW，lr 0.0005，梯度裁剪 5 |
| 数据 | 当前 rollout，detach 后按独立 RNG 打乱，无额外 replay buffer |
| minibatch | 4096 |
| SHAC 拟合 | 16 个完整数据 pass |
| SHAC target | 新权重占比 0.6，即旧权重占比 0.4 |
| AHAC 拟合 | 至少 6、最多 64 个 pass；最近 5 次相邻 loss 相对变化均不超过 0.005 才提前停止 |
| AHAC value snapshot | 每轮固定，拟合结束后 hard copy；不加延迟 EMA |

网络宽度、AdamW、裁剪及相对收敛判据为 Zhang 适配参数，不声称是原文全部超参。相对变化判据避免直接照搬 locomotion reward 的绝对 loss 阈值，也避免正负 loss 变化相互抵消后被误判收敛。

## 4. DCOL 风险约束与同步 H

给定 `gaps[T,N,K]`，距离单位为米，取每个 transition 的最近样本：

\[
g_{t,n}=\min_k g_{t,n,k},\qquad
r_{t,n}=\operatorname{softplus}\left(\frac{d-g_{t,n}}{s}\right).
\]

`+inf` gap 表示该方向无障碍，风险为零。默认 `d=0.10 m`、`s=0.05 m`、风险预算 `C=0.10`。预算不是十厘米距离阈值：恒定 gap 约 `0.2126 m` 时风险为 `0.10`；恒定 gap 约 `0.2485 m` 时风险为 `0.05`。平均风险可能掩盖少数危险样本，因此日志同时记录 maximum 和 over-budget fraction。

actor 额外目标为：

\[
L_{constraint}=\operatorname{mean}_{t,n}\big[\mu\,(r_{t,n}-C)\big],
\]

其中 `μ` 是本轮固定、无梯度的 scalar dual，风险保留几何/状态导数。critic 拟合原始 Zhang cost，不把随 dual 改变的约束项混入 target。

actor 更新后，设 `rbar = mean(r.detach())`，执行：

\[
\mu_{next}=\operatorname{clip}\big(\mu+\eta_\mu(\bar r-C),0,100\big).
\]

下一轮共同 H 使用以下显式反馈规则：

| 条件 | 下一轮 H |
| --- | --- |
| `rbar > C` | `max(Hmin, floor(0.8 × H))` |
| `rbar < 0.5 × C` | `min(Hmax, H + 4)` |
| 其余 | 保持 H |

默认 `Hinitial=32`、`Hmin=16`、`Hmax=150`、`μinitial=0`、`ημ=0.1`。风险阈值、dual 上限、增长步数和缩短系数均有独立设置。每轮所有并行环境共享一个 H；环境原有 reset 不会改变这个共同 rollout 长度。风险本身不会触发环境 reset，也不会异步切断某个环境的梯度图。

本规则不是论文 H 更新式的等价变形，必须以本文的适配名称记录。其收益要通过真实训练与固定时域 SHAC 对照验证，不能从规则本身宣称加速或提高成功率。

## 5. runner 接口

```python
values = CostActorCritic(feature_dim, ActorCriticSettings(...), device=device)
controller = AdaptiveHorizonController(AHACSettings(...))  # 仅 AHAC 对照
horizon = controller.horizon  # 本轮只读取一次

# 正常末端用 live features；reset 部分由 runner 切换成 detached reset value。
actor_next_values = values.actor_value(live_next_features)
td_next_values = values.bootstrap_value(detached_next_features)
parts = short_horizon_cost_objective(costs, actor_next_values, trace_cut, discount=gamma)
risk = controller.risk(gaps)  # [H, N, K] -> [H, N]
constraint = controller.penalty(risk)
actor_objective = parts["objective"] + constraint

# 原 actor optimizer 的 backward / clip / step 完成后：
critic_metrics = values.fit(critic_features, costs, td_next_values, trace_cut)
horizon_metrics = controller.update(risk.detach())
```

以上示意不替代 runner 对 reset bootstrap 的选择。TD next values 必须对每个 transition 都提供，不能复用仅在 segment endpoint 非零的稀疏 actor value 数组。

`state_dict()` / `load_state_dict()` 保存在线与 frozen critic、critic optimizer、私有打乱 RNG，以及 controller 的 H、dual、设置和更新次数。正常课程继承加载这些状态。同阶段退化恢复只回退学习状态与 H/dual，保留当前私有 RNG、累计 critic fits 和 controller updates；全局采样和实际报告也不回滚。controller 拒绝不兼容设置的 checkpoint，防止悄悄改变风险控制含义。

建议日志至少分开保存 `base_cost_total`、`discounted_cost`、`actor_bootstrap`、`ahac_constraint`、`actor_objective`、critic target/loss/iterations、`ahac_horizon_used/next`、dual 与风险统计。原七项 gradient diagnostic 只适用于 BPTT 的原目标，不能把其和当成带 bootstrap/constraint 的完整 AC 梯度。

## 6. 验证范围

`tests/test_zhang_actor_critic.py` 包含解析 TD(λ) 数值、reset trace 隔离、普通末端位置/记忆梯度、独立双 critic 的 max-cost 梯度、固定数据 critic 拟合、EMA 与硬快照、RNG/optimizer checkpoint，以及 DCOL 风险导数、全批时域增长/收缩/保持、dual 投影与恢复测试。

训练端已实际运行该测试文件，包含于首轮131项测试与102个子测试的通过结果。另完成真实64环境、各2次更新的 SHAC/AHAC 检查：执行动作误差为0、actor/critic目标有限；SHAC实际H为32/32，AHAC为32/36且下一轮计划40。原始记录见 logs/new_mechanisms_checks/unit.log 与 smokes_v1。短检查只验证实现，不作为收敛或正式成功率证据。

窗口归约有一项明确限制：为了保持 cost.mean() 与原 Zhang scalar total 相等，有效窗口endpoint使用 H*N/valid_count 缩放。同一窗口误差可能因整批有效比例不同而具有不同权重，因此这里的 TD cost 不是严格平稳的单环境即时成本。无reset时首个H32只有12个有效速度窗口，倍率32/12；H16首段没有有效速度窗口。后续detached前缀会补齐跨更新窗口。日志持续记录 velocity_sample_fraction，不应直接横比不同H的critic loss数值来判断训练优劣。
