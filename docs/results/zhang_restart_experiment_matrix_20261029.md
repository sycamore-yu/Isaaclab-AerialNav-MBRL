# Zhang 重启实验计划矩阵（2026-10-29）

## 状态与用途

本文记录下一轮 **12 组候选计划**，不表示已经启动、完成或通过验收。实际启动的组、顺序、源码提交、完整配置和运行目录，以最终启动 manifest 及对应运行回执为准。资源测量后可以将候选排队；是否实际执行 `fullwindowref` 由主实验安排确认，不能把未执行项记为失败或删去已经发生的结果。

本轮先使用 **256 个环境、seed=0、独立进程** 做比较。每组从随机初始化开始，沿完整五阶段课程运行，不设置最大优化更新次数，也不新增固定样本量或墙钟停止预算。256 环境属于显式 `--screening`：即使完成全部阶段，`acceptance_passed` 仍为 `false`。后续正式验收须使用 fresh 的至少 1024 环境运行，遵循同一五阶段要求。[课程与验收](../TRAINING_CURRICULUM.md)规定两种运行的标记和门槛。

可执行参数的计划文件为 [zhang_restart_controls_20261029.json](../../configs/experiments/zhang_restart_controls_20261029.json)，其 `purpose` 明确为计划而非启动回执。表内名称是展示标签：实际 run name 和路径按该文件保存，例如 `base_truncated` 对应 `dcol_truncated`，`step` 对应 `dcol_per_step`，`projection` 对应 `dcol_height_priority`，`fullwindowref` 对应 `dcol_full_window`。

## Zhang 基线的共同设置

除矩阵中明确修改的因素外，Zhang 各组固定下列设置。PPO 是单独的方法对照，不套用 Zhang 的损失、优化器或时域。

| 项目 | 共同计划 |
| --- | --- |
| 初始化与规模 | fresh，`num_envs=256`，`seed=0`；每组独立参数、优化器、RNG、回合与课程状态 |
| 场景 | 同阶段使用相同地图采样规则、障碍数量及共享扰动配置；共用同一冻结、经过验证的静态几何版本 |
| 基线几何与算法 | `variant=dcol`，`algorithm=bptt` |
| 物理与策略频率 | 100 Hz PhysX，10 Hz 策略；动作语义、控制器及执行动作对齐规则相同 |
| 感知与策略 | 原始深度 160×120，策略深度 1×12×16，原 CNN+GRU192；相机执行配置在 manifest 中显式固定 |
| 速度监督 | `velocity_loss_mode=truncated_window`；W=20 个策略步，即 2 s。默认 `variant_default` 在 dcol 下也解析为此模式 |
| 高度项 | `height_loss_mode=target_exp`，目标为当前回合导航目标 z，采样区间 0.5–2.5 m；系数保持 1 |
| actor 优化器 | `optimizer=adamw`，`weight_decay=0.01`；共同学习率及其阶段切换按配置记录 |
| 调度与归一化 | `lr_schedule=cosine`，`observation_normalization=false` |
| 恢复与投影 | `recovery_enabled=true`，`gradient_projection=none` |
| BPTT 时域 | H=150 个策略步；保留其余成本系数、梯度衰减和裁剪设置 |

参数定义见 [ZhangTrainCfg](../../src/isaaclab_navrl/zhang/config.py) 和 [新机制接口](../ZHANG_NEW_MECHANISMS_20261029.md)。本计划不把静态距离接口接入视为几何验证已经完成；正式使用的实现与验证证据须对应最终冻结源码。旧 unsigned 几何、旧高度边界项和旧实验默认值的历史结果不回写为本轮配置。

## 十二组候选

| 编号 / 计划名 | 相对基线的修改 | 展开时域 | 可以检验的因素与归因边界 |
| --- | --- | --- | --- |
| 01 `base_truncated` | 无；DCOL + truncated 速度监督 | BPTT H150 | 本轮共同参照。它本身尚无训练结果，也不是旧 window 组的续训。 |
| 02 `step` | 仅 `velocity_loss_mode=per_step`，仍为 `variant=dcol` | BPTT H150 | 检验逐策略步速度监督；名称指 **DCOL + per_step**，不切换为旧 `variant=step`，避免同时改变几何。 |
| 03 `RMS` | 仅开启 `observation_normalization` | BPTT H150 | 检验运行观测统计归一化；物理状态、动作解码和成本仍使用原物理量。统计独立保存，展开内使用冻结快照。 |
| 04 `plateau2` | `lr_schedule=plateau`，`plateau_patience=2` | BPTT H150 | 检验基于自然成功率报告的调度；其余 Plateau 参数保持默认，Recovery 继续开启，须记录两种机制的实际触发原因。 |
| 05 `Adam` | `optimizer=adam`，仍为 `weight_decay=0.01` | BPTT H150 | 比较这两个优化器配置的整体效果。Adam 的耦合 L2 与 AdamW 的去耦权重衰减不同，相同数值并不代表相同正则化轨迹。 |
| 06 `SHAC` | `algorithm=shac`，DCOL 与 truncated 保持 | 固定 H32 | 真实 critic、TD(λ)、target 更新和可微 bootstrap 的算法对照；不能将差异单独归因于缩短 H。critic head 保持共同默认 `linear`，不夹带 softplus 对照。 |
| 07 `AHAC300` | `algorithm=ahac`，DCOL 与 truncated 保持；共享 H 自适应 | 初始 H32，上限 H300 | **AHAC-derived DCOL synchronous adaptation**：双 critic、风险/dual 与同步时域反馈共同构成方法。当前下限为 H16；“32…300”表示初始值与上限，不表示强制下限32。不是论文逐行复现，也不将几何风险称为接触刚性。 |
| 08 `continuous` | `variant=continuous`；速度模式仍显式 truncated | BPTT H150 | 关闭无人机接触响应，碰撞和高度越界不再 reset；invalid 与 35.2 s timeout 仍终止。变体还涉及动态几何调用路径，属于联合方法对照，不能声称只改变 reset。首次失败统计保持锁存。 |
| 09 `Huber` | 仅 `height_loss_mode=target_huber` | BPTT H150 | 对同一回合目标高度和同一系数使用 beta=1 m 的 SmoothL1；不切换为旧 `boundary_huber`，也不同时改目标分布或系数。 |
| 10 `PPO` | 独立 NavRL/PPO 方法 | 原每环境32策略步更新采样 | 保留原202维 LiDAR/动态物体观测、奖励、速度动作与62.5 Hz时钟；不是 Zhang 优化器或损失的单因素消融。 |
| 11 `projection` | 仅 `gradient_projection=height_priority` | BPTT H150 | 检验原始参数梯度的高度优先投影。投影发生在 Adam 更新之前，不能据欧氏梯度条件保证实际参数位移降低高度成本。 |
| 12 `fullwindowref` | 仅显式 `velocity_loss_mode=full_window`，解析为 `window` | BPTT H150 | 保留原完整窗口和初始样本偏移，关闭新增终止短窗补项，作为历史稀疏监督的同源码参照。是否排入实际运行由资源安排与 manifest 确认。 |

SHAC/AHAC 保留 Zhang 七项成本语义；critic、bootstrap 和控制器的确切适配见 [算法说明](../migration/SHAC_AHAC_ADAPTATION.md)。SHAC 的 H32、AHAC 的实际 H 和 BPTT 的 H150 不应在图表中隐藏。矩阵中的 Adam/AdamW 对照保持同一个非零权重衰减数值，结果只能解释为这个完整设置的差异，不能解释为去掉权重衰减后的纯动量消融；两种衰减的定义见 PyTorch 2.11 的 [Adam](https://docs.pytorch.org/docs/2.11/generated/torch.optim.Adam.html) 与 [AdamW](https://docs.pytorch.org/docs/2.11/generated/torch.optim.AdamW.html) 文档。

### 三种速度模式的共同规则

`base_truncated`、`step`、`fullwindowref` 固定相同 DCOL 几何、BPTT、频率、动作、目标、高度项、其余成本系数和优化器。truncated 仅为真实终止、非 invalid、真实回合长度不超过 W、且尚未被完整窗口计分的 endpoint 补充一次速度误差。完整窗口和补充短窗共享有效 endpoint 分母；普通 rollout 尾部不补项，reset 两侧历史不混用。详细公式见 [速度监督协议](../ZHANG_TRUNCATED_PPO_CONTROLS_20261029.md)。

监督覆盖增加是一项机制观测，必须与首次到达率、失败类型及梯度一起分析。它本身不证明任务成功率提高。AHAC 较短 H 与跨更新历史也会影响窗口端点的分布；报告需同时记录实际 H、有效完整窗口数和补充短窗数。

### 高度和优化的判据

`target_exp` 使用 `1-exp(-abs(z-target_z))`，远离目标时梯度会衰减；`target_huber` 在同一米制误差上使用 SmoothL1。Huber 对照检验这种函数形状差异，不沿用旧 boundary loss 的局部系数阈值。projection 对照除原始高度梯度与总梯度关系外，应记录实际优化器位移的高度梯度投影；没有这项证据时，不能由投影代码的方向约束宣布 Adam 的真实高度更新改善。

所有 Zhang 组共同启用 [分档恢复](../migration/RECOVERY_LADDER_20261029.md)：3 份稳定自然报告建立5%起、每5个百分点的稳定档，退化先减半学习率，再观察后才可能恢复同阶段最高稳定档。它是所有组共享的训练规则。Plateau 改变调度与 Recovery 的实际交互，因此需保留 LR、触发原因、恢复档位和累计样本记录，不把恢复后的计数当作新起点。首次80%自然报告的阶段晋级规则不增加“三报告稳定80%”条件。

## 五阶段与样本比较

| 阶段 | 静态障碍 | 动态障碍 | 额外扰动 |
| --- | ---: | ---: | --- |
| `pretrain_s0` | 300 | 0 | 关闭 |
| `pretrain_s1` | 300 | 60 | 关闭 |
| `pretrain_s2` | 400 | 120 | 关闭 |
| `finetune_s1` | 300 | 60 | 开启 |
| `finetune_s2` | 400 | 100 | 开启 |

所有候选从 S0 开始，仅继承本 run 的直接前序阶段检查点；不使用历史实验、有限步检查或其他组权重。阶段切换按现有协议重建地图并清空回合、GRU、延迟和速度历史。每阶段累计至少 `num_envs` 个自然结束回合后生成报告，首次到达率达到80%才晋级。每组独立晋级，不为方便共用场景而强制同进度。独立进程和资源安排的依据见 [共享场景比较说明](../migration/SHARED_SCENE_COMPARISONS_20261029.md)。

比较同 seed、同阶段时，用真实日志累计 `environment_steps = sum(num_envs * actual_rollout_steps)`。AHAC 按每次实际 H 计数，不能按 H150×更新数推算；恢复训练不倒退已采样计数。对照在各组共同可用预算内选择完整更新和其真实自然报告，不插值，也不因为某组更新更快而给它假定等预算的结果。这是读日志的方法，不新增训练停止上限。

PPO 的策略步长为0.016 s，Zhang 为0.1 s，因此相同环境转移数不等于相同模拟时长；跨方法须同时报告转移数、模拟秒数与墙钟。共享 GPU 的墙钟受同时运行的实验、初始化和负载影响，不能当作独占设备吞吐。不同方法触发 reset 的时刻不同，同 seed 也不保证之后的随机采样与轨迹一一配对。

## 启动参数与汇总身份

[run_acceptance.py](../../scripts/run_acceptance.py) 每次执行一组实验；计划 JSON 不会被该脚本自动读取或启动。编排器需为每组创建独立的新 `--output`，传入 `--method`、`--screening --num_envs 256 --train_seed 0`、设备及微调学习率。所有父级参数放在 `--train_args` 之前；Zhang 的参数放在其后，布尔值编码为 `0/1`，不能直接传 JSON 的 `true/false` 字符串。PPO 使用 `--method navrl` 且不传 Zhang 的 `--train_args`。

本轮新增[容量队列](../../scripts/run_comparison_queue.py)读取该计划并调用上述父入口。启动时和每次派发前验证同一clean commit；使用独立固定提交工作树，训练导入路径显式指向该源码。最多同时运行两个自有pipeline，先DCOL基线、再PPO、再step；每次新初始化要求所有活动组在**当前阶段**至少完成一次完整update，并要求GPU空闲显存覆盖候选预留量再加2048 MiB。预留量分别为PPO3584、普通Zhang4096、AHAC5120 MiB，是容量准入值，不是已测正式运行峰值。

队列写出`queue.json`、原始计划和每组完整命令/PID/状态；`zhang_manifest.json`只加入实际启动的Zhang组。它不改变更新预算、不暂停续训轮转，也不终止其他会话的进程。收到停止信号或发现子进程错误后只停止继续派发，已有子进程继续运行并被记录；下一槽仅在此前pipeline自然结束或显式停止后释放。若活动组长期未完成课程，剩余候选会继续`queued`，不能宣称十二组已全部启动或已完成比较。

现有父脚本拒绝复用已存在的输出目录；S0 不传 checkpoint，后续四阶段只使用本组上一个已通过阶段的 `checkpoint_stage.pt`。`--train_args` 不允许 checkpoint、resume 或 max_iterations 绕过这条谱系。2026-10-29 的本地只读交叉检查已将计划12组全部传过父参数解析，并构造验证60条阶段命令；11组 Zhang 配置均通过 `ZhangTrainCfg.validate()`。该检查未启动训练进程。

启动回执除完整命令外，应明确记录 `name/path/method`、源码提交与工作树状态、screening/规模、训练和地形 seed、五阶段顺序、无限更新设置，以及 Zhang 的以下实际配置：

- raw/resolved 速度模式、窗口秒数、算法、BPTT/SHAC H 或 AHAC 的 min/initial/max H；AC 的 critic head、TD(λ)、discount 和拟合设置。
- 高度模式、目标高度区间及系数；actor 优化器、weight decay、预训练/微调学习率、调度、Plateau patience、归一化和 Recovery 开关。
- 梯度投影、诊断更新编号、裁剪/衰减；静态几何实现标识、相机分辨率/批次和两种时钟。

[Zhang 汇总工具](../../scripts/summarize_zhang_mechanisms.py) 使用另一份**扁平** manifest，必需字段为 `name/path/variant/algorithm/normalization/lr_schedule/seed`；其中 `normalization` 映射实际 `observation_normalization`。应同时声明工具支持的 `velocity_loss_mode`、`height_loss_mode`、`optimizer`、`weight_decay`、`gradient_projection`、`plateau_patience`、`ahac_max_horizon`、`num_envs`，让汇总核对本轮消融身份。其余完整配置仍需保留在启动回执和每阶段 `config.json` 中，不能因汇总器未比较某字段就认为它已经相同。

这个工具目前只读取 `run/zhang/<stage>`，因此扁平 manifest 只列11个 Zhang 候选中实际启动的组；PPO 按自己的日志单独报告。配置身份缺失或不匹配时不得进入同预算比较。尚未启动的候选留在计划中，不伪造成运行结果，也不因其缺日志阻塞已启动组的有效比较。

## 结果记录和可归因边界

每组保留原始 manifest、源码版本、完整配置、stage、每次实际 H、累计转移数、自然报告和检查点路径。汇总至少包括：

- 首次自然报告达到80%的位置、最新报告、历史最佳报告，以及最近5份自然报告按完成回合数加权的首次到达率；这些量分别展示。
- 同一口径的碰撞、bounds、invalid、timeout、成功到达时间、return/reward，以及窗口覆盖、各项成本、梯度非有限事件和优化器实际更新比例。
- SHAC/AHAC 的直接成本、reset 与普通末端 bootstrap、critic target/拟合指标；AHAC 另记风险、dual 和实际 H。
- 学习率、Plateau/Recovery 触发和恢复档位；projection 另记投影前后原始梯度与实际参数位移诊断。

首次结果沿用任务锁存：碰撞或高度失败后再到达目标不改记成功；continuous 即使继续运动，也在回合自然结束时按首次结果计数。没有凑齐报告、文件缺失或仍在追加的半行应明确标记 pending/partial，不能补成0%成功率。

这轮只有 seed0，筛选规模为256，且可能共享 GPU。即使出现同预算优势，也先形成可重复检验的候选，不由单次峰值、损失下降或覆盖率增加自动修改默认。最终正式运行的启动与验收证据独立保存；本文的计划表不能代替它们。
