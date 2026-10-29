# Zhang 新机制与干净对照（2026-10-29）

> 此文保留此前六组实验的设计与证据。当前方案已更新为 [几何修复后的八组对照](ZHANG_SIGNED_GEOMETRY_20261029.md)，以该文为准。

## 状态与基线

初轮新机制实验仅训练 Zhang。新机制前代码 f6d5fd4 已标记
`baseline/zhang-pre-new-mechanisms-20261029`；旧七组实验已停止并完整归档实际保存的数据，
见 [冻结记录](results/zhang_pre_new_mechanisms_archive_20261029.md)。
归档和相机经验已在 1597ffe 提交。新正式实验在实现提交后从随机初始化开始，
不使用旧实验或有限步检查的模型。PPO 当时未启动；最新用户已要求另开 fresh PPO 方法对照，
具体配置差异见 [truncated/PPO 对照](ZHANG_TRUNCATED_PPO_CONTROLS_20261029.md)。后续 775dc6 的 PPO launcher 于 11:40 UTC 启动；用户再次要求替换当前实验后，六个训练子进程已在完整更新处停止并归档，见[冻结记录](results/zhang_pre_restart_frozen_runs_20261029.md)。新一轮启动以独立回执为准。

旧 `variant=truncated_window` 已从活动几何配置移除。旧50更新记录中，其最好首次到达率为8.40%，
step为8.93%；它没有达到预期改善，用户当时决定取消后续实验。
这些早期单种子结果及旧几何条件继续保留，不证明两种方法的最终性能等价。
最新重启的是 DCOL 几何下独立的 `velocity_loss_mode=truncated_window`，
不恢复旧 variant，也不把旧结果作为新配置的正式结果。

## 最新重启接口（历史实验配置不回写）

下一轮从随机初始化使用新源码；下列默认不改变本文后半部已经发生的实验设置或数字：

| 接口 | 当前默认与可选项 |
| --- | --- |
| 速度监督 | `variant_default`：step 为 `per_step`，continuous/dcol 为 `truncated_window`；显式 `full_window` 恢复原完整窗口并解析为 `window` |
| 高度项 | `height_loss_mode=target_exp`；可选 `target_huber`、`boundary_huber`，三选一 |
| 目标高度 | 当前回合导航目标的 z，默认采样区间 0.5–2.5 m；`height_target_min/max` 必须严格位于任务高度边界内 |
| actor 优化器 | `optimizer=adamw`、`weight_decay=.01`；可显式选 `adam` 和非负权重衰减 |
| 梯度投影 | `gradient_projection=none`；`height_priority` 仅用于 BPTT，不能由原始梯度方向保证 Adam 的实际位移方向 |
| Plateau | 默认 `plateau_patience=2`，允许 0 等非负整数；未改 cosine 默认调度 |
| AHAC 共同 H | 初始 32，最小 16，最大 300；上限是允许值，不代表当前 H 或已达到此长度 |
| 稳定恢复 | 三份稳定报告，5% 起每 5 个百分点留档至 80%；完整规则见 [分档恢复协议](migration/RECOVERY_LADDER_20261029.md) |

小规模比较须显式使用 `run_acceptance.py --screening --num_envs N`。即使五阶段均达到原 80% 门槛，父结果也只能是 `purpose=training_comparison`、`stage_thresholds_passed=true`、`acceptance_passed=false`。正式验收仍要求至少 1024 环境且不使用 screening；两者均不能加载旧实验权重或用有限步检查代替自然报告。

## 课程与环境

| 阶段 | 静态 / 动态 | 额外扰动 |
| --- | --- | --- |
| pretrain_s0 | 300 / 0 | 关闭 |
| pretrain_s1 | 300 / 60 | 关闭 |
| pretrain_s2 | 400 / 120 | 关闭 |
| finetune_s1 | 300 / 60 | 开启 |
| finetune_s2 | 400 / 100 | 开启 |

每阶段至少累计 num_envs 个自然结束回合，首次到达率达到80%后晋级。
正式运行1024环境，不设置更新上限；没有独立评测环境或额外平滑晋级门槛。
每阶段使用本实验前一阶段检查点重建场景，清空回合、GRU、延迟和速度历史。
新S0不创建动态障碍集合或空聚合PhysX view，静态相机和距离查询继续工作。

保持100Hz PhysX、10Hz策略、160×120原始深度、12×16策略图像、
CNN+GRU192及六维交错输出。正式比较统一相机批次64，保留配置默认批次16。
相机优化的原因、接口映射和验证方法见
[批量位姿读取经验](migration/CAMERA_POSE_AGGREGATION_EXPERIENCE.md)。

### 连续展开

continuous 禁用无人机 PhysX 接触响应。碰撞和高度越界均只记录事件，不终止，
不重置位置、动力学、GRU或历史。无效状态和35.2秒超时仍重置。
首次失败按物理子步锁存；之后返回合法高度或到达目标都不会覆盖失败。
timeout时才把这个失败回合计入自然训练报告。

dcol 保持原逻辑碰撞/越界重置，先保留终止转移的损失和图，再清空下一回合历史。
动态凸体使用真实DCOL共同缩放。新重启接口将静态分支改为 `signed_heightfield_occupancy_exit_proxy`：以最高竖直投影定义保守占据，内部使用到原网格合法出口的负距和冻结出口梯度，外部保持原网格无符号距离。实际高度场含自交面和零厚薄片，因此不将它称为精确SDF；近似范围、非连续边界及真实场景检查见[静态几何审计](migration/SIGNED_STATIC_GEOMETRY_AUDIT_20261029.md)。旧实验使用 unsigned clearance，其结果保持原样；接口变更不证明训练改善。continuous 与 dcol 的动态几何和重置仍有差别，不能将全部差异只归因于是否重置。

### 三种高度项

当前默认 `target_exp` 精确使用 `1 - exp(-abs(z - target_z))`，指数尺度为 1 m。目标取本回合导航目标的 z，并与当前转移配对；发生 reset 时，新目标只属于下一回合。该损失远离目标时会饱和、恢复梯度会衰减；没有额外叠加线性或边界项。

`target_huber` 对同一米制高度误差使用 beta=1 m 的 SmoothL1。显式 `boundary_huber` 保留旧边界公式：`violation = relu(0.2 + margin - z) + relu(z - 4.0 + margin)`，再使用 `SmoothL1(violation / scale, 0)`；margin=scale=0.5 m 时内部 [0.7,3.5] m 为零。历史 height8 诊断针对这个旧边界模式，不能将其临界系数直接移到新目标高度损失。

日志与系数仍沿用 `height_boundary` / `coef_height_boundary` 名称，实际语义必须结合记录的 `height_loss_mode` 判断。

### 后续单因素诊断

短回合导致窗口速度监督覆盖不足时，可使用独立的 `--velocity_loss_mode per_step`，
在不变更DCOL几何的情况下给每个策略步提供速度误差。新默认映射与旧完整窗口对照由上表区分；
SHAC固定H32，BPTT固定H150，不能在比较时遗漏这个区别。
[窗口监督经验](migration/DCOL_STEP_VELOCITY_EXPERIENCE.md)保留原SHAC停止检查点与对照协议。
最新新增的 `--velocity_loss_mode truncated_window` 保留完整窗口，按真实回合长度补充合格终止短窗；
三种速度监督行为与PPO方法对照的公平因素、共同分母和终止语义见
[truncated/PPO 对照](ZHANG_TRUNCATED_PPO_CONTROLS_20261029.md)。
旧连续方案曾单独检验边界高度系数8；新默认系数仍为1。旧局部梯度依据及结论边界见
[U1/U7/U10分析](results/zhang_continuous_gradient_diagnosis_20261029.md)。

## 运行统计归一化

开关为 `--observation_normalization 0|1`，默认0。
对实际进入策略的深度和10维state采样；包括它们已有的噪声和输入延迟。
深度使用池化后单通道的共同统计，state每维独立统计。
不修改用于动力学、动作解码和损失的物理量。

每次展开开始复制固定统计快照，整段正向与反向重算使用同一快照。
实际策略输入每步只累计一次；统计detach且不放入activation checkpoint函数。
归一化保留输入梯度，不对结果裁剪，方差下限1e-4。
关闭时严格返回原输入；没有历史统计时为identity。

检查点保存统计、开关和方差下限。推理只读取固定统计，不继续更新。
阶段继承和同阶段恢复都加载与权重配套的统计。设置不一致时拒绝阶段继承。
只有真实对照支持改善后才修改默认，当前不因两次短检查宣布胜出。

## 学习率与参数监控

`--lr_schedule cosine|plateau|constant` 三选一，默认cosine。
预训练默认lr=0.001，微调默认0.00005；原余弦周期400更新不变。
Plateau直接使用
[PyTorch 2.11 ReduceLROnPlateau](https://docs.pytorch.org/docs/2.11/generated/torch.optim.lr_scheduler.ReduceLROnPlateau.html)：

| 设置 | 值 |
| --- | --- |
| 监控 | 新的自然训练报告 success_rate，mode=max |
| factor | 0.5 |
| patience | 默认2；建立基准后第3份无显著改善报告触发，可配置0 |
| threshold / threshold_mode | 0.005 / abs，成功率使用0–1 |
| cooldown | 2份报告 |
| min_lr | 1e-6 |

不按每次梯度更新重复调用scheduler，也不在Plateau模式下覆盖余弦学习率。
阶段变化时重置报告比较基准；微调S1恢复指定微调学习率。
Plateau与恢复控制在同一报告同时触发时，只降低一次学习率。

每次完整actor update记录：

- `Optim/param_norm`：更新前参数L2范数；
- `Optim/update_norm`：AdamW实际更新前后参数差L2范数；
- `Optim/update_ratio`：更新范数 / (参数范数 + 1e-12)；
- `Optim/layers/<layer>/*`：每层权重与bias共同统计；
- `Optim/head/{ax,vx,ay,vy,az,vz}/*`：六维输出行分别统计。

这些是实际参数位移，不把裁剪后梯度范数当成参数更新幅度。
critic另记录每次actor更新期间全部critic拟合产生的总参数位移，前缀为Critic/Optim。
保留已有的裁剪前梯度范数、非有限事件和可选七项梯度诊断。

## 退化后的稳定模型恢复

`--recovery_enabled 0|1`，默认1。恢复只使用当前阶段目录中的稳定检查点：

1. 最近连续三份自然报告跨度不超过5个百分点时，保存所有已达到且未保存的 5%、10%、…、80% 档。实际参考取窗口最小值，旧档不覆盖；一次跨档可保存同一个新状态。
2. 严格满足 `current < ref - min(.05, max(.02, .30*ref))` 连续三份报告后先把 lr 乘0.5；`ref` 取同阶段最高稳定档的实际参考。
3. 再观察三份新报告，全部仍退化才恢复最高档；否则继续训练。恢复后清空报告窗口与退化历史，保留各档文件并冷却三份报告。
4. 恢复当前阶段模型、优化器、归一化及适用的 critic/H/dual 学习状态，重置环境、GRU、延迟和速度历史；全局计数、报告、晋级事实、当前较低学习率、全局及私有 RNG 不回滚。

首次自然报告达到80%仍按原规则晋级；尚未获得三份稳定报告时，只能保存阶段检查点，不能标为 stable80。完整API、版本2状态与边界测试见 [分档恢复协议](migration/RECOVERY_LADDER_20261029.md)。恢复机制不允许从别的实验或阶段取模型，也不能代替独立评测证明策略稳定。

SIGINT/SIGTERM只请求停止。训练器完成当前完整actor/critic更新后，原子保存
checkpoint_stopped.pt，写status=stopped再退出。它不会导出半次optimizer更新，
也不会把人工暂停当成验收通过。

## SHAC 与 AHAC 对照

详情及作者代码审查见 [SHAC/AHAC适配](migration/SHAC_AHAC_ADAPTATION.md)。
两个对照都使用dcol配置和原七项Zhang成本，不使用NavRL reward替代成本。

SHAC固定H32，单critic、TDλ和EMA target；普通展开末端保留state及GRU的bootstrap梯度。
AHAC组明确命名为 **AHAC-derived DCOL synchronous adaptation**：
双独立critic、投影dual、几何风险约束、critic收敛拟合，以及全批共享H。
当前训练入口默认H从32开始、范围16–300；风险高时下一轮缩短，风险低时扩展。
它不是作者源码逐行复现，也不把几何风险称为接触刚性。

失败reset使用实际reset state的detached value，避免零终值免除未来正成本。
终止碰撞损失仍保留梯度，TD trace不会串入新回合。
critic包括显式位置、速度、执行器、GRU、窗口与延迟历史等增强输入，
但未建模完整可微PhysX接触，仍是近似value状态。

原窗口速度成本按有效endpoint均值归约。将它分配到真实transition时保留
H*N/valid_count缩放，以使cost.mean()等于原scalar total。
首H16可能没有完整速度窗口；H32首次无reset时有12个有效窗口。
这使critic成本随有效窗口比例变化，不能用不同H的critic loss大小直接判断优劣。
日志使用实际environment_steps，禁止用固定H150乘更新数推算AHAC采样。

## 首轮新机制对照矩阵（历史配置）

以下是已运行的旧配置；AHAC 150 上限等记录不随新默认300回写。各组seed0、1024环境，从随机初始化执行完整五阶段课程，余下损失系数和控制器相同：

| 名称 | variant / algorithm | H | RMS | LR调度 |
| --- | --- | --- | --- | --- |
| dcol_baseline | dcol / BPTT | 150 | 关闭 | cosine |
| dcol_rms | dcol / BPTT | 150 | 开启 | cosine |
| dcol_plateau | dcol / BPTT | 150 | 关闭 | Plateau |
| dcol_shac | dcol / SHAC | 32 | 关闭 | cosine |
| dcol_ahac | dcol / AHAC-derived | 16–150，自适应 | 关闭 | cosine |
| continuous | continuous / BPTT | 150 | 关闭 | cosine |

恢复与参数监控在各组一致开启。单种子用于初筛；
默认切换需要相同阶段、相同采样预算的持续表现和进一步种子复核。
同时报告首次80%的样本数与时间、近期加权成功率、失败构成、退化、非有限状态和更新幅度。
共享GPU的墙钟时间包含竞争，不能当成独占算法吞吐。
不同H每次优化频率和余弦进度不同，这是本轮训练配置比较的一部分，不声称只隔离一个数学因素。

## 验证证据

首轮CPU：131项测试与102个子测试通过，本次改动文件Ruff通过。
真实Isaac Lab 64环境、每组2次完整更新：

| 检查组 | 实际H | 环境策略步数 | 执行动作最大误差 | 非有限梯度 |
| --- | --- | ---: | ---: | ---: |
| dcol + RMS + Plateau | 150 / 150 | 19,200 | 0 | 0 |
| continuous | 150 / 150 | 19,200 | 0 | 0 |
| SHAC | 32 / 32 | 4,096 | 0 | 0 |
| AHAC-derived | 32 / 36 | 4,352 | 0 | 0 |

记录目录为logs/new_mechanisms_checks/smokes_v1。
这些有限步检查与合成报告测试均不计入正式成功率。
正式验收仍只认1024环境、全部五阶段依次通过，以及最终400/100扰动场景的自然训练报告。

补充检查已通过：SHAC从S0检查点继承到S1并完成第3次更新；AHAC在真实扰动环境中
同时开启RMS/Plateau，实际H32→25，采样3648步、动作误差0、非有限事件0。
连续方案在真实z=.15/4.3越界后保持episode与运动；实际静态LiDAR碰撞后继续约.05m，
失败事件未被之后的到达覆盖。dcol对照按原规则重置。invalid与timeout仍重置。
场景证据为continuous_scene_v2.json和dcol_scene_v2.json；初版只有退出码而无JSON的运行
不算通过。Isaac Sim关闭可能提前结束脚本，必须在关闭前保存成功/失败断言结果。
详细可复核索引见 [验证记录](results/zhang_new_mechanisms_validation_20261029.json)。

## 后续诊断修正

上述六组的冻结代码为50766ec。连续无接触/无越界重置使地面以下状态变为可达，
随后确认原无符号地面距离会产生反向恢复梯度；该组已在完整更新后保存停止。
AHAC线性成本critic的负外推也已保存为独立诊断，增加可选非负输出fresh对照。
修正不热更新原活动实验，详见[隔离修正、归档及新对照](ZHANG_DIAGNOSTIC_FOLLOWUPS_20261029.md)。
