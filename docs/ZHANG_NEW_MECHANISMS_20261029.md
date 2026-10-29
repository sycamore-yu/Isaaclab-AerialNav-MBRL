# Zhang 新机制与干净对照（2026-10-29）

## 状态与基线

本轮只训练 Zhang。新机制前代码 f6d5fd4 已标记
`baseline/zhang-pre-new-mechanisms-20261029`；旧七组实验已停止并完整归档实际保存的数据，
见 [冻结记录](results/zhang_pre_new_mechanisms_archive_20261029.md)。
归档和相机经验已在 1597ffe 提交。新正式实验在实现提交后从随机初始化开始，
不使用旧实验或有限步检查的模型。PPO 不启动。

`truncated_window` 已从活动训练配置移除。旧50更新记录中，其最好首次到达率为8.40%，
step为8.93%；它没有达到预期改善，用户决定取消后续实验。
这些早期单种子结果不证明两种方法的最终性能等价。

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
动态凸体使用真实DCOL共同缩放；静态仍使用实际mesh unsigned clearance。
因此 continuous 与 dcol 也有几何项差异，不能将它们的全部差异只归因于是否重置。

### 非饱和高度项

令
`violation = relu(0.2 + margin - z) + relu(z - 4.0 + margin)`，
使用 `SmoothL1(violation / scale, 0)`。
默认 margin=scale=0.5m、系数1，内部[0.7,3.5]m精确为零；
近边界平滑二次增长，远越界线性增长，梯度持续指向内部。
这是标准SmoothL1，近边界曲率为旧指数项的一半，没有暗中把系数乘二。

### 后续单因素诊断

短回合导致窗口速度监督覆盖不足时，可使用独立的 `--velocity_loss_mode per_step`，
在不变更DCOL几何的情况下给每个策略步提供速度误差。默认variant_default保持原映射；
SHAC固定H32，BPTT固定H150，不能在比较时遗漏这个区别。
[窗口监督经验](migration/DCOL_STEP_VELOCITY_EXPERIENCE.md)保留原SHAC停止检查点与对照协议。
连续方案另行检验高度系数8；默认高度系数仍为1，局部梯度依据及结论边界见
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
| patience | 5份无显著改善报告；第6份触发 |
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

1. 最近连续3份报告的成功率均至少20%，最大与最小相差不超过5个百分点时，建立稳定参考。
   参考分数取这3份的最小值，避免把单次尖峰当成稳定性能。
   后续参考至少改善0.5个百分点才覆盖。
2. 成功率比参考低 `max(5个百分点, 参考×20%)`，连续3份报告成立后先把lr乘0.5。
3. 再观察3份报告；只有每份仍退化才恢复。否则继续训练。
4. 恢复当前阶段稳定模型、AdamW、归一化及critic学习状态；AHAC还恢复H和dual。
   环境、GRU、动作与观测延迟历史、速度窗口全部重置。
5. 保留当前更低学习率及其持续上限，余弦重启不能把它重新抬高。
   全局更新数、采样数、耗时、训练报告、晋级事实、全局和critic私有RNG均不回滚；
   critic/AHAC的累计工作计数也不回滚。恢复后冷却3份报告。

稳定分数来自训练过程中自然结束的回合，不是固定策略独立评测。
已有少量成功但尚未满足稳定条件时，不自动锁定低水平hover策略。
所有动作记录在optimization_events.jsonl，TensorBoard记录下一学习率与恢复次数。
这一机制不改变80%晋级规则，不允许从别的实验或阶段取模型。

SIGINT/SIGTERM只请求停止。训练器完成当前完整actor/critic更新后，原子保存
checkpoint_stopped.pt，写status=stopped再退出。它不会导出半次optimizer更新，
也不会把人工暂停当成验收通过。

## SHAC 与 AHAC 对照

详情及作者代码审查见 [SHAC/AHAC适配](migration/SHAC_AHAC_ADAPTATION.md)。
两个对照都使用dcol配置和原七项Zhang成本，不使用NavRL reward替代成本。

SHAC固定H32，单critic、TDλ和EMA target；普通展开末端保留state及GRU的bootstrap梯度。
AHAC组明确命名为 **AHAC-derived DCOL synchronous adaptation**：
双独立critic、投影dual、几何风险约束、critic收敛拟合，以及全批共享H。
默认H从32开始、范围16–150；风险高时下一轮缩短，风险低时扩展。
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

## 新正式对照矩阵

各组seed0、1024环境，从随机初始化执行完整五阶段课程，余下损失系数和控制器相同：

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
