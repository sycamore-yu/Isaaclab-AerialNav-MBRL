# Zhang wave2 fresh 对照启动记录（2026-10-29）

## 当前结论

首批六组已实际启动并完成完整更新核对；其余六组仍queued。所有组N256、seed0、随机初始化、无最大更新上限。这里只完成screening启动和运行验证，尚未通过S0，更没有正式N1024验收。快照时间为 `2026-10-29T14:05:58.749598+00:00`；后续状态查看原始queue和训练日志。

训练源码固定 `6a5eb1d1ec1730e7fff9d6ffdcecdf8c0cea465b`，dispatcher固定 `42e47bc142d0077b27f0fe8d5727a8b18ec3542c`。它们分别位于 `/home/tong/tongworkspace/free/.worktrees/isaaclab-navrl/restart-training-20261029` 和 `/home/tong/tongworkspace/free/.worktrees/isaaclab-navrl/restart-dispatcher-20261029`，均为clean detached工作树。调度扩容没有修改训练代码。队列启动于 `2026-10-29T13:57:52.845064+00:00`，初始PID为 931536；PID只是历史身份，后续操作须核验cmdline和start ticks。

## 实际启动与早期结果

下表是不同实际采样量的状态快照，不用于算法排名。最新值、历史峰值、最近至多5份自然报告按完成回合数加权的值分别列出；尚未产生自然报告保留缺失。

| 组 | 完整更新 | 累计转移 | 最新成功率（报告更新） | 历史最高成功率 | 最近报告加权成功率 | 首完整更新核对 |
| --- | ---: | ---: | --- | ---: | ---: | --- |
| `dcol_truncated` | 10 | 384,000 | 0.321%（U9） | 4.573% | 1.157% | 34/34通过；首次观察U1 |
| `navrl_ppo` | 148 | 1,212,416 | 0.778%（U131） | 1.521% | 0.689% | 8/8通过；首次观察U5 |
| `dcol_per_step` | 7 | 268,800 | 0.326%（U7） | 1.060% | 0.607% | 34/34通过；首次观察U1 |
| `dcol_plateau2` | 5 | 192,000 | 1.199%（U5） | 1.199% | 0.794% | 34/34通过；首次观察U1 |
| `ahac300` | 8 | 94,208 | 0.332%（U8） | 0.895% | 0.516% | 37/37通过；首次观察U1 |
| `dcol_rms` | 1 | 38,400 | 未产生 | 未产生 | 未产生 | 34/34通过；首次观察U1 |

仍排队：`dcol_adam`、`continuous`、`shac`、`dcol_target_huber`、`dcol_height_priority`、`dcol_full_window`。这些组已保留完整新配置和启动计划，但没有运行结果。

六组均未报告非有限环境事件，五个Zhang组均未报告非有限梯度事件，执行动作误差为0。PPO没有Zhang的梯度事件计数，未把缺失字段补为0。Zhang首次更新核对包含实际训练参数、160×120深度、100Hz物理/10Hz策略和已验证几何文件hash。首次“观察”更新可能大于1；这不表示跳过了前面的训练更新。

采样GPU读数为 `18810, 5274, 99`（used MiB、free MiB、utilization %），即约18.4GiB已用、5.2GiB空闲、99%利用率。六槽为上限，每次派发仍受2048MiB余量、候选预留和当前阶段至少一次完整更新门控；共享GPU墙钟不能解释为独占吞吐。

## 同预算比较状态

现有汇总工具在共同预算 38400 转移处找到完整更新，但各组自然报告数为 `{"dcol_truncated": 0, "dcol_per_step": 0, "dcol_plateau2": 0, "ahac300": 0, "dcol_rms": 0}`。此时没有可用于成功率排名的共同自然报告；status=available只表示日志预算可对齐，不等于收敛结论。AHAC按实际H累加，不能以H150乘更新数；PPO另行汇总，策略步长0.016s与Zhang0.1s不同。

## 参数变化监控已在真实更新中生效

基线U7的全网络相对变化率为 2.0263%，六维输出层为 16.5536%；ay、vy两行分别为 35.0191%、34.9349%。裁剪前梯度范数 8.847919，配置裁剪阈值5。整体比率无法代替分层监控，但小初始参数分母会放大局部比率，因此这些值本身不证明更新过大或造成失败。完整P、Δ和逐层、逐输出行值写入同名JSON及原始updates/TensorBoard。

### U7投影诊断的具体含义

同一U7记录中，原始`g_height·g_total=-0.230082`，但真实AdamW位移的`g_height·Δθ=-0.00564518`。原始梯度冲突没有转化为这一步真实优化器位移的高度恶化：实际位移在当前代理目标的一阶意义上降低高度项。这里的历史日志名称`height_boundary`对应当前`target_exp`公式。

诊断用同一更新前网络和完整优化器状态、16个固定小间隙策略输入，构造`height_priority`反事实：`g_height·Δθ=-0.00663495`；投影命令减原命令的水平RMS为0.000195790m/s²，垂直RMS为0.007504262m/s²。实际训练仍为`gradient_projection=none`，原始诊断分支与真实参数/命令的误差为0。命令是延迟之前的解码世界坐标加速度；反事实没有推进PhysX轨迹。此结果说明本次投影效应主要改变垂直命令，不能宣布投影已经提升真实控制或首次到达率。同名JSON保存完整U7诊断行。

## 中断与谱系

前一队列的两组因SIGTERM退出，已保留检查点与结果，详情见[中断核查](zhang_restart_interrupted_run_20261029.md)。wave2重新随机初始化，没有加载旧权重；相同seed的重复起点不作为第二个独立seed。旧10个queued与2个stopped均在旧根目录 `superseded_by_wave2.json` 中映射到新目录。

## 原始证据与继续执行

根目录：`/home/tong/tongworkspace/free/isaaclab-navrl/logs/curriculum/zhang_restart_controls_wave2_20261029`。

- 相邻 `.launch.json` 和 `.plan.json`：实际启动命令、训练/dispatcher提交、初始资源和全12组计划。
- `queue.json`：实时PID、命令、状态；`zhang_manifest.json`只列实际已启动Zhang。
- `launch_validation.json`：首批六组首次观察完整更新的配置和数值核对。
- `status_snapshot_first_six.json`：本报告固定快照，含原始latest/best/recent reports和最后update。
- `comparison_summary_first_six.json`：完整日志预算对齐输出，未产生报告处保留null。
- 各运行 `<method>/<stage>/`：配置、自然报告、TensorBoard、检查点、优化器和几何诊断。

本报告的[同名JSON](zhang_restart_wave2_launch_20261029.json)记录摘要、证据hash与具体路径。所有五阶段均沿用80%晋级要求；正式验收仍需fresh、N≥1024，并通过最终400静态/100动态/扰动开启阶段。每小时后续任务已更新为本轮wave2状态；队列持续训练，不能将后台运行当作已验收。
