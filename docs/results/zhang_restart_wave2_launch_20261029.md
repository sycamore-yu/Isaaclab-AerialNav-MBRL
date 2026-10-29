# Zhang wave2 fresh 对照启动记录（2026-10-29）

## 当前结论

首批六组已实际启动并完成完整更新核对；其余六组仍queued。所有组N256、seed0、随机初始化、无最大更新上限。这里只完成screening启动和运行验证，尚未通过S0，更没有正式N1024验收。早期快照时间为 `2026-10-29T14:05:58.749598+00:00`。**后续六组均收到SIGTERM，队列于14:13:23 UTC停止；最新完整结果见文末终止记录。**

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

本报告的[同名JSON](zhang_restart_wave2_launch_20261029.json)记录早期启动摘要、证据hash与具体路径。五阶段仍采用80%晋级门槛。正式验收要求fresh、N≥1024，且最终400静态/100动态/扰动开启阶段真实通过。

## 终止记录（2026-10-29 14:15 UTC复核）

本轮队列在 `14:12:23.655181 UTC` 收到SIGTERM，`queue.json` 记录 `stop_dispatch_reason=signal:15`；六个已启动的trainer也各自记录 `stop_requested="15"`，在完整更新边界保存 `checkpoint_stopped.pt` 后退出。最后一个pipeline在 `14:13:23.553823 UTC` 结束。队列状态为 `dispatch_stopped`，六组均为 `stopped`，其余六组仍为 `queued`，并未启动。实时 `ps` 无相关trainer或dispatcher，GPU计算进程为空、显存使用92 MiB。训练和队列日志没有记录信号发送者，不能归因为算法崩溃或某位操作者。

六组均停在预训练S0（300静态、0动态、扰动关闭）。最新成功率取最后一份**完整自然回合报告**；它的更新号可能早于最后完整训练更新。最近5报告按完成回合数加权。历史最好只代表单份报告峰值。

| 组 | 完整更新／累计环境转移 | 最新完整报告的成功率 | 历史最好 | 最近5报告加权成功率 | 最新报告的主要失败 |
| --- | --- | --- | --- | --- | --- |
| `dcol_truncated` | 19／729,600 | U19：20/893，2.240% | 4.573%（U8） | 1.229% | 越界706/893；碰撞167/893 |
| `navrl_ppo` | 293／2,400,256 | U290：1/264，0.379% | 1.538%（U268） | 0.454% | 越界133/264；碰撞69/264；超时61/264 |
| `dcol_per_step` | 15／576,000 | U15：4/648，0.617% | 7.692%（U8） | 0.814% | 越界642/648 |
| `dcol_plateau2` | 12／460,800 | U12：4/468，0.855% | 4.702%（U8） | 1.662% | 越界454/468 |
| `ahac300` | 21／379,904（实际H累加） | U21：5/304，1.645% | 3.688%（U18） | 1.842% | 越界298/304 |
| `dcol_rms` | 9／345,600 | U9：30/304，9.868% | 9.868%（U9） | 2.125% | 越界140/304；碰撞134/304 |

已按实际启动的5个Zhang组重新执行 `summarize_zhang_mechanisms.py`，输出 `comparison_summary_at_stop_20261029.json`。S0共同可用预算是345,600次环境转移；AHAC使用真实可用完整更新327,680次，不虚构剩余转移。PPO单独汇总，不能按相同转移数直接等价比较模拟时长。以上早期结果尚不足以筛选胜者，未标记任何组为算法劣势淘汰。

五个Zhang组各自的 `nonfinite_grad_events=0`、`nonfinite_events=0`，已记录所有完整更新的 `executed_action_max_error=0`。四组 `truncated_window` 的更新日志里 `velocity_short_terminal_samples` 累计均为0；这只说明没有记录到合格短终止速度样本，现有汇总无法判定是否实际出现小于等于20策略步的合格终止回合。`dcol_per_step` 本来不使用短窗补项，不能混入这一判断。

末次更新的Zhang失败大多转为高度越界：`dcol_truncated` 的越界事件计数中，上越界733、下越界0；`dcol_per_step` 为647/0，`dcol_plateau2` 为459/0，`ahac300` 为303/0。它们是更新内的终止计数，不是上表自然报告的同一批样本。目标高度均值约1.5 m，基线末次平均高度2.287 m、平均目标高度绝对误差1.064 m。这是已观察到的垂直方向退化，尚不足以确定是高度损失、优化器或几何梯度的单独原因。RMS最后一次更新的上/下越界计数则为10/160，显示方向可以改变，不能将其U9单峰当作收敛证据。

两棵运行冻结工作树仍clean，源码和dispatcher分别固定在 `6a5eb1d1`、`42e47bc1`。所有 `checkpoint_stopped.pt`、原始JSONL、TensorBoard和早期诊断保留在 `logs/curriculum/zhang_restart_controls_wave2_20261029/`。没有复用已停止的权重或覆盖旧队列结果。由于本轮真实SIGTERM和用户后续的暂停／取消指令，没有自动重启队列或派发新的候选；正式验收仍未通过。
