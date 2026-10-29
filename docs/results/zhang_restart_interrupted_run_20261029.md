# Fresh首轮两组SIGTERM中断记录（2026-10-29）

## 结论

本次两个trainer均记录收到SIGTERM（stop_requested="15"），在完整更新边界正常保存checkpoint_stopped.pt后退出。queue记录signal:15并停止派发。两组均未通过S0；max_updates=null，没有达到更新上限，也没有非有限数值故障证据。当前训练/queue日志无法确认信号发送者。

root在13:55:39 UTC复核时已无这些训练进程，GPU计算进程列表为空。此前13:49快照还显示两组运行。root没有为本次扩容发出该停止信号；只读审阅代理也报告未发送信号。此事实不能用于推断其他操作者身份。

## 时间线

| 事件 | UTC时间 | 证据 |
| --- | --- | --- |
| queue收到SIGTERM | 2026-10-29T13:54:44.631461+00:00 | stop_dispatch_reason=signal:15 |
| PPO pipeline结束 | 2026-10-29T13:54:46.642827+00:00 | status=stopped，trainer记录15 |
| Zhang pipeline结束 | 2026-10-29T13:54:55.362999+00:00 | status=stopped，trainer记录15 |
| queue结束 | 2026-10-29T13:55:01.594118+00:00 | dispatch_stopped，两组返回码0 |

冻结6a5的run_comparison_queue.py中，SIGTERM handler只设置停止派发原因和时间，没有kill或向子进程转发信号。pipeline使用start_new_session=True。因此trainer自身的stop_requested=15不能由queue的正常drain逻辑解释。在本run根目录、method和S0目录中未见停止请求文件；queue stdout为0字节，启动回执也没有信号发起者字段。

## 最终统计

两组均为N256、seed0、fresh、300静态/0动态、无额外扰动。最新自然报告可能早于最后一次更新，不能用最后更新号代替报告更新号。

| 组 | 完成更新 | 环境转移 | 最新报告 | 首次成功 | 碰撞 | 越界 | 超时 | 最佳首次成功 |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | --- |
| `dcol_truncated` | 21 | 806,400 | U21，752回合 | 2.12766% | 43.08511% | 54.78723% | 0.00000% | 9.03010% @ U8 |
| `navrl_ppo` | 353 | 2,891,776 | U339，258回合 | 0.77519% | 91.08527% | 6.97674% | 1.16279% | 9.62963% @ U276 |

两个数值不同的峰值来自不同采样预算，不能据此判断PPO优于Zhang或相反。此次记录也不支持淘汰truncated或其他尚未启动的候选。两组nonfinite_events=0；Zhang另有nonfinite_grad_events=0，PPO未提供这个Zhang专用字段。

## 已保存检查点及原始证据

### dcol_truncated

检查点：`/home/tong/tongworkspace/free/isaaclab-navrl/logs/curriculum/zhang_restart_controls_20261029/zhang_restart_dcol_truncated_seed0_20261029/zhang/pretrain_s0/checkpoint_stopped.pt`，真实文件大小6,202,953字节。

最后报告成功到达平均用时2.14250s，平均episode return -17.86013。

原始结果：`/home/tong/tongworkspace/free/isaaclab-navrl/logs/curriculum/zhang_restart_controls_20261029/zhang_restart_dcol_truncated_seed0_20261029/zhang/pretrain_s0/training_result.json`。同目录training_statistics.jsonl、TensorBoard和配置保留；父级`/home/tong/tongworkspace/free/isaaclab-navrl/logs/curriculum/zhang_restart_controls_20261029/zhang_restart_dcol_truncated_seed0_20261029/training_result.json`保留阶段谱系。

### navrl_ppo

检查点：`/home/tong/tongworkspace/free/isaaclab-navrl/logs/curriculum/zhang_restart_controls_20261029/navrl_restart_ppo_seed0_20261029/navrl/pretrain_s0/checkpoint_stopped.pt`，真实文件大小2,171,737字节。

最后报告成功到达平均用时1.16000s，平均episode return 1157.36737。

原始结果：`/home/tong/tongworkspace/free/isaaclab-navrl/logs/curriculum/zhang_restart_controls_20261029/navrl_restart_ppo_seed0_20261029/navrl/pretrain_s0/training_result.json`。同目录training_statistics.jsonl、TensorBoard和配置保留；父级`/home/tong/tongworkspace/free/isaaclab-navrl/logs/curriculum/zhang_restart_controls_20261029/navrl_restart_ppo_seed0_20261029/training_result.json`保留阶段谱系。

旧队列根目录：`/home/tong/tongworkspace/free/isaaclab-navrl/logs/curriculum/zhang_restart_controls_20261029`。`queue.json`保留两个stopped及十个从未派发的queued项，不篡改为已运行。

## wave2替代关系

新队列于2026-10-29T13:57:52.845064+00:00在主目录`logs/curriculum/zhang_restart_controls_wave2_20261029/`启动，初始queue PID931536仅作为历史快照。训练源码固定6a5eb1d1ec1730e7fff9d6ffdcecdf8c0cea465b，dispatcher固定42e47bc142d0077b27f0fe8d5727a8b18ec3542c。十二组仍采用N256、seed0、fresh、五阶段、无更新上限；并发上限六组不等于十二组已同时启动。真实首批六组验证另见[wave2启动记录](zhang_restart_wave2_launch_20261029.md)。

旧根目录superseded_by_wave2.json记录全部十二个候选的新旧输出映射；旧数据完整保留。新轮没有复用此处两个stopped检查点，相同seed的重新随机初始化也不视为第二个独立seed的重复实验。正式验收仍须从随机初始化跑至少1024环境并合法通过全部五阶段。
