# 首次到达后等待 5 秒与末态成功判定

默认协议为 `arrival_wait_final_goal_v1`，训练和评测共用：

1. 距目标严格小于 0.5 m，设置 `arrived`，保存第一次进入的 `arrive_time`。
2. 从首次进入计时，等待 `wait_before_truncate=5.0 s`。离开、重新进入均不重启计时。
3. 等待到期或原有总时限先到时截断并 `reset_idx()`；碰撞、越界、无效状态按原终止规则处理。
4. 仅回合结束时仍在目标区且未发生失败才计成功。等待到期时已离开目标区归为超时；到达后碰撞仍计碰撞。
5. `terminal_event_code` 是最终评分，`first_event_code` 仅保留首次物理事件历史。训练课程与评测都等待最终结果；评测另存首次到达时间及结束时间。

时间判断在策略步返回时执行。当前四组为 20 Hz，GPU 控制试验从 0.05 s 首次到达至 5.05 s 截断，等待恰为 5.0 s；NavRL 的 0.016 s 时步在 5.008 s 后截断，误差小于一个策略步。

## 旧机制来源

[上游 NavRL，固定提交 3725bcc，env.py 583–597 行](https://github.com/Zhefan-Xu/NavRL/blob/3725bcc2e7c1be4ecf1455d922299ae85042603a/isaac-training/training/scripts/env.py#L583-L597) 单独计算 `reach_goal`，但 `terminated` 只包含高度越界和碰撞；`reach_goal` 写入当步统计。因此到达后不重置源于上游，不是 Isaac Lab 的强制要求。

本仓库初始迁移 `183e572` 沿用不重置规则。首次到达结果的持续锁存属于本仓库新增统计逻辑，导致到达后离开或后续失败仍可能按首次到达计成功。旧评测器也按首次事件锁定结果。新协议改为结束时目标占用与失败优先级判定。

## 暂停与继续

四组在完整优化更新边界保存并重新读取验证：

| 组别 | 阶段 | 累计更新 |
|---|---|---:|
| baseline 直接 | fS1 | 2008 |
| baseline 课程 | S1 | 2749 |
| 重塑直接 | fS1 | 1935 |
| 重塑课程 | S1 | 2614 |

暂停证据：`logs/curriculum/2026-11-01_00-02-33_rectangular_2x2/pause_for_immediate_success_reset/checkpoints.json`。目录名来自最初的暂停请求；最终采用的默认机制是本文件描述的等待 5 秒规则，未在立即重置机制下恢复四组训练。

继续训练使用新目录和冻结源码，`--resume_checkpoint` 严格核对同一阶段、环境数量、seed、配置与扰动。保留模型、优化器、归一化、学习率进度与缩放、Torch RNG、累计更新和采样数。仿真状态与循环历史重建；报告窗口、最佳成绩和性能恢复报告从新协议重新累计。旧 source、结果、最佳模型和停止检查点全部保留。课程组继续其剩余课程，直接组仍固定 fS1，N512、H150、至少 1024 回合一报告，无更新上限。

## 验证

- 主分支完整 CPU 回归：543 passed、1 skipped、531 subtests passed。
- `tests/test_arrival_wait.py` 覆盖首次到达不提前成功、离开再进入不重计时、等待到期末态离开、总时限提前结束、碰撞覆盖首次到达、历史 continuous 失败不被后续到达洗掉、参数验证。
- `tests/test_zhang_contract.py` 验证同阶段继续保留权重、Adam 步数、学习率缩放及累计更新/样本数。
- Newton MJWarp + Warp 的 Zhang/NavRL 训练及评测模式真实 GPU 检查通过：不提前 reset；计时不变；保持目标区与离开目标区分别得到 success/timeout；部分 reset 清空且只影响指定环境；总时限可提前截断。
- 证据目录：`logs/validation/2026-11-01_arrival_wait5/`。

这些检查验证回合机制与恢复契约，不代表新规则下的收敛、保持性或性能提升。旧成功率与新末态成功率语义不同，不能直接混合做通过判定。
