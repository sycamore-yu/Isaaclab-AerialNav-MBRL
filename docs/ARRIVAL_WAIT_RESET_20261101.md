# 首次到达后等待 5 秒与末态成功判定

当前默认协议为 `diffaero_arrival_wait_final_goal_v2`，训练和评测共用：

1. 距目标严格小于 0.5 m，设置 `arrived`，保存第一次进入的 `arrive_time`。
2. 从首次进入计时，等待 `wait_before_truncate=5.0 s`。离开、重新进入均不重启计时。
3. 对齐 DiffAero：当前仍在目标区，且当前时间严格大于 `arrive_time + 5 s` 时截断并 `reset_idx()`；原有总时限、碰撞、越界、无效状态仍按原规则处理。
4. 仅回合结束时仍在目标区且未发生失败才计成功。超过截止时刻但已离开目标区则继续运行，重入且已超过截止时刻时再截断；原有时限结束时仍在目标区外归为超时。到达后碰撞仍计碰撞。
5. `terminal_event_code` 是最终评分，`first_event_code` 仅保留首次物理事件历史。训练课程与评测都等待最终结果；评测另存首次到达时间及结束时间。

时间判断在策略步返回时执行，使用上游的严格 `>` 比较。四组实际 physics_dt=policy_dt=0.05 s，H150 为 7.5 s；不是 0.01 s 时步下的 1.5 s。5 秒等待可以推迟到达重置，不能保证之后所有优化窗口都不含 reset。H 步采集仍完整执行；reset 前损失保留，重置环境跨回合的状态梯度断开，这与 DiffAero APG 的处理一致。

`arrival_wait_final_goal_v1` 是之前已运行的“到期无条件截断”版本；其 5.0 s 控制试验与结果仍保留，但不再代表当前门控。v2 改为“当前在目标区且已超过首次到达截止时刻”。来源与直接验证见 `logs/diagnostics/2026-11-01_diffaero_arrival_h_audit/`。

## 旧机制来源

[上游 NavRL，固定提交 3725bcc，env.py 583–597 行](https://github.com/Zhefan-Xu/NavRL/blob/3725bcc2e7c1be4ecf1455d922299ae85042603a/isaac-training/training/scripts/env.py#L583-L597) 单独计算 `reach_goal`，但 `terminated` 只包含高度越界和碰撞；`reach_goal` 写入当步统计。因此到达后不重置源于上游，不是 Isaac Lab 的强制要求。

本仓库初始迁移 `183e572` 沿用不重置规则。首次到达结果的持续锁存属于本仓库新增统计逻辑，导致到达后离开或后续失败仍可能按首次到达计成功。旧评测器也按首次事件锁定结果。新协议改为结束时目标占用与失败优先级判定。

## 暂停与继续（首次 v1 切换记录）

四组在完整优化更新边界保存并重新读取验证：

| 组别 | 阶段 | 累计更新 |
|---|---|---:|
| baseline 直接 | fS1 | 2008 |
| baseline 课程 | S1 | 2749 |
| 重塑直接 | fS1 | 1935 |
| 重塑课程 | S1 | 2614 |

暂停证据：`logs/curriculum/2026-11-01_00-02-33_rectangular_2x2/pause_for_immediate_success_reset/checkpoints.json`。目录名来自最初的暂停请求；最终采用的默认机制是本文件描述的等待 5 秒规则，未在立即重置机制下恢复四组训练。

继续训练使用新目录和冻结源码，`--resume_checkpoint` 严格核对同一阶段、环境数量、seed、配置与扰动。保留模型、优化器、归一化、学习率进度与缩放、Torch RNG、累计更新和采样数。仿真状态与循环历史重建；报告窗口、最佳成绩和性能恢复报告从新协议重新累计。旧 source、结果、最佳模型和停止检查点全部保留。课程组继续其剩余课程，直接组仍固定 fS1，N512、H150、至少 1024 回合一报告，无更新上限。

## v1 验证记录

- 主分支完整 CPU 回归：543 passed、1 skipped、531 subtests passed。
- `tests/test_arrival_wait.py` 覆盖首次到达不提前成功、离开再进入不重计时、超过等待截止时刻仍在目标区外继续、总时限提前结束、碰撞覆盖首次到达、历史 continuous 失败不被后续到达洗掉、参数验证。
- `tests/test_zhang_contract.py` 验证同阶段继续保留权重、Adam 步数、学习率缩放及累计更新/样本数。
- Newton MJWarp + Warp 的 Zhang/NavRL 训练及评测模式真实 GPU 检查通过：不提前 reset；计时不变；保持目标区与离开目标区分别得到 success/timeout；部分 reset 清空且只影响指定环境；总时限可提前截断。
- 证据目录：`logs/validation/2026-11-01_arrival_wait5/`。

v1 的成功率按“到期无条件截断”统计，v2 使用不同等待门控；继续训练时重新累计报告窗口与最佳成绩。

## DiffAero 门控对齐

参考固定提交 `291ea14196aefbebcf7387dd71f7e096c83878b7` 的 [base_env.py:114–124](https://github.com/flyingbitac/diffaero/blob/291ea14196aefbebcf7387dd71f7e096c83878b7/env/base_env.py#L114-L124)。当前默认对齐到达等待门控，并保留用户要求的失败优先级：同一步碰撞或越界时不算成功。对 [APG.py:47–64](https://github.com/flyingbitac/diffaero/blob/291ea14196aefbebcf7387dd71f7e096c83878b7/algo/APG.py#L47-L64) 的执行验证中，第 60 步 reset 后仍采集 150 步损失再反传。

v2 验证：13 项相关回归通过；5 个直接执行上游方法的等待门控案例与本地结果逐项一致；真实 GPU 训练和评测均验证严格截止比较、超过截止时刻但在目标区外继续、重入后截断且首次时间不变。证据为 `logs/validation/2026-11-01_diffaero_gate/`。四组在 2060／2821／1987／2680 次累计更新处保存，随后从这些状态继续。
