# SRU-LSTM-Gated 适配与长期训练

Zhang BPTT 已支持可选 SRU-LSTM-Gated。
长期训练已启动；截至 2026-10-30 17:29:37（Asia/Shanghai），已完成 4 次更新、153600 个环境步。
训练仍在运行，NaN/Inf 状态事件和梯度事件均为 0。

## 代码与验证

训练源码固定为 `fe9684a4d27c4445305e9b0ae7691d6dc3b98029`。
上游源码固定为 `d537910684d5c73af2acd43db6c8b001b593fbc0`。
SRU 的完整策略参数量为 624896，GRU 为 514496。

- CPU：136 项测试、325 个子测试通过。4 项 CUDA 测试因显式使用 CPU 而跳过。
- 上游对照：通过 `SRU_REFERENCE_ROOT=/tmp/sru-pytorch-spatial-learning-audit` 执行，未跳过。初始化、多步输出、双状态和梯度完全一致。
- PhysX 训练：2 个环境，每次更新 150 步，共完成 2 次更新。损失分别为 5.231184、6.365896；梯度范数分别为 1.160691、0.873062。
- 诊断回放：实际参数更新误差和动作回放误差均为 0。
- 检查点推理：独立加载后完成 2 个回合，均因高度越界失败，无无效状态。该检查不构成成功率验收。
- Ruff 与 `git diff --check` 通过。

首次直接启动因缺少 `OMNI_KIT_ACCEPT_EULA` 退出，尚未进入仿真。
第二次启动复用现有队列的环境设置，完成上述训练和推理。
首次失败日志及第二次验证均保存在
[`logs/sru_validation/2026-10-30_17-20-24/`](../../logs/sru_validation/2026-10-30_17-20-24)。

## 长期训练

配置为 256 个环境、seed 0、随机初始化、固定 `finetune_s1`。
场景包含 300 个静态障碍和 60 个动态障碍，启用扰动。
BPTT 使用 20 Hz、150 步、完整 2 秒速度窗口。
训练不设更新上限，达到 80% 后继续运行。
该运行属于 256 环境 screening，不构成 1024 环境正式课程验收。

首份报告出现在第 2 次更新，包含 1090 个完整回合：

| 结果 | 回合数 |
|---|---:|
| 成功 | 3 |
| 碰撞 | 560 |
| 高度越界 | 527 |
| 无效状态 | 0 |
| 超时 | 0 |

首份报告成功率为 `3 / 1090 = 0.2752%`。
该数值来自随机初始化后的早期训练，尚不能判断收敛或收益。

队列 PID 为 `1865607`，管线 PID 为 `1865627`，训练 PID 为 `1865642`。
进程命令、导入路径、源码快照的全部 231 个文件和检查点模型类型已核验。

- [运行目录](../../logs/curriculum/2026-10-30_17-26-15_sru_lstm_gated)。
- [启动核验记录](../../logs/curriculum/2026-10-30_17-26-15_sru_lstm_gated/launch_verification.json)。
- [完整回合报告](../../logs/curriculum/2026-10-30_17-26-15_sru_lstm_gated/sru_lstm_gated/zhang/finetune_s1/training_statistics.jsonl)。
- [实时训练状态](../../logs/curriculum/2026-10-30_17-26-15_sru_lstm_gated/sru_lstm_gated/zhang/finetune_s1/training_result.json)。
- [TensorBoard 事件目录](../../logs/curriculum/2026-10-30_17-26-15_sru_lstm_gated/sru_lstm_gated/zhang/tensorboard)。
