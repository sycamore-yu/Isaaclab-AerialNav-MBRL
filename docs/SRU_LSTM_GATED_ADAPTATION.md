# Zhang 的 SRU-LSTM-Gated 适配

目标已确认：适配 Zhang 的 CNN + GRU 路径。
范围已确认：SRU 仅用于 BPTT；拒绝 SRU 与 SHAC/AHAC 的组合。
用户已确认设计，并追加启动 1 次长期训练。
适配代码已完成，当前正在验证训练与推理。

## 来源与现有接口

本地核查基于提交 `ab4fc6e3ac37431526743a91aa7a9c8a105f8308`。
上游固定为 `d537910684d5c73af2acd43db6c8b001b593fbc0`：

- [SRU-LSTM-Gated 单元与序列接口](https://github.com/leggedrobotics/sru-pytorch-spatial-learning/blob/d537910684d5c73af2acd43db6c8b001b593fbc0/network/lstm_sru_gate.py)。
- [MIT 许可](https://github.com/leggedrobotics/sru-pytorch-spatial-learning/blob/d537910684d5c73af2acd43db6c8b001b593fbc0/LICENSE)。
- [上游实验范围](https://github.com/leggedrobotics/sru-pytorch-spatial-learning/blob/d537910684d5c73af2acd43db6c8b001b593fbc0/README.md)：空间记忆实验与核心单元，不包含完整导航训练系统。

现有 [ZhangPolicy](../src/isaaclab_navrl/zhang/model.py) 把深度图编码与 10 维状态投影相加。
融合特征的宽度为 192，随后进入 GRUCell。
SRU 的变换门直接使用输入特征，不要求额外的位姿参数。
因此，接入 SRU 不必新增传感器、运动真值或观测字段。
沿用当前输入只能检验导航任务中的记忆模型替换效果。
该试验不等于复现上游点云空间记忆实验。

SRU-LSTM-Gated 的核心公式按上游源码保留：

```text
g_t = tanh(transform_gate(x_t) * g)
f_refined = i * (1 - (1 - f)^2) + (1 - i) * f^2
c_t = f_refined * c_prev + (1 - f_refined) * g_t
h_t = o * tanh(c_t)
```

## 已确认的范围与方案

用户已确认目标为 Zhang BPTT，采用下列方案：

- 新增可选 SRU-LSTM-Gated，复用 CNN、状态投影和 6 维输出头。
- 沿用 192 维记忆宽度及当前观测、动作、损失、时钟和课程。
- 使用上游单步单元，保留初始化、公式、版权与许可文本。
- 复用当前 PyTorch；不安装上游点云实验依赖。
- 在训练和推理中同时维护 `h`、`c`。实际回合重置时按环境清零；rollout 边界同时截断梯度。
- 将模型类型写入训练配置和检查点。禁止把 GRU 检查点当作 SRU 检查点继续训练。
- 新 SRU 实验从随机初始化开始。已有 GRU 模型继续按原类型加载。

启动参数为 `--recurrent_model sru_lstm_gated`，默认值仍为 `gru`。
双状态按 `[h, c]` 拼成每个环境 384 维的状态。
训练器和诊断探针沿用同一张量的重置、截断和快照流程。
输出头只读取 `h`；`c` 参与后续 SRU 状态更新。

同为 192 维不代表参数量相同。SRU 单元还包含变换门及细化机制。
若后续报告比较成功率，需要同时报告模型参数量与实际训练耗时。

## 决定：本轮仅支持 BPTT

当前 [critic_observation](../src/isaaclab_navrl/zhang/critic_state.py) 把 GRU 的 `h` 拼入价值网络输入。
SRU 的未来输出同时依赖 `h` 和 `c`。
用户选择本轮仅适配 BPTT。BPTT 不使用这个价值网络。

| 选择 | 原因与后果 |
|---|---|
| 未选：覆盖 BPTT、SHAC、AHAC，价值网络读取 `h、c` | 价值网络看到完整策略记忆。SRU 价值输入增加 192 维；对照同时包含价值网络变化。 |
| 未选：覆盖三种算法，价值网络仍只读取 `h` | 保留价值输入维数，但相同 `h`、不同 `c` 的历史可能产生不同未来动作。价值网络无法区分这种情况。 |
| 已选：本轮只适配 BPTT | 先验证当前基线。SRU 与 SHAC/AHAC 的组合须明确拒绝，不能沿用不完整状态。 |

旧 GRU 权重无法直接恢复为 SRU。增加 `c` 后，旧价值网络权重也不能直接恢复。
GRU 基线可以保留自己的原有接口与检查点。
决策与原因见 [ADR 0002](adr/0002-sru-lstm-gated-zhang-bptt.md)。

## 修改范围

| 位置 | 需要处理的行为 |
|---|---|
| `zhang/model.py`、新增 SRU 单元 | 网络选择、上游公式、双状态、激活重算。 |
| `zhang/config.py`、`zhang/cli.py` | 显式模型配置、参数验证、训练入口传递。 |
| `zhang/runner.py` | 状态初始化、rollout 截断、回合重置、阶段切换、恢复、诊断输入。 |
| `zhang/inference.py`、`scripts/evaluate.py` | 按检查点恢复模型类型，按 episode_id 重置完整状态。 |
| `zhang/projection.py` 及调用方 | 固定输入诊断需要完整记忆状态；投影公式不依赖 GRU 类型。 |
| 单元测试与 README | 上游一致性、状态生命周期、检查点、入口使用说明。 |

`scripts/train_zhang.py`、`scripts/run_acceptance.py`、`scripts/run_comparison_queue.py` 使用现有配置传递机制。
核查配置传递、日志和源码快照是否保留模型身份。
不新增另一套训练器。

## 验证

2026-10-30 已在当前 `.venv` 的 PyTorch `2.11.0+cu130` 上执行 CPU 参考检查。
输入为 `[4, 3, 192]`，对应 4 步、3 个样本、192 维特征：

- 上游序列接口与逐步 Cell 调用输出完全一致，最大绝对差为 0。
- `use_reentrant=False` 的激活重算与直接执行的梯度完全一致，最大绝对差为 0。
- 输入、初始双状态和参数的梯度均为有限值。
- 单层 SRU 单元的参数量为 332736。

上述检查只验证上游核心与本地 PyTorch 的接口。
适配完成后的必要检查如下：

1. 比较移植单元与固定上游的多步输出、最终双状态及梯度。
2. 在 2 个环境中只重置第 1 个，检查第 2 个状态保持不变，两个被重置状态均清零。
3. 检查 rollout 边界截断、激活重算和诊断输入的行为。
4. 检查 SRU 检查点保存、加载、阶段切换与模型不匹配拒绝。
5. 检查配置在创建环境前拒绝 SRU 与 SHAC/AHAC 的组合。
6. 运行有限步训练和推理，检查动作、梯度和损失均为有限值。

尚未验证 Isaac Sim 集成、GPU 吞吐或训练成功率。
正式成功率与保持情况仍使用项目现有完整回合统计协议。

适配后的 CPU 测试通过 136 项测试和 325 个子测试。
另有 4 项 CUDA 测试因本轮显式使用 CPU 而跳过。
检查覆盖上游数值与初始化、双状态重置与截断、激活重算和诊断探针。
检查还覆盖检查点读写、阶段切换、旧 GRU 加载、类型不匹配拒绝和配置传递。

## 长期训练配置

用户已授权启动 1 条长期训练。
[实验配置](../configs/experiments/zhang_sru_lstm_gated.json) 复用 `zhang_finetune_s1.json` 的 baseline，仅增加 SRU 模型选择。

- 256 个环境，seed 0，随机初始化。
- 固定 `finetune_s1`：300 个静态障碍、60 个动态障碍，启用扰动。
- BPTT，20 Hz，每次更新 150 步，完整 2 秒速度窗口。
- 沿用 AdamW、初始学习率 0.001、cosine 周期 400 次更新及恢复机制。
- 每份报告至少 1024 个完整回合；不设更新上限，达到 80% 后继续观察保持情况。
- 使用独立源码快照与结果目录；训练状态按实际日志汇报。
