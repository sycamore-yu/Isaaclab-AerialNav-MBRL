# Zhang 可微动力学编译候选检查（2026-10-29）

结论：默认 `torch.compile(policy_transition, fullgraph=True, dynamic=False)`
未通过预先设定的数值门限，正式训练继续使用原 eager 动力学。

## 检查范围

独立脚本 `scripts/benchmark_zhang_compile.py` 直接加载当前纯 PyTorch 动力学，
没有启动 Isaac Lab、训练策略或改动训练入口。固定 1024 环境、100 Hz 物理、
10 Hz 策略，完整链为 150 个策略步后统一反传。分别检查
`record_positions=False/True`，覆盖随机参数、同形状参数重采样、名义悬停、
近共线推力和完整链。

前向门限：`atol=1e-5, rtol=1e-4`。
梯度门限：`atol=2e-5, rtol=2e-4`。
使用固定随机 ±1 余切量求和，检查四类初始状态与所有 command 的梯度。

真实环境：PyTorch 2.11 + CUDA 13.0，RTX 4090。另有五组正式训练并发。
运行显式传入 `--benchmark_on_mismatch`，只用于测量未通过候选的速度；
该标志不改变检查结果或准入状态。

## 实际结果

两种分支的名义悬停均通过；其余八项检查未通过。全部失败明细、源文件 SHA256、
运行环境和计时样本保存在：

`logs/benchmark/zhang_compile_150_fivetrain.json`

部分最大绝对误差如下，来自 `record_positions=False`：

| 场景 | 位置 / m | 速度 / m/s | 加速度 / m/s² |
| --- | ---: | ---: | ---: |
| 随机单步 | 门限内 | 0.001081 | 0.029724 |
| 近共线单步 | 0.000110 | 0.002305 | 0.029870 |
| 150 步完整链 | 0.023929 | 0.002770 | 0.045469 |

完整链的 command 梯度最大绝对差为 0.000311，相对 L2 差约 7.87e-6，
仍有 696 个元素超过逐元素门限。位置记录分支也存在超差。

源码在 `no_grad` 中使用 `acos(cosine)` 计算 airmode；接近共线的推力对
浮点舍入敏感。该候选的结果与此风险相符，但本次检查没有逐算子定位所有差异，
不能把全部误差都归因于单一算子。

热运行微基准的中位墙钟加速比分别为 6.17× 和 6.48×，仅覆盖纯动力学、
历史堆叠与反传。它不包含 PhysX、相机、策略网络、重置或真实训练损失，
也不是完整训练加速比。完整链未做 PhysX 状态对齐，与实际训练每策略步对齐
测量的流程不同；该测试用于检查候选数学与梯度等价性，不是任务验收。

最终报告状态为 `validation_failed`，`validation_passed=false`，
`candidate_eligible_for_further_training_profile=false`。没有放宽门限或将该候选
接入正在运行的五组训练。
