# Zhang 损失梯度诊断（2026-10-29）

本工具用于解释训练回落；它不改变损失权重、原反向传播、梯度裁剪或 AdamW 更新。默认 `gradient_diagnostic_updates=()`，训练不执行额外求导。显式传入全局已完成更新编号，例如 `--gradient_diagnostic_updates 1,12,20,22,24,26,50`，才记录指定更新。

## 记录和解释

对同一条已完成的训练展开图，分别取得速度、速度预测、避障、碰撞、加速度、Jerk、高度边界七项原始损失的参数梯度。使用 `torch.autograd.grad(retain_graph=True, create_graph=False)`，不累加参数 `.grad`；随后仍执行原来的单次 `total.backward()`、全局裁剪和 AdamW。

`gradient_diagnostics.jsonl` 包含：

- 全参数、共享 CNN/GRU 主干以及输出头 z 两行的七项范数、Gram 矩阵、余弦和加权抵消比例。
- 七项加权梯度之和与原始反传梯度的误差，裁剪前范数及裁剪比例。
- 真实 AdamW 参数位移、相对更新范数、每项梯度对真实位移的投影，以及当前优化器设置。
- 原始/加权损失、边界激活比例、训练更新编号、完整训练配置、运行时间。
- 由同一 Gram 矩阵计算的高度权重阈值，以及高度权重 5 / 避障权重减半的原始梯度方向反事实。

反事实仅解释当前展开上的原始梯度，不能直接预测 AdamW 或下一条轨迹。梯度与实际参数位移的内积为负，表示该项损失在线性近似下下降；它不是新的训练成功率。诊断不增加环境采样、重置或优化更新。诊断带来的额外七次求导耗时单独记录，不将这些更新视为无诊断吞吐基准。

相机批量参数 `--camera_batch_size` 同时接入 Zhang 训练和验收入口；默认仍为 16，输入仅接受正整数，存入环境配置与检查点元数据。它不属于损失/优化配置，也不改变 160×120 原始分辨率。

## 实际验证

在当前远端 Python 3.12 / PyTorch 2.11.0+cu130 环境执行：

1. 新 CLI 及原验收协议测试：17 项通过，60 个子测试通过。证据：`logs/sim_to_real_checks/gradient_diagnostic_cli.xml`。
2. CPU 梯度诊断测试：3 项通过。使用真实 checkpoint 策略和七项损失，先建立非空 AdamW 状态，再比较诊断开关。CPU 原梯度、参数、优化器状态使用零容差；另覆盖零梯度与 AdamW 权重衰减、默认关闭和非法编号。证据：`logs/sim_to_real_checks/gradient_diagnostic_cpu.xml`。
3. RTX 4090 CUDA 小规模等价检查：N=2、H=4，实际发生梯度裁剪。原梯度、裁剪前范数、模型参数、参数位移、完整 AdamW 状态的最大绝对误差均为 0；CPU/CUDA RNG 不变，七项梯度和及更新投影闭合。证据：`logs/sim_to_real_checks/gradient_diagnostic_cuda.json`。

CUDA 测试固定容差 rtol=1e-5 / atol=1e-7，未按结果放宽。额外峰值 allocated：关闭诊断 4,123,648 字节，开启诊断 113,175,552 字节；两个模型/优化器副本同时驻留，该数字不是正式 N=1024/H=150 的内存或吞吐结果。正式长展开的检查将在选中的真实训练更新中记录，不能由此小规模测试替代。

## 官方 API 依据

- [PyTorch 2.11 checkpoint](https://docs.pytorch.org/docs/2.11/checkpoint.html)：当前策略使用 non-reentrant checkpoint，重算仅执行无外部状态写入的网络。
- [PyTorch 2.11 autograd.grad](https://docs.pytorch.org/docs/2.11/generated/torch.autograd.grad.html)：返回指定输入的梯度；保留图供原始训练反传继续使用。
