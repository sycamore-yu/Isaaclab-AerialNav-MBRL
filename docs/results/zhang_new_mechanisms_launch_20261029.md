# Zhang 新机制正式实验启动记录（2026-10-29）

## 可复核的身份

- 训练代码提交：`50766ecfc9b7971ee2a7e64d9f5ed45fd3cd43f5`。
- 基线标签：`baseline/zhang-new-mechanisms-20261029`。
- 六组启动时间：2026-10-29T09:58:24.307097+00:00（UTC）；启动时 Git 工作区干净。
- 所有组从随机初始化开始，seed=0、num_envs=1024；不载入旧实验或短检查权重。
- 五阶段顺序、80% 自然训练首次到达率门槛和完整更新后保存规则保持一致。
- 不设最大更新数，不启动 PPO，不使用独立评测成功率替代训练报告。
- 完整命令、PID、日志路径：`logs/curriculum/zhang_new_mechanisms_20261029.json`。
- 本文件是启动与早期观测记录，**不是训练验收通过报告**。

## 实验矩阵

| 名称 | 变体 / 算法 | 实际展开配置 | 观测 RMS | 学习率调度 | 初始管线 PID |
| --- | --- | --- | --- | --- | ---: |
| dcol_baseline | dcol / bptt | 150 | 关 | cosine | 745892 |
| dcol_rms | dcol / bptt | 150 | 开 | cosine | 745893 |
| dcol_plateau | dcol / bptt | 150 | 关 | plateau | 745894 |
| dcol_shac | dcol / shac | 32 | 关 | cosine | 745895 |
| dcol_ahac | dcol / AHAC-derived DCOL synchronous adaptation | 16–150，自适应，从32开始 | 关 | cosine | 745896 |
| continuous | continuous / bptt | 150 | 关 | cosine | 745897 |

各组均开启参数更新监控与同阶段稳定模型恢复，相机 batch=64，原始深度160×120。
物理100Hz、策略10Hz。预训练默认学习率0.001，微调0.00005，梯度裁剪5。
代码实现、算法适配边界和检查证据见
[新机制说明](../ZHANG_NEW_MECHANISMS_20261029.md)、
[验证索引](zhang_new_mechanisms_validation_20261029.json)。

## 课程

`pretrain_s0(300/0/off) → pretrain_s1(300/60/off) → pretrain_s2(400/120/off)
→ finetune_s1(300/60/on) → finetune_s2(400/100/on)`。

课程通过必须来自本实验自然结束的至少1024个回合，首次到达率≥80%。
continuous的碰撞/高度越界锁存首次失败，但不重置；回合到超时后报告失败。
无效状态仍重置。其他dcol组保持终止转移损失后再重置。

## 初期同采样预算快照

快照生成时间：2026-10-29T10:06:57.163252+00:00。共同采样预算为921,600个环境策略步，
仅取预算以内完整更新与自然报告；SHAC取其最近一个完整更新，不插值。
各组仍在S0，尚无组达到80%。下表是最近最多5份报告按回合数加权的成功率，
不是固定策略独立测评，也不据此切换默认配置。

| 组 | 已用策略步 | 加权首次到达率 | 加权高度越界率 | 本预算内最好单报告 |
| --- | ---: | ---: | ---: | ---: |
| dcol_baseline | 921,600 | 0.717% | 89.629% | 0.994% |
| dcol_rms | 921,600 | 0.803% | 93.504% | 1.256% |
| dcol_plateau | 921,600 | 0.678% | 89.509% | 0.994% |
| dcol_shac | 917,504 | 0.646% | 94.954% | 1.118% |
| dcol_ahac | 921,600 | 0.321% | 99.213% | 0.675% |
| continuous | 921,600 | 0.293% | 94.873% | 0.391% |

机器可读、带原始文件和行号的快照：
[zhang_new_mechanisms_early_20261029.json](zhang_new_mechanisms_early_20261029.json)。

初次BPTT更新的整体相对参数变化约2.7584%，六维输出层中az行约234.05%。
后者的初始参数范数很小，单看比例不能证明训练不稳定；需要结合输出加速度、
上下边界事件、实际损失和后续趋势。监控记录的是AdamW实际更新，不是裁剪后的梯度。
早期上下边界事件以越过上边界为主，连续组随后仍继续运动，非饱和高度项保持恢复方向的梯度。
所有组初期执行动作对齐误差为0，未记录非有限梯度或状态。

六组共享同一RTX4090。启动后实测GPU利用率99%、显存约21–22GiB；
各组墙钟时间包含GPU竞争。只用同阶段实际采样预算比较学习效率，
共享GPU时间不作为独占算法速度结论。

## 持续比较与默认选择

汇总命令：

```bash
.venv/bin/python scripts/summarize_zhang_mechanisms.py \
  --root logs/curriculum \
  --manifest logs/curriculum/zhang_new_mechanisms_20261029.json \
  --output logs/curriculum/zhang_new_mechanisms_summary.json
```

汇总只包含manifest列出的正式组，排除短检查和旧实验。
持续记录首次80%的采样步数/耗时、最近加权成功率、失败构成、非有限事件、更新幅度和恢复事件。
归一化和Plateau仍为可选配置；只有持续改善并经进一步种子复核后才修改默认。
训练中不直接修改源代码；必要的修正或新调参组使用单独记录的干净提交和新随机初始化。
