# 隔离修正后的正式训练启动记录

启动：2026-10-29T10:37:25.909716+00:00（UTC）。

- 训练代码提交：`980b43fcc630f009cc82936dd59df367c7a71167`。
- 标签：`baseline/zhang-diagnostic-followups-20261029`。
- 源码目录：`/home/tong/tongworkspace/free/isaaclab-navrl-followups-20261029`。
- 启动时工作树干净，实际导入位置已验证为该工作树的src目录。
- 两组均seed0、1024环境、随机初始化，无最大更新数；不继承旧实验或短检查模型。
- 原始160×120深度、相机batch64、100Hz物理/10Hz策略、完整五阶段课程与80%晋级门槛保持。
- 运行配置包含 `PYTORCH_ALLOC_CONF=expandable_segments:True`；共享GPU时间不用于独占性能结论。
- 本记录表示正式运行已启动，**不表示训练验收通过**。

| 名称 | 初始管线PID | 变体 / 算法 | 学习机制变更 |
| --- | ---: | --- | --- |
| continuous_signed_ground | 772401 | continuous / bptt | 地面以下有符号半空间距离 |
| dcol_ahac_softplus | 772402 | dcol / ahac | cost-value最后输出softplus |

continuous的全局更新1、7、10额外记录七项梯度诊断；诊断不参与更新，但其额外时间需单独解释。
其余损失系数、学习率、RMS关闭、cosine及恢复控制保持原对照配置。
AHAC仍按实际H累计采样，不能以配置H150乘更新次数推算。

完整可复现命令和环境字段见[启动清单](zhang_diagnostic_followups_launch_20261029.json)。
运行日志根为主仓库 `logs/curriculum/`，两组路径分别为：

- `zhang_followup_continuous_signed_seed0_20261029`
- `zhang_followup_dcol_ahac_softplus_seed0_20261029`

新旧源码必须继续隔离。旧基线、RMS、Plateau和SHAC的活动管线保留原脚本路径，
不能覆盖其后续课程会再次读取的源码。新代码的严格阶段继承会验证新增critic字段，
不接受把旧实验检查点伪装为新实验。

检查证据见[267项CPU检查及真实Isaac Lab检查](zhang_followup_validation_20261029.json)，
方法与归档解释见[诊断修正](../ZHANG_DIAGNOSTIC_FOLLOWUPS_20261029.md)。
