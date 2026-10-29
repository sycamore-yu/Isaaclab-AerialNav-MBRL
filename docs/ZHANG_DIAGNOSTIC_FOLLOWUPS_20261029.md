# Zhang 早期诊断后的隔离修正与新对照

日期：2026-10-29。原新机制实验使用代码 `50766ecfc9b7971ee2a7e64d9f5ed45fd3cd43f5`；启动记录提交为 `955a608fd66332f382a24a7b2725fede4abd8602`。本轮在独立 Git 工作树 `/home/tong/tongworkspace/free/isaaclab-navrl-followups-20261029` 和分支 `codex/zhang-training-followups-20261029` 上开发，原目录的活动训练及后续课程仍加载原源码。

## 已确认的问题与变更

1. **连续方案地面以下的梯度方向。** 原无符号平面距离在穿过地面后使avoidance与collision项奖励继续下降。SmoothL1高度项本身方向正确，但会与这两项冲突。只对continuous低于地面的原look-ahead点换成有符号半空间距离；地面以上、step和dcol保留原值与梯度。见[复现和修正经验](migration/CONTINUOUS_GROUND_SIGN_EXPERIENCE.md)。
2. **成本critic的负外推。** 七项成本均非负，AHAC线性critic曾产生较大的负TD目标。增加 `--critic_value_transform softplus` 可选对照，默认仍为linear；只变换critic最后标量，不改网络参数、优化器、成本或H控制。见[数学依据、限制及实验协议](migration/COST_VALUE_SOFTPLUS_EXPERIMENT.md)。
3. **诊断可解释性。** 新增detached reset/live-endpoint bootstrap分量、旧next-value的最小值/均值/负值比例，以及PyTorch allocated/reserved及各自peak显存指标。它们不参与目标或反向传播。

参数监控和RMS的实测解释见[早期更新尺度诊断](results/zhang_early_optimizer_diagnosis_20261029.md)。首步输出行相对更新很大源于很小的初始化尺度；这不是监控公式错误，也不能直接等同于命令加速度的同幅变化。

## 被停止的两组与完整保留的证据

停止信号只请求结束当前完整actor/critic更新，然后原子保存检查点。两组均没有非有限事件，也没有通过S0；停止不是最终算法能力排序。

| 原组 | 完整更新 | 环境策略步 | 停止原因 |
| --- | ---: | ---: | --- |
| continuous | 21 | 3,225,600 | 已确认地面以下无符号几何的反向梯度，需要修正定义域后重零对照 |
| dcol_ahac / linear | 33 | 3,225,600 | 保存负成本预测的诊断轨迹，并释放增长的显存占用，转入非负输出fresh对照 |

AHAC的负目标并非一直单调恶化：U15均值约-834.59，U33已回升到-37.69；U33平均七项成本5.1605、bootstrap -2.4593，最后自然报告成功率1.1424%。保留这一恢复趋势，不能把早期负外推表述为已经证明永远不能收敛。非负输出也不自动证明更快或更稳定，必须继续看自然首次到达率。

原停止检查点的SHA-256：

```text
continuous: d1a883166c7628bab555ebd24b9412873df9ac93a94315903fc2ea8973436509
AHAC linear: b19001464764fe99dd78b8171fceffe33c4560c4e0c81af6c856e94e7a2f126a
```

两组的完整目录、启动清单、launcher日志均已归档到主仓库：
`logs/archive/20261029_post_new_mechanisms_diagnostics/frozen_runs.tar.gz`。
大小54,864,420字节；SHA-256：

```text
5c845eb49b153a387d0e3ba40a23f2cea2bb9720a02ac29b35f85247a721ebf9
```

同目录的manifest保留训练结果、计数与检查点hash；原结果目录仍可读取。其他DCOL基线、RMS、Plateau及SHAC不在本次停止清单中。

## 新对照约定

新运行在此分支提交完成、工作区干净之后启动，均随机初始化、seed0、1024环境、无最大更新数，依次完成S0→S1→S2→微调S1→微调S2，仍以每阶段至少1024个自然回合的80%首次到达率晋级。

| 新组 | 对应原组 | 学习机制差异 |
| --- | --- | --- |
| continuous_signed_ground | 原continuous | 仅地面以下的avoidance/collision距离符号修正 |
| dcol_ahac_softplus | 原dcol_ahac | 仅critic最终输出改为softplus非负值 |

其余学习率、原七项损失系数、RMS关闭、cosine、恢复控制、执行动作与随机化对齐、原始160×120深度不变。continuous仍不因碰撞或高度越界重置。AHAC仍是已记录的DCOL同步适配，不声称完整可微PhysX接触或作者代码逐行复现。

continuous新组可在指定更新记录已实现的七项梯度诊断；该诊断不参与更新，但会增加耗时。以实际环境策略步比较学习进展，对含额外诊断的更新不能直接比较训练速度。

## 显存控制及比较限制

原AHAC在H增长到150附近，nvidia-smi报告进程占用9,426 MiB，而记录的PyTorch峰值allocated约1,502.6 MiB。两者差额可能包含缓存、碎片和非PyTorch分配，仅凭这两个数不能确定全部原因。新代码同时记录allocated和reserved，以便区分这些来源。

新对照进程使用 `PYTORCH_ALLOC_CONF=expandable_segments:True`。这是[PyTorch 2.11官方说明](https://docs.pytorch.org/docs/2.11/notes/cuda.html#optimizing-memory-usage-with-pytorch-alloc-conf)针对动态分配尺寸的可选分配器机制；保持native后端和计算精度不变。这个运行时资源设置与学习机制分别记录，不据此宣称算法本身的墙钟加速。多组共享同一GPU，竞争程度也会变化。

## 复核与验证

CPU集成测试包含默认linear数值/RNG保持、非负critic的输入梯度与TD目标性质、checkpoint拒绝跨变换、地面以上精确等价、地面下恢复方向、真正的pre-step查询时刻、阶段继承与参数监控。严格继承不允许跨阶段偷偷改变展开长度。

执行端CPU已通过124项测试与143个子测试，所有本次修改的Python文件Ruff通过。
真实Isaac Lab两组64环境各完成两次完整训练更新：continuous采样19,200步，
AHAC+softplus实际H32/36、采样4,352步；执行动作对齐误差与非有限事件均为0。
手工地面下位姿的连续PhysX推进、实际mesh查询、恢复梯度和深度有限性检查通过。
有限步检查不用于正式验收；完整证据见[验证索引](results/zhang_followup_validation_20261029.json)。
