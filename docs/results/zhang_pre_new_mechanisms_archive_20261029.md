# 新机制前的实验冻结与归档（2026-10-29）

## 基线

代码基线为 `f6d5fd4`，标记为
`baseline/zhang-pre-new-mechanisms-20261029`。这是旧对照的结束点。
新实验必须在新机制实现并提交后从随机初始化开始，不加载本次归档的模型。
同一新实验内部的课程继承和已授权的同阶段退化恢复另行记录。

## 停止与保留范围

六组仍在运行的训练已向核实过的训练子进程发送 SIGINT，父子进程均已退出。
continuous 对照此前在第 53 次更新完成后停止。停止后 GPU 计算进程列表为空。
归档时决定不再启动旧 `variant=truncated_window` 对照；保留其原始证据，不将早期单种子结果表述为最终收敛结论。后续用户重启的是DCOL几何下独立的速度损失模式，见 [新对照规范](../ZHANG_TRUNCATED_PPO_CONTROLS_20261029.md)，不改变本归档的历史配置和结果。

| 旧实验 | 已完成更新 | 历史最好首次到达率 | 最好检查点更新 |
| --- | ---: | ---: | ---: |
| zhang_four_step_s0_20261029 | 62 | 8.92587% | 24 |
| zhang_four_continuous_s0_20261029 | 53 | 4.14228% | 41 |
| zhang_four_dcol_s0_20261029 | 59 | 15.52588% | 37 |
| zhang_four_truncated_window_s0_20261029 | 61 | 8.40336% | 24 |
| zhang_step_h5_s0_20261029 | 47 | 14.76474% | 20 |
| zhang_step_h5_lr0p0005_s0_20261029 | 16 | 6.63415% | 9 |
| zhang_step_h5_lr0p0001_s0_20261029 | 8 | 0.52204% | 7 |

所有实验均停留在旧 pretrain_s1，未达到验收。表中的已完成更新数来自 updates.jsonl；
最好成功率与检查点来自训练统计和实际可加载的检查点，不代表最后一次更新的权重。

归档包括所有实际保存的 best/latest 权重、训练报告、更新指标、TensorBoard、
配置、启动命令及操作停止记录。旧训练器在 SIGINT 下记录 KeyboardInterrupt/failed，
原始记录保持不改；operator_stop_new_mechanisms.json 说明这是用户要求的实验冻结。
旧训练器未在中断时导出内存权重，因此不得把最后一行更新编号当成最终模型编号。
各 latest 检查点仍为更新 1，best 检查点编号见表。

## 可复核归档

- 文件：`logs/archive/20261029_pre_new_mechanisms/experiments.tar.gz`
- 大小：81,798,146 bytes
- SHA256：`28e18aa6ff09ee1590b7317a41e35cd46f1d7e00341d986912a48efc2279ffdb`
- 索引：同目录 `manifest.json`
- 校验文件：同目录 `experiments.sha256`

所有保留检查点已用 torch.load(weights_only=True) 验证可加载。
原日志目录保留用于 TensorBoard 历史检索；“清理”指退出旧计算进程和监视进程，
释放 GPU，不删除唯一实验数据。后续实验使用全新的目录和已提交代码。
