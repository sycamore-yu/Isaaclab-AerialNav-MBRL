# Baseline 场景对照（2026-10-30）

状态：原有六组实验已暂停，新对照已配置、未启动。暂停回执见
`logs/curriculum/2026-10-30_13-19-18_finetune_s1/pause_20261030.json`；原日志、冻结源码及 checkpoint 保留。

可执行计划：`configs/experiments/zhang_baseline_scene_controls_20261030.json`。

| Run | 固定场景 | 静态障碍 | 动态障碍 | 指定速度采样 m/s | 扰动 |
| --- | --- | ---: | ---: | --- | --- |
| baseline_fs1_speed_2_6 | fS1 / finetune_s1 | 300 | 60 | U(2, 6) | 开 |
| baseline_s0 | S0 / pretrain_s0 | 300 | 0 | 原 U(2.25, 9.75) | 关 |
| baseline_s1_dyn40 | S1 / pretrain_s1 | 300 | 40 | 原 U(2.25, 9.75) | 关 |
| baseline_fs1_dyn40 | fS1 / finetune_s1 | 300 | 40 | 原 U(2.25, 9.75) | 开 |

四组均从相同 seed=0 随机初始化，256 环境，固定各自场景，不晋级；每份自然结束回合报告至少1024回合，不设最大更新数。所有算法、损失、优化器与20 Hz时钟参数复用当前 `zhang_finetune_s1.json` 中的 baseline。既有 baseline（fS1/60、原速度范围）保留为共同参照，不覆盖旧结果。

这里将用户的两项要求分别作为速度对照与场景/障碍数量对照，避免同时改变两项因素；没有新增 fS1/40 + U(2,6) 联合组。指定速度为 reset 时采样的 `max_speed` 指令上限，不是实测飞行速度，也不是动态障碍速度。

`num_dynamic_obstacles` 覆盖同时作用于场景容量、课程有效数量和自然回合报告。速度范围保存在环境配置中，每次 `configure_scene()` 重建 reset 事件时重新传入。全局五阶段默认配置保持原值；父入口只允许在固定场景 Zhang screening 中传入这些覆盖项，不能据此获得标准课程验收。

后续恢复前需要将 runtime 修改提交为冻结源码，并使用新的输出目录运行现有 `scripts/run_comparison_queue.py --plan ... --commit <完整SHA>`。当前保持暂停；本文件不是启动回执或训练比较结果。
