# 默认 fS1/40 与 S1 学习率试验暂停

用户指定后续默认使用 `baseline_fs1_dyn40`，S1 与 fS1 均默认40个动态障碍。

`STAGE_CONFIG` 保留五阶段顺序，S1/fS1 的动态数量改为40。默认 Zhang 环境为固定 `finetune_s1`、300静态、40动态、开启扰动。队列不指定 `--plan` 时使用 `configs/experiments/zhang_baseline_fs1_dyn40.json`，单组256环境、seed0、随机初始化；直接训练省略 `--stage` 时选择同一场景并在无checkpoint时从随机权重启动。显式阶段继续使用原来的checkpoint/`--fresh_stage`验证；正式五阶段验收入口仍从S0开始。训练统计沿用至少1024个自然结束回合。

历史降速对照 `baseline_fs1_speed_2_6` 显式保留60个动态障碍，避免复跑原来的速度单因素对照时同时改变数量。已有日志、冻结源码、checkpoint保持原实验身份。损失、优化器、物理后端、循环模型默认值不在本次选择范围。

当前学习率研究目录：`/home/tong/tongworkspace/free/.worktrees/isaaclab-navrl/lr-schedules-2000-20261030/.optim-agent-runs/lr-cos1000-constant-2000-20261030`。

| 暂停试验 | 累计更新 | 本轮更新 | 停止checkpoint |
|---|---:|---:|---|
| S1/40 constant_lr | 484 | 184 | `baseline_s1_dyn40/constant_lr/train/checkpoint_stopped.pt` |
| S1/40 cosine_1000 | 485 | 185 | `baseline_s1_dyn40/cosine_1000/train/checkpoint_stopped.pt` |

两条训练均在更新边界退出；checkpoint已在CPU加载并验证模型张量有限；研究按 `pruned` 保留提前结束的记录。具体PID、checkpoint SHA256与暂停原因见研究目录的 `pause_s1_20261031.json`。fS1 的两条原训练PID继续运行；其冻结训练源码哈希已重新验证。

协调进程只将 `baseline_fs1_dyn40` 纳入后续任务，不自动恢复S1。fS1首批2000更新和终点评测结束后，按原计划继续一个学习率单因素试验，同时恢复权重筛选候选（避障权重2.625）的512更新独立种子确认。旧权重研究仍保留24次训练启动与12小时活动时间上限；S1的历史结果及候选保留为暂停材料。

验证：`PYTHONPATH=src PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest -q --tb=short tests`：483 passed、1 skipped、522 subtests passed。Ruff及`git diff --check`通过。测试覆盖实际CLI解析与配置到启动边界；未额外启动GPU训练。回归中同步了60障碍旧断言，并修正了既有Hydra stub与传感器mixin方法位置的测试夹具。
