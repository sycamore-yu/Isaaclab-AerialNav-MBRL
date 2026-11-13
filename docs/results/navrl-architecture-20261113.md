# NavRL 架构重构验证 — 2026-11-13

起点为 `625afcef08706b8d76af711ad087e051d48ea4a6`。
修改范围为本地开发仓库 `main`，不涉及公开发布仓库。

## 实现

| 边界 | 实现 | 保留的语义 |
|---|---|---|
| 任务状态 | `tasks/navigation/task.py::NavigationTask` | 目标、首次到达、最终结果、局部重置、回合编号；Warp 缓冲区地址不变 |
| 观测来源 | `tasks/navigation/context.py::ObservationContext` | `policy` 与 `privileged` 分开；网络输入数值、维度及检查点格式不变 |
| 重置前数据 | `tasks/navigation/transition.py::TransitionBatch` | 保存旧回合状态、动作、事件和参数；自动重置不覆盖快照 |
| 梯度计算入口 | `TransitionPublisher.set_consumer()` | 一次消费后清除回调；训练器保留可微代理图，环境不持有损失实现 |
| 几何查询 | `tasks/navigation/geometry.py::query_clearance` | 米制带符号净间距；机体半径只减一次；动态障碍按查询时间预测 |

训练、推理、PointNet 录制、Torch/Warp 环境和现有仿真检查脚本均使用新接口。
历史 DCOL 对照保留原数学定义，只更新查询时间接口。
接口用法见 `docs/NAVIGATION_ARCHITECTURE.md`。

以下内容没有修改：实验默认配置、课程参数、控制频率、控制器、电机模型、
损失公式、成功规则、检查点格式、原有实验记录及运行中的实验工作树。
首次进入目标后的等待仍为原有规则：超过 5 s 且当前仍在目标区时截断。
已发生失败的回合不会因后续到达而改记成功。

## 数值等价

修改训练器前，保存原实现的两组确定性短程结果。
两组均使用 `DepthEnv` 解析执行夹具、真实 `ZhangTrainer`、seed 7、2 个环境、
4 步展开、0.1 s 速度窗口及 2 次参数更新。

| 损失时序 | 全部网络参数最大差值 | 归一化参数 | 非有限梯度 |
|---|---:|---|---:|
| `deferred` | 0 | 逐项完全相同 | 0 |
| `pre_reset` | 0 | 逐项完全相同 | 0 |

比较使用 `rtol=0, atol=0`。结果为
`logs/architecture_20261113/golden_comparison.json`。
这验证了该短程配置的数值等价，不是新的长期训练性能结果。

## 真实仿真

所有检查复用仓库 `.venv`，并采用项目已有的运行环境变量。

### Newton/Warp 环境契约

4 个环境、300 个静态障碍、8 个动态障碍，Zhang 20 Hz。
通过延迟缓存、扰动、局部重置、超时自动重置、界面重置、越界、无效事件、
观测及终止快照生命周期、奖励缓存和实际 CUDA 图复用检查。
结果为 `logs/architecture_20261113/warp_zhang.json`。

```bash
OMNI_KIT_ACCEPT_EULA=YES PYTHONPATH=src OMP_NUM_THREADS=1 \
  MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python \
  tests/warp_environment_contract.py --method zhang --variant dcol \
  --timing unified_20hz --steps 4 \
  --output logs/architecture_20261113/warp_zhang.json
```

### Newton/Torch 环境检查

4 个环境、16 个控制步、8 个动态障碍。
状态和观测保持有限，共享障碍不受单个无人机重置影响。
结果为 `logs/architecture_20261113/torch_navrl.json`。

```bash
OMNI_KIT_ACCEPT_EULA=YES PYTHONPATH=src OMP_NUM_THREADS=1 \
  MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/smoke.py \
  --method navrl --num_envs 4 --steps 16 --dynamic_obstacles 8 \
  --frontend torch physics=newton_mjwarp \
  --output logs/architecture_20261113/torch_navrl.json
```

### 实际 BPTT 训练

Newton/Warp、cuRoboV2、4 个环境、4 步展开、2 次更新。
完成 32 个环境步并生成 `checkpoint_final.pt`。
非有限状态和非有限梯度均为 0，执行动作对齐误差为 0。
两次更新的梯度范数分别为 `0.0458844081` 和 `0.0439455807`。
结果为 `logs/architecture_20261113/bptt_warp/training_result.json`。

```bash
OMNI_KIT_ACCEPT_EULA=YES PYTHONPATH=src OMP_NUM_THREADS=1 \
  MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/train_zhang.py \
  --log_dir logs/architecture_20261113/bptt_warp \
  --num_envs 4 --stage pretrain_s0 --max_iterations 2 --timesteps 4 \
  --velocity_window_s 0.1 --camera_batch_size 4 \
  --frontend warp physics=newton_mjwarp
```

该训练仅验证完整训练链路。它没有达到课程晋级条件，也没有启动长期训练。
重跑时应使用新的日志目录。

### PointNet 原入口

同一训练入口加 `presets=pointnet`，使用 4 个环境、4 步展开和 1 次更新。
完成 16 个环境步并保存检查点，非有限状态和非有限梯度均为 0。
点云通过 `ObservationContext.policy["points"]` 进入原 PointNet–GRU 模型。
结果为 `logs/architecture_20261113/pointnet_warp/training_result.json`。

```bash
OMNI_KIT_ACCEPT_EULA=YES PYTHONPATH=src OMP_NUM_THREADS=1 \
  MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/train_zhang.py \
  presets=pointnet --log_dir logs/architecture_20261113/pointnet_warp \
  --num_envs 4 --stage pretrain_s0 --max_iterations 1 --timesteps 4 \
  --velocity_window_s 0.1 --camera_batch_size 4 \
  --frontend warp physics=newton_mjwarp
```

## 测试与审查记录

最终相关回归为 **142 passed**，用时 47.75 s。
结果保存于 `logs/architecture_20261113/final_regression.xml` 和同名 `.txt`。
其中新增 17 项测试，覆盖状态所有权、部分重置、观测命名空间、推理输入隔离、
终止快照、一次性消费、代理梯度、距离数值/梯度、动态查询时间和共享几何内存形状。
现有训练器回归覆盖 GRU、SRU、BPTT、SHAC/AHAC 端点和检查点推理。

原始提交的测试副本位于 `logs/architecture_20261113/baseline_source/`。
该副本通过 `git archive` 创建，没有切换或修改任何实验工作树。
原始结果、重构结果及测试日志均保留在同一验证目录。

原始提交的 4 个旧测试文件复跑结果为 **29 failed、29 passed、95 subtests passed**。
失败原因包括测试配置缺少 `perception_cfg`、旧课程数量预期、
Hydra 测试替身缺少 `resolve_task_config`、环境测试替身缺少 `apply_training_stage`。
结果保存于 `baseline_known_failures.xml` 和 `.txt`。
本次没有修改这些旧预期，也没有为通过旧测试改变研究默认参数。

全量初跑结果为 **607 passed、32 failed、1 skipped、511 subtests passed**。
其中 3 项为本次新增的零时刻查询回归：查询当前距离时不应依赖障碍速度字段。
已修复并在最终 142 项回归中通过，原有缓存梯度及穿透逃逸梯度断言保持不变。
逐项比对 JUnit 的测试标识与失败消息后，其余失败与原始提交完全相同。
汇总为 `logs/architecture_20261113/regression_assessment.json`。
最后一次修复后复跑相关测试，没有再次运行完整测试集。

本次改动的 Ruff 检查通过。全仓库另有 3 处既有 `I001` 导入排序问题，
分别位于 `signed_static_scene.py`、`test_geometry_contract.py` 和
`test_signed_terrain_geometry.py`。这 3 个文件与起点提交的 Git 对象哈希相同。

审查发现的历史 DCOL 方法签名遗漏已修复，并加入数值及梯度等价测试。
其余审查重点为按行重置、Warp 指针稳定性、传感器采样次数、终止数据生命周期、
动作/参数快照及旧检查点输入布局。
代理工具未能启动，代码审查在当前会话内完成。
