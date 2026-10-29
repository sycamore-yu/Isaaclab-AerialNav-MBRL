# 160×120 相机批次测量（2026-10-29）

## 改动与使用范围

共同环境配置新增 `depth_camera_batch_size`，默认保持 16。`benchmark_camera.py`
和 `tests/camera_parity.py` 接受 `--camera_batch_size`。重复调用
`configure_scene()` 会保留显式选择的批次。原始 160×120 射线、相机视场、
噪声处理及 12×16 池化均未改变。

正在运行的四组正式对比使用批次 16；本次测量不更改这些训练的配置。

## 一致性检查：通过

真实 Isaac Lab 3.0 / PhysX 场景使用 65 个环境，覆盖 64 个相机的完整批次与
1 个相机的尾批次。七个场景包括初始姿态、三轴姿态变化、全部动态物体移动、
局部重置、池化前像素噪声、局部内参变化、恢复共享内参。

与官方 `MultiMeshRayCasterCamera` 的原始深度及同算法池化结果相比，全部场景
`max_abs_error = 0`，`mismatches = 0`。局部重置保留其他环境的图像与帧计数，
缓存局部射线完全一致。

原始证据：`logs/sim_to_real_checks/camera_160x120_batch64_parity.json`。

## 性能测量

条件：RTX 4090，1024 环境，300 静态 / 60 动态障碍，160×120 深度，名义动力学，
像素噪声关闭。每项预热 3 次、采样 25 次；先测批次 16，再测 64。
两次测量期间另有四组正式 Zhang 训练并发，**GPU 非独占**。

| 指标 | 批次 16 | 批次 64 |
| --- | ---: | ---: |
| 固定场景相机中位数 / ms | 129.542 | 96.099 |
| 固定场景相机均值 / ms | 143.937 | 98.649 |
| 完整环境步中位数 / ms | 458.858 | 432.814 |
| 完整环境步均值 / ms | 460.165 | 434.320 |
| 完整环境步 p95 / ms | 497.390 | 466.331 |

在本次并发条件下，相机中位耗时降低约 25.8%，完整环境步中位耗时降低约 5.7%。
这不是独占 GPU 吞吐，也不是完整 BPTT 更新的加速比；测量顺序和其他训练的工作量
会影响差值。不同批次改变像素噪声 RNG 的取样排列，分布保持不变，因此比较实验
应显式记录并统一批次。

原始样本与测量时间：

- `logs/benchmark/camera_d160_b16_fourtrain_1024.json`
- `logs/benchmark/camera_d160_b64_fourtrain_1024.json`

三份修改文件的 Ruff 与 `git diff --check` 通过。

## 后续：合并未命中初始化

相机射线 kernel 现在直接为未命中像素写入与官方填充 kernel 相同的 `inf`，
删除每批独立的填充调用。命中分支、射线坐标变换、几何查询与池化不变；
未激活环境仍直接返回。每批 Warp kernel 调用数从 4 减到 3。

该改动另行完成 65 环境、批次 64+1 的同一七场景真实 parity 检查，全部原始
深度与池化输出误差仍为 0，局部重置保持其他环境结果。
证据：`logs/sim_to_real_checks/camera_160x120_batch64_miss_fusion_parity.json`。

这个合并尚未测出独立性能增益，不能将前述 16→64 的测量归因于它。
原四组正式运行与后续 h5 运行均在改动前已载入相机代码。

## 后续：聚合动态物体的 PhysX 位姿读取

在五组正式训练并发的 profile 中，旧路径逐个读取 60 个动态物体位姿，
仅 `_update_mesh_transforms()` 的中位耗时已达 67.987 ms。相机现在保留官方
逐 target 的初始化与几何表，从旧单体 view 的公开 `prim_paths` 取得刚体
owner 路径，再创建一个聚合 view。映射依据新 view 的真实 `prim_paths`，
不假定请求列表或 glob 的顺序。每帧一次 `get_transforms()`，按映射重排后
复用官方 `copy_mesh_transforms_to_table_kernel` 广播到原有的完整表。

静态 mesh、相机射线、几何局部坐标和深度变换不变。物理 view 失效后清理
引用与映射；重新初始化时重建。实际路径和列／行映射保存在
`world_mesh_info.mesh_transform_tracking`。

### 正确性

100 与 120 个动态物体分别完成 65 环境、64+1 批次的八场景检查。
每个场景将完整位置／四元数表与原官方逐 target 更新逐元素比较，
另将 raw／pooled 深度与官方传感器比较。第八个场景故意反序请求聚合 view，
再依据实际返回路径重建映射。两次检查的全部位姿表、原始深度与池化最大误差
均为 0，mismatch 均为 0。

- `logs/sim_to_real_checks/camera_160x120_aggregate_view_parity.json`
- `logs/sim_to_real_checks/camera_160x120_aggregate_view_120_parity.json`

### 性能

两次 profile 均使用批次 64、1024 环境、300 静态 / 60 动态障碍、无像素噪声，
预热 3 次、采样 25 次，五组正式训练持续并发，GPU 非独占。

| 中位耗时 / ms | 逐物体读取 | 聚合读取 |
| --- | ---: | ---: |
| mesh 位姿更新 | 67.987 | 1.344 |
| 固定场景完整相机 | 119.016 | 39.569 |
| 完整环境步 | 477.865 | 423.409 |

本次完整环境步中位耗时降低约 11.4%。分项独立计时并同步，不能严格相加；
这些也不是独占 GPU 或完整 BPTT 更新的加速比。当前五组的现有阶段在修改前
已导入旧代码；后续新进程使用该优化，默认相机批次仍为 16。

- `logs/benchmark/camera_d160_b64_fusion_fivetrain_profile.json`
- `logs/benchmark/camera_d160_b64_aggregate_fivetrain_profile.json`

API 依据：

- [Isaac Lab 3.0 PhysxManager](https://isaac-sim.github.io/IsaacLab/v3.0.0/source/api/lab_physx/isaaclab_physx.physics.html)
- [Omni Physics Tensor API](https://docs.omniverse.nvidia.com/kit/docs/omni_physics/108.0/extensions/runtime/source/omni.physics.tensors/docs/api/python.html)
