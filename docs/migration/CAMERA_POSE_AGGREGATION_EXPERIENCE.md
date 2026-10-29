# 相机动态位姿批量读取的优化经验

## 可重复使用的结论

当许多全局动态障碍共享给大量相机环境时，先检查每帧是否逐刚体跨 Python /
PhysX Tensor API 读取位姿。本项目的主要瓶颈是多次小读取及调度开销。
将它们合并为一个官方 RigidBodyView，每帧一次 get_transforms()，再复用官方
Warp 表复制 kernel，能保留实际 PhysX 位姿和相机几何语义。

实现提交：`a4f2a66`。代码位于
`src/isaaclab_navrl/tasks/navigation/depth_camera.py`。
完整测量及逐项一致性证据见
[相机测量记录](../results/zhang_camera_batch_20261029.md)。

## 本次测量

RTX 4090；1024 环境；300 静态、60 动态障碍；160×120 原始深度；相机批次64；
噪声关闭；预热3次、采样25次；同时有五组正式训练，GPU非独占。

| 中位耗时 | 原逐刚体路径 | 聚合路径 |
| --- | ---: | ---: |
| 位姿更新 | 67.986573 ms | 1.343962 ms |
| 完整相机帧 | 119.015542 ms | 39.568573 ms |
| 完整环境策略步 | 477.864928 ms | 423.409269 ms |

对应位姿约50.6倍、相机约3.01倍，环境步中位耗时降低约11.4%。
这些数字仅对应上述并发负载；不能当作独占 GPU 吞吐或整次 BPTT 训练加速比。
分项独立同步计时，不能严格相加。

## 实现顺序

1. 保留官方逐 target 初始化。以各单体 view 的公开 prim_paths 取得真实刚体 owner。
2. 检查每个原 view 的 count 和目标宽度均为1，并检查 owner 唯一性。
3. 使用 Isaac Lab PhysxManager 管理的物理句柄创建一个聚合 RigidBodyView。
4. 读取新 view 实际返回的 prim_paths，校验数量、集合和唯一性，建立行到目标列的映射。
5. 每帧只调用一次 get_transforms()。映射为恒等时直接使用，否则 index_select 重排。
6. 使用官方 copy_mesh_transforms_to_table_kernel 写入原有 mesh 位姿表。
   保持同一 CUDA stream，保留转换缓冲区引用直到 kernel 使用结束。
7. 物理 view 失效时清理聚合引用和映射，重新初始化时重建。
   将实际路径及映射写入 world_mesh_info.mesh_transform_tracking。

不能假定请求列表或 glob 的顺序等于 PhysX view 的行顺序。
也不能直接把全局 M 个 mesh 作为一个 glob 塞入原 BaseMultiMeshRayCaster：
其部分初始化逻辑按每环境目标解释 M/N；共享全局障碍与这一假设不同。
本次只替换位姿刷新路径，不改变官方几何初始化、局部坐标或传感器输出定义。

## 正确性门槛

使用真实 Isaac Lab / PhysX 场景，比对旧官方路径与新路径的完整位置表、
四元数表、原始深度和相同池化结果。测试包括动态物体移动、相机三轴姿态、
局部环境重置、内参变化、像素噪声，以及故意反序请求 view 来检验映射。

实际已完成100和120动态障碍、65环境、64+1尾批次、各8个场景；
上述全部误差与 mismatch 均为0。证据：

- `logs/sim_to_real_checks/camera_160x120_aggregate_view_parity.json`
- `logs/sim_to_real_checks/camera_160x120_aggregate_view_120_parity.json`

零动态障碍必须单独走空集合路径；不应为其创建空聚合物理 view。
新增课程使用这一条件，需在新课程 smoke 中继续验证。

## 相关优化与边界

相机批次16→64减少调度开销，深度分辨率和池化形状不变。
未命中 inf 初始化并入 raycast kernel，减少一次 kernel 调用。
这些改动分别验证；不能把总性能差值全归因于其中一项。
批次可能改变噪声 RNG 的取样排列，对照实验应统一并记录批次。

优化时优先记录分项时间、定位实际接口调用次数，再修改最小路径。
每次改动均先通过实际传感器 parity，再测整帧和整环境步，最后观察完整训练。
不要用纯动力学微基准替代真实训练，也不要为吞吐放宽数值一致性标准。

## 官方 API

- [Isaac Lab 3.0 PhysxManager](https://isaac-sim.github.io/IsaacLab/v3.0.0/source/api/lab_physx/isaaclab_physx.physics.html)
- [Omni Physics Tensor API](https://docs.omniverse.nvidia.com/kit/docs/omni_physics/108.0/extensions/runtime/source/omni.physics.tensors/docs/api/python.html)
