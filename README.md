# NavRL · Isaac Lab 3.0 迁移

本目录保存已批准的迁移决策、原版控制器与
Isaac Lab 3.0 原生多旋翼接口的比较、电机单元测试及 PhysX 闭环施力对照。
完整的仿真环境与 RSL-RL 训练尚未迁移。

- [迁移方案、冲突与验收](docs/NAVRL_ISAACLAB_3_MIGRATION.md)
- [架构决策记录（ADR 0001）](docs/adr/0001-port-navrl-with-original-task-semantics.md)
- [PhysX 闭环动力学差异报告](docs/PHYSX_CLOSED_LOOP_COMPARISON.md)
- [可重复的 PhysX 对照脚本](tests/physx_closed_loop_compare.py)
- [RotorGroup 与 Thruster 的实际单元测试](tests/test_thruster_parity.py)
- [原版 RotorGroup 批量执行与零延迟推力接口测试](tests/test_original_rotor_integration.py)
- [旧版与 RSL-RL Beta 动作分布的对照测试](tests/test_rsl_beta_contract.py)

基线源码位于兄弟目录 `../NavRL`，Isaac Lab 3.0
的参考源码位于 `/home/tong/tongworkspace/reference_repos/IsaacLab-v3.0.0`。
