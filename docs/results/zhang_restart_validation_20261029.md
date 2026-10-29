# Zhang 重启前验证索引（2026-10-29）

## 当前状态

已完成真实PhysX环境检查、N256有限更新容量检查、最终v6静态几何300/400地图验证，以及一次整合回归：366项通过、2项跳过、522个子测试通过；37个修改过的Python文件Ruff通过。尚未启动新比较，也没有训练验收通过的结论。历史失败保留，启动证据在实际发生后追加。

工作树验证根目录：`/home/tong/tongworkspace/free/.worktrees/isaaclab-navrl/restart-controls-20261029/logs/restart_checks/`。容量检查根目录：`/home/tong/tongworkspace/free/isaaclab-navrl/logs/smoke/restart_controls_20261029/`。下表路径相对对应根目录。

## 分轮单元回归

| 回执 | 实际结果 | 范围 |
| --- | --- | --- |
| `initial_unit.log/xml` | 159通过、16失败 | 失败均为后来修正的 ground fixture；不回写为全绿。 |
| `controls_unit.log/xml` | 111通过、73个子测试 | 配置、runner、ground、critic state、actor-critic、DCOL；覆盖稳定档不可覆盖、最高同阶段恢复、Adam和目标高度身份。 |
| `all_unit.log/xml` | 345通过、14失败、2跳过、514个子测试 | 5个旧动态shape fixture及9个参考文件路径失败。 |
| `dynamic_geometry_unit.log/xml` | 10通过 | 修正旧动态fixture后的实际CPU/CUDA验证。 |
| `unchanged_reference_unit.log/xml`及manifest | 10通过 | 四个参考测试与主工作区逐字一致；因其相对路径依赖，从主测试目录调用，以显式PYTHONPATH测试本工作树源码。 |
| `geometry_proxy_unit_v1.log/xml` | 40通过，8.88 s | 初版出口代理。 |
| `geometry_proxy_v2/focused_unit.log/xml` | 43通过，9.35 s | 新增自交、部分遮盖和薄片；不能代替真实地图检查。 |
| `geometry_proxy_float64_v4/focused_unit.log/xml` | 45通过，32.20 s | 加入实测竖直面CPU/CUDA精度回归。 |
| `geometry_proxy_local_refine_v5/focused_unit.log/xml` | 45通过，35.45 s | 加入近壁局部选面复核；仍有实际400擦边出口问题。 |
| `geometry_proxy_guard_v6/focused_unit.log/xml` | 45通过，4.29 s | 加入出口数值余量；保持原场景门槛。 |
| `final_validation/all_unit.log/xml`及`manifest.json` | 366通过、2跳过、522个子测试，56.70 s | 一次整合执行；全部待提交源码。四个未修改参考测试使用本工作树逐字副本放在上游NavRL的同级临时测试目录，以保留原相对路径；显式PYTHONPATH指向本工作树，没有使用另一会话后来修改的主目录测试。 |
| `final_validation/ruff.log`及manifest | 37个Python文件通过 | 全部修改/新增Python文件，未自动修改代码。 |
| `queue_validation/unit.log/xml`及manifest | 15通过，0.09 s | 后续新增容量队列：12组fresh命令、布尔参数、当前阶段首update门控、显存/槽位边界、实际Zhang汇总身份；不启动仿真。队列脚本及测试两文件Ruff通过。 |

本地 stdlib 交叉检查另有协议24、CLI15、配置11、汇总12项通过；12组计划、60条阶段命令和11组Zhang配置通过检查，未启动仿真。配置身份缺失或不匹配均排除同预算比较；PPO单独汇总。上述不同轮次不能相加冒作一次整套通过。

## 真实环境集成

| JSON | 实际设置及结果 |
| --- | --- |
| `integration_proxy_v2/s0_dcol.json` | N64、300/0；通过0.5–2.5 m目标高度重采样、局部reset隔离、回合快照不变等检查。 |
| `integration_proxy_v2/s0_continuous.json` | N64、300/0；碰撞和上下越界不reset，继续运动；首次失败保持，invalid/timeout仍reset；地面下方梯度指向上方。 |
| `integration_proxy_v2/randomized_dcol.json` | N16、60动态；14策略步/140物理步，执行动作最大误差0、非有限事件0、深度160×120；扰动与局部reset检查通过。 |

初轮 `integration_proxy/s0_dcol.json` 发现目标高度未从观测接口返回，已补齐；失败文件保留。以上人工位姿及事件检查不计入训练成功回合，也不代表最终几何或100动态扰动阶段验收。

## 完整更新与容量检查

以下均为随机初始化的有限步 smoke，模型不会用于新比较。Zhang检查使用2 m搜索封顶及float64最近点修正之前的几何v1；这些容量数据不能冒作最终版本的性能认证。

| manifest | 设置 | 实际转移 | 采样峰值MiB | wrapper秒 | trainer秒 |
| --- | --- | ---: | ---: | ---: | ---: |
| `navrl_256_manifest.json` | PPO，N256，2更新 | 16,384 | 2,770 | 31.1876 | 27.6804 |
| `dcol256_h150_manifest.json` | DCOL，N256，H150，2更新 | 76,800 | 3,190 | 106.913 | 76.578 |
| `ahac256_h300_manifest.json` | AHAC，N256，initial H300/max H300，1更新 | 76,800 | 3,894 | 99.708 | 71.703 |
| `adam256_projection_manifest.json` | N256，H150，Adam+height_priority，诊断U1，1更新 | 38,400 | 3,110 | 77.819 | 50.023 |

三项Zhang检查均无非有限状态或梯度。AHAC主动从H300开始测容量，不是从H32自然增长的学习结果；Adam+projection是组合路径检查，不是两个单因素实验的效果结论。计时包含各自初始化、首轮编译或诊断，不能直接用来比较收敛速度；采样显存也不是后续400/120和扰动阶段的保证上限。

## 静态几何原始失败及精度定位

`geometry/static_300.json` 对25,770个地面以上面发现74个面两侧inside、43个面两侧outside。严格自交另有独立证据；保留原网格，采用明确命名的占据/出口训练代理，详见[几何审计](../migration/SIGNED_STATIC_GEOMETRY_AUDIT_20261029.md)。

`geometry_proxy_v2/static_300.json` 与 `geometry_proxy_diagnostic_v3/static_300.json` 对51,657个查询核对：25,875内部点，231次异常出口搜索；内部出口无效数0，方向复查错误数0，但再次closest查询残差最大0.00014355649 m，整体失败。

独立NumPy float64复核显示所导出的5个anchor都在原三角形上；Warp的两次单精度投影沿墙面漂移。原查询距墙约0.1 mm，最近点z向切向误差61.035 μm，距离高估约17%，错误向上分量约0.524、角度偏差最大31.62°。记录与复核入口见[精度证据](zhang_static_projection_precision_20261029.json)和[audit_static_anchor_precision.py](../../tests/audit_static_anchor_precision.py)。

修复对BVH选中的原三角面采用float64平面/边投影，结果存回float32，两种查询入口共用；距离不超过2 mm时另在2.01 mm AABB内复核选面。不改原PhysX、深度网格或首次事件判据，不宣称深部BVH全局最优。400实测还有一个约0.358 μm的擦边判定分歧；最终增加6.675720 μm候选clearance guard，拒绝数值不稳定出口，保持严格占据定义与100 μm探针。不能据这些近壁点认定此前全部训练上漂的原因。

最终 `geometry_proxy_guard_v6/static_300.json` 与 `static_400.json` 均写出 `status=passed`，完整证据和源码hash见[最终几何验证](zhang_restart_geometry_validation_20261029.json)。300/400分别检查51,657/64,354点，内部25,875/32,297，异常出口搜索219/415；无效内部出口和方向复查失败均为0，最大原网格残差均9.536743e-7 m。普通侧面与顶面、外部parity、地面、PhysX手工位姿与深度检查通过。

两图固定2560个z=-0.5 m压力查询的CUDA中位数为92.710/95.867 ms；同卡另有四个N1024训练进程，不能当作独占训练吞吐。原网格折叠/薄片74/43及138/58保留为实际地图事实，代理并非精确SDF。

## 待完成证据

实际源码提交、fresh筛选启动manifest仍须按真实回执更新。N256属于screening；正式验收仍需至少N1024、随机初始化、完整五阶段，最终400静态/100动态/扰动开启环境达到80%自然首次到达率。见[实验矩阵](zhang_restart_experiment_matrix_20261029.md)、[课程](../TRAINING_CURRICULUM.md)和[Recovery协议](../migration/RECOVERY_LADDER_20261029.md)。
