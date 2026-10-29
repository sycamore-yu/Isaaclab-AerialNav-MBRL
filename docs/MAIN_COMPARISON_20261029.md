# Main 整合与九组对照（2026-10-29）

当前唯一启动入口为 `scripts/run_comparison_queue.py`，计划为
`configs/experiments/zhang_main_comparison.json`。训练源码和调度器均来自干净的
`main`，启动时固定完整提交号；分支、提交或工作区状态不符时拒绝派发。
训练运行期间保持该源码不变。日志、检查点和容量记录写入忽略的 `logs/`。

## 整合范围

合并 `codex/navrl-zhang-curriculum` 的 `d5c4d5d` 与
`codex/zhang-restart-controls-20261029` 的 `a537960`，保留双方历史。
旧六条 wave2 训练已在完整优化更新后停止，均保存 `checkpoint_stopped.pt`；
停止更新分别为 DCOL truncated 19、PPO 293、per-step 15、Plateau 12、AHAC 21、RMS 9。
旧四条 signed-geometry 训练的停止检查点也保留。暂停清单、源代码 bundle、
工作树保留文件及清理回执位于 `logs/consolidation_main_20261029/`。
新实验从随机权重开始，不加载这些停止检查点。

- 碰撞与普通避障共用完整长方体、有限圆柱体、地面与静态距离；覆盖全部动态物体。
  无人机碰撞球半径为 0.3 m，策略观测仍为最近五个动态物体。
- 静态地形保留 restart 分支验证过的内部占据判断、原三角面出口和双精度投影修正。
  原 heightfield 存在折叠及内部面，内部距离为有验证出口的有符号代理，不能声称为精确 SDF。
- 动态有符号距离和 DCOL 保留中心/轴线的确定性非零逃逸次梯度。
- 高度默认每回合按导航目标 z 从 `[0.5, 2.5]` m 采样，
  `L_z = 1 - exp(-abs(z - z_target))`，权重 1。终止步使用重置前的目标。
- 短窗通过独立的 `velocity_loss_mode` 配置，不再使用同名环境变体；
  合格的碰撞、越界、超时短回合均计分，invalid 不计分。

## 当前矩阵

全部 seed=0、256 环境，每次自然报告至少 1024 个完成回合；五阶段、晋级阈值 80%，
无最大优化更新数。小批量记录为 screening，不能冒充原 1024 并行环境正式验收。
RMS、Adam、height-priority 等历史候选不进入此次九组计划。容量不足的组保持 queued。

| 组 | 速度监督 | H | 其他差异 |
|---|---|---:|---|
| unified_20hz（优先） | 原完整 2 秒窗口，40 步 | 150 | 物理、策略动作、两传感器均 20 Hz |
| dcol | 原完整 2 秒窗口，20 步 | 150 | DCOL 基线，物理 100 Hz、策略 10 Hz |
| step | 每策略步 | 150 | 仅改变速度监督，保留 DCOL |
| truncated | 完整窗口加合格终止短窗 | 150 | 仅改变速度监督，保留 DCOL |
| plateau | 完整窗口 | 150 | 仅改变调度器，patience=2 |
| shac | 完整窗口 | 32 | cost actor-critic |
| ahac | 完整窗口 | 初始32，16–300 | AHAC-derived 自适应展开 |
| navrl_ppo | 原 PPO 奖励 | 32 | 保留 NavRL 的观测、速度动作和频率 |
| continuous | 完整窗口 | 150 | 碰撞/越界不重置，普通有符号避障损失 |

Zhang 共同使用 target_exp、AdamW、weight_decay=.01、无 RMS、无梯度投影。
“20 Hz 策略”指动作推理频率；梯度更新仍发生在每次完整 rollout 后。
因此 H=150 在 20 Hz 下覆盖 7.5 秒，10 Hz 下覆盖 15 秒；该组不能单独证明 H 的因果影响。
20 Hz 将电机离散系数由 .43 调整为 `1-(1-.43)^5`，保持相同物理时间常数。

## 验证和启动

真实 16 环境、60 动态物体的 20 Hz 场景检查通过：20 个动作步对应20个物理步；
深度及 LiDAR 的更新时间差为 0.04999995–0.05000001 秒，电机系数 .93983078。
它是时钟及有限状态检查，不是学习成功率验收。合并后395项测试和522个子测试通过，队列追加20项检查通过，变更51个Python文件的Ruff通过。
64环境真实DCOL场景的目标高度重置、碰撞与有符号距离梯度检查通过。合并测试结果记录于
`logs/consolidation_main_20261029/pytest_final.log`；几何、时钟场景均有独立 JSON。

```bash
cd /home/tong/tongworkspace/free/isaaclab-navrl
.venv/bin/python scripts/run_comparison_queue.py \
  --plan configs/experiments/zhang_main_comparison.json \
  --output logs/curriculum/zhang_main_20261029 \
  --commit "$(git rev-parse HEAD)" --gpu 0 --device cuda:0 --slots 6 --reserve_mib 2048
```

`queue.json` 区分 running、queued 和失败状态，启动命令本身不代表已完成训练。
实际状态以运行回执和阶段 `training_result.json` 为准。TensorBoard 继续读取
`logs/curriculum/`，历史运行与此次新运行分目录保留。
