# SHAC 终止、截断与速度监督修复

参考只读上游 DiffAero `291ea14` 的 `algo/SHAC.py:record_loss`、
`bootstrap_tdlambda` 和避障环境的重置前观测接口。

- 真正 terminated（碰撞、高度越界、无效状态）：next value=0，同时终止TD递推。
- timeout/truncated：使用重置前的状态、目标、深度、RNN、延迟队列、速度历史、
  回合步数及随机参数计算冻结Critic价值；TD递推不跨新回合。
- 普通H末端：冻结Critic参数，保留输入状态和记忆对actor的梯度。
- 时域末端与真正终止同时发生时，真正终止优先，不能添加未来价值。
- 默认SHAC使用per_step速度成本，覆盖每条转移，避免H32/20Hz不足2秒窗口。

DirectRLEnv复用done缓冲区；在采样后立即复制terminated和timeout，避免后续步改写
反向传播所保存的布尔掩码。旧失败检查保留在logs/20hz_default_20261029/。

验证：可控runner用不同的重置前后位置、目标和深度，验证无重置状态串入、真正终止
零bootstrap、超时及普通末端的输入梯度、TD目标，以及真正终止与H末端重合。
真实PhysX为16环境/60动态障碍，2次更新、256转移，注入2次吸收终止和62次超时，
吸收bootstrap绝对值总和为0、per_step覆盖全部样本、无非有限状态和梯度。
真实注入检查不是导航成功率验收。新SHAC训练必须在干净main提交后随机初始化。

用户随后要求这些语义也用于AHAC，并修复风险聚合、时域控制、速度监督和Critic预算；
这些改动在下一独立提交完成，均验证后再启动新组。
