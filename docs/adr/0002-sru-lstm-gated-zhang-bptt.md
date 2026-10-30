---
status: accepted
date: 2026-10-30
---

# SRU-LSTM-Gated 首轮仅接入 Zhang BPTT

用户选择在 Zhang 的 CNN + GRU 路径新增 SRU-LSTM-Gated，首轮只支持 BPTT。
SHAC/AHAC 的价值网络目前读取 GRU 的隐藏状态。
SRU 同时具有隐藏状态和记忆状态，接入价值网络需要另定输入范围。
因此，本轮对 SRU 与 SHAC/AHAC 的组合明确报错。
该范围保留当前 BPTT 的训练目标，同时避免引入价值网络变化。
