# Copyright (c) 2025 Fan Yang, Robotic Systems Lab, ETH Zurich
# Licensed under the MIT License (see licenses/SRU.txt).
"""SRU-LSTM-Gated cell from leggedrobotics/sru-pytorch-spatial-learning.

Source: network/lstm_sru_gate.py at d537910684d5c73af2acd43db6c8b001b593fbc0.
Only the single-step cell is needed by the existing BPTT rollout.
"""

import torch
from torch import nn


class LSTMSRUGateCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.linear_all = nn.Linear(input_size + hidden_size, 4 * hidden_size)
        with torch.no_grad():
            self.linear_all.bias[hidden_size : 2 * hidden_size] = 1.0 + torch.randn(hidden_size)
        nn.init.orthogonal_(self.linear_all.weight)
        self.transform_gate = nn.Linear(input_size, hidden_size)
        nn.init.orthogonal_(self.transform_gate.weight)

    def forward(self, x, h, c):
        gates = self.linear_all(torch.cat((x, h), dim=1))
        tx = self.transform_gate(x)
        i, f, o, g = gates.split(self.hidden_size, dim=1)
        i, f, o = i.sigmoid(), f.sigmoid(), o.sigmoid()
        candidate = torch.tanh(tx * g)
        f = i * (1.0 - (1.0 - f) ** 2) + (1.0 - i) * f**2
        c_next = f * c + (1.0 - f) * candidate
        return o * torch.tanh(c_next), c_next
