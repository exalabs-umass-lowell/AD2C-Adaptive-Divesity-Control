#  Copyright (c) 2024.
#  ProrokLab (https://www.proroklab.org/)
#  All rights reserved.

import torch
from torch import nn
import numpy as np

from AD2C.utils import clamp_preserve_gradients


def squash(loc, action_spec, clamp: bool):
    """
    Squash loc into the action_spec bounds
    """
    f = clamp_squash if clamp else tanh_squash
    return f(loc, action_spec)


def tanh_squash(loc, action_spec):
    tanh_loc = torch.nn.functional.tanh(loc)
    scale = (action_spec.space.high - action_spec.space.low) / 2
    add = (action_spec.space.high + action_spec.space.low) / 2
    return tanh_loc * scale + add


def clamp_squash(loc, action_spec):
    loc = clamp_preserve_gradients(loc, action_spec.space.low, action_spec.space.high)
    return loc


import torch
from torch import nn
import numpy as np

class LowPassFilter(nn.Module):
    def __init__(self, sampling_period: float, cutoff_frequency_rad_per_s: float, initial_value: float = 0.0):
        super().__init__()
        # The parameter names now match their usage below
        self.coefficient = np.exp(-sampling_period * cutoff_frequency_rad_per_s)
        self.register_buffer("previous_value", torch.tensor(initial_value, dtype=torch.float32))

    # Corrected indentation
    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        output = self.coefficient * self.previous_value + (1 - self.coefficient) * input_tensor
        self.previous_value = output.detach() # Update state without tracking gradients
        return output

class HighPassFilter(nn.Module):
    def __init__(self, sampling_period: float, cutoff_frequency_rad_per_s: float):
        super().__init__()
        dt = sampling_period
        wc = cutoff_frequency_rad_per_s
        self.a1 = dt * wc + 2.0
        self.b1 = dt * wc - 2.0
        
        # Register buffers to correctly manage state
        self.register_buffer("u_prev", torch.tensor(0.0, dtype=torch.float32))
        self.register_buffer("y_prev", torch.tensor(0.0, dtype=torch.float32))

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        output = (1.0 / self.a1) * (-self.b1 * self.y_prev + 2.0 * (input_tensor - self.u_prev))
        self.u_prev = input_tensor.detach()
        self.y_prev = output.detach()
        return output