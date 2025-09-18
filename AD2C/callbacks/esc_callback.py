import os
import pickle
from typing import List, Dict,Any, Callable

from PIL import Image
from matplotlib import pyplot as plt
import numpy as np
import io
import torch
import wandb
from tensordict import TensorDictBase, TensorDict
from typing import List, Dict, Union

from benchmarl.experiment.callback import Callback
from AD2C.models.het_control_mlp_empirical import HetControlMlpEmpirical
from AD2C.snd import compute_behavioral_distance

import numpy as np


from .utils import *


class esc:
    # cutoff_frequencies in rad/s
    def __init__(
        self,
        sampling_period,
        disturbance_frequency,
        disturbance_magnitude,
        integrator_gain,
        initial_search_value,
        high_pass_cutoff_frequency,
        low_pass_cutoff_frequency,
        use_adapter,
    ):
        self.dt = sampling_period  # in [s]
        self.disturbance_frequency = disturbance_frequency  # in rad/s
        self.disturbance_magnitude = disturbance_magnitude
        # negative for gradient descent
        self.integrator_gain = integrator_gain
        self.initial_search_value = initial_search_value
        # boolean, true or false (use the adapter or not)
        self.use_adapter = use_adapter

        self.high_pass_filter = High_pass_filter_first_order(
            sampling_period, high_pass_cutoff_frequency, 0, 0
        )
        self.low_pass_filter = Low_pass_filter_first_order(
            sampling_period, low_pass_cutoff_frequency, 0
        )
        # current phase of perturbation
        self.wt = 0
        # integrator output
        self.integral = 0
        # estimated second moment
        self.m2 = 0
        self.b2 = 0.9
        # to prevent from dividing by zero
        self.epsilon = 1e-8

        return

    def update(self, cost):
        high_pass_output = self.high_pass_filter.apply(cost)
        low_pass_input = high_pass_output * np.sin(self.wt)
        low_pass_output = self.low_pass_filter.apply(low_pass_input)
        gradient = 0

        if self.use_adapter:
            self.m2 = self.b2 * self.m2 + (1 - self.b2) * np.power(low_pass_output, 2)
            if np.sqrt(self.m2) > 1:
                gradient = low_pass_output / (np.sqrt(self.m2) + self.epsilon)
            else:
                gradient = low_pass_output * np.sqrt(self.m2)

        else:
            gradient = low_pass_output

        self.integral += self.integrator_gain * gradient * self.dt
        setpoint = self.initial_search_value + self.integral
        output = self.disturbance_magnitude * np.sin(self.wt) + setpoint
        perturbation = self.disturbance_magnitude * np.sin(self.wt)

        # update wt
        self.wt += self.disturbance_frequency * self.dt
        if self.wt > 2 * np.pi:
            self.wt -= 2 * np.pi

        return (
            output,
            high_pass_output,
            low_pass_output,
            (np.sqrt(self.m2) + self.epsilon),
            gradient,
            setpoint,
        )

class ExtremumSeekingController(Callback):
    """
    Implements an Extremum Seeking Controller to optimize a performance metric by
    periodically adjusting the desired SND.
    """
    def __init__(
        self,
        control_group: str,
        initial_snd: float,
        
        # ESC parameters
        dither_magnitude: float, # Renamed from dither_amplitude for clarity
        dither_frequency_rad_s: float, # Explicitly state units
        integral_gain: float,
        high_pass_cutoff_rad_s: float,
        low_pass_cutoff_rad_s: float,
        use_adapter: bool = True,
        sampling_period: float = 1.0
        ):

        super().__init__()
        self.control_group = control_group
        
        self.initial_snd = initial_snd

        self.esc_params = {
            "sampling_period": sampling_period,
            "disturbance_frequency": dither_frequency_rad_s,
            "disturbance_magnitude": dither_magnitude,
            "integrator_gain": integral_gain,
            "initial_search_value": initial_snd,
            "high_pass_cutoff_frequency": high_pass_cutoff_rad_s,
            "low_pass_cutoff_frequency": low_pass_cutoff_rad_s,
            "use_adapter": use_adapter,
        }
        # Controller state variables
        self.model = None
        self.controller = None

    def on_setup(self):
        """Initializes the controller and logs hyperparameters."""
        hparams = {
            # "controller_type": "ExtremumSeeking_v2",
            "control_group": self.control_group,
            **self.esc_params
        }
        self.experiment.logger.log_hparams(**hparams)

        if self.control_group not in self.experiment.group_policies:
            print(f"\nWARNING: Controller group '{self.control_group}' not found. Disabling controller.\n")
            return

        policy = self.experiment.group_policies[self.control_group]
        self.model = get_het_model(policy)

        if isinstance(self.model, HetControlMlpEmpirical):
            print(f"\n✅ SUCCESS: Extremum Seeking Controller initialized for group '{self.control_group}'.")
            self.controller = esc(**self.esc_params)
            self.model.desired_snd[:] = float(self.initial_snd)
        else:
            print(f"\nWARNING: A compatible model was not found for group '{self.control_group}'. Disabling controller.\n")
            self.model = None

    def on_evaluation_end(self, rollouts: List[TensorDictBase]):
        if self.model is None or self.controller is None:
            return
        logs_to_push = {}
        
        # 1. Collect rewards + compute actual diversity for logging
        episode_rewards = []
        with torch.no_grad():
            for r in rollouts:
                reward_key = ('next', self.control_group, 'reward')
                total_reward = r.get(reward_key).sum().item() if reward_key in r.keys(include_nested=True) else 0
                episode_rewards.append(total_reward)

        if not episode_rewards:
            print("\nWARNING: No episode rewards found. Cannot update controller.\n")
            self.experiment.logger.log(logs_to_push, step=self.experiment.n_iters_performed)
            return

        reward_mean = np.mean(episode_rewards)
        cost = -reward_mean  # Assuming we want to maximize reward, so cost is negative reward
        
        # 2. Call the core ES function
        (
            uk, 
            hpf_out, 
            lpf_out, 
            m2_sqrt, 
            gradient, 
            setpoint
        ) = self.controller.update(cost)
        
        # 3. Update diversity parameter
        previous_snd = self.model.desired_snd.item()
        self.model.desired_snd[:] = torch.clamp(torch.tensor(uk), min=0.0)

        print(f"[ESC] Updated SND: {self.model.desired_snd.item()} "
              f"(Reward: {reward_mean:.3f}, Update Step: {uk - previous_snd:.4f})")

        # 4. Logging
        logs_to_push.update({
            "esc/mean_reward": reward_mean,
            "esc/cost": cost,
            "esc/diversity_output": uk,
            "esc/diversity_setpoint": setpoint,
            "esc/gradient_estimate": gradient,
            "esc/hpf_output": hpf_out,
            "esc/lpf_output": lpf_out,
            "esc/m2_sqrt": m2_sqrt,
            "esc/update_step": uk - previous_snd
        })
        self.experiment.logger.log(logs_to_push, step=self.experiment.n_iters_performed)