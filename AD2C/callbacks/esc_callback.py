#  Copyright (c) 2024.
#  ProrokLab (https://www.proroklab.org/)
#  All rights reserved.

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
        dither_amplitude: float,
        dither_frequency: float,
        integral_gain: float,
        low_pass_filter_alpha: float = 0.1,
        max_update_step: float = 0.2,
        beta: float = 0.5,
    ):
        super().__init__()
        self.control_group = control_group
        self.initial_snd = initial_snd
        
        # ESC parameters
        self.dither_amplitude = dither_amplitude
        self.dither_frequency = dither_frequency
        self.integral_gain = integral_gain
        self.low_pass_filter_alpha = low_pass_filter_alpha
        
        self.max_update_step = max_update_step
        self.beta = beta
        
        # Controller state variables
        self.model = None
        self._n_iters = 0
        self._low_pass_filtered_signal = 0.0
        self._estimated_gradient = 0.0

    def on_setup(self):
        """Initializes the controller and logs hyperparameters."""
        hparams = {
            "controller_type": "ExtremumSeeking",
            "control_group": self.control_group,
            "initial_snd": self.initial_snd,
            "dither_amplitude": self.dither_amplitude,
            "dither_frequency": self.dither_frequency,
            "integral_gain": self.integral_gain,
            "low_pass_filter_alpha": self.low_pass_filter_alpha,
            "max_update_step": self.max_update_step,
            "beta": self.beta,
        }
        self.experiment.logger.log_hparams(**hparams)

        if self.control_group not in self.experiment.group_policies:
            print(f"\nWARNING: Controller group '{self.control_group}' not found. Disabling controller.\n")
            return

        policy = self.experiment.group_policies[self.control_group]
        self.model = get_het_model(policy)

        if isinstance(self.model, HetControlMlpEmpirical):
            print(f"\n✅ SUCCESS: Extremum Seeking Controller initialized for group '{self.control_group}'.")
            self.model.desired_snd[:] = float(self.initial_snd)
        else:
            print(f"\nWARNING: A compatible model was not found for group '{self.control_group}'. Disabling controller.\n")
            self.model = None
    
                
    def on_evaluation_end(self, rollouts: List[TensorDictBase]):
        if self.model is None:
            return

        logs_to_push = {}

        episode_snd = []
        episode_returns = []
        
        with torch.no_grad():
            for r in rollouts:
                obs = r.get((self.control_group, "observation"))
                agent_actions = []
                for i in range(self.model.n_agents):
                    temp_td = TensorDict({
                        (self.control_group, "observation"): obs
                    }, batch_size=obs.shape[:-1])
                    action_td = self.model._forward(temp_td, agent_index=i, compute_estimate=False)
                    agent_actions.append(action_td.get(self.model.out_key))

                pairwise_distances_tensor = compute_behavioral_distance(agent_actions, just_mean=False)
                episode_snd.append(pairwise_distances_tensor.mean().item())

                reward_key = ('next', self.control_group, 'reward')
                total_reward = r.get(reward_key).sum().item() if reward_key in r.keys(include_nested=True) else 0
                episode_returns.append(total_reward)

        if not episode_returns:
            print("\nWARNING: No episode returns found. Cannot update controller.\n")
            self.experiment.logger.log(logs_to_push, step=self.experiment.n_iters_performed)
            return

        actual_snd = np.mean(episode_snd)
        mean_return = np.mean(episode_returns)
        correlation_score = correlation_score_f(episode_snd, episode_returns)
        
        # Calculate the performance score to be optimized
        performance_score = (self.beta * mean_return) + ((1 - self.beta) * correlation_score)
        
        # 1. Dither signal
        self._n_iters += 1
        dither_signal = self.dither_amplitude * np.sin(self.dither_frequency * self._n_iters)
        
        # 2. Demodulation
        demodulated_signal = performance_score * dither_signal
        
        # 3. Low-pass filter (recursive average)
        self._low_pass_filtered_signal = (1 - self.low_pass_filter_alpha) * self._low_pass_filtered_signal + \
                                         self.low_pass_filter_alpha * demodulated_signal
        
        # 4. Integration
        self._estimated_gradient += self.integral_gain * self._low_pass_filtered_signal
        
        # 5. Update the desired SND
        update_step = self._estimated_gradient
        update_step_clamped = self.max_update_step * np.tanh(update_step / self.max_update_step)
        
        new_snd_tensor = self.initial_snd + dither_signal + update_step_clamped
        self.model.desired_snd[:] = torch.clamp(torch.tensor(new_snd_tensor), min=0.0)
        
        print(f"Updated SND: {self.model.desired_snd.item()} (Update Step: {update_step_clamped})")

        print_plt(episode_snd, episode_returns, "SND vs. Reward per Episode", self.model.desired_snd.item())
        
        logs_to_push[f"extremum_seeking/snd_actual"] = actual_snd
        logs_to_push[f"extremum_seeking/target_snd"] = self.model.desired_snd.item()
        logs_to_push.update({
            "extremum_seeking/mean_return": mean_return,
            "extremum_seeking/performance_score": performance_score,
            "extremum_seeking/dither_signal": dither_signal,
            "extremum_seeking/demodulated_signal": demodulated_signal,
            "extremum_seeking/low_pass_filtered_signal": self._low_pass_filtered_signal,
            "extremum_seeking/estimated_gradient": self._estimated_gradient,
            "extremum_seeking/update_step_clamped": update_step_clamped,
        })
        self.experiment.logger.log(logs_to_push, step=self.experiment.n_iters_performed)