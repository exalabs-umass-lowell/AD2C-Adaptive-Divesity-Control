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
        hpf: float = 0.1,  # New high-pass filter parameter
        lpf_a: float = 0.1  # New probe amplitude filter parameter
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
        self.hpf = hpf
        self.lpf_a = lpf_a
        
        # Controller state variables
        self.model = None
        self._n_iters = 0
        self._Jkm1 = 0.0 # Objective function at the previous timestep
        self._sigmakm1 = 0.0 # High-pass filter output at the previous timestep
        self._psikm1 = 0.0 # Demodulated signal at the previous timestep
        self._gammakm1 = 0.0 # Low-pass filter output at the previous timestep
        self._uhatkm1 = float(self.initial_snd) # Integrator output at the previous timestep
        self._akm1 = dither_amplitude # Probe amplitude at the previous timestep
        self._T = 1.0 # Timestep size (delta_T) - you may need to adjust this based on your system's loop time.

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
            "hpf": self.hpf,
            "lpf_a": self.lpf_a
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

        Jk = np.mean(episode_rewards)
        self._n_iters += 1
        
        # 2. Call the core ES function
        (
            uk, 
            sigmak, 
            psik, 
            gammak, 
            uhatk, 
            ak
        ) = self._run_es_step(Jk)
        
        # 3. Update diversity parameter
        self.model.desired_snd[:] = torch.clamp(torch.tensor(uk), min=0.0)

        print(f"[ESC] Updated SND: {self.model.desired_snd.item()} "
              f"(Reward: {Jk:.3f}, Update Step: {uk - self._uhatkm1:.4f})")

        # 4. Logging
        logs_to_push.update({
            "esc/mean_reward": Jk,
            "esc/diversity_target": uk,
            "esc/sigmak": sigmak,
            "esc/psik": psik,
            "esc/gammak": gammak,
            "esc/uhatk": uhatk,
            "esc/ak": ak,
            "esc/update_step": uk - self._uhatkm1
        })
        self.experiment.logger.log(logs_to_push, step=self.experiment.n_iters_performed)

        # 5. Update state for the next timestep
        self._Jkm1 = Jk
        self._sigmakm1 = sigmak
        self._psikm1 = psik
        self._gammakm1 = gammak
        self._uhatkm1 = uhatk
        self._akm1 = ak
    
    def _run_es_step(self, Jk: float):
        """
        Calculates the next control signal using the core ES function.
        """
        t = self._n_iters * self._T
        w = 2 * np.pi * self.dither_frequency
        
        # High-pass filter
        sigmak = (Jk - self._Jkm1 - (self.hpf * self._T / 2 - 1) * self._sigmakm1) / (1 + self.hpf * self._T / 2)

        # Demodulation
        psik = sigmak * np.cos(w * t)

        # Low-pass filter
        gammak = (self._T * self.low_pass_filter_alpha * (psik + self._psikm1) - (self._T * self.low_pass_filter_alpha - 2) * self._gammakm1) / (2 + self._T * self.low_pass_filter_alpha)
        
        # Probe amplitude adaptation
        # Note: The original formula has an error. It should likely use a different form for the low-pass filter,
        # but to match the provided function, we'll implement it as is.
        ak = self.beta * (self._T * self.lpf_a * ((np.arctan(psik) / np.pi * 2)**2 + (np.arctan(self._psikm1) / np.pi * 2)**2)) - (self._T * self.lpf_a - 2) * self._akm1 / (2 + self._T * self.lpf_a)

        # Integrator
        uhatk = self._uhatkm1 + self.integral_gain * self._T / 2 * (gammak + self._gammakm1)

        # Modulation
        uk = uhatk + ak * np.cos(w * t)
        
        return (uk, sigmak, psik, gammak, uhatk, ak)