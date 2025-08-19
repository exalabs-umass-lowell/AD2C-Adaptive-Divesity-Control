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

class SimpleProportionalController(Callback):
    """
    A simplified callback that implements a proportional controller to adjust desired SND.
    It calculates risk-adjusted performance and directly updates the SND based on a rolling baseline.
    """
    def __init__(
        self,
        control_group: str,
        proportional_gain: float,
        initial_snd: float,
        baseline_update_rate_alpha: float = 0.05,
        max_update_step: float = 0.2,
        beta: float = 0.5,
    ):
        super().__init__()
        self.control_group = control_group
        self.proportional_gain = proportional_gain
        self.initial_snd = initial_snd
        self.baseline_update_rate_alpha = baseline_update_rate_alpha
        self.max_update_step = max_update_step
        self.beta = beta
        
        # Controller state variables
        self._r_baseline = 0.0
        self._is_first_step = True
        self.model = None

    def on_setup(self):
        """Initializes the controller and logs hyperparameters."""
        hparams = {
            "controller_type": "SimpleProportional",
            "control_group": self.control_group,
            "proportional_gain": self.proportional_gain,
            "initial_snd": self.initial_snd,
            # "baseline_alpha": self.baseline_update_rate_alpha,
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
            print(f"\n✅ SUCCESS: Simple Proportional Controller initialized for group '{self.control_group}'.")
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
        update_step_clamped = 0.0

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


        if self._is_first_step:
            update = self.beta * (correlation_score + mean_return)
            improvement = 0
            self._r_baseline = mean_return
            self._is_first_step = False
        else:
            improvement = (mean_return - self._r_baseline)
            update = correlation_score + improvement
            self._r_baseline = mean_return
            update_step = self.proportional_gain * update
            update_step = torch.tensor(update_step)
            update_step_clamped = self.max_update_step * torch.tanh(update_step / self.max_update_step).item()
            new_snd_tensor = self.model.desired_snd.clone().detach() + update_step_clamped
            self.model.desired_snd[:] = torch.clamp(new_snd_tensor, min=0.0)
            print(f"Updated SND: {self.model.desired_snd.item()} (Update Step: {update_step_clamped})")

        print_plt(episode_snd, episode_returns, "SND vs. Reward per Episode", self.model.desired_snd.item())
        
        logs_to_push[f"simple_control/snd_actual"] = actual_snd
        logs_to_push[f"simple_control/target_snd"] = self.model.desired_snd.item()
        logs_to_push.update({
            "simple_control/mean_return": mean_return,
            "simple_control/improvement": improvement,
            "simple_control/score": correlation_score,
            "simple_control/Update": update,
            "simple_control/update_step_clamped": update_step_clamped,
        })
        self.experiment.logger.log(logs_to_push, step=self.experiment.n_iters_performed)
