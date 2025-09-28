from typing import List

from PIL import Image
from matplotlib import pyplot as plt
import numpy as np
import io
import torch
import wandb
from tensordict import TensorDictBase, TensorDict

from benchmarl.experiment.callback import Callback
from AD2C.models.het_control_mlp_empirical import HetControlMlpEmpirical
from AD2C.snd import compute_behavioral_distance

from .plots import *
from callbacks.utils import *
import networkx as nx
import pandas as pd

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class performaceLoggerCallback(Callback):
    def __init__(
        self,
        control_group: str,
        initial_snd: float,
    ):
        super().__init__()
        self.control_group = control_group
        self.initial_snd = initial_snd

        self.eps_target_diversity = []
        self.eps_actual_diversity = []
        self.eps_number = []
        self.eps_mean_returns = []
        self.distance_history = []

        # Controller state variables
        self.model = None

    def on_setup(self):
        """Initializes the controller and logs hyperparameters."""
        hparams = {
            "control_group": self.control_group,
            "initial_snd": self.initial_snd,
        }
        self.experiment.logger.log_hparams(**hparams)

        if self.control_group not in self.experiment.group_policies:
            print(f"\nWARNING: Controller group '{self.control_group}' not found. Disabling callback.\n")
            return

        policy = self.experiment.group_policies[self.control_group]
        self.model = get_het_model(policy)

        if isinstance(self.model, HetControlMlpEmpirical):
            print(f"\n SUCCESS: Performance Logger initialized for group '{self.control_group}'.")
            self.model.desired_snd[:] = float(self.initial_snd)
        else:
            print(f"\n WARNING: A compatible model was not found for group '{self.control_group}'. Disabling callback.\n")
            self.model = None

    # NOTE: The loop in this function calls the model's forward pass N times for N agents.
    # For a significant performance improvement, check if your model supports a batch
    # forward pass to get all agent actions in a single call.
    def _get_agent_actions_for_rollout(self, rollout):
        """Compute actions for all agents given an observation sequence."""
        obs = rollout.get((self.control_group, "observation"))
        actions = []
        for i in range(self.model.n_agents):
            temp_td = TensorDict(
                {(self.control_group, "observation"): obs},
                batch_size=obs.shape[:-1]
            )
            action_td = self.model._forward(temp_td, agent_index=i, compute_estimate=False)
            actions.append(action_td.get(self.model.out_key))
        return actions

    def on_evaluation_end(self, rollouts: List[TensorDictBase]):
        if self.model is None:
            return

        logs_to_push = {}
        episode_snd = []
        episode_returns = []
        final_distances_tensor = None  # Will hold distances from the first rollout for graphing

        with torch.no_grad():
            for i, r in enumerate(rollouts):
                agent_actions = self._get_agent_actions_for_rollout(r)
                
                # Compute behavioral distance ONCE per rollout
                pairwise_distances_tensor = compute_behavioral_distance(agent_actions, just_mean=False)
                
                # For the first rollout, save the detailed distance tensor for the agent graph
                if i == 0 and self.model.n_agents > 1:
                    final_distances_tensor = pairwise_distances_tensor.mean(dim=0).flatten()
                    self.distance_history.append(final_distances_tensor.detach().clone())

                # The mean SND for this episode
                episode_snd.append(pairwise_distances_tensor.mean().item())
                
                # The total reward for this episode
                reward_key = ('next', self.control_group, 'reward')
                total_reward = r.get(reward_key).sum().item() if reward_key in r.keys(include_nested=True) else 0
                episode_returns.append(total_reward)

        if not episode_returns:
            print("\nWARNING: No episode returns found. Skipping performance logging.\n")
            return

        x, y = np.array(episode_snd), np.array(episode_returns)
        mean_x, mean_y = np.mean(x), np.mean(y)

        # --- Metrics calculations ---
        initial_correlation_score = correlation_score_f(x, y)
        centroid = find_centroid(x, y)
        distances = calculate_cohesion(x, y, centroid)
        initial_cohesion = np.mean(distances)

        mask = y > mean_y
        filtered_x, filtered_y = x[mask], y[mask]

        next_target_diversity = mean_x
        filtered_cohesion = 0
        performance_score = 0
        
        if len(filtered_x) > 1:
            filtered_centroid = find_centroid(filtered_x, filtered_y)
            next_target_diversity = filtered_centroid[0]
            filtered_distances = calculate_cohesion(filtered_x, filtered_y, filtered_centroid)
            filtered_cohesion = np.mean(filtered_distances)
            
            normalized_reward = z_score_normalize(y)
            normalized_cohesion_distances = z_score_normalize(distances)
            w1, w2 = 1.0, 0.5
            performance_score = w1 * np.mean(normalized_reward) - w2 * np.mean(normalized_cohesion_distances)

        # --- Logging and plotting ---
        self.eps_actual_diversity.append(mean_x)
        self.eps_number.append(self.experiment.n_iters_performed)
        self.eps_target_diversity.append(next_target_diversity)
        self.eps_mean_returns.append(mean_y)
        
        # Correctly call the plotting function with keyword arguments
        plot_title = f"SND vs. Reward (Iteration {self.experiment.n_iters_performed})"
        plot = plot_snd_vs_reward(
            snd_values=x,
            reward_values=y,
            title=plot_title,
            filtered_snd_values=filtered_x,
            filtered_reward_values=filtered_y
        )

        logs_to_push.update({
            "Performance/snd_actual": mean_x,
            "Performance/mean_return": mean_y,
            "Performance/score": initial_correlation_score,
            "Performance/initial_cohesion": initial_cohesion,
            "Performance/target_diversity": next_target_diversity,
            "Performance/filtered_cohesion": filtered_cohesion,
            "Performance/performance_score": performance_score,
            "Performance/Plot": plot,
        })

        # Generate agent graph from the saved tensor without re-computing
        if final_distances_tensor is not None:
            graph_plot = plot_agent_distances(final_distances_tensor, self.model.n_agents)
            logs_to_push["Performance/distances graph"] = graph_plot

        # Trajectory plots
        if len(self.eps_number) > 1:
            eps_actual_diversity_np = np.array(self.eps_actual_diversity)
            eps_mean_returns_np = np.array(self.eps_mean_returns)
            
            if not np.all(np.isfinite(eps_actual_diversity_np)) or not np.all(np.isfinite(eps_mean_returns_np)):
                print("\nWARNING: Non-finite data found for trajectory plot. Skipping.\n")
            else:
                plot_2d = plot_trajectory_2d(
                    snd=self.eps_actual_diversity,
                    returns=self.eps_mean_returns,
                    episodes=self.eps_number,
                    # target_diversity=self.eps_target_diversity
                )
                logs_to_push["Performance/Trajectory Plot"] = plot_2d
                
                if self.distance_history:
                    distance_growth_plot = plot_distance_history(
                        distance_history=self.distance_history,
                        n_agents=self.model.n_agents
                    )
                    if distance_growth_plot:
                        logs_to_push["Performance/Distance Growth Plot"] = distance_growth_plot
    
        self.experiment.logger.log(logs_to_push, step=self.experiment.n_iters_performed)