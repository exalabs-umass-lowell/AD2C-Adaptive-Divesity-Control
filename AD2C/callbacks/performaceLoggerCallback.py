from typing import List

from PIL import Image
from matplotlib import pyplot as plt
import numpy as np
import io
import torch
import wandb
from tensordict import TensorDictBase, TensorDict
from typing import List

from benchmarl.experiment.callback import Callback
from AD2C.models.het_control_mlp_empirical import HetControlMlpEmpirical
from AD2C.models.het_control_mlp_snd import HetControlMlpEscSnd
from AD2C.snd import compute_behavioral_distance

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.spatial.distance import euclidean

from .plots import *
from callbacks.utils import *
import networkx as nx 
import pandas as pd

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import io
from PIL import Image
import wandb


class performaceLoggerCallback(Callback):
    def __init__(
        self,
        control_group: str,
        # proportional_gain: float,
        initial_snd: float,
        ):
        super().__init__()
        self.control_group = control_group
        # self.proportional_gain = proportional_gain
        self.initial_snd = initial_snd
        
        self.eps_target_Diversity = []
        self.eps_actual_diversity = []
        self.eps_number = []
        self.eps_mean_returns = []

        
        # Controller state variables
        self._r_baseline = 0.0
        self._is_first_step = True
        self.model = None

    def on_setup(self):
        """Initializes the controller and logs hyperparameters."""
        hparams = {
            # "controller_type": "SndCallback",
            "control_group": self.control_group,
            "initial_snd": self.initial_snd,
        }
        self.experiment.logger.log_hparams(**hparams)

        if self.control_group not in self.experiment.group_policies:
            print(f"\nWARNING: Controller group '{self.control_group}' not found. Disabling controller.\n")
            return

        policy = self.experiment.group_policies[self.control_group]
        self.model = get_het_model(policy)

        if isinstance(self.model, HetControlMlpEscSnd):
            print(f"\n SUCCESS: ClusterBase Controller initialized for group '{self.control_group}'.")
            self.model.desired_snd[:] = float(self.initial_snd)
        else:
            print(f"\n WARNING: A compatible model was not found for group '{self.control_group}'. Disabling controller.\n")
            self.model = None  

    def _get_agent_actions_for_rollout(self, rollout):
        # Compute actions for all agents given observation
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
        
        # This will hold the correctly processed distances for the agent graph and CSV
        final_distances_tensor = None

        with torch.no_grad():
            for r in rollouts:
                agent_actions = self._get_agent_actions_for_rollout(r)
                
                # The raw tensor has shape [time_steps, num_agents, num_pairs]
                pairwise_distances_tensor = compute_behavioral_distance(agent_actions, just_mean=False)
                
                # The mean across all dimensions gives a single SND value for each episode
                episode_snd.append(pairwise_distances_tensor.mean().item())
                
                reward_key = ('next', self.control_group, 'reward')
                total_reward = r.get(reward_key).sum().item() if reward_key in r.keys(include_nested=True) else 0
                episode_returns.append(total_reward)

        if not episode_returns:
            print("\nWARNING: No episode returns found. Cannot update controller.\n")
            self.experiment.logger.log(logs_to_push, step=self.experiment.n_iters_performed)
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

        next_target_diversity = mean_x # Default value
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
        self.eps_target_Diversity.append(next_target_diversity)
        self.eps_mean_returns.append(mean_y)
        
        plot = plot_snd_vs_reward(x, y, mean_y, next_target_diversity, mean_x, filtered_x, filtered_y)

        logs_to_push.update({
            "ClusterBase/snd_actual": mean_x,
            "ClusterBase/mean_return": mean_y,
            "ClusterBase/score": initial_correlation_score,
            "ClusterBase/initial_cohesion": initial_cohesion,
            "ClusterBase/target_diversity": next_target_diversity,
            "ClusterBase/filtered_cohesion": filtered_cohesion,
            "ClusterBase/performance_score": performance_score,
            "ClusterBase/Plot": plot,
        })

        # Process data for agent graph and CSV only once
        if self.model and self.model.n_agents > 1:
            sample_rollout = rollouts[0]
            agent_actions = self._get_agent_actions_for_rollout(sample_rollout)
            raw_pairwise_distances = compute_behavioral_distance(agent_actions, just_mean=False)
            
            # Average over the time dimension and flatten for a clean 1D tensor of distances
            final_distances_tensor = raw_pairwise_distances.mean(dim=0).flatten()
            
            graph_plot = plot_agent_distances(final_distances_tensor, self.model.n_agents)
            logs_to_push["Performance/distances graph"] = graph_plot

            # save_pairwise_diversity_to_csv(
            #     pairwise_distances_tensor=final_distances_tensor,
            #     episode_number=self.experiment.n_iters_performed,
            #     n_agents= self.model.n_agents
            # )

        # Trajectory plot if more than one episode
        if len(self.eps_number) > 1:
            eps_actual_diversity_np = np.array(self.eps_actual_diversity)
            eps_mean_returns_np = np.array(self.eps_mean_returns)
            
            if not np.all(np.isfinite(eps_actual_diversity_np)) or not np.all(np.isfinite(eps_mean_returns_np)):
                print("\nWARNING: Non-finite data found for trajectory plot. Skipping plot for this iteration.\n")
            else:
                plot_2d = plot_trajectory_2d(
                    snd = self.eps_actual_diversity,
                    returns = self.eps_mean_returns,
                    episodes=self.eps_number,
                    target_diversity=self.eps_target_Diversity
                )
                logs_to_push["Performace/Trajectory Plot"] = plot_2d
                
                # save_trajectory_data_to_csv(
                #     episodes=self.eps_number,
                #     snd=self.eps_actual_diversity,
                #     returns=self.eps_mean_returns,
                #     target_diversity=self.eps_target_Diversity,
                #     run_name_suffix=f'initial_snd_{self.initial_snd:.2f}'
                # )
                
                # trajectory_plot = plot_trajectory_3d(
                #     snd=self.eps_actual_diversity,
                #     returns=self.eps_mean_returns,
                #     episodes=self.eps_number,
                #     target_diversity=self.eps_target_Diversity
                # )
                # logs_to_push["ClusterBase/Trajectory_Plot_3D"] = trajectory_plot
    
        self.experiment.logger.log(logs_to_push, step=self.experiment.n_iters_performed)

        # if next_target_diversity:
        #     self.model.desired_snd[:] = float(next_target_diversity)