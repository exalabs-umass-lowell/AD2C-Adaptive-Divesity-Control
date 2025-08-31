from typing import List
import torch
import numpy as np
from AD2C.models.het_control_mlp_esc import HetControlMlpEsc
from tensordict import TensorDictBase, TensorDict

from benchmarl.experiment.callback import Callback
from AD2C.models.het_control_mlp_empirical import HetControlMlpEmpirical
from AD2C.snd import compute_behavioral_distance

# Assuming these plotting functions are defined in another file (e.g., plots.py)
# and can handle being called without the 'target_diversity' argument.
from .plots import plot_trajectory_2d, plot_trajectory_3d, plot_agent_distances
from callbacks.utils import get_het_model


class TrajectoryLoggerCallback(Callback):
    def __init__(
        self,
        control_group: str,
    ):
        super().__init__()
        self.control_group = control_group
        
        # Lists to store the trajectory data over time
        self.eps_actual_diversity = []
        self.eps_mean_returns = []
        self.eps_number = []
        
        self.model = None

    def on_setup(self):
        """Finds the policy model for the specified group."""
        if self.control_group not in self.experiment.group_policies:
            print(f"\nWARNING: Logger group '{self.control_group}' not found. Disabling logger.\n")
            return

        policy = self.experiment.group_policies[self.control_group]
        self.model = get_het_model(policy)

        # MODIFIED LINE: Check for either model type
        if isinstance(self.model, (HetControlMlpEmpirical, HetControlMlpEsc)):
            print(f"\nSUCCESS: Trajectory Logger initialized for group '{self.control_group}'.")
        else:
            print(f"\nWARNING: A compatible model was not found for group '{self.control_group}'. Disabling logger.\n")
            self.model = None

    def _get_agent_actions_for_rollout(self, rollout: TensorDictBase) -> List[torch.Tensor]:
        """Computes actions for all agents based on the rollout's observations."""
        obs = rollout.get((self.control_group, "observation"))
        actions = []
        for i in range(self.model.n_agents):
            temp_td = TensorDict({(self.control_group, "observation"): obs}, batch_size=obs.shape[:-1])
            # Set compute_estimate to False to avoid unnecessary calculations
            action_td = self.model._forward(temp_td, agent_index=i, compute_estimate=False)
            actions.append(action_td.get(self.model.out_key))
        return actions

    def on_evaluation_end(self, rollouts: List[TensorDictBase]):
        """
        Calculates actual SND and reward from rollouts, then logs metrics and plots.
        """
        if self.model is None:
            return

        episode_snd = []
        episode_returns = []
        plot_generated = False  # Flag to ensure we only plot for one episode

        # 4. Prepare logs for this step (initialize dict before the loop)
        logs_to_push = {}

        # 1. Collect SND and Reward data from all evaluation episodes
        with torch.no_grad():
            for r in rollouts:
                agent_actions = self._get_agent_actions_for_rollout(r)
                
                # Compute the pairwise distances for the episode
                pairwise_distances = compute_behavioral_distance(agent_actions, just_mean=False)
                episode_snd.append(pairwise_distances.mean().item())

                # Generate the agent distance plot for the first episode only
                if not plot_generated:
                    agent_distance_plot = plot_agent_distances(
                        pairwise_distances_tensor=pairwise_distances,
                        n_agents=self.model.n_agents
                    )
                    logs_to_push["Trajectory/Agent_Distances"] = agent_distance_plot
                    plot_generated = True

                # Compute the total return for the episode
                reward_key = ('next', self.control_group, 'reward')
                total_reward = r.get(reward_key).sum().item() if reward_key in r.keys(include_nested=True) else 0
                episode_returns.append(total_reward)

        if not episode_returns:
            print("\nWARNING: No episode returns found. Skipping logging for this iteration.\n")
            return

        # 2. Calculate the average actual SND and return for this evaluation step
        mean_actual_snd = np.mean(episode_snd)
        mean_return = np.mean(episode_returns)

        # 3. Store the data for the trajectory plot
        self.eps_actual_diversity.append(mean_actual_snd)
        self.eps_mean_returns.append(mean_return)
        self.eps_number.append(self.experiment.n_iters_performed)

        # Add main trajectory metrics to the log dictionary
        logs_to_push["Trajectory/actual_snd"] = mean_actual_snd
        logs_to_push["Trajectory/mean_return"] = mean_return

        # 5. Generate and log the trajectory plot if we have enough data
        if len(self.eps_number) > 1:
            # Check for non-finite values to prevent plotting errors
            if np.isfinite(self.eps_actual_diversity).all() and np.isfinite(self.eps_mean_returns).all():
                plot_2d = plot_trajectory_2d(
                    snd=self.eps_actual_diversity,
                    returns=self.eps_mean_returns,
                    episodes=self.eps_number
                )
                logs_to_push["Trajectory/SND_vs_Reward_2D"] = plot_2d

        # 6. Log everything to the experiment logger (e.g., W&B)
        self.experiment.logger.log(logs_to_push, step=self.experiment.n_iters_performed)