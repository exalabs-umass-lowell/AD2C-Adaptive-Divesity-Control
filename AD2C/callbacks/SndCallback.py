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
from AD2C.utils import overflowing_logits_norm

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.spatial.distance import euclidean

from .utils import get_het_model, print_plt

class SndCallback(Callback):
    """
    Callback used to compute SND during evaluations
    """
    def __init__(
        self,
        control_group: str,
        initial_snd: float,
    ):
        super().__init__()
        self.control_group = control_group
        self.initial_snd = initial_snd

        self.eval_results = []
        self.eval_diversity = []
        
        # Controller state variables
        self._is_first_step = True
        self.model = None

    def on_setup(self):
        """Initializes the controller and logs hyperparameters."""
        hparams = {
            "controller_type": "SndCallback",
            "control_group": self.control_group,
            "initial_snd": self.initial_snd,
        }
        self.experiment.logger.log_hparams(**hparams)

        policy = self.experiment.group_policies[self.control_group]
        self.model = get_het_model(policy)

        if isinstance(self.model, HetControlMlpEmpirical):
            print(f"\n SUCCESS: Simple SND Callback initialized for group '{self.control_group}'.")
            self.model.desired_snd[:] = float(self.initial_snd)
        else:
            print(f"\n WARNING: A compatible model was not found for group '{self.control_group}'. Disabling controller.\n")
            self.model = None
                
    def on_evaluation_end(self, rollouts: List[TensorDictBase]):
        if self.model is None:
            return

        logs_to_push = {}

        episode_snd = [] # List of SND for Each Evalution Episode
        episode_returns = [] # List of Rewards For each Evalution episode
        
        
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
        
        self.eval_results.append(mean_return)
        self.eval_diversity.append(actual_snd)
        
        
        plot = print_plt(episode_snd, episode_returns, "SND vs. Reward per Episode", self.model.desired_snd.item())
        # episodes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] # 
        # target_values = [0.5, 0.6, 0.7, 0.75, 0.8, 0.82, 0.85, 0.87, 0.88, 0.9]
        # actual_values = [0.4, 0.55, 0.6, 0.7, 0.78, 0.8, 0.83, 0.85, 0.87, 0.89]

        # plot_trajectory(
        #     x_blue=episodes,
        #     y_blue=target_values,
        #     title='Reward and Diversity Over Episodes',
        #     x_red=episodes,
        #     y_red=actual_values
        # )
        
        
        logs_to_push[f"Dico/snd_actual"] = actual_snd
        logs_to_push[f"Dico/target_snd"] = self.model.desired_snd.item()
        logs_to_push.update({
            "Dico/mean_return": mean_return,
            "Dico/DiversityVreward": plot,
        })
        self.experiment.logger.log(logs_to_push, step=self.experiment.n_iters_performed)

    def on_experiment_end(self):
        """
        Plots the trajectory of diversity and reward over time.
        """
        # Ensure there is data to plot
        if self.eval_results and self.eval_diversity:
            # Create a figure and a primary axes
            fig, ax1 = plt.subplots(figsize=(10, 6))

            # Plot Diversity on the first y-axis
            color = 'tab:blue'
            ax1.set_xlabel('Evaluation Run')
            ax1.set_ylabel('Mean Diversity (SND)', color=color)
            ax1.plot(self.eval_diversity, color=color, label='Mean Diversity')
            ax1.tick_params(axis='y', labelcolor=color)

            # Create a second y-axis that shares the same x-axis
            ax2 = ax1.twinx()  
            color = 'tab:red'
            ax2.set_ylabel('Mean Reward', color=color)  
            ax2.plot(self.eval_results, color=color, label='Mean Reward')
            ax2.tick_params(axis='y', labelcolor=color)

            # Add a title and combine legends
            fig.suptitle('Trajectory of Diversity and Reward Over Time')
            fig.legend(loc="upper left")

            # Save the plot to a buffer and log to WandB
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            img = Image.open(buf)
            logs_to_push = {"Dico/Diversity_Reward_Trajectory": wandb.Image(img)}
            self.experiment.logger.log(logs_to_push, step=self.experiment.n_iters_performed)
            plt.close(fig) # Close the figure to free up memory
        else:
            print("No evaluation data to plot.")