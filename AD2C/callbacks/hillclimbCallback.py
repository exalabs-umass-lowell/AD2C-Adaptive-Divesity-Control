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
from AD2C.utils import overflowing_logits_norm

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.spatial.distance import euclidean

from callbacks.utils import *

def calculate_costs(episode_snd, episode_returns, target_diversity, alpha):
    """
    Calculates the cost for each episode in a batch.
    """
    mean_snd = np.mean(episode_snd)
    
    # The diversity penalty term is constant for all episodes in a batch
    diversity_term = (1 - alpha) * (mean_snd - target_diversity)**2
    
    # FIX: Convert episode_returns to a NumPy array for element-wise multiplication
    reward_term = alpha * np.array(episode_returns)
    
    # The cost for each episode is the reward term minus the diversity penalty.
    costs = reward_term - diversity_term
    
    return costs

class gradientBaseSndCallback(Callback):
    def __init__(self, control_group: str, initial_snd: float, learning_rate: float, alpha: float):
        super().__init__()
        self.control_group = control_group
        self.initial_snd = initial_snd
        self.learning_rate = learning_rate
        self.alpha = alpha

        # Controller state variables
        self.model = None
        self.target_diversity = float(self.initial_snd)
        self.target_diversity_for_next_eval = self.target_diversity + 1e-4 # Add a small epsilon
        self.prev_target_diversity = None
        self.prev_episode_rewards = None
        self.prev_actual_snd = None

    def on_setup(self):
        hparams = {
            "controller_type": "GradientBaseSndCallback",
            "control_group": self.control_group,
            "initial_snd": self.target_diversity,
            "learning_rate": self.learning_rate,
            "alpha": self.alpha,
        }
        self.experiment.logger.log_hparams(**hparams)

        if self.control_group not in self.experiment.group_policies:
            print(f"\nWARNING: Controller group '{self.control_group}' not found. Disabling controller.\n")
            return

        policy = self.experiment.group_policies[self.control_group]
        self.model = get_het_model(policy)

        if isinstance(self.model, HetControlMlpEmpirical):
            print(f"\n SUCCESS: Gradient-Based SND Controller initialized for group '{self.control_group}'.")
            self.model.desired_snd[:] = float(self.target_diversity)
        else:
            print(f"\n WARNING: A compatible model was not found for group '{self.control_group}'. Disabling controller.\n")
            self.model = None
        
    def on_evaluation_end(self, rollouts: List[TensorDictBase]):
        if self.model is None:
            return

        logs_to_push = {}
        episode_snd = []
        episode_returns = []
        episode_costs = []
        gradient = 0.0

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

        episode_costs = calculate_costs(episode_snd, episode_returns, self.target_diversity, self.alpha)
        mean_snd = np.mean(episode_snd)
        mean_return = np.mean(episode_returns)

        # Gradient-based update
        reward_term = self.alpha * mean_return
        diversity_term = (1 - self.alpha) * (mean_snd - self.target_diversity) ** 2
        cost = reward_term - diversity_term

        current_target_diversity = self.target_diversity

        # Check if we have previous data to compute the gradient
        if self.prev_target_diversity is None:
            self.prev_target_diversity = self.target_diversity - 1e-2
            self.prev_episode_rewards = mean_return
            self.prev_actual_snd = mean_snd
            print("First evaluation step, initializing previous values.")
        else:
            delta_Dtarget = self.target_diversity - self.prev_target_diversity

            print(f"Current Target: {delta_Dtarget}")
            # if abs(delta_Dtarget) > 1e-6:
            EPS = 1e-6
            # delta_Dtarget = self.target_diversity - self.prev_target_diversity
            
            if abs(delta_Dtarget) > EPS:
                dR_dTarget = (mean_return - self.prev_episode_rewards) / delta_Dtarget
                dD_achieved_dTarget = (mean_snd - self.prev_actual_snd) / delta_Dtarget
                gradient = (self.alpha * dR_dTarget) - (
                    2 * (1 - self.alpha) * (mean_snd - self.target_diversity) * dD_achieved_dTarget
                )
                print(f"Gradient: {gradient}, dR_dTarget: {dR_dTarget}, dD_achieved_dTarget: {dD_achieved_dTarget}")
                self.target_diversity += self.learning_rate * gradient
            else:
                # skip update; keep diversity unchanged, avoid nans/infs
                print(f"SKIP: delta_Dtarget too small: {delta_Dtarget}")
                
        # Store the current values for the next evaluation step
        # self.prev_target_diversity = current_target_diversity
        self.prev_episode_rewards = mean_return
        self.prev_actual_snd = mean_snd
        
        # self.target_diversity_for_next_eval = self.target_diversity + 1e-4

        # Clamp the diversity to prevent it from going out of bounds
        self.target_diversity = max(0.0, self.target_diversity)
        # print(f"Current Target Diversity: {self.target_diversity}, Gradient: {gradient}")
        # Plotting logic
        x_plot = np.array(episode_snd)
        y_plot = np.array(episode_returns)

        filtered_indices = y_plot > np.mean(y_plot)
        filtered_x = x_plot[filtered_indices]
        filtered_y = y_plot[filtered_indices]

        plot = plot_snd_vs_reward(
            x=x_plot,
            y=y_plot,
            title=f'SND vs. Reward (Iter {self.experiment.n_iters_performed})',
            next_target_diversity=self.target_diversity,
            curent_target_diversity=current_target_diversity,
            filtered_x=filtered_x,
            filtered_y=filtered_y
        )

        logs_to_push.update({
            "GradientBase/snd_actual": mean_snd,
            "GradientBase/mean_return": mean_return,
            "GradientBase/target_diversity": self.target_diversity,
            "GradientBase/cost": cost,
            "GradientBase/Plot": plot,
            "GradientBase/gradient": gradient,
        })

        self.experiment.logger.log(logs_to_push, step=self.experiment.n_iters_performed)
        self.model.desired_snd[:] = float(self.target_diversity)

# Other experimental Settings
# For Hill Climbing MAB Controller
class HillClimbingMABCallback(Callback):
    """
    A hybrid MAB controller that combines stable hill-climbing with global exploration.
    (Full docstring...)
    """
    def __init__(
        self,
        control_group: str,
        snd_arms: List[float] = None,
        c_mab: float = 2.0,
        beta: float = 1.0,
        exploration_epsilon: float = 0.1,
    ):
        super().__init__()
        self.control_group = control_group
        self.beta = beta
        self.c_mab = c_mab
        self.exploration_epsilon = exploration_epsilon

        if snd_arms is None:
            self.snd_arms = sorted([0.0, 0.25, 0.5, 0.75, 1.0])
        else:
            self.snd_arms = sorted(snd_arms)
        self.arm_indices = {arm: i for i, arm in enumerate(self.snd_arms)}

        self.arm_counts = {snd: 0 for snd in self.snd_arms}
        self.arm_rewards = {snd: 0.0 for snd in self.snd_arms}
        self.total_pulls = 0
        self.current_arm = self.snd_arms[0]
        self.model = None

    def on_setup(self):
        """Initializes the MAB controller and logs hyperparameters."""
        hparams = {
            "controller_type": "HillClimbingMAB",
            "control_group": self.control_group,
            "snd_arms": self.snd_arms,
            "c_mab": self.c_mab,
            "beta": self.beta,
            "exploration_epsilon": self.exploration_epsilon,
        }
        self.experiment.logger.log_hparams(**hparams)

        if self.control_group not in self.experiment.group_policies:
            print(f"\nWARNING: Controller group '{self.control_group}' not found. Disabling controller.\n")
            self.current_arm = -1
            return

        policy = self.experiment.group_policies[self.control_group]
        self.model = get_het_model(policy)

        if isinstance(self.model, HetControlMlpEmpirical):
            print(f"\n✅ SUCCESS: Hill-Climbing MAB Controller initialized for group '{self.control_group}'.")
            self.model.desired_snd[:] = float(self.current_arm)
        else:
            print(f"\nWARNING: A compatible model was not found for group '{self.control_group}'. Disabling controller.\n")
            self.current_arm = -1

    def _select_next_arm(self):
        """Selects the next arm to pull based on a hybrid hill-climbing and UCB strategy."""
        for arm in self.snd_arms:
            if self.arm_counts[arm] == 0:
                print(f"[MAB] Bootstrap: Pulling un-pulled arm {arm}")
                return arm

        is_global_exploration_step = np.random.rand() < self.exploration_epsilon

        if is_global_exploration_step:
            candidate_arms = self.snd_arms
            # self.experiment.logger.log_scalar("mab_control/step_type", 1, step=self.experiment.n_iters_performed)
            print(f"[MAB] Global Exploration Step: Considering all arms.")
        else:
            current_idx = self.arm_indices[self.current_arm]
            neighbor_indices = {current_idx}
            if current_idx > 0:
                neighbor_indices.add(current_idx - 1)
            if current_idx < len(self.snd_arms) - 1:
                neighbor_indices.add(current_idx + 1)
            candidate_arms = [self.snd_arms[i] for i in sorted(list(neighbor_indices))]
            # print(f"mab_control/step_type", 0, step=self.experiment.n_iters_performed)
            print(f"[MAB] Local Hill-Climb Step: Considering neighbors {candidate_arms}")

        best_arm = None
        max_ucb_value = -float('inf')

        for arm in candidate_arms:
            # Handle division by zero for unpulled arms in the candidate set
            if self.arm_counts[arm] == 0:
                ucb_value = float('inf')
            else:
                avg_reward = self.arm_rewards[arm] / self.arm_counts[arm]
                confidence_bound = self.c_mab * np.sqrt(np.log(self.total_pulls) / self.arm_counts[arm])
                ucb_value = avg_reward + confidence_bound
            
            if ucb_value > max_ucb_value:
                max_ucb_value = ucb_value
                best_arm = arm
        
        return best_arm


    def on_evaluation_end(self, rollouts: List[TensorDict]):
        """
        Updates MAB state, logs actual SND, and selects the next desired_snd.
        """
        if self.current_arm == -1:
            return  # Controller is disabled
        
        logs_to_push = {}

        # Loop to calculate actual SND
        for group in self.experiment.group_map.keys():
            if len(self.experiment.group_map[group]) <= 1:
                continue

            policy = self.experiment.group_policies.get(group)
            if policy is None:
                continue
            
            model = get_het_model(policy)
            if not isinstance(model, HetControlMlpEmpirical):
                continue

            obs = torch.cat([r.get((group, "observation")) for r in rollouts], dim=0)

            agent_actions = []
            with torch.no_grad():
                for i in range(model.n_agents):
                    # The fix is here: wrap the observation tensor in a TensorDict
                    temp_td = TensorDict({
                        (group, "observation"): obs
                    }, batch_size=obs.shape[:-1])

                    action_td = model._forward(temp_td, agent_index=i, compute_estimate=False)
                    agent_actions.append(action_td.get(model.out_key))

            snd_mean = compute_behavioral_distance(agent_actions, just_mean=True).mean().item()
            logs_to_push[f"eval/{group}/actual_snd"] = snd_mean
            logs_to_push[f"eval/{group}/snd_desired_used"] = model.desired_snd.item()

        # The rest of the method updates the MAB controller state
        reward_key = ('next', 'agents', 'reward')
        episode_returns = [r.get(reward_key).mean().item() for r in rollouts if reward_key in r.keys(include_nested=True)]
        
        if not episode_returns:
            print("\nWARNING: No episode returns found. Cannot update MAB.\n")
            if logs_to_push:
                 self.experiment.logger.log(logs_to_push, step=self.experiment.n_iters_performed)
            return
        
        mean_return = torch.tensor(episode_returns).mean().item()
        std_return = torch.tensor(episode_returns).std().item()
        performance = mean_return - self.beta * std_return

        self.arm_counts[self.current_arm] += 1
        self.total_pulls += 1
        
        # This part has been corrected to accumulate the sum of rewards
        self.arm_rewards[self.current_arm] += performance
        
        logs_to_push.update({
            f"mab_rewards/arm_{self.current_arm}": self.arm_rewards[self.current_arm] / self.arm_counts[self.current_arm],
            f"mab_pulls/arm_{self.current_arm}": self.arm_counts[self.current_arm],
            "mab_control/last_performance": performance,
            "mab_control/total_pulls": self.total_pulls,
        })

        next_arm = self._select_next_arm()
        if next_arm is not None:
            self.current_arm = next_arm
            self.model.desired_snd[:] = float(self.current_arm)

        logs_to_push["mab_control/selected_snd_des"] = self.current_arm
        
        self.experiment.logger.log(logs_to_push, step=self.experiment.n_iters_performed)

# TO find the best SND from exploration results
def find_best_snd(results_dir: str):
    best_snd = None
    max_score = -float('inf')
    found_results = False

    print(f"Searching for exploration results in: {results_dir}\n")

    if not os.path.isdir(results_dir):
        print(f"WARNING: The results directory '{results_dir}' does not exist.")
        return None

    for folder_name in os.listdir(results_dir):
        run_folder = os.path.join(results_dir, folder_name)

        if os.path.isdir(run_folder):
            result_file = os.path.join(run_folder, "final_score.pkl")
            
            if not os.path.exists(result_file):
                print(f"Skipping '{run_folder}': No 'final_score.pkl' file found.")
                continue

            try:
                with open(result_file, 'rb') as f:
                    result = pickle.load(f)

                # Check for required keys to prevent a KeyError
                if 'score' in result and 'snd' in result:
                    found_results = True
                    score = result['score']
                    snd = result['snd']
                    print(f"Found result for SND {snd}: Score = {score}")

                    if score > max_score:
                        max_score = score
                        best_snd = snd
                        print(f"--> New best SND found: {best_snd} with score {max_score}")
                else:
                    print(f"WARNING: Skipping '{result_file}'. Missing 'score' or 'snd' keys.")

            except (pickle.UnpicklingError, EOFError) as e:
                print(f"ERROR: Failed to load pickled file '{result_file}'. Reason: {e}")
            except Exception as e:
                print(f"ERROR: An unexpected error occurred with '{result_file}'. Reason: {e}")

    print("\n--- RANKING COMPLETE ---")
    if found_results:
        print(f"The best SND found is {best_snd} with score {max_score:.4f}\n")
    else:
        print("No valid exploration results were found.\n")

    return best_snd


class EloEvaluationCallback(Callback):
    def __init__(
        self,
        control_group: str,
        snd: float,
        beta: float = 1.0,
        run_name: str = "exploration_run",
        save_folder: str = None, 
    ):
        super().__init__()
        self.control_group = control_group
        self.snd = snd
        self.beta = beta
        self.run_name = run_name
        self.model = None
        self.performance_data = []
        self.save_folder = save_folder

    def on_setup(self):
        hparams = {
            "controller_type": "EloEvaluation",
            "control_group": self.control_group,
            "snd": self.snd,
            "beta": self.beta,
        }
        self.experiment.logger.log_hparams(**hparams)

        if self.control_group not in self.experiment.group_policies:
            print(f"\nWARNING: Controller group '{self.control_group}' not found. Disabling callback.\n")
            return

        policy = self.experiment.group_policies[self.control_group]
        self.model = get_het_model(policy)
        if isinstance(self.model, HetControlMlpEmpirical):
            print(f"\n✅ SUCCESS: Elo Evaluation Callback initialized for group '{self.control_group}'.")
            self.model.desired_snd[:] = float(self.snd)
        else:
            print(f"\nWARNING: A compatible model was not found for group '{self.control_group}'. Disabling callback.\n")
            self.model = None

    def on_evaluation_end(self, rollouts: List[TensorDictBase]):
        if self.model is None:
            return

        reward_key = ('next', 'agents', 'reward')
        episode_returns = [r.get(reward_key).mean().item() for r in rollouts if reward_key in r.keys(include_nested=True)]
        
        if not episode_returns:
            print("\nWARNING: No episode returns found. Cannot log performance.\n")
            return
        
        mean_return = torch.tensor(episode_returns).mean().item()
        std_return = torch.tensor(episode_returns).std().item()
        performance = mean_return - self.beta * std_return
        
        self.performance_data.append(performance)

        self.experiment.logger.log({
            "eval/performance_per_step": performance
        }, step=self.experiment.n_iters_performed)

        # --- NEW LOGIC: Check if this is the final evaluation and save the score ---
        # Assuming that the final evaluation happens when max_n_frames is reached.
        # This is a robust way to check for experiment termination.
        if self.experiment.total_frames >= self.experiment.config.max_n_frames:
            print("--- Final evaluation detected. Saving final score. ---")
            final_score = np.nan
            if self.performance_data:
                final_score = np.mean(self.performance_data)
            else:
                print(f"WARNING: No performance data collected for SND {self.snd}. Saving NaN score.")
            
            if self.save_folder is None:
                print(f"ERROR: No save_folder was provided. Skipping file save.")
                return

            results_path = os.path.join(self.save_folder, "final_score.pkl")
            with open(results_path, 'wb') as f:
                pickle.dump({"snd": self.snd, "score": final_score}, f)

            self.experiment.logger.log({
                "final_score": final_score,
                "snd_value": self.snd,
                "run_name": self.run_name
            }, step=self.experiment.n_iters_performed)
            print(f"Final score for SND {self.snd}: {final_score}")