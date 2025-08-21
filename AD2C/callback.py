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


def get_het_model(policy):
    model = policy.module[0]
    while not isinstance(model, HetControlMlpEmpirical):
        model = model[0]
    return model


def correlation_score_f(x: Union[List, np.ndarray], y: Union[List, np.ndarray]) -> float:
    # Convert lists to NumPy arrays for element-wise operations
    x = np.array(x)
    y = np.array(y)

    if len(x) != len(y) or len(x) <= 1:
        return 0.0
    n = len(x)
    
    sum_x = np.sum(x)
    sum_y = np.sum(y)
    sum_xy = np.sum(x * y)
    sum_x_squared = np.sum(x**2)
    sum_y_squared = np.sum(y**2)

    n = len(x)

    numerator = n * sum_xy - sum_x * sum_y
    denominator_term_x = n * sum_x_squared - sum_x**2
    denominator_term_y = n * sum_y_squared - sum_y**2
    
    denominator = np.sqrt(denominator_term_x * denominator_term_y)

    # Avoid division by zero if the denominator is close to zero
    if denominator == 0:
        return 0.0

    return numerator / denominator


def print_plt(x: List[float], y: List[float], title: str, desired_diversity: float):
    mean_y = np.mean(y)
    fig, ax = plt.subplots()
    ax.scatter(x, y)
    ax.set_xlabel("Episode SND")
    ax.set_ylabel("Episode Reward")
    ax.set_title("SND vs. Reward per Episode")
    ax.axhline(y=mean_y, color='gray', linestyle='--', label=f'Mean Reward ({mean_y:.2f})')
    ax.axvline(x=desired_diversity, color='red', linestyle='--', label='Desired Diversity')
    ax.legend()

    ax.set_ylim(-2, 6)
    ax.set_xlim(0, 4)

    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    plt.close(fig)
    pil_image = Image.open(buf)
    wandb_image = wandb.Image(pil_image)
    
    return wandb_image

def plot_snd_vs_reward(
    x: list, 
    y: list, 
    title: str, 
    next_target_diversity: float, 
    filtered_x: list = None, 
    filtered_y: list = None
):
    mean_y = np.mean(y)

    plt.figure(figsize=(10, 6))
    plt.scatter(x, y, alpha=0.6, label='All Episodes')
    plt.axhline(y=mean_y, color='gray', linestyle='--', label=f'Mean Reward ({mean_y:.2f})')
    plt.axvline(x=next_target_diversity, color='red', linestyle='-', 
                label=f'Next Target Diversity ({next_target_diversity:.2f})')

    
    # Plot filtered episodes above mean reward (if provided)
    if filtered_x is not None and filtered_y is not None:
        plt.scatter(
            filtered_x, filtered_y, 
            color='gold', edgecolor='black', s=100, 
            label='Episodes Above Mean Reward'
        )

    plt.title(title)
    plt.xlabel('Episode SND')
    plt.ylabel('Episode Reward')
    plt.legend()
    plt.grid(True)
    plt.xlim(0, 4)
    plt.ylim(-2, 6)

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close()
    pil_image = Image.open(buf)
    wandb_image = wandb.Image(pil_image)
    return wandb_image
        
def z_score_normalize(data):
    """Normalizes a list or array of data using the Z-score method."""
    mean_val = np.mean(data)
    std_val = np.std(data)
    if std_val == 0:
        return np.zeros_like(data)
    normalized_data = (data - mean_val) / std_val
    return normalized_data

def calculate_cohesion(x, y, center):
    """Calculates the average distance of points from a center."""
    points = np.column_stack((x, y))
    distances = np.array([euclidean(point, center) for point in points])
    return distances

def find_centroid(x, y):
    """Finds the centroid of a set of points."""
    points = np.column_stack((x, y))
    if len(points) == 0:
        return np.array([np.nan, np.nan])
    kmeans = KMeans(n_clusters=1, random_state=0, n_init='auto')
    kmeans.fit(points)
    return kmeans.cluster_centers_[0]

class NormLoggerCallback(Callback):
    """
    Callback to log some training metrics
    """

    def on_batch_collected(self, batch: TensorDictBase):
        for group in self.experiment.group_map.keys():
            keys_to_norm = [
                (group, "f"),
                (group, "g"),
                (group, "fdivg"),
                (group, "logits"),
                (group, "observation"),
                (group, "out_loc_norm"),
                (group, "estimated_snd"),
                (group, "scaling_ratio"),
            ]
            to_log = {}

            for key in keys_to_norm:
                value = batch.get(key, None)
                if value is not None:
                    to_log.update(
                        {"/".join(("collection",) + key): torch.mean(value).item()}
                    )
            self.experiment.logger.log(
                to_log,
                step=self.experiment.n_iters_performed,
            )

class TagCurriculum(Callback):
    """
    Tag curriculum used to freeze the green agents' policies during training
    """

    def __init__(self, simple_tag_freeze_policy_after_frames, simple_tag_freeze_policy):
        super().__init__()
        self.n_frames_train = simple_tag_freeze_policy_after_frames
        self.simple_tag_freeze_policy = simple_tag_freeze_policy
        self.activated = not simple_tag_freeze_policy

    def on_setup(self):
        # Log params
        self.experiment.logger.log_hparams(
            simple_tag_freeze_policy_after_frames=self.n_frames_train,
            simple_tag_freeze_policy=self.simple_tag_freeze_policy,
        )
        # Make agent group homogeneous
        policy = self.experiment.group_policies["agents"]
        model = get_het_model(policy)
        # Set the desired SND of the green agent team to 0
        # This is not important as the green agent team is composed of 1 agent
        model.desired_snd[:] = 0

    def on_batch_collected(self, batch: TensorDictBase):
        if (
            self.experiment.total_frames >= self.n_frames_train
            and not self.activated
            and self.simple_tag_freeze_policy
        ):
            del self.experiment.train_group_map["agents"]
            self.activated = True

class ActionSpaceLoss(Callback):
    """
    Loss to disincentivize actions outside of the space
    """

    def __init__(self, use_action_loss, action_loss_lr):
        super().__init__()
        self.opt_dict = {}
        self.use_action_loss = use_action_loss
        self.action_loss_lr = action_loss_lr

    def on_setup(self):
        # Log params
        self.experiment.logger.log_hparams(
            use_action_loss=self.use_action_loss, action_loss_lr=self.action_loss_lr
        )

    def on_train_step(self, batch: TensorDictBase, group: str) -> TensorDictBase:
        if not self.use_action_loss:
            return
        policy = self.experiment.group_policies[group]
        model = get_het_model(policy)
        if group not in self.opt_dict:
            self.opt_dict[group] = torch.optim.Adam(
                model.parameters(), lr=self.action_loss_lr
            )
        opt = self.opt_dict[group]
        loss = self.action_space_loss(group, model, batch)
        loss_td = TensorDict({"loss_action_space": loss}, [])

        loss.backward()

        grad_norm = self.experiment._grad_clip(opt)
        loss_td.set(
            f"grad_norm_action_space",
            torch.tensor(grad_norm, device=self.experiment.config.train_device),
        )

        opt.step()
        opt.zero_grad()

        return loss_td

    def action_space_loss(self, group, model, batch):
        logits = model._forward(
            batch.select(*model.in_keys), compute_estimate=True, update_estimate=False
        ).get(
            model.out_key
        )  # Compute logits from batch
        if model.probabilistic:
            logits, _ = torch.chunk(logits, 2, dim=-1)
        out_loc_norm = overflowing_logits_norm(
            logits, self.experiment.action_spec[group, "action"]
        )  # Compute how much they overflow outside the action space bounds

        # Penalise the maximum overflow over the agents
        max_overflowing_logits_norm = out_loc_norm.max(dim=-1)[0]

        loss = max_overflowing_logits_norm.pow(2).mean()
        return loss


# This callback is used to learn the optimal desired_snd using a multi-armed bandit approach.
class SndCallback(Callback):
    """
    Callback used to compute SND during evaluations
    """
    def __init__(
        self,
        control_group: str,
        proportional_gain: float,
        initial_snd: float,
    ):
        super().__init__()
        self.control_group = control_group
        self.proportional_gain = proportional_gain
        self.initial_snd = initial_snd

        
        # Controller state variables
        self._r_baseline = 0.0
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

        if self.control_group not in self.experiment.group_policies:
            print(f"\nWARNING: Controller group '{self.control_group}' not found. Disabling controller.\n")
            return

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

        plot = print_plt(episode_snd, episode_returns, "SND vs. Reward per Episode", self.model.desired_snd.item())
        
        logs_to_push[f"Dico/snd_actual"] = actual_snd
        logs_to_push[f"Dico/target_snd"] = self.model.desired_snd.item()
        logs_to_push.update({
            "Dico/mean_return": mean_return,
            "Dico/score": correlation_score,
            "Dico/DiversityVreward": plot,
        })
        self.experiment.logger.log(logs_to_push, step=self.experiment.n_iters_performed)

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

        image = print_plt(episode_snd, episode_returns, "SND vs. Reward per Episode", self.model.desired_snd.item())
        
        logs_to_push[f"simple_control/snd_actual"] = actual_snd
        logs_to_push[f"simple_control/target_snd"] = self.model.desired_snd.item()
        logs_to_push.update({
            "simple_control/mean_return": mean_return,
            "simple_control/improvement": improvement,
            "simple_control/score": correlation_score,
            "simple_control/Update": update,
            "simple_control/update_step_clamped": update_step_clamped,
            "simple_control/plot": image,
        })
        self.experiment.logger.log(logs_to_push, step=self.experiment.n_iters_performed)

class pIControllerCallback(Callback):
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
        self.integral_score = 0.0

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
            "integral_score": self.integral_score,
        }
        self.experiment.logger.log_hparams(**hparams)

        if self.control_group not in self.experiment.group_policies:
            print(f"\nWARNING: Controller group '{self.control_group}' not found. Disabling controller.\n")
            return

        policy = self.experiment.group_policies[self.control_group]
        self.model = get_het_model(policy)

        if isinstance(self.model, HetControlMlpEmpirical):
            print(f"\n✅ SUCCESS: PI Controller initialized for group '{self.control_group}'.")
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
            self.integral_score = 0.0
            self._r_baseline = mean_return
            self._is_first_step = False
            improvement = 0.0
        else:
            improvement = mean_return - self._r_baseline
            self.integral_score += correlation_score
            self._r_baseline = mean_return

        # The integral term uses its own gain
        update = correlation_score + improvement + 0.01 * self.integral_score
        update_step = self.proportional_gain * update
        update_step = torch.tensor(update_step)
        update_step_clamped = self.max_update_step * torch.tanh(update_step / self.max_update_step).item()

        new_snd_tensor = self.model.desired_snd.clone().detach() + update_step_clamped
        self.model.desired_snd[:] = torch.clamp(new_snd_tensor, min=0.0)
        
        print(f"Updated SND: {self.model.desired_snd.item()}, Current_snd: {actual_snd}), (Update Step: {update_step_clamped}) ")

        image = print_plt(episode_snd, episode_returns, "SND vs. Reward per Episode", self.model.desired_snd.item())
        
        logs_to_push[f"Pi_Control/snd_actual"] = actual_snd
        logs_to_push[f"Pi_Control/target_snd"] = self.model.desired_snd.item()
        logs_to_push.update({
            "Pi_Control/mean_return": mean_return,
            "Pi_Control/improvement": improvement,
            "Pi_Control/score": correlation_score,
            "Pi_Control/Update": update,
            "Pi_Control/Integral_score": self.integral_score,
            # "Pi_Control/update_step_clamped": update_step_clamped,
            "Pi_Control/Plot": image,
        })
        self.experiment.logger.log(logs_to_push, step=self.experiment.n_iters_performed)

class clusterSndCallback(Callback):
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

        
        # Controller state variables
        self._r_baseline = 0.0
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

        if self.control_group not in self.experiment.group_policies:
            print(f"\nWARNING: Controller group '{self.control_group}' not found. Disabling controller.\n")
            return

        policy = self.experiment.group_policies[self.control_group]
        self.model = get_het_model(policy)

        if isinstance(self.model, HetControlMlpEmpirical):
            print(f"\n SUCCESS: ClusterBase Controller initialized for group '{self.control_group}'.")
            self.model.desired_snd[:] = float(self.initial_snd)
        else:
            print(f"\n WARNING: A compatible model was not found for group '{self.control_group}'. Disabling controller.\n")
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

        # --- Move your metric calculation here ---
        x = np.array(episode_snd)
        y = np.array(episode_returns)

        mean_x = np.mean(x)
        mean_y = np.mean(y)
        initial_correlation_score = correlation_score_f(x, y)
        centroid = find_centroid(x, y)
        distances = calculate_cohesion(x, y, centroid)
        initial_cohesion = np.mean(distances)

        # Filter points above mean_y
        filtered_x = x[y > mean_y]
        filtered_y = y[y > mean_y]

        # Only proceed if more than one filtered point
        if len(filtered_x) > 1:
            filtered_centroid = find_centroid(filtered_x, filtered_y)
            next_target_diversity = filtered_centroid[0]
            filtered_distances = calculate_cohesion(filtered_x, filtered_y, filtered_centroid)

            # Normalize reward and cohesion
            normalized_reward = z_score_normalize(y)
            normalized_cohesion_distances = z_score_normalize(distances)
            w1 = 1.0
            w2 = 0.5
            performance_score = w1 * np.mean(normalized_reward) - w2 * np.mean(normalized_cohesion_distances)
            filtered_cohesion = np.mean(filtered_distances)
        else:
            next_target_diversity = None
            filtered_cohesion = None
            performance_score = None

        
        # Optionally plot
        plot = plot_snd_vs_reward(x, y, mean_y, next_target_diversity, filtered_x, filtered_y)

        # --- Logging results ---
        logs_to_push.update({
            "ClusterBase/snd_actual": mean_x,
            "ClusterBase/mean_return": mean_y,
            "ClusterBase/score": initial_correlation_score,
            "ClusterBase/initial_cohesion": initial_cohesion,
            "ClusterBase/target_diversity": next_target_diversity if next_target_diversity is not None else 0,
            "ClusterBase/filtered_cohesion": filtered_cohesion if filtered_cohesion is not None else 0,
            "ClusterBase/performance_score": performance_score if performance_score is not None else 0,
            "ClusterBase/Plot": plot,
        })

        self.experiment.logger.log(logs_to_push, step=self.experiment.n_iters_performed)

        # Optionally update desired SND if you want an adaptive controller!
        if next_target_diversity is not None:
            self.model.desired_snd[:] = float(next_target_diversity)



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