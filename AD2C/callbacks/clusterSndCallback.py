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
from AD2C.snd import compute_behavioral_distance

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.spatial.distance import euclidean

from callbacks.utils import *
import networkx as nx 

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import io
from PIL import Image
import wandb

# def plot_agent_distances(pairwise_distances_tensor, n_agents):
#     G = nx.complete_graph(n_agents)
#     # Corrected line: move tensor to CPU and convert to numpy
#     distances = pairwise_distances_tensor.cpu().flatten().numpy()
    
#     # Assign distances to edges
#     edge_idx = 0
#     for u, v in G.edges():
#         if u != v:
#             G.edges[(u, v)]['weight'] = distances[edge_idx]
#             edge_idx += 1

#     edge_labels = {
#         (u, v): f"{G.edges[(u, v)]['weight']:.2f}" for u, v in G.edges()
#     }
    
#     # Invert weights for the spring layout
#     # The 'weight' parameter in spring_layout will use the values you assign
#     # so we don't need to create a separate inverted_weights dict.
    
#     # Use the spring layout
#     pos = nx.spring_layout(G, k=0.8, weight='weight')
    
#     # Draw the graph
#     fig = plt.figure(figsize=(10, 10))
    
#     # Draw nodes and labels
#     nx.draw_networkx_nodes(G, pos, node_color='lightblue', node_size=1000)
#     nx.draw_networkx_labels(G, pos, labels={i: f'Agent {i+1}' for i in range(n_agents)}, font_size=12)
    
#     # Draw edges and labels
#     nx.draw_networkx_edges(G, pos, edge_color='gray', width=2)
#     nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='red', font_size=10)
    
#     plt.title('Relative Distances Between Agents')
#     plt.axis('off')
    
#     # Convert plot to wandb image
#     buffer = io.BytesIO()
#     plt.savefig(buffer, format='png')
#     buffer.seek(0)
#     plt.close(fig)
#     return wandb.Image(Image.open(buffer))

def plot_agent_distances(pairwise_distances_tensor, n_agents):
    G = nx.complete_graph(n_agents)
    
    # Ensure the input is a 1D array of distances
    # This reshapes it to a 1D array if it's not already
    distances = pairwise_distances_tensor.cpu().numpy().flatten()
    actual_snd = pairwise_distances_tensor.mean().item()
    
    edge_idx = 0
    # Assign a scalar distance to each edge
    for u, v in G.edges():
        if edge_idx < len(distances):
            G.edges[(u, v)]['distance'] = distances[edge_idx].item()
            G.edges[(u, v)]['weight'] = 1.0 / (distances[edge_idx].item() + 1e-5)
            edge_idx += 1
    
    edge_labels = {
        (u, v): f"{data['distance']:.2f}" for u, v, data in G.edges(data=True)
    }
    
    pos = nx.spring_layout(G, k=0.8, weight='weight')
    
    fig = plt.figure(figsize=(10, 10))
    
    nx.draw_networkx_nodes(G, pos, node_color='lightblue', node_size=1000)
    nx.draw_networkx_labels(G, pos, labels={i: f'Agent {i+1}' for i in range(n_agents)}, font_size=12)
    nx.draw_networkx_edges(G, pos, edge_color='gray', width=2)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='red', font_size=10)
    
    plt.title(f'Diversity: {actual_snd:.2f}')
    plt.axis('off')
    
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    plt.close(fig)
    return wandb.Image(Image.open(buffer))  

def plot_trajectory_3d_no_number(snd, returns, episodes, target_diversity):
    """
    Generates a 3D line chart of SND, mean return, and episode number, including the target diversity trajectory.
    """

    fig = plt.figure(figsize=(14, 12))
    ax = fig.add_subplot(111, projection='3d')

    # Plot Actual Diversity Trajectory
    ax.plot(episodes, snd, returns, c='b', marker='o', label='Actual Diversity Trajectory')
    for ep, s, r in zip(episodes, snd, returns):
        ax.text(ep, s, r, f'({ep}, {s:.2f}, {r:.2f})',
                color='black', ha='right', va='bottom', fontsize=5)

    # Plot Target Diversity Trajectory if it is not None and has same length as episodes
    if target_diversity is not None and len(target_diversity) == len(episodes):
        ax.plot(episodes, target_diversity, returns, c='r', marker='x', linestyle='--', label='Target Diversity Trajectory')
        for ep, td, r in zip(episodes, target_diversity, returns):
            ax.text(ep, td, r, f'({ep}, {td:.2f}, {r:.2f})',
                    color='red', ha='left', va='top', fontsize=5)

    ax.set_ylim(0, 3)    # SND axis scale
    ax.set_zlim(-2, 5)           # Return axis scale

    ax.set_xlabel('Episode Number')
    ax.set_ylabel('SND (Behavioral Distance)')
    ax.set_zlabel('Mean Return')
    ax.set_title('3D Trajectory of Mean Return and SND')
    ax.legend()

    # Save to bytes buffer for wandb
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    plt.close(fig)
    return wandb.Image(Image.open(buffer))

def plot_trajectory_3d(snd, returns, episodes, target_diversity):
    """
    Generates a 3D line chart of SND, mean return, and episode number, including the target diversity trajectory.
    """

    fig = plt.figure(figsize=(14, 12))
    ax = fig.add_subplot(111, projection='3d')

    # Plot Actual Diversity Trajectory
    ax.plot(episodes, snd, returns, c='b', marker='o', label='Actual Diversity Trajectory')
    for ep, s, r in zip(episodes, snd, returns):
        ax.text(ep, s, r, f'({ep}, {s:.2f}, {r:.2f})',
                color='black', ha='right', va='bottom', fontsize=8)

    # Plot Target Diversity Trajectory if it is not None and has same length as episodes
    if target_diversity is not None and len(target_diversity) == len(episodes):
        ax.plot(episodes, target_diversity, returns, c='r', marker='x', linestyle='--', label='Target Diversity Trajectory')
        for ep, td, r in zip(episodes, target_diversity, returns):
            ax.text(ep, td, r, f'({ep} ,{td:.2f}, {r:.2f})',
                    color='red', ha='left', va='top', fontsize=8)

    ax.set_ylim(0, 4)    # SND axis scale
    ax.set_zlim(-2, 5)           # Return axis scale

    ax.set_xlabel('Episode Number')
    ax.set_ylabel('SND (Behavioral Distance)')
    ax.set_zlabel('Mean Return')
    ax.set_title('3D Trajectory of Mean Return and SND')
    ax.legend()
    
    ax.view_init(elev=20, azim=-60)  

    # Save to bytes buffer for wandb
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    plt.close(fig)
    return wandb.Image(Image.open(buffer))


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
        
        self.eps_target_Diversity = []
        self.eps_actual_diversity = []
        self.eps_number = []

        
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

        def get_agent_actions(obs):
            # Compute actions for all agents given observation
            actions = []
            for i in range(self.model.n_agents):
                temp_td = TensorDict(
                    {(self.control_group, "observation"): obs},
                    batch_size=obs.shape[:-1]
                )
                action_td = self.model._forward(temp_td, agent_index=i, compute_estimate=False)
                actions.append(action_td.get(self.model.out_key))
            return actions

        with torch.no_grad():
            for r in rollouts:
                obs = r.get((self.control_group, "observation"))
                agent_actions = get_agent_actions(obs)
                pairwise_distances_tensor = compute_behavioral_distance(agent_actions, just_mean=False)
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

        # Filter points above mean_y
        mask = y > mean_y
        filtered_x, filtered_y = x[mask], y[mask]

        # If enough filtered points, calculate further metrics
        if len(filtered_x) > 1:
            filtered_centroid = find_centroid(filtered_x, filtered_y)
            next_target_diversity = filtered_centroid[0]
            filtered_distances = calculate_cohesion(filtered_x, filtered_y, filtered_centroid)
            filtered_cohesion = np.mean(filtered_distances)
            
            # Normalize for performance score
            normalized_reward = z_score_normalize(y)
            normalized_cohesion_distances = z_score_normalize(distances)
            w1, w2 = 1.0, 0.5
            performance_score = w1 * np.mean(normalized_reward) - w2 * np.mean(normalized_cohesion_distances)
        else:
            next_target_diversity = 0
            filtered_cohesion = 0
            performance_score = 0

        # --- Logging and plotting ---
        self.eps_actual_diversity.append(mean_x)
        self.eps_number.append(self.experiment.n_iters_performed)
        self.eps_target_Diversity.append(next_target_diversity)

        if not hasattr(self, 'eps_mean_returns'):
            self.eps_mean_returns = []
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

        # Agent distances graph (only if multiple agents)
        if self.model and self.model.n_agents > 1:
            sample_rollout = rollouts[0]
            obs = sample_rollout.get((self.control_group, "observation"))
            agent_actions = get_agent_actions(obs)
            pairwise_distances_tensor = compute_behavioral_distance(agent_actions, just_mean=False)
            graph_plot = plot_agent_distances(pairwise_distances_tensor, self.model.n_agents)
            logs_to_push["ClusterBase/Agent_Distances_Graph"] = graph_plot

        # Trajectory plot if more than one episode
        if len(self.eps_number) > 1:
            trajectory_plot = plot_trajectory_3d(
                snd=self.eps_actual_diversity,
                returns=self.eps_mean_returns,
                episodes=self.eps_number,
                target_diversity=self.eps_target_Diversity
            )
            logs_to_push["ClusterBase/Trajectory_Plot_3D"] = trajectory_plot

        self.experiment.logger.log(logs_to_push, step=self.experiment.n_iters_performed)

        # Optionally update desired SND if you want an adaptive controller!
        if next_target_diversity:
            self.model.desired_snd[:] = float(next_target_diversity)
