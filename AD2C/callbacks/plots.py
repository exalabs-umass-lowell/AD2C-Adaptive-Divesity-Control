from typing import List

from PIL import Image
from matplotlib import pyplot as plt
import numpy as np
import io
import wandb
from AD2C.snd import compute_behavioral_distance

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.spatial.distance import euclidean

from callbacks.utils import *
import networkx as nx 
import pandas as pd

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import io
from PIL import Image
import wandb
import csv
import itertools

def plot_distance_history(distance_history, n_agents):
    """
    Plots the growth of distances for each agent pair over time on a single graph.

    Args:
        distance_history (list): A list of 1D tensors, where each tensor contains the 
                                 pairwise distances at a specific time step.
        n_agents (int): The total number of agents.

    Returns:
        wandb.Image: An image object of the plot for logging.
    """
    # 1. Identify all unique agent pairs (e.g., (0,1), (0,2), (1,2), etc.)
    agent_pairs = list(itertools.combinations(range(n_agents), 2))
    
    # 2. Restructure the data for easier plotting
    # Stack the list of tensors into a single [num_steps, num_pairs] tensor
    if not distance_history or not isinstance(distance_history[0], torch.Tensor):
        print("Warning: distance_history is empty or does not contain tensors. Skipping plot.")
        return None
        
    history_tensor = torch.stack([d.cpu() for d in distance_history])
    # Transpose to get a [num_pairs, num_steps] tensor, making it easy to plot each pair's history
    history_by_pair = history_tensor.T
    
    # 3. Create the plot
    fig, ax = plt.subplots(figsize=(12, 8))
    time_steps = range(len(distance_history))
    
    for i, pair in enumerate(agent_pairs):
        agent1, agent2 = pair
        # Plot the distance history for the current pair
        ax.plot(time_steps, history_by_pair[i], label=f'Agents {agent1+1}-{agent2+1}')
        
    # 4. Format the plot for clarity
    ax.set_title('Growth of Pairwise Agent Distances Over Time', fontsize=16)
    ax.set_xlabel('Time Step', fontsize=12)
    ax.set_ylabel('Distance', fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.6)
    
    # Place legend outside the plot area to avoid clutter, especially with many agents
    ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), borderaxespad=0.)
    
    # Adjust layout to make room for the legend
    fig.tight_layout()
    
    # 5. Save the plot to a memory buffer to return as an image
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png', bbox_inches='tight')
    buffer.seek(0)
    plt.close(fig) # Close the figure to free up memory
    
    return wandb.Image(Image.open(buffer))



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


def plot_trajectory_2d(episodes, snd, returns, target_diversity=None):
    """
    Generates a single 2D line chart plotting SND and Mean Return trajectory.
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot Actual Trajectory
    ax.plot(snd, returns, c='b', marker='o', label='Actual Trajectory')

    # Add text annotations for episode numbers to show the time progression
    for i, ep in enumerate(episodes):
        ax.annotate(f'Ep: {ep}', (snd[i], returns[i]), fontsize=8, ha='right')

    # Plot Target Diversity Trajectory if available
    if target_diversity is not None and len(target_diversity) == len(episodes):
        ax.plot(target_diversity, returns, c='r', marker='x', linestyle='--', label='Target Diversity Trajectory')
    
    # Set labels, title, and legend
    ax.set_xlabel('SND (Behavioral Distance)')
    ax.set_ylabel('Mean Return')
    ax.set_title('Trajectory of Mean Return vs. SND')
    ax.legend()
    ax.grid(True)

    # Save to a bytes buffer for wandb
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

def save_trajectory_data_to_csv(episodes, snd, returns, target_diversity=None, run_name_suffix=""):
    """
    Creates a pandas DataFrame and saves it as a CSV file in a
    new directory named 'Saved Run Tables'.
    The run_name_suffix is added to the filename for better organization.
    """
    # Define the directory name and create it if it doesn't exist
    save_dir = 'Saved Run Tables'
    os.makedirs(save_dir, exist_ok=True)
    
    # Convert input lists to NumPy arrays to enable numerical operations
    episodes_np = np.array(episodes)
    
    # Create a pandas DataFrame
    data_dict = {
        'episode_number': episodes_np,
        'reward': returns,
        'actual_snd': snd
    }
    
    if target_diversity is not None and len(target_diversity) == len(episodes):
        data_dict['target_snd'] = target_diversity
    
    # Calculate eval_iter using the NumPy array
    eval_iter = np.floor(episodes_np / 200).astype(int)
    data_dict['eval_iter'] = eval_iter
    
    df = pd.DataFrame(data_dict)
    
    # Construct the full file path and save the CSV
    # The filename now includes the run_name_suffix
    file_name = f'trajectory_data_{run_name_suffix}.csv'
    file_path = os.path.join(save_dir, file_name)
    df.to_csv(file_path, index=False)
    
    print(f"Data table saved to {file_path}")
    

def save_pairwise_diversity_to_csv(pairwise_distances_tensor, episode_number, n_agents, file_prefix='agent_distances'):
    folder = '/home/svarp/Desktop/Projects/AD2C/ControllingBehavioralDiversity/Saved Pairwise Diversities'
    os.makedirs(folder, exist_ok=True)
    distances = pairwise_distances_tensor.cpu().numpy().flatten()
    filename = f"{file_prefix}_ep{episode_number}_n{n_agents}.csv"
    filepath = os.path.join(folder, filename)
    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['edge_index', 'distance'])
        for idx, dist in enumerate(distances):
            writer.writerow([idx, dist])
    print(f"Saved agent distances to {filepath}")
    return filepath
