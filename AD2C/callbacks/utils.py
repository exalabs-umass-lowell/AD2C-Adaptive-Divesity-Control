import os
import pickle
from typing import List, Dict,Any, Callable

from PIL import Image
from matplotlib import pyplot as plt
import numpy as np
import io
import wandb
from tensordict import TensorDictBase, TensorDict, nn
from typing import List, Dict, Union

from AD2C.models.het_control_mlp_empirical import HetControlMlpEmpirical

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.spatial.distance import euclidean

from models.het_control_mlp_esc import HetControlMlpEsc
from typing import List, Tuple, Type, Optional

import os
import pickle
from typing import List, Dict,Any, Callable

from PIL import Image
from matplotlib import pyplot as plt
import numpy as np
import io
import torch
import torch.nn as nn
import wandb
from tensordict import TensorDictBase, TensorDict
from typing import List, Dict, Union

from benchmarl.experiment.callback import Callback
from AD2C.models.het_control_mlp_empirical import HetControlMlpEmpirical
# Make sure to import all compatible models
from AD2C.models.het_control_mlp_esc import HetControlMlpEsc
from AD2C.snd import compute_behavioral_distance
from AD2C.utils import overflowing_logits_norm




HET_CONTROL_MODELS: Tuple[Type[nn.Module]] = (HetControlMlpEmpirical, HetControlMlpEsc)



def get_het_model(policy):
    # This assumes policy.module is an nn.Sequential
    model = policy.module
    # FIX 2: Make the function more robust by checking for the base module type
    if isinstance(model, HET_CONTROL_MODELS):
        return model
    # If wrapped in a Sequential, find the correct module
    if isinstance(model, nn.Sequential):
        for module in model:
            if isinstance(module, HET_CONTROL_MODELS):
                return module
    # Return None if no compatible model is found
    return None



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

    ax.set_ylim(-1, 4)
    ax.set_xlim(-0.5, 1.5)

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
    curent_target_diversity: float, 
    filtered_x: list = None, 
    filtered_y: list = None
):
    mean_y = np.mean(y)

    plt.figure(figsize=(10, 6))
    plt.scatter(x, y, alpha=0.6, label='All Episodes')
    plt.axhline(y=mean_y, color='gray', linestyle='--', label=f'Mean Reward ({mean_y:.2f})')
    plt.axvline(x=next_target_diversity, color='red', linestyle='-', 
                label=f'Next Target Diversity ({next_target_diversity:.2f})')
    plt.axvline(x=curent_target_diversity, color='blue', linestyle='--', 
                label=f'Current Target Diversity ({curent_target_diversity:.2f})')

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
    plt.xlim(-0.5, 2)
    plt.ylim(-1, 7)

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close()
    pil_image = Image.open(buf)
    wandb_image = wandb.Image(pil_image)
    return wandb_image

def plot_trajectory(x_blue: list, y_blue: list, title: str, x_red: list, y_red: list):
    # Create the plot
    plt.figure(figsize=(10, 6))

    # Plot the first line (blue)
    plt.plot(x_blue, y_blue, label='Target', color='blue', marker='o')

    # Plot the second line (red)
    plt.plot(x_red, y_red, label='Actual', color='red', marker='x')

    # Set the plot title and axis labels
    plt.title(title, fontsize=16)
    plt.xlabel('Episode', fontsize=12)
    plt.ylabel('Reward / Diversity', fontsize=12)

    # Add a legend to distinguish the lines
    plt.legend(fontsize=10)

    # Add a grid for better readability
    plt.grid(True, linestyle='--', alpha=0.6)

    # Save the plot to a file
    plt.savefig('trajectory_plot.png')




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

