# Standard library imports
import io
import os
import pickle
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union

# Third-party imports
import numpy as np
import torch
import torch.nn as nn
import wandb
from benchmarl.experiment.callback import Callback
from matplotlib import pyplot as plt
from PIL import Image
from tensordict import TensorDict, TensorDictBase

# Local application imports
# Note: Ensure these paths are correct for your project structure.
from AD2C.models.het_control_mlp_empirical import HetControlMlpEmpirical
from AD2C.models.het_control_mlp_esc import HetControlMlpEsc
from AD2C.models.het_control_mlp_snd import HetControlMlpEscSnd

from AD2C.snd import compute_behavioral_distance
from AD2C.utils import overflowing_logits_norm


# This tuple defines the model classes that the recursive search will look for.
HET_CONTROL_MODELS = (HetControlMlpEmpirical, HetControlMlpEsc, HetControlMlpEscSnd)

def _find_model_recursively(module: nn.Module) -> Optional[nn.Module]:
    """A helper function to recursively search for a compatible model."""
    if isinstance(module, HET_CONTROL_MODELS):
        return module
    
    for child in module.children():
        if (found_model := _find_model_recursively(child)) is not None:
            return found_model
            
    return None

def get_het_model(policy: nn.Module) -> Optional[nn.Module]:
    """
    Safely and recursively extracts a compatible heterogeneous model from a policy module.
    """
    # This pattern is common when dealing with models wrapped by utilities
    # like DataParallel or DistributedDataParallel.
    if hasattr(policy, 'module'):
        return _find_model_recursively(policy.module)
    return _find_model_recursively(policy)


def correlation_score_f(x: Union[List, np.ndarray], y: Union[List, np.ndarray]) -> float:
    """Calculates the Pearson correlation coefficient using NumPy's optimized function."""
    if len(x) != len(y) or len(x) <= 1:
        return 0.0
    
    # np.corrcoef is fast and handles edge cases well.
    correlation_matrix = np.corrcoef(x, y)
    correlation = correlation_matrix[0, 1]

    # Return 0.0 if the correlation is NaN (e.g., due to zero variance in inputs).
    return correlation if not np.isnan(correlation) else 0.0


def print_plt(x: List[float], y: List[float], title: str, desired_diversity: float) -> wandb.Image:
    """Generates a scatter plot and returns it as a Weights & Biases Image object."""
    mean_y = np.mean(y)
    fig, ax = plt.subplots()
    ax.scatter(x, y)
    ax.set_xlabel("Episode SND")
    ax.set_ylabel("Episode Reward")
    ax.set_title(title)
    ax.axhline(y=mean_y, color='gray', linestyle='--', label=f'Mean Reward ({mean_y:.2f})')
    ax.axvline(x=desired_diversity, color='red', linestyle='--', label='Desired Diversity')
    ax.legend()
    ax.set_ylim(-1, 4)
    ax.set_xlim(-0.5, 1.5)

    # Save plot to an in-memory buffer to avoid writing to disk.
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    plt.close(fig) # Important: Frees up memory.
    
    return wandb.Image(Image.open(buf))


def plot_snd_vs_reward(
    x: list, 
    y: list, 
    title: str, 
    next_target_diversity: float,
    current_target_diversity: float, # Corrected typo from "curent"
    filtered_x: list = None, 
    filtered_y: list = None
) -> wandb.Image:
    """Generates a more detailed SND vs. Reward plot and returns it as a wandb.Image."""
    mean_y = np.mean(y)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(x, y, alpha=0.6, label='All Episodes')
    ax.axhline(y=mean_y, color='gray', linestyle='--', label=f'Mean Reward ({mean_y:.2f})')
    ax.axvline(x=next_target_diversity, color='red', linestyle='-', label=f'Next Target Diversity ({next_target_diversity:.2f})')
    ax.axvline(x=current_target_diversity, color='blue', linestyle='--', label=f'Current Target Diversity ({current_target_diversity:.2f})')

    if filtered_x is not None and filtered_y is not None:
        ax.scatter(filtered_x, filtered_y, color='gold', edgecolor='black', s=100, label='Episodes Above Mean Reward')

    ax.set_title(title)
    ax.set_xlabel('Episode SND')
    ax.set_ylabel('Episode Reward')
    ax.legend()
    ax.grid(True)
    ax.set_xlim(-0.5, 2)
    ax.set_ylim(-1, 7)

    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    plt.close(fig)
    
    return wandb.Image(Image.open(buf))


def plot_trajectory(x_blue: list, y_blue: list, title: str, x_red: list, y_red: list) -> wandb.Image:
    """
    Plots two trajectories and returns the plot as a wandb.Image for consistent logging.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(x_blue, y_blue, label='Target', color='blue', marker='o')
    ax.plot(x_red, y_red, label='Actual', color='red', marker='x')

    ax.set_title(title, fontsize=16)
    ax.set_xlabel('Episode', fontsize=12)
    ax.set_ylabel('Reward / Diversity', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.6)
    
    # Save to buffer and return as wandb.Image, just like the other plot functions.
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    plt.close(fig)
    
    return wandb.Image(Image.open(buf))


def z_score_normalize(data: np.ndarray) -> np.ndarray:
    """Normalizes an array of data using the Z-score method."""
    mean_val = np.mean(data)
    std_val = np.std(data)
    if std_val == 0:
        return np.zeros_like(data)
    return (data - mean_val) / std_val


def calculate_cohesion(x: np.ndarray, y: np.ndarray, center: np.ndarray) -> np.ndarray:
    """
    Calculates the Euclidean distance of each point from a center using vectorization.
    """
    points = np.column_stack((x, y))
    # Vectorized calculation is much faster than iterating through points.
    return np.linalg.norm(points - center, axis=1)


def find_centroid(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Finds the centroid (geometric mean) of a set of points directly and efficiently.
    """
    points = np.column_stack((x, y))
    if points.shape[0] == 0:
        return np.array([np.nan, np.nan])
        
    # Using np.mean is the direct definition of a centroid and is far more
    # efficient than using a clustering algorithm like KMeans for this task.
    return np.mean(points, axis=0)