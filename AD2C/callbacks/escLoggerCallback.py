# Copyright (c) 2024.
# ProrokLab (https://www.proroklab.org/)
# All rights reserved.

from __future__ import annotations
import matplotlib.pyplot as plt
import torch
import numpy as np
import wandb
from tensordict import TensorDictBase
from benchmarl.experiment import Experiment
from benchmarl.experiment.callback import Callback


class EscLoggerCallback(Callback):
    """
    A callback to generate and log ESC diagnostic plots and metrics to W&B.
    """

    def __init__(self):
        self.history_data = []
        self.run_number = 0
        self.is_wandb_init = False

    def on_run_start(self, experiment: Experiment):
        """Called at the start of the experiment run."""
        self._init_wandb(experiment)

    def on_collect_end(self, experiment: Experiment):
        """
        Called after each data collection phase. This is where we log our data.
        """
        if not self.is_wandb_init: self._init_wandb(experiment)

        # Get the tensordict with the latest collected data
        latest_td = experiment.collector.get_filled_td()
        
        self._log_esc_diagnostics(latest_td)
        self.run_number += 1

    def on_run_end(self, experiment: Experiment):
        """Called at the end of the experiment run."""
        if self.is_wandb_init:
            wandb.finish()

    def _log_esc_diagnostics(self, latest_td: TensorDictBase):
        """
        Generates diagnostic plots and logs them and scalar metrics to W&B.
        """
        # --- 1. Extract Data ---
        # The same logic from our standalone plotting function
        td_episode = latest_td[:, 0]
        agent_group = "agents"
        
        try:
            scaling_ratio = td_episode.get((agent_group, "scaling_ratio"))[:, 0, 0].cpu().numpy()
            logits = td_episode.get((agent_group, "logits")).cpu()
            k_hat = td_episode.get((agent_group, "k_hat"))[0, 0, 0].item()
        except KeyError as e:
            print(f"Warning: Could not find key {e} in tensordict. Skipping W&B logging for this step.")
            return

        behavioral_diversity = logits.norm(dim=-1).std(dim=-1).numpy()

        # --- 2. Create the Figure ---
        fig, axes = plt.subplots(2, 2, figsize=(20, 12), constrained_layout=True)
        fig.suptitle(f"ESC Diagnostics - Collection Step #{self.run_number}", fontsize=20)
        
        # (Plotting code for Time-Series, Scatter, and Evolution is identical to before)
        # Plot 1: Time-Series
        ax1 = axes[0, 0]
        ax1_twin = ax1.twinx()
        ax1.plot(range(len(scaling_ratio)), scaling_ratio, color="blue", label="Scaling Ratio")
        ax1_twin.plot(range(len(behavioral_diversity)), behavioral_diversity, color="purple", linestyle="--", label="Diversity")
        ax1.set_title("A) Control Dynamics")
        ax1.set_ylabel("Scaling Ratio", color="blue")
        ax1_twin.set_ylabel("Diversity", color="purple")
        ax1.legend(); ax1_twin.legend(loc="upper right")

        # Plot 2: Scatter
        ax2 = axes[0, 1]
        abs_scaling_ratio = np.abs(scaling_ratio)
        ax2.scatter(abs_scaling_ratio, behavioral_diversity, alpha=0.4)
        slope, _ = np.polyfit(abs_scaling_ratio, behavioral_diversity, 1)
        ax2.set_title(f"B) Base Diversity ≈ {slope:.2f}")
        ax2.set_xlabel("Absolute Scaling Ratio")

        # Plot 3: Evolution
        ax3 = axes[1, 0]
        current_avg_ratio = scaling_ratio.mean()
        current_avg_diversity = behavioral_diversity.mean()
        self.history_data.append({"ratio": current_avg_ratio, "diversity": current_avg_diversity})
        if len(self.history_data) > 1:
            run_indices = range(len(self.history_data))
            avg_ratios = [d['ratio'] for d in self.history_data]
            avg_diversities = [d['diversity'] for d in self.history_data]
            ax3_twin = ax3.twinx()
            ax3.plot(run_indices, avg_ratios, 'o-', color="blue", label="Avg. Ratio")
            ax3_twin.plot(run_indices, avg_diversities, 's--', color="purple", label="Avg. Diversity")
            ax3.set_title("C) Learning Evolution")
            ax3.set_xlabel("Collection Step")
            ax3.legend(); ax3_twin.legend(loc="upper right")
        
        axes[1, 1].set_visible(False)

        # --- 3. Log to W&B ---
        wandb.log({
            "ESC/Diagnostics Plot": fig,
            "ESC/Average Scaling Ratio": current_avg_ratio,
            "ESC/Average Behavioral Diversity": current_avg_diversity,
            "ESC/k_hat": k_hat
        }, step=self.run_number)
        
        plt.close(fig)