from typing import List
import torch
import numpy as np
import matplotlib.pyplot as plt

import plotly.graph_objects as go
from plotly.subplots import make_subplots

# benchmarl and tensordict imports
from tensordict import TensorDictBase, TensorDict
from benchmarl.experiment import Experiment
from benchmarl.experiment.callback import Callback

# Your project-specific imports
from AD2C.models.het_control_mlp_empirical import HetControlMlpEmpirical
from AD2C.models.het_control_mlp_esc import HetControlMlpEsc
from AD2C.snd import compute_behavioral_distance
from .plots import plot_trajectory_2d, plot_agent_distances
from callbacks.utils import get_het_model


class TrajectoryLoggerCallback(Callback):
    """
    A comprehensive callback that logs:
    1. Evaluation metrics (SND, Return) from evaluation rollouts.
    2. Detailed ESC diagnostics (plots, k_hat) from training data collection.
    """
    def __init__(self, control_group: str):
        super().__init__()
        self.control_group = control_group
        
        # --- State for Evaluation Plots ---
        self.eps_actual_diversity = []
        self.eps_mean_returns = []
        self.eps_number = []
        
        # --- State for ESC Training Plots ---
        self.esc_history_data = []
        self.collect_step_count = 0
        
        self.model = None

    # CORRECTED SIGNATURE: Added the 'experiment' argument
    def on_setup(self):
        """Finds the policy model for the specified group."""
        if self.control_group not in self.experiment.group_policies:
            print(f"\nWARNING: Logger group '{self.control_group}' not found. Disabling logger.\n")
            return

        policy = self.experiment.group_policies[self.control_group]
        self.model = get_het_model(policy)

        if isinstance(self.model, (HetControlMlpEmpirical, HetControlMlpEsc)):
            print(f"\nSUCCESS: Trajectory Logger initialized for group '{self.control_group}'.")
        else:
            print(f"\nWARNING: A compatible model was not found for '{self.control_group}'. Disabling logger.\n")
            self.model = None

    # This hook runs after TRAINING data has been collected
    def on_batch_collected(self, batch: TensorDictBase):
        """Logs ESC-specific diagnostics from the training data collector."""
        if not isinstance(self.model, HetControlMlpEsc):
            return

        # Log both scalars and plots using the collected data
        self._log_esc_scalars(self.experiment, batch)
        self._log_esc_plot(self.experiment, batch)

        self.collect_step_count += 1

    # CORRECTED SIGNATURE: Added the 'experiment' argument
    def on_evaluation_end(self, rollouts: List[TensorDictBase]):
        """Calculates and logs metrics from EVALUATION rollouts."""
        if self.model is None: return

        episode_snd = []
        episode_returns = []
        plot_generated = False
        logs_to_push = {}

        with torch.no_grad():
            for r in rollouts:
                agent_actions = self._get_agent_actions_for_rollout(r)
                
                pairwise_distances = compute_behavioral_distance(agent_actions, just_mean=False)
                episode_snd.append(pairwise_distances.mean().item())

                if not plot_generated:
                    agent_distance_plot = plot_agent_distances(
                        pairwise_distances_tensor=pairwise_distances,
                        n_agents=self.model.n_agents
                    )
                    logs_to_push["eval/Agent_Distances"] = agent_distance_plot
                    plot_generated = True

                reward_key = ('next', self.control_group, 'reward')
                total_reward = r.get(reward_key).sum().item()
                episode_returns.append(total_reward)

        if not episode_returns: return

        mean_actual_snd = np.mean(episode_snd)
        mean_return = np.mean(episode_returns)

        self.eps_actual_diversity.append(mean_actual_snd)
        self.eps_mean_returns.append(mean_return)
        self.eps_number.append(self.experiment.n_iters_performed)

        logs_to_push["eval/actual_snd"] = mean_actual_snd
        logs_to_push["eval/mean_return"] = mean_return

        if len(self.eps_number) > 1:
            if np.isfinite(self.eps_actual_diversity).all() and np.isfinite(self.eps_mean_returns).all():
                plot_2d = plot_trajectory_2d(
                    snd=self.eps_actual_diversity,
                    returns=self.eps_mean_returns,
                    episodes=self.eps_number
                )
                logs_to_push["eval/SND_vs_Reward_Trajectory"] = plot_2d

        self.experiment.logger.log(logs_to_push, step=self.experiment.n_iters_performed)

    def _log_esc_scalars(self, experiment: Experiment, batch: TensorDictBase):
        """
        Logs the mean of important ESC tensors as scalars, based on your reference.
        """
        # Define all the keys your model logs to the tensordict
        keys_to_log = [
            (self.control_group, "logits"),
            (self.control_group, "out_loc_norm"),
            (self.control_group, "scaling_ratio"),
            (self.control_group, "k_hat"),
            (self.control_group, "esc_dither"),
            (self.control_group, "esc_reward_J"),
            (self.control_group, "esc_grad_estimate"),
            (self.control_group, "esc_k_hat_update"),
        ]
        
        to_log_dict = {}
        for key in keys_to_log:
            value = batch.get(key, None)
            if value is not None:
                # Format the key for W&B, e.g., "collection/agents/k_hat_mean"
                log_name = f"collection/{'/'.join(key)}"
                to_log_dict[log_name] = torch.mean(value.float()).item()
        
        if to_log_dict:
            experiment.logger.log(to_log_dict, step=self.collect_step_count)

    def _log_esc_plot(self, experiment: Experiment, latest_td: TensorDictBase):
        """
        Generates and logs a visually improved and interactive ESC diagnostic plot using Plotly,
        including mean diversity lines for analysis.
        """
        # 1. --- Data Extraction ---
        try:
            td_episode = latest_td[:, 0]
            scaling_ratio = td_episode.get((self.control_group, "scaling_ratio")).mean(dim=(1, -1)).cpu().numpy()
            logits = td_episode.get((self.control_group, "logits")).cpu()
            k_hat_tensor = td_episode.get((self.control_group, "k_hat"))
            k_hat = k_hat_tensor[0, 0].item()
            behavioral_diversity = logits.norm(dim=-1).std(dim=1).cpu().numpy()
            
            if scaling_ratio.shape != behavioral_diversity.shape:
                print("Warning: Shape mismatch. Skipping plot.")
                return
                
        except (KeyError, IndexError) as e:
            print(f"Warning: Could not find key for plotting ({e}). Skipping plot.")
            return

        # --- Calculate mean diversity for plotting horizontal lines ---
        mean_diversity = np.mean(behavioral_diversity)

        # 2. --- Create the Plot using Plotly ---
        fig = make_subplots(
            rows=2, cols=2,
            specs=[[{"secondary_y": True}, {}],
                [{"secondary_y": True}, None]], # 'None' hides the bottom-right plot
            subplot_titles=("A) Control Dynamics", "B) Base Diversity Estimate", "C) Learning Evolution")
        )

        # Plot A: Time-Series Dynamics
        fig.add_trace(go.Scatter(y=scaling_ratio, name="Scaling Ratio", line=dict(color='royalblue')),
                    row=1, col=1, secondary_y=False)
        fig.add_trace(go.Scatter(y=behavioral_diversity, name="Diversity", line=dict(color='purple', dash='dash')),
                    row=1, col=1, secondary_y=True)
        
        # Add horizontal mean line to Plot A (on the secondary y-axis)
        fig.add_shape(type="line", xref="x1", yref="y2",
                    x0=0, y0=mean_diversity, x1=len(behavioral_diversity)-1, y1=mean_diversity,
                    line=dict(color="limegreen", width=2, dash="dot"))

        # Plot B: Base Diversity Estimate
        abs_scaling_ratio = np.abs(scaling_ratio)
        fig.add_trace(go.Scatter(x=abs_scaling_ratio, y=behavioral_diversity, mode='markers',
                                name='Data Points', marker=dict(opacity=0.6, color='royalblue')),
                    row=1, col=2)
        if len(abs_scaling_ratio) > 1:
            slope, intercept = np.polyfit(abs_scaling_ratio, behavioral_diversity, 1)
            fit_line = slope * abs_scaling_ratio + intercept
            fig.add_trace(go.Scatter(x=abs_scaling_ratio, y=fit_line, mode='lines',
                                    name=f'Fit (Slope ≈ {slope:.2f})', line=dict(color='red')),
                        row=1, col=2)

        # Add horizontal mean line to Plot B
        fig.add_shape(type="line", xref="x2", yref="y3",
                    x0=np.min(abs_scaling_ratio), y0=mean_diversity, x1=np.max(abs_scaling_ratio), y1=mean_diversity,
                    line=dict(color="limegreen", width=2, dash="dot"))
                    
        # Add an annotation to label the mean line in Plot B
        fig.add_annotation(xref="paper", yref="y3", x=0.98, y=mean_diversity,
                        text=f"Mean Div: {mean_diversity:.2f}", showarrow=False, yshift=10,
                        font=dict(color="limegreen"), bgcolor="rgba(255,255,255,0.8)")

        # Plot C: Learning Evolution
        self.esc_history_data.append({"ratio": scaling_ratio.mean(), "diversity": behavioral_diversity.mean()})
        if len(self.esc_history_data) > 1:
            indices = list(range(len(self.esc_history_data)))
            ratios = [d['ratio'] for d in self.esc_history_data]
            diversities = [d['diversity'] for d in self.esc_history_data]
            
            fig.add_trace(go.Scatter(x=indices, y=ratios, name="Avg. Ratio", line=dict(color='royalblue')),
                        row=2, col=1, secondary_y=False)
            fig.add_trace(go.Scatter(x=indices, y=diversities, name="Avg. Diversity", line=dict(color='purple', dash='dash')),
                        row=2, col=1, secondary_y=True)

        # 3. --- Update Layout and Axis Titles ---
        fig.update_layout(
            title_text=f"ESC Diagnostics - Collection Step #{self.collect_step_count}",
            legend_title_text="Metrics",
            height=700,
            template="plotly_white"
        )
        fig.update_xaxes(title_text="Time Step", row=1, col=1)
        fig.update_yaxes(title_text="Scaling Ratio", row=1, col=1, secondary_y=False)
        fig.update_yaxes(title_text="Diversity", showgrid=False, row=1, col=1, secondary_y=True)
        fig.update_xaxes(title_text="Absolute Scaling Ratio", row=1, col=2)
        fig.update_yaxes(title_text="Behavioral Diversity", row=1, col=2)
        fig.update_xaxes(title_text="Collection Epoch", row=2, col=1)
        fig.update_yaxes(title_text="Avg. Ratio", row=2, col=1, secondary_y=False)
        fig.update_yaxes(title_text="Avg. Diversity", showgrid=False, row=2, col=1, secondary_y=True)
        
        # 4. --- Logging ---
        logs_to_push = {
            "ESC/Diagnostics_Plot": fig,
            "ESC/average_scaling_ratio": scaling_ratio.mean(),
            "ESC/average_behavioral_diversity": behavioral_diversity.mean(),
            "ESC/k_hat": k_hat
        }
        experiment.logger.log(logs_to_push, step=self.collect_step_count)
        
    def _get_agent_actions_for_rollout(self, rollout: TensorDictBase) -> List[torch.Tensor]:
        """Computes actions for all agents based on the rollout's observations."""
        obs = rollout.get((self.control_group, "observation"))
        actions = []
        # This re-computation can be slow. If performance is an issue, consider logging actions during the rollout.
        for i in range(self.model.n_agents):
            temp_td = TensorDict({(self.control_group, "observation"): obs}, batch_size=obs.shape[:-1])
            action_td = self.model._forward(temp_td, agent_index=i, compute_estimate=False)
            actions.append(action_td.get(self.model.out_key))
        return actions