from typing import List, Union
from collections import deque
import torch
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from tensordict import TensorDictBase
from benchmarl.experiment import Experiment
from benchmarl.experiment.callback import Callback
from AD2C.models.het_control_mlp_snd import HetControlMlpEscSnd
from callbacks.utils import get_het_model

import matplotlib.pyplot as plt



class TrajectorySNDLoggerCallback(Callback):
    def __init__(self, control_group: str, max_history_points: int = 60000):
        super().__init__()
        self.control_group = control_group
        self.esc_history_data =deque(maxlen=10000)
        self.global_history_data = deque(maxlen=max_history_points)
        # self.reward_scaling_data = deque(maxlen=max_scatter_points)
        # self.continuous_scaling_data = deque(maxlen=max_continuous_points)
        # self.continuous_target_diversity_data = deque(maxlen=max_continuous_points)
        # self.continuous_measured_diversity_data = deque(maxlen=max_continuous_points)
        self.collect_step_count = 0
        self.model = None

    def on_setup(self):
        """Correctly identifies the HetControlMlpEscSnd model."""
        if self.control_group not in self.experiment.group_policies:
            print(f"\nWARNING: Logger group '{self.control_group}' not found. Disabling logger.\n")
            return
        policy = self.experiment.group_policies[self.control_group]
        self.model = get_het_model(policy)
        if isinstance(self.model, HetControlMlpEscSnd):
            print(f"\nSUCCESS: Logger initialized for HetControlMlpEscSnd on group '{self.control_group}'.")
        else:
            print(f"\nWARNING: A compatible HetControlMlpEscSnd model was not found. Disabling logger.\n")
            self.model = None

    def on_batch_collected(self, batch: TensorDictBase):
        """Performs the model update and then logs the results."""
        if not isinstance(self.model, HetControlMlpEscSnd): return

        self.model._update_esc(batch) # This updates k_hat (the target diversity) based on reward
        self._log_esc_scalars(self.experiment, batch)
        self._log_global_history(batch)
        self._log_esc_plot(self.experiment, batch)
        self.collect_step_count += 1

    # @staticmethod
    # def _process_trajectory(tensor: torch.Tensor) -> Union[np.ndarray, None]:
        # if tensor is None or tensor.numel() == 0: return None
        # # Extract from the first element of the batch
        # trajectory = tensor[0].detach()
        # # Average out any extra dimensions (e.g., agent dimension)
        # while trajectory.ndim > 1:
        #     trajectory = trajectory.mean(dim=-1)
        # return trajectory.cpu().numpy()
    def _log_global_history(self, batch: TensorDictBase):
        """Logs step-by-step data from the batch for the global history plot."""
        try:
            if (self.control_group, "esc_learning") not in batch.keys(include_nested=True):
                return

            # Extract entire trajectories from the batch
            dither_traj = batch.get((self.control_group, "current_dither")).flatten().cpu().numpy()
            target_div_traj = batch.get((self.control_group, "target_diversity")).flatten().cpu().numpy()
            
            # Get the single mean reward for this batch
            reward_mean = batch.get((self.control_group, "esc_learning", "reward_mean")).mean().item()
            
            # Create a reward array of the same length as the trajectories
            reward_traj = np.full_like(dither_traj, fill_value=reward_mean)
            
            # Append the data points to the global history
            self.global_history_data.extend(zip(dither_traj, target_div_traj, reward_traj))
        except (KeyError, AttributeError):
            # Fail silently if a key is missing on a particular step
            pass

    def _log_esc_scalars(self, experiment: Experiment, batch: TensorDictBase):
        """Logs scalar values from the batch based on the confirmed key structure."""
        to_log = {}

        # Keys at the top level of the agent group
        top_level_keys = ["estimated_snd", "scaling_ratio", "current_dither", "target_diversity", "k_hat"]
        for key in top_level_keys:
             val = batch.get((self.control_group, key), None)
             if val is not None:
                to_log[f"collection/{self.control_group}/{key}"] = val.float().mean().item()

        # Keys inside the 'esc_learning' namespace
        if (self.control_group, "esc_learning") in batch.keys(include_nested=True):
            esc_learning_keys = [
                "reward_mean", "hpf_out", "lpf_out", "gradient_final",
                "k_hat", "integral", "m2_sqrt", "wt", 
            ]
            for key in esc_learning_keys:
                val = batch.get((self.control_group, "esc_learning", key), None)
                if val is not None:
                    # Log the post-update k_hat under a distinct name to avoid confusion
                    log_key = "target_diversity" if key == "k_hat" else key
                    to_log[f"controller_learning/{self.control_group}/{log_key}"] = val.float().mean().item()

        if to_log:
            experiment.logger.log(to_log, step=self.collect_step_count)


    def _log_esc_plot(self, experiment: 'Experiment', batch: TensorDictBase):
        metrics = self._extract_esc_metrics(batch)
        if metrics is None: return
        
        # History uses the POST-update k_hat for tracking learning progress
        self.esc_history_data.append({
            "k_hat_update": metrics["k_hat_update"],
            "j_mean": metrics["j_mean"],
            "estimated_diversity" : metrics['estimated_diversity']
        })

        # reward_tensor = latest_td.get(("next", self.control_group, "reward"))
        # reward_traj = self._process_trajectory(reward_tensor)
        # if self.collect_step_count % 10 == 0 and reward_traj is not None:
        #     self.reward_scaling_data.extend(zip(metrics["measured_snd"], reward_traj))
        
        # Continuous plots use the PRE-update k_hat (base_target_snd) as this was the target for the episode
        # self.continuous_scaling_data.extend(metrics["scaling_ratio"])
        # self.continuous_target_diversity_data.extend(np.full_like(metrics["scaling_ratio"], metrics["base_target_snd"]))
        # self.continuous_measured_diversity_data.extend(metrics["measured_snd"])

        step_count = self.collect_step_count
        
        logs_to_push = {
            # "esc/1_Control_Dynamics": self._plot_control_dynamics(metrics, step_count),
            # "esc/2_Continuous_Diversity_Control": self._plot_diversity_control_signal(step_count),
            # "esc/3_Target_Diversity_Evolution": self._plot_learning_evolution(step_count),
            # "collection/current_target_diversity": metrics["k_hat_after_update"],
            # "collection/mean_measured_diversity": metrics["mean_measured_diversity"]
            "esc/1_Convergence_Over_Time": self._plot_convergence_over_time(step_count),
            "esc/2_Performance_Landscape": self._plot_performance_landscape(step_count),
            "esc/3_Global_Training_History": self._plot_global_training_history(step_count),
            "esc/4_Agent_Distance_Heatmap": self._plot_agent_distance_heatmap(batch, step_count),

        }

        # if self.collect_step_count % 100 == 0:
        #     logs_to_push["esc/4_Reward_vs_Diversity_Scatter"] = self._plot_reward_scaling_scatter(step_count)
        #     logs_to_push["esc/5_Gradient_Landscape"] = self._plot_gradient_landscape(step_count)
        
        final_logs = {k: v for k, v in logs_to_push.items() if v is not None}
        if final_logs:
            experiment.logger.log(final_logs, step=step_count)
    
    def _extract_esc_metrics(self, batch: TensorDictBase) -> Union[dict, None]:
        """
        Extracts and processes all necessary metrics for ESC plotting from the batch,
        using the specific keys provided.
        """
        try:
            # --- Map your batch keys to the variables we need for plotting ---
            print("Available keys:", batch[self.control_group].keys(include_nested=True))

            
            # Measured Diversity for the episode -> (group, "estimated_snd")
            estimated_diversity_t = batch.get((self.control_group, "estimated_snd"))
            
            # Mean Reward for the episode -> (group, "esc_reward_J")
            j_mean_t = batch.get((self.control_group, "esc_learning", "reward_mean"))
            
            # The target diversity value AFTER the ESC update -> (group, "esc_k_hat_update")
            k_hat_update_t = batch.get((self.control_group, "k_hat"))

            if estimated_diversity_t is None or j_mean_t is None or k_hat_update_t is None:
                # If any of the core metrics are missing, we can't plot
                return None

            # Process the tensors to get single scalar values for logging
            # We calculate the mean over the batch and sequence dimensions
            estimated_diversity = torch.mean(estimated_diversity_t.float()).item()
            j_mean = torch.mean(j_mean_t.float()).item()
            k_hat_update = torch.mean(k_hat_update_t.float()).item()

            return {
                "estimated_diversity": estimated_diversity,
                "j_mean": j_mean,
                "k_hat_update": k_hat_update,
            }
            
        except (KeyError, IndexError, AttributeError) as e:
            print(f"Warning: Could not extract ESC metrics for plotting. Error: {e}")
            return None

    
    def _plot_global_training_history(self, global_step_count: int) -> go.Figure:
        """NEW: Plots the continuous evolution of signals over the entire training history."""
        if not self.global_history_data:
            return None
        
        history = list(self.global_history_data)
        frames = np.arange(len(history))
        dithers, targets, rewards = zip(*history)
        
        fig = make_subplots(specs=[[{"secondary_y": True}]])

        # Plot the dither signal
        fig.add_trace(go.Scatter(
            x=frames, y=dithers, name="Dither Signal",
            line=dict(color='orange', dash='dot', width=1.5), opacity=0.8
        ), secondary_y=False)

        # Plot the actual target diversity (k_hat + dither)
        fig.add_trace(go.Scatter(
            x=frames, y=targets, name="Actual Target Diversity",
            line=dict(color='royalblue', width=2)
        ), secondary_y=False)
        
        # Plot the batch reward on the secondary axis
        fig.add_trace(go.Scatter(
            x=frames, y=rewards, name="Batch Reward",
            line=dict(color='green', width=2)
        ), secondary_y=True)
        
        fig.update_layout(
            title_text=f"<b>Global Training History up to Frame #{global_step_count}</b>",
            xaxis_title="Global Frame (Timestep)",
            template="plotly_white", height=500
        )
        fig.update_yaxes(title_text="Diversity", secondary_y=False)
        fig.update_yaxes(title_text="Mean Batch Reward", secondary_y=True)
        
        return fig

    
    def _plot_performance_landscape(self, step_count: int) -> go.Figure:
        """
        Plots a heatmap of the average reward landscape, inspired by Figure 4 of the paper.
        """

        history_unfiltered = list(self.esc_history_data)


        valid_hist  = [
            d for d in history_unfiltered
            if not np.isnan(d.get("k_hat_update")) and not np.isnan(d.get("estimated_diversity"))
        ]

        if len(valid_hist) < 10: # Need a bit of data for a meaningful heatmap
            return None

        history = valid_hist


        target_diversities = np.array([d["k_hat_update"] for d in history])
        measured_diversities = np.array([d["estimated_diversity"] for d in history])
        rewards = np.array([d["j_mean"] for d in history])
        
        # Use a 2D histogram to bin data and find the average reward in each bin
        reward_sum, x_edges, y_edges = np.histogram2d(
            target_diversities, measured_diversities, bins=25, weights=rewards
        )
        counts, _, _ = np.histogram2d(
            target_diversities, measured_diversities, bins=25
        )
        
        with np.errstate(divide='ignore', invalid='ignore'):
            avg_reward = reward_sum / counts
        avg_reward = avg_reward.T # Transpose for Plotly's coordinate system

        fig = go.Figure(data=go.Heatmap(
            z=avg_reward,
            x=(x_edges[:-1] + x_edges[1:]) / 2, # Center bin edges for labels
            y=(y_edges[:-1] + y_edges[1:]) / 2,
            colorscale='Viridis',
            colorbar_title="Average Reward"
        ))
        fig.update_layout(
            title_text=f"<b>Reward Landscape up to Step #{step_count}</b>",
            xaxis_title="Target Diversity (k̂)",
            yaxis_title="Mean Measured Diversity",
            template="plotly_white", height=450
        )
        return fig

    def _plot_convergence_over_time(self, step_count: int) -> go.Figure:
        """
        Plots the evolution of diversity and reward over time, inspired by Figures 5 & 6.
        """
        if len(self.esc_history_data) < 2:
            return None

        history = list(self.esc_history_data)
        episodes = np.arange(len(history))
        
        target_diversities = np.array([d["k_hat_update"] for d in history])
        measured_diversities = np.array([d["estimated_diversity"] for d in history])
        rewards = np.array([d["j_mean"] for d in history])
        
        fig = make_subplots(
            rows=2, cols=1, shared_xaxes=True,
            vertical_spacing=0.1, subplot_titles=("Diversity Convergence", "Reward Evolution")
        )

        # Top Panel: Diversity
        fig.add_trace(go.Scatter(
            x=episodes, y=measured_diversities, name="Measured Diversity",
            line=dict(color='royalblue'), mode='lines'
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=episodes, y=target_diversities, name="Target Diversity (k̂)",
            line=dict(color='firebrick', dash='dash'), mode='lines'
        ), row=1, col=1)

        # Bottom Panel: Reward
        fig.add_trace(go.Scatter(
            x=episodes, y=rewards, name="Mean Reward (J)",
            line=dict(color='green'), mode='lines'
        ), row=2, col=1)

        fig.update_layout(
            title_text=f"<b>System Performance up to Step #{step_count}</b>",
            template="plotly_white", height=500, showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        fig.update_yaxes(title_text="Diversity", row=1, col=1)
        fig.update_yaxes(title_text="Reward", row=2, col=1)
        fig.update_xaxes(title_text="Episode", row=2, col=1)
        return fig

    def _plot_agent_distance_heatmap(self, batch: TensorDictBase, step_count: int) -> plt.Figure:
        """
        Generates a heatmap of pairwise agent distances from the batch data.
        """
        try:
            # Extract the raw pairwise distances from the batch
            distances_tensor = batch.get((self.control_group, "pairwise_distance"))
            if distances_tensor is None:
                return None
            
            # Use the model to get the number of agents
            n_agents = self.model.n_agents
            pairwise_distances = distances_tensor.cpu().numpy().flatten()
            
            # Create an empty NxN matrix to store the distances
            distance_matrix = np.zeros((n_agents, n_agents))

            # Populate the matrix with the distances
            indices = np.triu_indices(n_agents, k=1)
            distance_matrix[indices] = pairwise_distances
            distance_matrix = distance_matrix + distance_matrix.T

            # --- Plotting ---
            fig, ax = plt.subplots(figsize=(10, 8))
            im = ax.imshow(distance_matrix, cmap="viridis")

            cbar = ax.figure.colorbar(im, ax=ax)
            cbar.ax.set_ylabel("Pairwise pairwise_distance", rotation=-90, va="bottom")

            agent_labels = [f'Agent {i}' for i in range(n_agents)]
            ax.set_xticks(np.arange(n_agents))
            ax.set_yticks(np.arange(n_agents))
            ax.set_xticklabels(agent_labels)
            ax.set_yticklabels(agent_labels)
            plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

            for i in range(n_agents):
                for j in range(n_agents):
                    color = "black" if distance_matrix[i, j] < distance_matrix.max() / 2 else "white"
                    ax.text(j, i, f"{distance_matrix[i, j]:.2f}", ha="center", va="center", color=color)

            mean_distance = pairwise_distances.mean()
            ax.set_title(f"Agent Pairwise Distance Matrix (Step: {step_count})\nMean Distance (SND): {mean_distance:.2f}", fontsize=16)

            fig.tight_layout()
            
            # Return the figure object for the logger
            return fig

        except (KeyError, AttributeError, IndexError) as e:
            # Fail silently if data is missing or something goes wrong
            # print(f"Could not generate heatmap: {e}")
            return None



    # def _plot_control_dynamics(self, metrics: dict, step_count: int) -> go.Figure:
    #     """CORRECTED: Plots the TRUE dynamic target (`k_hat` + dither) for the episode."""
    #     fig = make_subplots(specs=[[{"secondary_y": True}]])
        
    #     # Calculate the actual, time-varying target the controller was aiming for
    #     dynamic_target = metrics["base_target_snd"] + metrics["dither"]
        
    #     fig.add_trace(go.Scatter(x=metrics["time_steps"], y=metrics["measured_snd"], name="Measured Diversity", line=dict(color='green')), secondary_y=False)
    #     fig.add_trace(go.Scatter(x=metrics["time_steps"], y=dynamic_target, name="Dynamic Target (k̂+dither)", line=dict(color='firebrick', dash='dash')), secondary_y=False)
    #     fig.add_trace(go.Scatter(x=metrics["time_steps"], y=metrics["scaling_ratio"], name="Resulting Scaling Ratio", line=dict(color='royalblue', dash='dot')), secondary_y=True)
        
    #     fig.update_layout(title_text=f"<b>Intra-Episode Diversity Dynamics at Step #{step_count}</b>", template="plotly_white", height=400)
    #     fig.update_yaxes(title_text="Diversity Value", secondary_y=False, color='black')
    #     fig.update_yaxes(title_text="Scaling Ratio", secondary_y=True, color='royalblue')
    #     return fig

    # def _plot_learning_evolution(self, step_count: int) -> go.Figure:
    #     """Shows how the base target diversity (k_hat) evolves over training."""
    #     if len(self.esc_history_data) < 2: return go.Figure()
    #     history, steps = self.esc_history_data, np.arange(len(self.esc_history_data))
    #     # k_hats now correctly tracks the post-update value
    #     k_hats = np.array([d["k_hat"] for d in history])
    #     fig = go.Figure()
    #     fig.add_trace(go.Scatter(x=steps, y=k_hats, name='Target Diversity (k̂)', mode='lines+markers', line=dict(color='firebrick')))
    #     fig.update_layout(title_text=f"<b>Target Diversity (k̂) Evolution up to Step #{step_count}</b>", xaxis_title="Episode", yaxis_title="Target Diversity Value", template="plotly_white", height=400)
    #     return fig
    
    # def _plot_diversity_control_signal(self, step_count: int) -> go.Figure:
    #     """The primary plot showing continuous diversity control over time."""
    #     if not self.continuous_scaling_data: return go.Figure()
    #     fig = make_subplots(specs=[[{"secondary_y": True}]])
        
    #     timesteps = np.arange(len(self.continuous_scaling_data))
    #     scaling_data = np.array(self.continuous_scaling_data)
    #     target_div_data = np.array(self.continuous_target_diversity_data)
    #     measured_div_data = np.array(self.continuous_measured_diversity_data)

    #     fig.add_trace(go.Scatter(x=timesteps, y=measured_div_data, name="Measured Diversity", line=dict(color='green', width=1.5)), secondary_y=False)
    #     fig.add_trace(go.Scatter(x=timesteps, y=target_div_data, name="Base Target Diversity (k̂)", line=dict(color='firebrick', dash='dash', width=2)), secondary_y=False)
        
    #     fig.add_trace(go.Scatter(
    #         x=timesteps, 
    #         y=scaling_data, 
    #         name="Scaling Ratio", 
    #         line=dict(color='royalblue', width=1), 
    #         opacity=0.7), 
    #         secondary_y=True
    #     )

    #     fig.update_layout(
    #         title_text=f"<b>Continuous Diversity Control Signal up to Step #{step_count}</b>",
    #         xaxis_title="Global Timestep", template="plotly_white"
    #     )
    #     fig.update_yaxes(title_text="Diversity", secondary_y=False)
    #     fig.update_yaxes(title_text="Scaling Ratio", secondary_y=True)
    #     return fig

    # def _plot_gradient_landscape(self, step_count: int) -> go.Figure:
    #     if len(self.esc_history_data) < 50: return go.Figure()
    #     history = self.esc_history_data
    #     k_hats = np.array([d["k_hat"] for d in history])
    #     j_means = np.array([d["j_mean"] for d in history])
    #     grads = np.array([d["gradient_estimate"] for d in history])
        
    #     gradient_sum, x_edges, y_edges = np.histogram2d(k_hats, j_means, bins=20, weights=grads)
    #     counts, _, _ = np.histogram2d(k_hats, j_means, bins=20)
        
    #     with np.errstate(divide='ignore', invalid='ignore'):
    #         avg_gradient = gradient_sum / counts
    #     avg_gradient[np.isnan(avg_gradient)] = 0
        
    #     fig = go.Figure(data=go.Heatmap(
    #         z=avg_gradient.T, x=(x_edges[:-1]+x_edges[1:])/2, y=(y_edges[:-1]+y_edges[1:])/2,
    #         colorscale='RdBu', zmid=0
    #     ))
    #     fig.update_layout(
    #         title_text=f"<b>Gradient Landscape up to Step #{step_count}</b>",
    #         xaxis_title="Target Diversity (k̂)", yaxis_title="J_mean (Average Reward)"
    #     )
    #     return fig

    # def _plot_reward_scaling_scatter(self, step_count: int) -> go.Figure:
    #     if not self.reward_scaling_data: return go.Figure()
    #     diversities, rewards = zip(*self.reward_scaling_data)
    #     fig = go.Figure(go.Scatter(
    #         x=diversities, y=rewards, mode='markers',
    #         marker=dict(size=5, opacity=0.6, color=np.arange(len(rewards)), colorscale='Viridis', showscale=True, colorbar_title="Time")
    #     ))
    #     fig.update_layout(
    #         title_text=f"<b>Reward vs. Measured Diversity up to Step #{step_count}</b>",
    #         xaxis_title="Measured Diversity", yaxis_title="Reward", template="plotly_white"
    #     )
    #     return fig
