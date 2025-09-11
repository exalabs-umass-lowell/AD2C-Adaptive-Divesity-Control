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


class TrajectorySNDLoggerCallback(Callback):
    def __init__(self, control_group: str, max_scatter_points: int = 20000, max_continuous_points: int = 50000):
        super().__init__()
        self.control_group = control_group
        self.esc_history_data = []
        self.reward_scaling_data = deque(maxlen=max_scatter_points)
        self.continuous_scaling_data = deque(maxlen=max_continuous_points)
        self.continuous_target_diversity_data = deque(maxlen=max_continuous_points)
        self.continuous_measured_diversity_data = deque(maxlen=max_continuous_points)
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
        self._log_esc_plot(self.experiment, batch)
        self.collect_step_count += 1

    @staticmethod
    def _process_trajectory(tensor: torch.Tensor) -> Union[np.ndarray, None]:
        if tensor is None or tensor.numel() == 0: return None
        # Extract from the first element of the batch
        trajectory = tensor[0].detach()
        # Average out any extra dimensions (e.g., agent dimension)
        while trajectory.ndim > 1:
            trajectory = trajectory.mean(dim=-1)
        return trajectory.cpu().numpy()

    def _log_esc_scalars(self, experiment: Experiment, batch: TensorDictBase):
        """Logs scalar values from both the output and learning steps."""
        keys_to_log = [
            ("output", "scaling_ratio"), ("output", "dither"),
            ("output", "measured_snd"), ("output", "target_snd"),
    
            ("esc_learning", "k_hat"), ("esc_learning", "s_reward"), ("esc_learning", "J_mean"),
            ("esc_learning", "grad_estimate"), ("esc_learning", "k_hat_update"),
            
            ("esc_learning","reward_mean"),("esc_learning", "hpf_out"),("esc_learning", "lpf_out"),
            ("esc_learning", "gradient_final"),("esc_learning","k_hat"),("esc_learning","integral")

        ]

        # "logits": out, "out_loc_norm": out_loc_norm, "actual_snd": distance, "target_snd": self.k_hat


        to_log = {}
        for ns, key in keys_to_log:
            val = batch.get((self.control_group, ns, key), None)
            if val is not None:
                to_log[f"controller_learning/{self.control_group}/{key}"] = val.float().mean().item()
        if to_log: experiment.logger.log(to_log, step=self.collect_step_count)

    def _log_esc_plot(self, experiment: 'Experiment', latest_td: TensorDictBase):
        metrics = self._extract_esc_metrics(latest_td)
        if metrics is None: return
        
        # History uses the POST-update k_hat for tracking learning progress
        self.esc_history_data.append({
            "k_hat": metrics["k_hat_after_update"],
            "gradient_estimate": metrics.get("gradient_estimate", 0.0),
            "j_mean": metrics.get("j_mean", 0.0)
        })

        reward_tensor = latest_td.get(("next", self.control_group, "reward"))
        reward_traj = self._process_trajectory(reward_tensor)
        if self.collect_step_count % 10 == 0 and reward_traj is not None:
            self.reward_scaling_data.extend(zip(metrics["measured_snd"], reward_traj))
        
        # Continuous plots use the PRE-update k_hat (base_target_snd) as this was the target for the episode
        self.continuous_scaling_data.extend(metrics["scaling_ratio"])
        self.continuous_target_diversity_data.extend(np.full_like(metrics["scaling_ratio"], metrics["base_target_snd"]))
        self.continuous_measured_diversity_data.extend(metrics["measured_snd"])

        step_count = self.collect_step_count
        
        logs_to_push = {
            "esc/1_Control_Dynamics": self._plot_control_dynamics(metrics, step_count),
            "esc/2_Continuous_Diversity_Control": self._plot_diversity_control_signal(step_count),
            "esc/3_Target_Diversity_Evolution": self._plot_learning_evolution(step_count),
            "collection/current_target_diversity": metrics["k_hat_after_update"],
            "collection/mean_measured_diversity": metrics["mean_measured_diversity"]
        }

        if self.collect_step_count % 100 == 0:
            logs_to_push["esc/4_Reward_vs_Diversity_Scatter"] = self._plot_reward_scaling_scatter(step_count)
            logs_to_push["esc/5_Gradient_Landscape"] = self._plot_gradient_landscape(step_count)
        
        final_logs = {k: v for k, v in logs_to_push.items() if v is not None}
        experiment.logger.log(final_logs, step=step_count)
        
    def _extract_esc_metrics(self, latest_td: TensorDictBase) -> Union[dict, None]:
        try:
            # --- Extract data from the "output" namespace (used DURING the episode) ---
            sr_tensor = latest_td.get((self.control_group, "output", "scaling_ratio"))
            measured_snd_tensor = latest_td.get((self.control_group, "output", "measured_snd"))
            dither_tensor = latest_td.get((self.control_group, "output", "dither"))
            target_snd_tensor = latest_td.get((self.control_group, "output", "target_snd")) # This is the pre-update k_hat

            scaling_ratio = self._process_trajectory(sr_tensor)
            measured_snd = self._process_trajectory(measured_snd_tensor)
            dither = self._process_trajectory(dither_tensor)
            
            if scaling_ratio is None or measured_snd is None or dither is None or target_snd_tensor is None:
                return None
            
            T = len(scaling_ratio)
            if T == 0: return None
            
            metrics = {
                "T": T, "time_steps": np.arange(T), 
                "scaling_ratio": scaling_ratio,
                "measured_snd": measured_snd,
                "dither": dither,
                "base_target_snd": float(target_snd_tensor.mean().item()), # The base k_hat for this episode
                "mean_measured_diversity": float(np.nanmean(measured_snd))
            }
            
            # --- Extract data from the "esc_learning" namespace (AFTER the update) ---
            grad_tensor = latest_td.get((self.control_group, "esc_learning", "grad_estimate"))
            if grad_tensor is not None:
                metrics["gradient_estimate"] = float(grad_tensor.mean().item())

            j_mean_tensor = latest_td.get((self.control_group, "esc_learning", "J_mean"))
            if j_mean_tensor is not None: metrics["j_mean"] = float(j_mean_tensor.mean().item())

            # This is the k_hat value AFTER the learning step, used for the evolution plot
            k_hat_after_update_tensor = latest_td.get((self.control_group, "esc_learning", "k_hat"))
            metrics["k_hat_after_update"] = float(k_hat_after_update_tensor.mean().item()) if k_hat_after_update_tensor is not None else metrics["base_target_snd"]
            
            return metrics
        except (KeyError, IndexError, AttributeError):
            return None
            
    def _plot_control_dynamics(self, metrics: dict, step_count: int) -> go.Figure:
        """CORRECTED: Plots the TRUE dynamic target (`k_hat` + dither) for the episode."""
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        # Calculate the actual, time-varying target the controller was aiming for
        dynamic_target = metrics["base_target_snd"] + metrics["dither"]
        
        fig.add_trace(go.Scatter(x=metrics["time_steps"], y=metrics["measured_snd"], name="Measured Diversity", line=dict(color='green')), secondary_y=False)
        fig.add_trace(go.Scatter(x=metrics["time_steps"], y=dynamic_target, name="Dynamic Target (k̂+dither)", line=dict(color='firebrick', dash='dash')), secondary_y=False)
        fig.add_trace(go.Scatter(x=metrics["time_steps"], y=metrics["scaling_ratio"], name="Resulting Scaling Ratio", line=dict(color='royalblue', dash='dot')), secondary_y=True)
        
        fig.update_layout(title_text=f"<b>Intra-Episode Diversity Dynamics at Step #{step_count}</b>", template="plotly_white", height=400)
        fig.update_yaxes(title_text="Diversity Value", secondary_y=False, color='black')
        fig.update_yaxes(title_text="Scaling Ratio", secondary_y=True, color='royalblue')
        return fig

    def _plot_learning_evolution(self, step_count: int) -> go.Figure:
        """Shows how the base target diversity (k_hat) evolves over training."""
        if len(self.esc_history_data) < 2: return go.Figure()
        history, steps = self.esc_history_data, np.arange(len(self.esc_history_data))
        # k_hats now correctly tracks the post-update value
        k_hats = np.array([d["k_hat"] for d in history])
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=steps, y=k_hats, name='Target Diversity (k̂)', mode='lines+markers', line=dict(color='firebrick')))
        fig.update_layout(title_text=f"<b>Target Diversity (k̂) Evolution up to Step #{step_count}</b>", xaxis_title="Episode", yaxis_title="Target Diversity Value", template="plotly_white", height=400)
        return fig
    
    def _plot_diversity_control_signal(self, step_count: int) -> go.Figure:
        """The primary plot showing continuous diversity control over time."""
        if not self.continuous_scaling_data: return go.Figure()
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        timesteps = np.arange(len(self.continuous_scaling_data))
        scaling_data = np.array(self.continuous_scaling_data)
        target_div_data = np.array(self.continuous_target_diversity_data)
        measured_div_data = np.array(self.continuous_measured_diversity_data)

        fig.add_trace(go.Scatter(x=timesteps, y=measured_div_data, name="Measured Diversity", line=dict(color='green', width=1.5)), secondary_y=False)
        fig.add_trace(go.Scatter(x=timesteps, y=target_div_data, name="Base Target Diversity (k̂)", line=dict(color='firebrick', dash='dash', width=2)), secondary_y=False)
        
        fig.add_trace(go.Scatter(
            x=timesteps, 
            y=scaling_data, 
            name="Scaling Ratio", 
            line=dict(color='royalblue', width=1), 
            opacity=0.7), 
            secondary_y=True
        )

        fig.update_layout(
            title_text=f"<b>Continuous Diversity Control Signal up to Step #{step_count}</b>",
            xaxis_title="Global Timestep", template="plotly_white"
        )
        fig.update_yaxes(title_text="Diversity", secondary_y=False)
        fig.update_yaxes(title_text="Scaling Ratio", secondary_y=True)
        return fig

    def _plot_gradient_landscape(self, step_count: int) -> go.Figure:
        if len(self.esc_history_data) < 50: return go.Figure()
        history = self.esc_history_data
        k_hats = np.array([d["k_hat"] for d in history])
        j_means = np.array([d["j_mean"] for d in history])
        grads = np.array([d["gradient_estimate"] for d in history])
        
        gradient_sum, x_edges, y_edges = np.histogram2d(k_hats, j_means, bins=20, weights=grads)
        counts, _, _ = np.histogram2d(k_hats, j_means, bins=20)
        
        with np.errstate(divide='ignore', invalid='ignore'):
            avg_gradient = gradient_sum / counts
        avg_gradient[np.isnan(avg_gradient)] = 0
        
        fig = go.Figure(data=go.Heatmap(
            z=avg_gradient.T, x=(x_edges[:-1]+x_edges[1:])/2, y=(y_edges[:-1]+y_edges[1:])/2,
            colorscale='RdBu', zmid=0
        ))
        fig.update_layout(
            title_text=f"<b>Gradient Landscape up to Step #{step_count}</b>",
            xaxis_title="Target Diversity (k̂)", yaxis_title="J_mean (Average Reward)"
        )
        return fig

    def _plot_reward_scaling_scatter(self, step_count: int) -> go.Figure:
        if not self.reward_scaling_data: return go.Figure()
        diversities, rewards = zip(*self.reward_scaling_data)
        fig = go.Figure(go.Scatter(
            x=diversities, y=rewards, mode='markers',
            marker=dict(size=5, opacity=0.6, color=np.arange(len(rewards)), colorscale='Viridis', showscale=True, colorbar_title="Time")
        ))
        fig.update_layout(
            title_text=f"<b>Reward vs. Measured Diversity up to Step #{step_count}</b>",
            xaxis_title="Measured Diversity", yaxis_title="Reward", template="plotly_white"
        )
        return fig
