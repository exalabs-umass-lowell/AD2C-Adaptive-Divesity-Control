from typing import List, Union
from collections import deque
import torch
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from tensordict import TensorDictBase
from benchmarl.experiment import Experiment
from benchmarl.experiment.callback import Callback
from AD2C.models.het_control_mlp_esc import HetControlMlpEsc
from callbacks.utils import get_het_model


class TrajectoryLoggerCallback(Callback):
    def __init__(self, control_group: str, max_scatter_points: int = 20000, max_continuous_points: int = 50000):
        super().__init__()
        self.control_group = control_group
        self.esc_history_data = []
        self.reward_scaling_data = deque(maxlen=max_scatter_points)
        self.continuous_scaling_data = deque(maxlen=max_continuous_points)
        self.continuous_k_hat_data = deque(maxlen=max_continuous_points)
        self.continuous_diversity_data = deque(maxlen=max_continuous_points)
        self.collect_step_count = 0
        self.model = None

    def on_setup(self):
        """Correctly identifies the model."""
        if self.control_group not in self.experiment.group_policies:
            print(f"\nWARNING: Logger group '{self.control_group}' not found. Disabling logger.\n")
            return
        policy = self.experiment.group_policies[self.control_group]
        self.model = get_het_model(policy)
        if isinstance(self.model, HetControlMlpEsc):
            print(f"\nSUCCESS: Logger initialized for group '{self.control_group}'.")
        else:
            print(f"\nWARNING: A compatible model was not found. Disabling logger.\n")
            self.model = None

    def on_batch_collected(self, batch: TensorDictBase):
        """Performs the model update and then logs the results."""
        if not isinstance(self.model, HetControlMlpEsc): return
        self.model._update_esc(batch)
        self._log_esc_scalars(self.experiment, batch)
        self._log_esc_plot(self.experiment, batch)
        self.collect_step_count += 1

    @staticmethod
    def _process_trajectory(tensor: torch.Tensor) -> Union[np.ndarray, None]:
        if tensor is None or tensor.numel() == 0: return None
        trajectory = tensor[0].detach()
        while trajectory.ndim > 1:
            trajectory = trajectory.mean(dim=-1)
        return trajectory.cpu().numpy()

    def _log_esc_scalars(self, experiment: Experiment, batch: TensorDictBase):
        keys_to_log = [
            ("output", "logits"), ("output", "scaling_ratio"), ("output", "dither"),
            ("esc", "k_hat"), ("esc", "s_reward"), ("esc", "J_mean"),
            ("esc", "grad_estimate"), ("esc", "k_hat_update"),("esc", "s_reward_new")
        ]
        to_log = {}
        for ns, key in keys_to_log:
            val = batch.get((self.control_group, ns, key), None)
            if val is not None:
                to_log[f"collection/{self.control_group}/{ns}/{key}"] = val.float().mean().item()
        if to_log: experiment.logger.log(to_log, step=self.collect_step_count)

    # In your TrajectoryLoggerCallback class...

    def _log_esc_plot(self, experiment: 'Experiment', latest_td: TensorDictBase):
        metrics = self._extract_esc_metrics(latest_td)
        if metrics is None: return
        
        # Always store history data at every step
        self.esc_history_data.append({
            "k_hat": metrics["k_hat"],
            "gradient_estimate": metrics.get("gradient_estimate", 0.0),
            "j_mean": metrics.get("j_mean", 0.0)
        })

        # Always store continuous data at every step
        self.continuous_scaling_data.extend(metrics["scaling_ratio"])
        self.continuous_k_hat_data.extend(np.full_like(metrics["scaling_ratio"], metrics["k_hat"]))
        if "behavioral_diversity" in metrics:
            self.continuous_diversity_data.extend(metrics["behavioral_diversity"])
        
        # Periodically collect scatter plot data
        if self.collect_step_count % 10 == 0:
            reward_tensor = latest_td.get(("next", self.control_group, "reward"))
            reward_traj = self._process_trajectory(reward_tensor)
            if reward_traj is not None:
                self.reward_scaling_data.extend(zip(metrics["scaling_ratio"], reward_traj))

        step_count = self.collect_step_count
        
        # --- These plots show data from the LATEST episode, so they update every step ---
        logs_to_push = {
            "ESC/1_Control_Dynamics": self._plot_control_dynamics(metrics, step_count),
            "ESC/4_Intra-Episode_Gradient": self._plot_gradient_estimate(metrics, step_count),
            "ESC/7_Continuous_Signal": self._plot_continuous_control_signal(step_count),
            "ESC/current_k_hat": metrics["k_hat"],
        }

        # --- These plots show HISTORY, so we only generate them periodically ---
        if self.collect_step_count % 1 == 0:
            logs_to_push["ESC/2_k_hat_Evolution"] = self._plot_learning_evolution(step_count)
            logs_to_push["ESC/3_Gradient_Evolution"] = self._plot_gradient_evolution(step_count)
            logs_to_push["ESC/5_Reward_vs_Scaling_Scatter"] = self._plot_reward_scaling_scatter(step_count)
            logs_to_push["ESC/6_Gradient_Landscape"] = self._plot_gradient_landscape(step_count)
        
        final_logs = {k: v for k, v in logs_to_push.items() if v is not None}
        experiment.logger.log(final_logs, step=step_count)
        
    def _extract_esc_metrics(self, latest_td: TensorDictBase) -> Union[dict, None]:
        try:
            sr_tensor = latest_td.get((self.control_group, "output", "scaling_ratio"))
            logits_tensor = latest_td.get((self.control_group, "output", "logits"))
            scaling_ratio = self._process_trajectory(sr_tensor)
            if scaling_ratio is None or logits_tensor is None: return None
            T = len(scaling_ratio)
            if T == 0: return None
            
            metrics = {"T": T, "time_steps": np.arange(T), "scaling_ratio": scaling_ratio}
            diversity = logits_tensor[0].norm(dim=-1).std(dim=-1).cpu().numpy()
            metrics["behavioral_diversity"] = diversity
            metrics["mean_diversity"] = float(np.nanmean(diversity))
            
            grad_tensor = latest_td.get((self.control_group, "esc", "grad_estimate"))
            if grad_tensor is not None:
                metrics["gradient_estimate"] = float(grad_tensor.mean().item())
                metrics["ge_magnitude"] = self._process_trajectory(grad_tensor.abs())
            
            j_mean_tensor = latest_td.get((self.control_group, "esc", "J_mean"))
            if j_mean_tensor is not None: metrics["j_mean"] = float(j_mean_tensor.mean().item())

            k_hat_tensor = latest_td.get((self.control_group, "esc", "k_hat"))
            metrics["k_hat"] = float(k_hat_tensor.mean().item()) if k_hat_tensor is not None else 1.0
            return metrics
        except (KeyError, IndexError, AttributeError):
            return None
        
    def _plot_control_dynamics(self, metrics: dict, step_count: int) -> go.Figure:
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Scatter(x=metrics["time_steps"], y=metrics["scaling_ratio"], name="Scaling Ratio", line=dict(color='royalblue')), secondary_y=False)
        fig.add_trace(go.Scatter(x=metrics["time_steps"], y=np.full(metrics["T"], metrics["k_hat"]), name="Learned k̂", line=dict(color='firebrick', dash='dash')), secondary_y=False)
        if "behavioral_diversity" in metrics:
            fig.add_trace(go.Scatter(x=metrics["time_steps"], y=metrics["behavioral_diversity"], name="Diversity", line=dict(color='green', dash='dot')), secondary_y=True)
        fig.update_layout(title_text=f"<b>Control Dynamics at Step #{step_count}</b>", template="plotly_white", height=400)
        fig.update_yaxes(title_text="Control Value", secondary_y=False)
        fig.update_yaxes(title_text="Diversity", secondary_y=True)
        return fig

    def _plot_learning_evolution(self, step_count: int) -> go.Figure:
        if len(self.esc_history_data) < 2: return go.Figure()
        history, steps = self.esc_history_data, np.arange(len(self.esc_history_data))
        k_hats = np.array([d["k_hat"] for d in history])
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=steps, y=k_hats, name='Learned k̂', mode='lines+markers', line=dict(color='firebrick')))
        fig.update_layout(title_text=f"<b>k̂ Evolution up to Step #{step_count}</b>", xaxis_title="Episode", yaxis_title="k̂ Value", template="plotly_white", height=400)
        return fig
    
    def _plot_gradient_evolution(self, step_count: int) -> go.Figure:
        if len(self.esc_history_data) < 2: return go.Figure()
        history, steps = self.esc_history_data, np.arange(len(self.esc_history_data))
        grads = np.array([d["gradient_estimate"] for d in history])
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=steps, y=grads, name='Grad Estimate', mode='lines', line=dict(color='purple')))
        fig.update_layout(title_text=f"<b>Gradient Estimate Evolution up to Step #{step_count}</b>", xaxis_title="Episode", yaxis_title="Gradient Estimate", template="plotly_white", height=400)
        return fig
    
    def _plot_gradient_estimate(self, metrics: dict, step_count: int) -> Union[go.Figure, None]:
        """
        Visualizes the core components of the ESC gradient calculation.
        Shows the dynamic Dither Signal that leads to the final, constant Gradient Estimate.
        """
        # Check if both required metrics are available
        if "ge_magnitude" not in metrics or "dither" not in metrics:
            return None

        # Create a figure with two rows to show the relationship clearly
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1,
                        subplot_titles=("Dynamic Input", "Calculated Output"))

        # Row 1: The oscillating Dither Signal
        fig.add_trace(go.Scatter(
            x=metrics["time_steps"],
            y=metrics["dither"],
            name="Dither Signal",
            line=dict(color='green')
        ), row=1, col=1)

        # Row 2: The constant resulting Gradient Estimate magnitude
        fig.add_trace(go.Scatter(
            x=metrics["time_steps"],
            y=metrics["ge_magnitude"],
            name="|Grad Estimate|",
            line=dict(color='purple', dash='dash')
        ), row=2, col=1)

        fig.update_layout(
            title_text=f"<b>Intra-Episode Gradient Calculation at Step #{step_count}</b>",
            template="plotly_white",
            height=500,
            showlegend=False
        )
        fig.update_yaxes(title_text="Dither Value", row=1, col=1)
        fig.update_yaxes(title_text="Gradient Mag.", row=2, col=1)
        
        return fig

    def _plot_continuous_control_signal(self, step_count: int) -> go.Figure:
        if not self.continuous_scaling_data: return go.Figure()
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        timesteps = np.arange(len(self.continuous_scaling_data))
        scaling_data = np.array(self.continuous_scaling_data)
        k_hat_data = np.array(self.continuous_k_hat_data)
        diversity_data = np.array(self.continuous_diversity_data)

        fig.add_trace(go.Scatter(x=timesteps, y=scaling_data, name="Scaling Ratio", line=dict(color='royalblue', width=1)), secondary_y=False)
        fig.add_trace(go.Scatter(x=timesteps, y=k_hat_data, name="Learned k̂", line=dict(color='firebrick', dash='dash', width=2)), secondary_y=False)
        fig.add_trace(go.Scatter(x=timesteps, y=diversity_data, name="Diversity", line=dict(color='green', dash='dot', width=1)), secondary_y=True)

        fig.update_layout(
            title_text=f"<b>Continuous Control Signal up to Step #{step_count}</b>",
            xaxis_title="Global Timestep", template="plotly_white"
        )
        fig.update_yaxes(title_text="Control Value", secondary_y=False)
        fig.update_yaxes(title_text="Diversity", secondary_y=True)
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
            xaxis_title="k̂ (Learned Parameter)", yaxis_title="J_mean (Average Reward)"
        )
        return fig

    def _plot_reward_scaling_scatter(self, step_count: int) -> go.Figure:
        if not self.reward_scaling_data: return go.Figure()
        ratios, rewards = zip(*self.reward_scaling_data)
        fig = go.Figure(go.Scatter(
            x=ratios, y=rewards, mode='markers',
            marker=dict(size=5, opacity=0.6, color=np.arange(len(rewards)), colorscale='Viridis', showscale=True, colorbar_title="Time")
        ))
        fig.update_layout(
            title_text=f"<b>Reward vs. Scaling Ratio up to Step #{step_count}</b>",
            xaxis_title="Scaling Ratio", yaxis_title="Reward", template="plotly_white"
        )
        return fig

