from typing import List, Union
import torch
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from tensordict import TensorDictBase, TensorDict
from benchmarl.experiment import Experiment
from benchmarl.experiment.callback import Callback

from AD2C.models.het_control_mlp_empirical import HetControlMlpEmpirical
from AD2C.models.het_control_mlp_esc import HetControlMlpEsc
from AD2C.snd import compute_behavioral_distance
from .plots import plot_trajectory_2d, plot_agent_distances
from callbacks.utils import get_het_model


class TrajectoryLoggerCallback(Callback):
    def __init__(self, control_group: str):
        super().__init__()
        self.control_group = control_group
        # History for DiCo model (eval)
        self.eps_actual_diversity, self.eps_mean_returns, self.eps_number = [], [], []
        
        # History for ESC model (collection)
        self.esc_history_data = []
        self.esc_3d_landscape_buffer = []

        self.collect_step_count = 0
        self.model = None

    def on_setup(self):
        """Correctly identifies either model type."""
        if self.control_group not in self.experiment.group_policies:
            print(f"\nWARNING: Logger group '{self.control_group}' not found. Disabling logger.\n")
            return
        policy = self.experiment.group_policies[self.control_group]
        self.model = get_het_model(policy)
        if isinstance(self.model, (HetControlMlpEmpirical, HetControlMlpEsc)):
            print(f"\nSUCCESS: Trajectory Logger initialized for group '{self.control_group}' (Model: {self.model.__class__.__name__}).")
        else:
            print(f"\nWARNING: A compatible model was not found for '{self.control_group}'. Disabling logger.\n")
            self.model = None

    def on_batch_collected(self, batch: TensorDictBase):
        """Routes logging logic based on the active model type."""
        if self.model is None:
            return

        if isinstance(self.model, HetControlMlpEsc):
            self._log_esc_scalars(self.experiment, batch)
            self._log_esc_plot(self.experiment, batch)
        elif isinstance(self.model, HetControlMlpEmpirical):
            self._log_dico_scalars(self.experiment, batch)
        
        self.collect_step_count += 1

    def on_evaluation_end(self, rollouts: List[TensorDictBase]):
        """This logic is specific to the Empirical (DiCo) model to log SND."""
        if not isinstance(self.model, HetControlMlpEmpirical):
            return

        episode_snd, episode_returns, logs_to_push = [], [], {}
        with torch.no_grad():
            for r in rollouts:
                agent_actions = self._get_agent_actions_for_rollout(r) 
                pairwise_distances = compute_behavioral_distance(agent_actions, just_mean=False)
                episode_snd.append(pairwise_distances.mean().item())

                if "eval/Agent_Distances" not in logs_to_push:
                    logs_to_push["eval/Agent_Distances"] = plot_agent_distances(
                        pairwise_distances_tensor=pairwise_distances, n_agents=self.model.n_agents
                    )

                reward_key = ('next', self.control_group, 'reward')
                episode_returns.append(r.get(reward_key).sum().item())

        if not episode_returns:
            return

        mean_actual_snd = float(np.mean(episode_snd))
        mean_return = float(np.mean(episode_returns))

        self.eps_actual_diversity.append(mean_actual_snd)
        self.eps_mean_returns.append(mean_return)
        self.eps_number.append(self.experiment.n_iters_performed)

        logs_to_push["eval/actual_snd"] = mean_actual_snd
        logs_to_push["eval/mean_return"] = mean_return

        if len(self.eps_number) > 1 and np.isfinite(self.eps_actual_diversity).all() and np.isfinite(self.eps_mean_returns).all():
            logs_to_push["eval/SND_vs_Reward_Trajectory"] = plot_trajectory_2d(
                snd=self.eps_actual_diversity, returns=self.eps_mean_returns, episodes=self.eps_number
            )

        self.experiment.logger.log(logs_to_push, step=self.experiment.n_iters_performed)

    def _log_dico_scalars(self, experiment: Experiment, batch: TensorDictBase):
        """Logs scalar metrics specific to the DiCo (HetControlMlpEmpirical) model."""
        keys_to_log = [
            (self.control_group, "logits"), (self.control_group, "out_loc_norm"),
            (self.control_group, "scaling_ratio"), (self.control_group, "estimated_snd"),
        ]
        to_log = {}
        for key in keys_to_log:
            val = batch.get(key, None)
            if val is not None:
                to_log[f"collection/{'/'.join(key)}"] = val.float().mean().item()
        if hasattr(self.model, "desired_snd"):
            to_log[f"collection/{self.control_group}/desired_snd"] = self.model.desired_snd.item()
        if to_log:
            experiment.logger.log(to_log, step=self.collect_step_count)

    def _log_esc_scalars(self, experiment: Experiment, batch: TensorDictBase):
        """Logs scalar metrics specific to the ESC (HetControlMlpEsc) model."""
        keys_to_log = [
            (self.control_group, "logits"), (self.control_group, "out_loc_norm"),
            (self.control_group, "scaling_ratio"), (self.control_group, "k_hat"),
            (self.control_group, "esc_dither"), (self.control_group, "esc_reward_J"),
            (self.control_group, "esc_J_raw"), (self.control_group, "esc_grad_estimate"),
            (self.control_group, "esc_k_hat_update"),
        ]
        to_log = {}
        for key in keys_to_log:
            val = batch.get(key, None)
            if val is not None:
                to_log[f"collection/{'/'.join(key)}"] = val.float().mean().item()
        if to_log:
            experiment.logger.log(to_log, step=self.collect_step_count)

    def _log_esc_plot(self, experiment: 'Experiment', latest_td: TensorDictBase):
        """Orchestrator method to extract ESC metrics, generate all plots, and log them."""
        metrics = self._extract_esc_metrics(latest_td)
        if metrics is None:
            return

        self.esc_history_data.append({
            "ratio": float(np.nanmean(metrics["scaling_ratio"])),
            "diversity": metrics["mean_diversity"],
        })

        if isinstance(self.model, HetControlMlpEsc) and metrics.get("J_raw") is not None:
            self.esc_3d_landscape_buffer.append({
                "J": float(np.nanmean(metrics["J_raw"])),
                # === FIX START ===
                # Removed .item() because the model stores these as plain floats
                "amplitude": self.model.esc_amplitude,
                "frequency": self.model.esc_frequency,
                # === FIX END ===
            })

        step_count = self.collect_step_count

        fig_ts = self._plot_control_dynamics(metrics, step_count)
        fig_scatter = self._plot_diversity_scatter(metrics, step_count)
        fig_line = self._plot_learning_evolution(step_count)
        fig_grad = self._plot_gradient_estimate(metrics, step_count)
        fig_3d_landscape = self._plot_esc_reward_landscape(step_count)

        logs_to_push = {
            "ESC/Control_Dynamics": fig_ts,
            "ESC/Diversity_vs_AbsRatio": fig_scatter,
            "ESC/Learning_Evolution": fig_line,
            "ESC/Gradient Estimate": fig_grad,
            "ESC/Reward_Landscape_3D": fig_3d_landscape,
            "ESC/average_scaling_ratio": float(np.nanmean(metrics["scaling_ratio"])),
            "ESC/average_behavioral_diversity": metrics["mean_diversity"],
            "ESC/k_hat": metrics["k_hat"],
        }
        
        final_logs = {k: v for k, v in logs_to_push.items() if v is not None}
        experiment.logger.log(final_logs, step=step_count)

    # ... (the rest of your file remains unchanged) ...
    def _extract_esc_metrics(self, latest_td: TensorDictBase) -> Union[dict, None]:
        """
        Extracts and processes all necessary metrics from the latest TensorDict
        FOR THE ESC MODEL.
        """
        try:
            td_episode = latest_td[:, 0]

            # --- Extract Core Metrics ---
            sr_tensor = td_episode.get((self.control_group, "scaling_ratio"))
            scaling_ratio = sr_tensor.detach().flatten(1).mean(dim=1).cpu().numpy()
            logits = td_episode.get((self.control_group, "logits")).detach()
            behavioral_diversity = logits.norm(dim=-1).std(dim=1).cpu().numpy()
            k_hat_tensor = td_episode.get((self.control_group, "k_hat"))
            k_hat = float(k_hat_tensor.reshape(-1)[0].item())

            if scaling_ratio.shape != behavioral_diversity.shape:
                print("Warning: scaling_ratio and diversity shapes differ; skipping ESC plots.")
                return None

            T = len(scaling_ratio)
            if T == 0:
                print("Warning: ESC metrics have length 0. Skipping plots.")
                return None

            metrics = {
                "T": T,
                "scaling_ratio": scaling_ratio,
                "behavioral_diversity": behavioral_diversity,
                "k_hat": k_hat,
                "mean_diversity": float(np.nanmean(behavioral_diversity)),
                "abs_scaling_ratio": np.abs(scaling_ratio),
            }

            # --- Extract Optional Gradient Metrics ---
            try:
                metrics["ge"] = td_episode.get((self.control_group, "esc_grad_estimate")).float().mean(dim=1).cpu().numpy()
                metrics["J_hat"] = td_episode.get((self.control_group, "esc_reward_J")).float().mean(dim=1).cpu().numpy()
                metrics["k_upd"] = td_episode.get((self.control_group, "esc_k_hat_update")).float().mean(dim=1).cpu().numpy()
                metrics["J_raw"] = td_episode.get((self.control_group, "esc_J_raw")).float().mean(dim=1).cpu().numpy()
            except KeyError:
                metrics.update({"ge": None, "J_hat": None, "k_upd": None, "J_raw": None})

            return metrics

        except (KeyError, IndexError, AttributeError) as e:
            print(f"Warning: Could not find/process expected keys for ESC plotting ({e}). Skipping.")
            return None

    def _plot_control_dynamics(self, metrics: dict, step_count: int) -> go.Figure:
        """Plot 1: Control Dynamics (Time Series)"""
        T = metrics["T"]
        x_t = np.arange(T)
        mean_diversity = metrics["mean_diversity"]

        fig = make_subplots(rows=1, cols=1, specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Scatter(x=x_t, y=metrics["scaling_ratio"], name="Scaling Ratio"),
                      row=1, col=1, secondary_y=False)
        fig.add_trace(go.Scatter(x=x_t, y=metrics["behavioral_diversity"], name="Diversity", line=dict(dash="dash")),
                      row=1, col=1, secondary_y=True)
        
        fig.add_shape(type="line", x0=0, x1=T-1, y0=mean_diversity, y1=mean_diversity,
                      xref="x1", yref="y2", line=dict(width=2, dash="dot"))
        fig.add_annotation(x=0.98, xref="paper", y=mean_diversity, yref="y2",
                           text=f"Mean Diversity = {mean_diversity:.3f}", showarrow=False,
                           xanchor="right", yshift=8, bgcolor="rgba(255,255,255,0.7)")
        
        fig.update_layout(title=f"ESC Control Dynamics — step #{step_count}",
                          template="plotly_white", height=350, legend_title_text="Metrics")
        fig.update_xaxes(title_text="Time Step")
        fig.update_yaxes(title_text="Scaling Ratio", secondary_y=False)
        fig.update_yaxes(title_text="Diversity", secondary_y=True, showgrid=False)
        return fig

    def _plot_diversity_scatter(self, metrics: dict, step_count: int) -> go.Figure:
        """Plot 2: Diversity vs |Scaling Ratio| (Scatter Plot)"""
        abs_ratio = metrics["abs_scaling_ratio"]
        diversity = metrics["behavioral_diversity"]
        mean_diversity = metrics["mean_diversity"]
        T = metrics["T"]

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=abs_ratio, y=diversity, mode="markers", name="Data"))

        if T > 1 and np.isfinite(abs_ratio).all() and np.isfinite(diversity).all():
            x_min, x_max = abs_ratio.min(), abs_ratio.max()
            if x_min != x_max:
                slope, intercept = np.polyfit(abs_ratio, diversity, 1)
                xs = np.array([x_min, x_max])
                ys = slope * xs + intercept
                fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", name=f"Fit (m≈{slope:.2f})"))

        fig.add_shape(type="line", x0=0, x1=1, xref="x domain", y0=mean_diversity, y1=mean_diversity,
                      yref="y", line=dict(width=2, dash="dot"))
        fig.add_annotation(x=1, xref="x domain", y=mean_diversity, yref="y",
                           text=f"mean diversity = {mean_diversity:.3f}", showarrow=False,
                           xshift=-6, yshift=10, bgcolor="rgba(255,255,255,0.7)")

        fig.update_layout(title=f"Diversity vs |Scaling Ratio| — step #{step_count}",
                          xaxis_title="|Scaling Ratio|", yaxis_title="Behavioral Diversity",
                          template="plotly_white", height=350)
        return fig

    def _plot_learning_evolution(self, step_count: int) -> go.Figure:
        """Plot 3: Avg Scaling Ratio vs Avg Diversity (History Plot)"""
        fig = go.Figure()
        
        history = self.esc_history_data
        if len(history) < 1:
            fig.update_layout(title=f"Diversity & Scaling Ratio Over Time (No History) — step #{step_count}")
            return fig

        ratios = np.array([d["ratio"] for d in history], dtype=float)
        diversities = np.array([d["diversity"] for d in history], dtype=float)
        mask = np.isfinite(ratios) & np.isfinite(diversities)
        
        y_ratios, y_diversities = ratios[mask], diversities[mask]
        steps = np.arange(len(y_ratios))

        if len(steps) > 0:
            fig.add_trace(go.Scatter(x=steps, y=y_diversities, mode='lines+markers', name='Avg. Diversity', yaxis='y1'))
            fig.add_trace(go.Scatter(x=steps, y=y_ratios, mode='lines+markers', name='Avg. Scaling Ratio', yaxis='y2'))

        fig.update_layout(
            title=f"Diversity & Scaling Ratio Over Time — up to step #{step_count}",
            xaxis_title="Collection Step", template="plotly_white", height=400, legend_title_text="Metrics",
            yaxis=dict(title=dict(text="Avg. Diversity", font=dict(color="#1f77b4")), tickfont=dict(color="#1f77b4")),
            yaxis2=dict(title=dict(text="Avg. Scaling Ratio", font=dict(color="#ff7f0e")), overlaying="y", side="right", tickfont=dict(color="#ff7f0e"))
        )
        return fig

    def _plot_gradient_estimate(self, metrics: dict, step_count: int) -> Union[go.Figure, None]:
        """Plot 4: Gradient Estimate & J (per time step)"""
        if metrics.get("ge") is None or metrics.get("J_hat") is None:
            return None

        ge, j_hat = metrics["ge"], metrics["J_hat"]
        k_upd, j_raw = metrics["k_upd"], metrics["J_raw"]
        
        t_axis = np.arange(len(j_hat))
        fig = make_subplots(rows=1, cols=1, specs=[[{"secondary_y": True}]])

        fig.add_trace(go.Scatter(x=t_axis, y=j_hat, name="Ĵ (EMA reward)"), row=1, col=1, secondary_y=False)
        if j_raw is not None and np.size(j_raw) == np.size(j_hat):
            fig.add_trace(go.Scatter(x=t_axis, y=j_raw, name="J (raw)", line=dict(width=1)), row=1, col=1, secondary_y=False)
        
        fig.add_trace(go.Scatter(x=t_axis, y=ge, name="Gradient estimate", line=dict(dash="dash")), row=1, col=1, secondary_y=True)
        fig.add_trace(go.Scatter(x=t_axis, y=k_upd, name="k̂ update = g·grad", line=dict(dash="dot")), row=1, col=1, secondary_y=True)

        fig.add_shape(type="line", x0=0, x1=1, y0=0.0, y1=0.0, xref="paper", yref="y2", line=dict(width=1))
        
        fig.update_layout(title=f"ESC Gradient Estimate & J — step #{step_count}",
                          template="plotly_white", height=350, legend_title_text="Metrics")
        fig.update_xaxes(title_text="Time Step")
        fig.update_yaxes(title_text="J / Ĵ", secondary_y=False)
        fig.update_yaxes(title_text="Gradient / Δk̂", secondary_y=True, showgrid=False)
        return fig

    def _plot_esc_reward_landscape(self, step_count: int) -> Union[go.Figure, None]:
        """Plot 5: 3D Reward Landscape (J vs. Amplitude & Frequency)"""
        if len(self.esc_3d_landscape_buffer) < 3:
            return None

        try:
            data = self.esc_3d_landscape_buffer
            amplitudes = np.array([d["amplitude"] for d in data])
            frequencies = np.array([d["frequency"] for d in data])
            j_values = np.array([d["J"] for d in data])

            mask = np.isfinite(amplitudes) & np.isfinite(frequencies) & np.isfinite(j_values)
            if not np.any(mask):
                return None
            
            amplitudes, frequencies, j_values = amplitudes[mask], frequencies[mask], j_values[mask]

        except (KeyError, IndexError):
            return None

        fig = go.Figure(data=[go.Scatter3d(
            x=amplitudes, y=frequencies, z=j_values, mode='markers',
            marker=dict(
                size=5, color=j_values, colorscale='Viridis', 
                colorbar_title='Reward (J)', opacity=0.8, showscale=True
            )
        )])

        fig.update_layout(
            title=f"ESC Reward Landscape — up to step #{step_count}",
            scene=dict(xaxis_title='ESC Amplitude', yaxis_title='ESC Frequency', zaxis_title='Reward (J)'),
            margin=dict(l=0, r=0, b=0, t=40), height=450
        )
        return fig

    def _get_agent_actions_for_rollout(self, rollout: TensorDictBase) -> List[torch.Tensor]:
        """Helper for on_evaluation_end for the Empirical model."""
        obs = rollout.get((self.control_group, "observation"))
        actions = []
        for i in range(self.model.n_agents):
            temp_td = TensorDict({(self.control_group, "observation"): obs}, batch_size=obs.shape[:-1])
            action_td = self.model._forward(temp_td, agent_index=i, compute_estimate=False) 
            actions.append(action_td.get(self.model.out_key))
        return actions

