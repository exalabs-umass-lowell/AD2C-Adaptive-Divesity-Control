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
        Create and log THREE separate Plotly figures:
        1) Control Dynamics (time series): scaling ratio & diversity
        2) Diversity vs |Scaling Ratio| (scatter) + linear fit
        3) Learning Evolution (running averages per collection step)
        """
        # ---------- 1) Extract data from latest_td ----------
        try:
            # Take first environment in the collector batch (you can adapt if needed)
            td_episode = latest_td[:, 0]

            # scaling_ratio: shape ~ [T, n_agents, action_dim] or [T, n_agents, ...]
            # We reduce across agent dim and feature dim to get a single series over time
            sr = td_episode.get((self.control_group, "scaling_ratio"))  # tensor
            scaling_ratio = sr.mean(dim=(1, -1)).detach().cpu().numpy()  # [T]

            # logits used to estimate a diversity proxy
            logits = td_episode.get((self.control_group, "logits")).detach()
            # diversity proxy: std over agents of the logits' L2 norm per agent
            behavioral_diversity = logits.norm(dim=-1).std(dim=1).cpu().numpy()  # [T]

            # k_hat (scalar-like). If it has [T, n_agents] we just pick the first value.
            k_hat_tensor = td_episode.get((self.control_group, "k_hat"))
            k_hat = k_hat_tensor.reshape(-1)[0].item()

            # Basic sanity
            if scaling_ratio.shape != behavioral_diversity.shape:
                print("Warning: scaling_ratio and diversity shapes differ; skipping ESC plots.")
                return

        except (KeyError, IndexError) as e:
            print(f"Warning: Could not find expected keys for ESC plotting ({e}). Skipping.")
            return

        T = len(scaling_ratio)
        if T == 0:
            return

        mean_diversity = float(np.mean(behavioral_diversity))
        abs_scaling_ratio = np.abs(scaling_ratio)

        # Keep a running history for plot (3)
        self.esc_history_data.append({
            "ratio": float(np.mean(scaling_ratio)),
            "diversity": float(np.mean(behavioral_diversity)),
        })

        # ---------- 2) FIGURE 1: Control Dynamics (time series) ----------
        fig_ts = make_subplots(rows=1, cols=1, specs=[[{"secondary_y": True}]])
        fig_ts.add_trace(
            go.Scatter(x=np.arange(T), y=scaling_ratio, name="Scaling Ratio"),
            row=1, col=1, secondary_y=False,
        )
        fig_ts.add_trace(
            go.Scatter(x=np.arange(T), y=behavioral_diversity, name="Diversity", line=dict(dash="dash")),
            row=1, col=1, secondary_y=True,
        )
        # Mean diversity line on secondary y
        fig_ts.add_shape(
            type="line", x0=0, x1=T-1, y0=mean_diversity, y1=mean_diversity,
            xref="x1", yref="y2", line=dict(width=2, dash="dot")
        )
        fig_ts.update_layout(
            title=f"ESC Control Dynamics — step #{self.collect_step_count}",
            template="plotly_white", height=350, legend_title_text="Metrics",
        )
        fig_ts.add_annotation(
            x=0.98, xref="paper",        # pin near the right edge regardless of x-range
            y=mean_diversity, yref="y2", # reference the secondary y-axis
            text=f"Mean Diversity = {mean_diversity:.3f}",
            showarrow=False,
            xanchor="right",
            yshift=8,
            bgcolor="rgba(255,255,255,0.7)"
        )
        fig_ts.update_xaxes(title_text="Time Step", row=1, col=1)
        fig_ts.update_yaxes(title_text="Scaling Ratio", row=1, col=1, secondary_y=False)
        fig_ts.update_yaxes(title_text="Diversity", row=1, col=1, secondary_y=True, showgrid=False)

        # ---------- 3) FIGURE 2: Diversity vs |Scaling Ratio| (scatter + fit) ----------
        fig_scatter = go.Figure()
        fig_scatter.add_trace(
            go.Scatter(x=abs_scaling_ratio, y=behavioral_diversity, mode="markers", name="Data")
        )

        # optional linear fit
        if T > 1 and np.isfinite(abs_scaling_ratio).all() and np.isfinite(behavioral_diversity).all():
            slope, intercept = np.polyfit(abs_scaling_ratio, behavioral_diversity, 1)
            xs = (np.linspace(abs_scaling_ratio.min(), abs_scaling_ratio.max(), 100)
                if abs_scaling_ratio.max() > abs_scaling_ratio.min() else abs_scaling_ratio)
            ys = slope * xs + intercept
            fig_scatter.add_trace(go.Scatter(x=xs, y=ys, mode="lines", name=f"Fit (m≈{slope:.2f})"))

        # --- horizontal mean diversity line across the whole plot width ---
        y_mean = float(np.nanmean(behavioral_diversity))
        fig_scatter.add_shape(
            type="line",
            x0=0, x1=1, xref="x domain",   # spans the full x domain
            y0=y_mean, y1=y_mean, yref="y",
            line=dict(width=2, dash="dot")
        )
        fig_scatter.add_annotation(
            x=1, xref="x domain", y=y_mean, yref="y",
            text=f"mean diversity = {y_mean:.2f}",
            showarrow=False, xshift=-6, yshift=10
        )

        fig_scatter.update_layout(
            title=f"Diversity vs |Scaling Ratio| — step #{self.collect_step_count}",
            xaxis_title="|Scaling Ratio|",
            yaxis_title="Behavioral Diversity",
            template="plotly_white",
            height=350,
        )

        # ---------- 4) FIGURE 3: Learning Evolution (history over collection steps) ----------
        fig_hist = make_subplots(rows=1, cols=1, specs=[[{"secondary_y": True}]])
        if len(self.esc_history_data) >= 1:
            idx = np.arange(len(self.esc_history_data))
            ratios = np.array([d["ratio"] for d in self.esc_history_data], dtype=float)
            diversities = np.array([d["diversity"] for d in self.esc_history_data], dtype=float)

            fig_hist.add_trace(
                go.Scatter(x=idx, y=ratios, name="Avg. Ratio"),
                row=1, col=1, secondary_y=False,
            )
            fig_hist.add_trace(
                go.Scatter(x=idx, y=diversities, name="Avg. Diversity", line=dict(dash="dash")),
                row=1, col=1, secondary_y=True,
            )

            # --- Means as TRACES (show in legend) ---
            x_left = float(idx.min()) if idx.size else 0.0
            x_right = float(idx.max()) if idx.size else 0.0
            if x_left == x_right:       # single point -> give some width
                x_left -= 0.5
                x_right += 0.5
                fig_hist.update_xaxes(range=[x_left, x_right])

            mean_div_hist = float(np.nanmean(diversities))
            mean_ratio_hist = float(np.nanmean(ratios))

            fig_hist.add_trace(
                go.Scatter(x=[x_left, x_right], y=[mean_ratio_hist, mean_ratio_hist],
                        name=f"Mean Ratio ({mean_ratio_hist:.2f})",
                        line=dict(dash="dot"), hoverinfo="skip"),
                row=1, col=1, secondary_y=False,
            )
            fig_hist.add_trace(
                go.Scatter(x=[x_left, x_right], y=[mean_div_hist, mean_div_hist],
                        name=f"Mean Diversity ({mean_div_hist:.3f})",
                        line=dict(dash="dot"), hoverinfo="skip"),
                row=1, col=1, secondary_y=True,
            )

            # Label the avg diversity value so it's readable even when zoomed
            fig_hist.add_annotation(
                x=0.98, xref="paper",
                y=mean_div_hist, yref="y2",
                text=f"Avg Diversity = {mean_div_hist:.3f}",
                showarrow=False, xanchor="right", yshift=8,
                bgcolor="rgba(255,255,255,0.7)"
            )

        fig_hist.update_layout(
            title=f"ESC Learning Evolution — up to step #{self.collect_step_count}",
            template="plotly_white", height=350, legend_title_text="Metrics",
        )
        fig_hist.update_xaxes(title_text="Collection Epoch", row=1, col=1)
        fig_hist.update_yaxes(title_text="Avg. Ratio", row=1, col=1, secondary_y=False)
        fig_hist.update_yaxes(title_text="Avg. Diversity", row=1, col=1, secondary_y=True, showgrid=False)

        # ---------- 5) Log all three plots + key scalars ----------
        logs_to_push = {
            "ESC/Control_Dynamics": fig_ts,
            "ESC/Diversity_vs_AbsRatio": fig_scatter,
            "ESC/Learning_Evolution": fig_hist,
            "ESC/average_scaling_ratio": float(np.mean(scaling_ratio)),
            "ESC/average_behavioral_diversity": float(np.mean(behavioral_diversity)),
            "ESC/k_hat": k_hat,
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