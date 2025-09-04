from typing import List
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
        self.eps_actual_diversity, self.eps_mean_returns, self.eps_number = [], [], []
        self.esc_history_data = []
        self.collect_step_count = 0
        self.model = None

    def on_setup(self):
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

    def on_batch_collected(self, batch: TensorDictBase):
        if not isinstance(self.model, HetControlMlpEsc):
            return
        self._log_esc_scalars(self.experiment, batch)
        self._log_esc_plot(self.experiment, batch)
        self.collect_step_count += 1

    def on_evaluation_end(self, rollouts: List[TensorDictBase]):
        if self.model is None:
            return
        episode_snd, episode_returns, logs_to_push = [], [], {}
        with torch.no_grad():
            for r in rollouts:
                agent_actions = self._get_agent_actions_for_rollout(r)
                pairwise_distances = compute_behavioral_distance(agent_actions, just_mean=False)
                episode_snd.append(pairwise_distances.mean().item())

                if "eval/Agent_Distances" not in logs_to_push:
                    logs_to_push["eval/Agent_Distances"] = plot_agent_distances(
                        pairwise_distances_tensor=pairwise_distances,
                        n_agents=self.model.n_agents
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

    def _log_esc_scalars(self, experiment: Experiment, batch: TensorDictBase):
        keys_to_log = [
            (self.control_group, "logits"),
            (self.control_group, "out_loc_norm"),
            (self.control_group, "scaling_ratio"),
            (self.control_group, "k_hat"),
            (self.control_group, "esc_dither"),
            (self.control_group, "esc_reward_J"),
            (self.control_group, "esc_J_raw"),          # NEW
            (self.control_group, "esc_grad_estimate"),
            (self.control_group, "esc_k_hat_update"),
        ]
        to_log = {}
        for key in keys_to_log:
            val = batch.get(key, None)
            if val is not None:
                to_log[f"collection/{'/'.join(key)}"] = val.float().mean().item()
        if to_log:
            experiment.logger.log(to_log, step=self.collect_step_count)

    def _log_esc_plot(self, experiment: Experiment, latest_td: TensorDictBase):
        # --- Extract from latest collector episode (env index 0) ---
        try:
            td_episode = latest_td[:, 0]

            # scaling ratio may be [T, n_agents] or [T, n_agents, action_dim, ...]
            sr = td_episode.get((self.control_group, "scaling_ratio"))         # [T, ...]
            scaling_ratio = sr.detach().flatten(1).mean(dim=1).cpu().numpy()   # [T]

            logits = td_episode.get((self.control_group, "logits")).detach()   # [T, n_agents, action_dim]
            behavioral_diversity = logits.norm(dim=-1).std(dim=1).cpu().numpy()  # [T]

            k_hat_tensor = td_episode.get((self.control_group, "k_hat"))
            k_hat = float(k_hat_tensor.reshape(-1)[0].item())

            if scaling_ratio.shape != behavioral_diversity.shape:
                print("Warning: scaling_ratio and diversity shapes differ; skipping ESC plots.")
                return
        except (KeyError, IndexError) as e:
            print(f"Warning: Could not find expected keys for ESC plotting ({e}). Skipping.")
            return

        T = len(scaling_ratio)
        if T == 0:
            return

        mean_diversity = float(np.nanmean(behavioral_diversity))
        abs_scaling_ratio = np.abs(scaling_ratio)

        # History for plot 3
        self.esc_history_data.append({
            "ratio": float(np.nanmean(scaling_ratio)),
            "diversity": float(np.nanmean(behavioral_diversity)),
        })

        # ---------- Plot 1: Control Dynamics ----------
        x_t = np.arange(T)
        fig_ts = make_subplots(rows=1, cols=1, specs=[[{"secondary_y": True}]])
        fig_ts.add_trace(go.Scatter(x=x_t, y=scaling_ratio, name="Scaling Ratio"),
                         row=1, col=1, secondary_y=False)
        fig_ts.add_trace(go.Scatter(x=x_t, y=behavioral_diversity, name="Diversity", line=dict(dash="dash")),
                         row=1, col=1, secondary_y=True)
        fig_ts.add_shape(type="line", x0=0, x1=T-1, y0=mean_diversity, y1=mean_diversity,
                         xref="x1", yref="y2", line=dict(width=2, dash="dot"))
        fig_ts.add_annotation(x=0.98, xref="paper", y=mean_diversity, yref="y2",
                              text=f"Mean Diversity = {mean_diversity:.3f}", showarrow=False,
                              xanchor="right", yshift=8, bgcolor="rgba(255,255,255,0.7)")
        fig_ts.update_layout(title=f"ESC Control Dynamics — step #{self.collect_step_count}",
                             template="plotly_white", height=350, legend_title_text="Metrics")
        fig_ts.update_xaxes(title_text="Time Step")
        fig_ts.update_yaxes(title_text="Scaling Ratio", secondary_y=False)
        fig_ts.update_yaxes(title_text="Diversity", secondary_y=True, showgrid=False)

        # ---------- Plot 2: Diversity vs |Scaling Ratio| ----------
        fig_scatter = go.Figure()
        fig_scatter.add_trace(go.Scatter(x=abs_scaling_ratio, y=behavioral_diversity, mode="markers", name="Data"))
        if T > 1 and np.isfinite(abs_scaling_ratio).all() and np.isfinite(behavioral_diversity).all():
            slope, intercept = np.polyfit(abs_scaling_ratio, behavioral_diversity, 1)
            xs = np.linspace(abs_scaling_ratio.min(), abs_scaling_ratio.max(), 100) if abs_scaling_ratio.max() > abs_scaling_ratio.min() else abs_scaling_ratio
            ys = slope * xs + intercept
            fig_scatter.add_trace(go.Scatter(x=xs, y=ys, mode="lines", name=f"Fit (m≈{slope:.2f})"))
        y_mean = float(np.nanmean(behavioral_diversity))
        fig_scatter.add_shape(type="line", x0=0, x1=1, xref="x domain", y0=y_mean, y1=y_mean, yref="y",
                              line=dict(width=2, dash="dot"))
        fig_scatter.add_annotation(x=1, xref="x domain", y=y_mean, yref="y",
                                   text=f"mean diversity = {y_mean:.3f}", showarrow=False, xshift=-6, yshift=10)
        fig_scatter.update_layout(title=f"Diversity vs |Scaling Ratio| — step #{self.collect_step_count}",
                                  xaxis_title="|Scaling Ratio|", yaxis_title="Behavioral Diversity",
                                  template="plotly_white", height=350)

        # ---------- Plot 3: Avg Scaling Ratio vs Avg Diversity (history correlation) ----------
        fig_hist = go.Figure()
        if len(self.esc_history_data) >= 1:
            ratios = np.array([d["ratio"] for d in self.esc_history_data], dtype=float)
            diversities = np.array([d["diversity"] for d in self.esc_history_data], dtype=float)

            mask = np.isfinite(ratios) & np.isfinite(diversities)
            x, y = ratios[mask], diversities[mask]

            fig_hist.add_trace(go.Scatter(x=x, y=y, mode="markers", name="Epoch Averages"))
            if x.size > 1 and (x.max() > x.min()):
                slope, intercept = np.polyfit(x, y, 1)
                xs = np.linspace(x.min(), x.max(), 100)
                ys = slope * xs + intercept
                fig_hist.add_trace(go.Scatter(x=xs, y=ys, mode="lines", name=f"Fit (m≈{slope:.2f})"))

            if x.size > 0:
                x_mean = float(np.nanmean(x))
                y_mean_hist = float(np.nanmean(y))
                y_min, y_max = float(np.nanmin(y)), float(np.nanmax(y))
                x_min, x_max = float(np.nanmin(x)), float(np.nanmax(x))
                if y_min == y_max: y_min -= 1e-6; y_max += 1e-6
                if x_min == x_max: x_min -= 1e-6; x_max += 1e-6
                fig_hist.add_shape(type="line", x0=x_mean, x1=x_mean, y0=y_min, y1=y_max, xref="x", yref="y", line=dict(width=2, dash="dot"))
                fig_hist.add_shape(type="line", x0=x_min, x1=x_max, y0=y_mean_hist, y1=y_mean_hist, xref="x", yref="y", line=dict(width=2, dash="dot"))
                fig_hist.add_annotation(x=x_mean, y=y_max, yshift=10, text=f"Mean Ratio = {x_mean:.3f}", showarrow=False)
                fig_hist.add_annotation(x=x_max, y=y_mean_hist, xshift=8, text=f"Mean Diversity = {y_mean_hist:.3f}", showarrow=False)

        fig_hist.update_layout(title=f"Avg Scaling Ratio vs Avg Diversity — up to step #{self.collect_step_count}",
                               xaxis_title="Avg. Scaling Ratio (per collection step)",
                               yaxis_title="Avg. Diversity (per collection step)",
                               template="plotly_white", height=350, legend_title_text="Metrics")

        # ---------- Plot 4: Gradient Estimate & J (per time step) ----------
        fig_grad = None
        try:
            # Per-time-step, avg over agents
            ge = td_episode.get((self.control_group, "esc_grad_estimate")).float().mean(dim=1).cpu().numpy()
            J_hat = td_episode.get((self.control_group, "esc_reward_J")).float().mean(dim=1).cpu().numpy()
            k_upd = td_episode.get((self.control_group, "esc_k_hat_update")).float().mean(dim=1).cpu().numpy()
            J_raw = td_episode.get((self.control_group, "esc_J_raw")).float().mean(dim=1).cpu().numpy()  # NEW (optional)

            t_axis = np.arange(len(J_hat))
            fig_grad = make_subplots(rows=1, cols=1, specs=[[{"secondary_y": True}]])
            # Left axis: objectives
            fig_grad.add_trace(go.Scatter(x=t_axis, y=J_hat, name="Ĵ (EMA reward)"), row=1, col=1, secondary_y=False)
            if J_raw is not None and np.size(J_raw) == np.size(J_hat):
                fig_grad.add_trace(go.Scatter(x=t_axis, y=J_raw, name="J (raw)", line=dict(width=1)),
                                   row=1, col=1, secondary_y=False)
            # Right axis: gradient and parameter update
            fig_grad.add_trace(go.Scatter(x=t_axis, y=ge, name="Gradient estimate", line=dict(dash="dash")),
                               row=1, col=1, secondary_y=True)
            fig_grad.add_trace(go.Scatter(x=t_axis, y=k_upd, name="k̂ update = g·grad", line=dict(dash="dot")),
                               row=1, col=1, secondary_y=True)

            # Zero line on right axis
            fig_grad.add_shape(type="line", x0=0, x1=max(1, len(t_axis)-1), y0=0.0, y1=0.0,
                               xref="x1", yref="y2", line=dict(width=1))
            fig_grad.update_layout(title=f"ESC Gradient Estimate & J — step #{self.collect_step_count}",
                                   template="plotly_white", height=350, legend_title_text="Metrics")
            fig_grad.update_xaxes(title_text="Time Step")
            fig_grad.update_yaxes(title_text="J / Ĵ", secondary_y=False)
            fig_grad.update_yaxes(title_text="Gradient / Δk̂", secondary_y=True, showgrid=False)
        except KeyError:
            pass  # Keys not present on this batch (e.g., eval) — skip

        # ---------- Log everything ----------
        logs_to_push = {
            "ESC/Control_Dynamics": fig_ts,
            "ESC/Diversity_vs_AbsRatio": fig_scatter,
            "ESC/Learning_Evolution": fig_hist,
            "ESC/average_scaling_ratio": float(np.nanmean(scaling_ratio)),
            "ESC/average_behavioral_diversity": float(np.nanmean(behavioral_diversity)),
            "ESC/k_hat": k_hat,
        }
        if fig_grad is not None:
            logs_to_push["ESC/GradEstimate_vs_Jhat"] = fig_grad

        experiment.logger.log(logs_to_push, step=self.collect_step_count)

    def _get_agent_actions_for_rollout(self, rollout: TensorDictBase) -> List[torch.Tensor]:
        obs = rollout.get((self.control_group, "observation"))
        actions = []
        for i in range(self.model.n_agents):
            temp_td = TensorDict({(self.control_group, "observation"): obs}, batch_size=obs.shape[:-1])
            action_td = self.model._forward(temp_td, agent_index=i, compute_estimate=False)
            actions.append(action_td.get(self.model.out_key))
        return actions
