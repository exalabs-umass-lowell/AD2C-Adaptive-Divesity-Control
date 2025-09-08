#  Copyright (c) 2024.
#  ProrokLab (https://www.proroklab.org/)
#  All rights reserved.

from __future__ import annotations

from dataclasses import dataclass, MISSING
from typing import Optional, Sequence, Tuple, Type, Dict

import torch
from tensordict import TensorDictBase
from tensordict.nn import NormalParamExtractor
from torch import nn
from torchrl.modules import MultiAgentMLP

from benchmarl.models.common import Model, ModelConfig
# We assume these utility functions are available in your project's specified paths
from AD2C.snd import compute_behavioral_distance
from AD2C.utils import overflowing_logits_norm
from .utils import squash


class HetControlMlpEscSnd(Model):
    """
    A hierarchical model that uses Extremum Seeking Control (ESC) to learn the
    optimal target diversity (desired_snd) that maximizes extrinsic reward.

    This model implements a two-layered control system:
    1.  An inner loop (inspired by DiCo) that adjusts a scaling factor to force
        the policy's behavioral diversity to match a given target.
    2.  An outer loop (ESC) that perturbs this diversity target and uses the
        resulting reward signal to perform gradient ascent, automatically finding
        the best diversity target for the task.
    """

    def __init__(
        self,
        # Common network parameters
        activation_class: Type[nn.Module],
        num_cells: Sequence[int],
        probabilistic: bool,
        scale_mapping: Optional[str],
        process_shared: bool,
        # ESC parameters for the outer loop
        esc_gain: float,
        esc_amplitude: float,
        esc_frequency: float,
        initial_desired_snd: float,
        reward_tau: float,
        # Diversity controller parameters for the inner loop
        diversity_tau: float,
        **kwargs,
    ):
        super().__init__(**kwargs)

        # Store parameters
        self.activation_class = activation_class
        self.num_cells = num_cells
        self.probabilistic = probabilistic
        self.scale_mapping = scale_mapping
        self.process_shared = process_shared

        # Setup the control state for both loops
        self._setup_control_state(
            initial_desired_snd, esc_gain, esc_amplitude, esc_frequency, reward_tau, diversity_tau
        )
        # Setup the neural networks
        self._setup_networks()

        self.scale_extractor = (
            NormalParamExtractor(scale_mapping=scale_mapping)
            if scale_mapping is not None
            else None
        )
        self._current_device = torch.device(self.device)

    def _setup_control_state(
        self, initial_desired_snd, esc_gain, esc_amplitude, esc_frequency, reward_tau, diversity_tau
    ):
        # --- State for the Outer Loop (ESC) ---
        self.esc_gain = float(esc_gain)
        self.esc_amplitude = float(esc_amplitude)
        self.esc_frequency = float(esc_frequency)
        self.reward_tau = float(reward_tau)

        # This buffer is the learned parameter (equivalent to k_hat in the old model)
        self.register_buffer(
            "learned_desired_snd",
            torch.tensor([initial_desired_snd], device=self.device, dtype=torch.float),
        )
        # Buffers for ESC mechanics
        self.register_buffer("time_step", torch.tensor(0, device=self.device, dtype=torch.long))
        self.register_buffer("last_dither", torch.tensor([0.0], device=self.device, dtype=torch.float))
        self.register_buffer("s_reward", torch.tensor(0.0, device=self.device, dtype=torch.float))

        # --- State for the Inner Loop (Diversity Controller) ---
        self.diversity_tau = float(diversity_tau)
        self.register_buffer(
            "estimated_snd", torch.tensor([float("nan")], device=self.device, dtype=torch.float)
        )

    def _setup_networks(self):
        self.input_features = self.input_leaf_spec.shape[-1]
        self.output_features = self.output_leaf_spec.shape[-1]

        # Shared part of the policy
        self.shared_mlp = MultiAgentMLP(
            n_agent_inputs=self.input_features,
            n_agent_outputs=self.output_features,
            n_agents=self.n_agents,
            centralised=False,
            share_params=True,
            activation_class=self.activation_class,
            num_cells=self.num_cells,
            device=self.device,
        )
        # Heterogeneous (per-agent) part of the policy
        agent_outputs = (
            self.output_features // 2 if self.probabilistic else self.output_features
        )
        self.agent_mlps = MultiAgentMLP(
            n_agent_inputs=self.input_features,
            n_agent_outputs=agent_outputs,
            n_agents=self.n_agents,
            centralised=False,
            share_params=False,
            activation_class=self.activation_class,
            num_cells=self.num_cells,
            device=self.device,
        )

    def _forward(self, tensordict: TensorDictBase, **kwargs) -> TensorDictBase:
        input_tensor = tensordict.get(self.in_key)
        self._ensure_device(input_tensor.device)

        # 1. OUTER LOOP: Get the perturbed diversity target from the ESC controller.
        current_target_snd, current_dither = self._get_esc_perturbed_target()

        # 2. INNER LOOP: Estimate the policy's current, unscaled behavioral diversity.
        with torch.no_grad():
            measured_diversity = self._estimate_snd(input_tensor)
            if self.estimated_snd.isnan().any():
                self.estimated_snd.copy_(measured_diversity)
            else:
                # Soft update of the diversity estimate
                self.estimated_snd.copy_(
                    (1 - self.diversity_tau) * self.estimated_snd + self.diversity_tau * measured_diversity
                )
        
        # 3. INNER LOOP: Compute the scaling ratio to match the target diversity.
        # Add a small epsilon for numerical stability.
        scaling_ratio = current_target_snd / (self.estimated_snd + 1e-6)

        # 4. POLICY EXECUTION: Apply the scaling ratio.
        shared_out = self.shared_mlp(input_tensor)
        agent_out = self.agent_mlps(input_tensor)
        shared_out = self.process_shared_out(shared_out)

        if self.probabilistic:
            shared_loc, shared_scale = shared_out.chunk(2, -1)
            agent_loc = shared_loc + agent_out * scaling_ratio
            out = torch.cat([agent_loc, shared_scale], dim=-1)
        else:
            out = shared_out + scaling_ratio * agent_out
        
        # 5. LOGGING: Store all intermediate values for analysis.
        self._log_outputs(tensordict, out, scaling_ratio, current_dither, current_target_snd)
        tensordict.set(self.out_key, out)
        return tensordict

    def _get_esc_perturbed_target(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generates the perturbed target diversity for the current timestep."""
        with torch.no_grad():
            self.time_step.add_(1)
            t = self.time_step.to(dtype=torch.float32)
            dither = self.esc_amplitude * torch.sin(self.esc_frequency * t)
            perturbed_target = self.learned_desired_snd + dither
            self.last_dither.copy_(dither)
            # Ensure the target diversity is non-negative
            return torch.relu(perturbed_target), dither

    def _estimate_snd(self, obs: torch.Tensor) -> torch.Tensor:
        """Estimates the behavioral diversity of the unscaled heterogeneous policies."""
        agent_actions = [agent_net(obs) for agent_net in self.agent_mlps.agent_networks]
        distance = compute_behavioral_distance(agent_actions=agent_actions, just_mean=True).mean()
        return distance

    def _update_esc(self, tensordict: TensorDictBase) -> None:
        """
        Performs the ESC update on the `learned_desired_snd` parameter.
        This should be called by a callback *after* a batch has been collected.
        """
        reward_key = ("next", self.agent_group, "reward")
        if reward_key not in tensordict.keys(include_nested=True):
            return
        with torch.no_grad():
            reward = tensordict.get(reward_key)
            if reward.abs().sum() < 1e-6:
                return

            esc_updates = self._calculate_esc_update(reward, self.s_reward, self.last_dither)
            self.s_reward.copy_(esc_updates["s_reward_new"])
            self.learned_desired_snd.add_(esc_updates["snd_update"])

            # Log all ESC-related metrics to the tensordict for the logger callback
            log_data = {
                "J_mean": esc_updates["J_mean"],
                "s_reward": self.s_reward,
                "grad_estimate": esc_updates["gradient_estimate"],
                "snd_update": esc_updates["snd_update"],
                "learned_desired_snd": self.learned_desired_snd,
            }
            self._log_to_tensordict(tensordict, "esc", log_data)

    def _calculate_esc_update(self, reward: torch.Tensor, s_reward_old: torch.Tensor, last_dither: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Calculates the gradient ascent update for the learned diversity target."""
        j_mean = reward.mean()
        s_reward_new = s_reward_old * (1 - self.reward_tau) + j_mean * self.reward_tau
        gradient_estimate = s_reward_new * last_dither
        snd_update = self.esc_gain * gradient_estimate
        return {
            "J_mean": j_mean,
            "s_reward_new": s_reward_new,
            "gradient_estimate": gradient_estimate,
            "snd_update": snd_update,
        }

    def _log_to_tensordict(self, td: TensorDictBase, namespace: str, data: Dict[str, torch.Tensor]):
        """Helper function to log tensors, expanding scalars to match batch shape."""
        agent_group_shape = td.get_item_shape(self.agent_group)
        for key, tensor in data.items():
            tensor_to_log = (
                tensor.expand(agent_group_shape)
                if tensor.numel() == 1 and len(agent_group_shape) > 0
                else tensor
            )
            td.set((self.agent_group, namespace, key), tensor_to_log)

    def _log_outputs(self, td: TensorDictBase, out: torch.Tensor, scaling_ratio: torch.Tensor, dither: torch.Tensor, target_snd: torch.Tensor):
        """Logs all relevant outputs from the forward pass."""
        if self.probabilistic:
            loc_to_normalize, _ = out.chunk(2, dim=-1)
        else:
            loc_to_normalize = out
        out_loc_norm = overflowing_logits_norm(
            loc_to_normalize, self.action_spec[self.agent_group, "action"]
        )
        log_data = {
            "logits": out,
            "scaling_ratio": scaling_ratio,
            "dither": dither,
            "out_loc_norm": out_loc_norm,
            "estimated_snd": self.estimated_snd,
            "target_snd": target_snd,
        }
        self._log_to_tensordict(td, "output", log_data)

    def process_shared_out(self, logits: torch.Tensor) -> torch.Tensor:
        """Applies squashing and extracts scale for probabilistic policies."""
        if not self.probabilistic and self.process_shared:
            return squash(logits, action_spec=self.action_spec[self.agent_group, "action"], clamp=False)
        elif self.probabilistic:
            loc, scale = self.scale_extractor(logits)
            if self.process_shared:
                loc = squash(loc, action_spec=self.action_spec[self.agent_group, "action"], clamp=False)
            return torch.cat([loc, scale], dim=-1)
        return logits

    def _ensure_device(self, target: torch.device) -> None:
        """Ensures the model is on the same device as the input data."""
        if self._current_device != target:
            self.to(target)
            self._current_device = target

# --- The Config class for the combined model ---
@dataclass
class HetControlMlpEscSndConfig(ModelConfig):
    # Common network parameters
    activation_class: Type[nn.Module] = MISSING
    num_cells: Sequence[int] = MISSING
    process_shared: bool = MISSING
    probabilistic: Optional[bool] = MISSING
    scale_mapping: Optional[str] = MISSING
    
    # ESC (Outer Loop) parameters
    esc_gain: float = MISSING
    esc_amplitude: float = MISSING
    esc_frequency: float = MISSING
    initial_desired_snd: float = MISSING
    reward_tau: float = MISSING  # EMA smoothing for the reward signal

    # Diversity Controller (Inner Loop) parameters
    diversity_tau: float = MISSING  # EMA smoothing for the diversity estimate

    @staticmethod
    def associated_class():
        return HetControlMlpEscSnd