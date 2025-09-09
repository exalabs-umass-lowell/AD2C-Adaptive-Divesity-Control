from __future__ import annotations
from dataclasses import dataclass, MISSING
from typing import Optional, Sequence, Tuple, Type, Dict

import torch
from tensordict import TensorDictBase
from tensordict.nn import NormalParamExtractor
from torch import nn
from torchrl.modules import MultiAgentMLP

from benchmarl.models.common import Model, ModelConfig
# Make sure these imports are correct for your project structure
from AD2C.snd import compute_behavioral_distance
from AD2C.utils import overflowing_logits_norm
from .utils import squash


class HetControlMlpEscSnd(Model):
    """
    A model that combines Extremum Seeking Control (ESC) with diversity control (SND).
    - It learns the optimal target diversity (`k_hat`) via ESC based on environment rewards.
    - It controls the agent-specific policy contributions to match this learned target diversity.
    """

    def __init__(
        self,
        activation_class: Type[nn.Module],
        num_cells: Sequence[int],
        esc_gain: float,
        esc_amplitude: float,
        esc_frequency: float,
        initial_k: float,           # This is now the initial target diversity
        tau: float,                 # This is the ESC reward smoothing tau
        snd_tau: float,             # This is the SND measurement smoothing tau
        probabilistic: bool,
        scale_mapping: Optional[str],
        process_shared: bool,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_cells = num_cells
        self.activation_class = activation_class
        self.probabilistic = probabilistic
        self.scale_mapping = scale_mapping
        self.process_shared = process_shared

        # ESC parameters for learning k_hat
        self.esc_gain = float(esc_gain)
        self.esc_amplitude = float(esc_amplitude)
        self.esc_frequency = float(esc_frequency)
        
        # SND parameter for smoothing the diversity measurement
        self.snd_tau = snd_tau

        self._setup_esc_and_snd_state(initial_k, tau)
        self._setup_networks()
        self.scale_extractor = (
            NormalParamExtractor(scale_mapping=scale_mapping) if scale_mapping is not None else None
        )
        self._current_device = torch.device(self.device)

    def _setup_esc_and_snd_state(self, initial_k: float, esc_tau: float):
        # State for learning the target diversity (k_hat) via ESC
        self.esc_tau = float(esc_tau)
        self.register_buffer("k_hat", torch.tensor([initial_k], device=self.device, dtype=torch.float))
        self.register_buffer("time_step", torch.tensor(0, device=self.device, dtype=torch.long))
        self.register_buffer("last_dither", torch.tensor([0.0], device=self.device, dtype=torch.float))
        self.register_buffer("s_reward", torch.tensor(0.0, device=self.device, dtype=torch.float))
        
        # State for measuring diversity (SND)
        self.register_buffer(
            "estimated_snd", torch.tensor([float("nan")], device=self.device, dtype=torch.float)
        )

    def _setup_networks(self):
        self.input_features = self.input_leaf_spec.shape[-1]
        self.output_features = self.output_leaf_spec.shape[-1]
        self.shared_mlp = MultiAgentMLP(
            n_agent_inputs=self.input_features,
            n_agent_outputs=self.output_features,
            n_agents=self.n_agents,
            centralised=False, share_params=True, activation_class=self.activation_class,
            num_cells=self.num_cells, device=self.device,
        )
        agent_outputs = self.output_features // 2 if self.probabilistic else self.output_features
        self.agent_mlps = MultiAgentMLP(
            n_agent_inputs=self.input_features,
            n_agent_outputs=agent_outputs,
            n_agents=self.n_agents,
            centralised=False, share_params=False, activation_class=self.activation_class,
            num_cells=self.num_cells, device=self.device,
        )

    def _forward(self, tensordict: TensorDictBase, **kwargs) -> TensorDictBase:
        input_tensor = tensordict.get(self.in_key)
        self._ensure_device(input_tensor.device)

        # 1. Get the dither signal to perturb the target diversity
        current_dither = self._get_dither_signal()

        # 2. Measure the current behavioral diversity of the agent networks
        distance = self.estimate_snd(input_tensor)
        
        # 3. Calculate the scaling ratio
        # The target diversity is our learned k_hat, perturbed by the dither
        dithered_target_diversity = self.k_hat + current_dither
        # The scaling ratio tries to match the measured distance to the dithered target
        scaling_ratio = dithered_target_diversity / distance.clamp(min=1e-6)

        # 4. Apply the scaling to the network outputs
        shared_out = self.shared_mlp(input_tensor)
        agent_out = self.agent_mlps(input_tensor)
        shared_out = self.process_shared_out(shared_out)

        if self.probabilistic:
            shared_loc, shared_scale = shared_out.chunk(2, -1)
            agent_loc = shared_loc + agent_out * scaling_ratio
            out = torch.cat([agent_loc, shared_scale], dim=-1)
        else:
            out = shared_out + scaling_ratio * agent_out
        
        self._log_outputs(tensordict, out, scaling_ratio, current_dither, distance)
        tensordict.set(self.out_key, out)
        return tensordict

    def _update_esc(self, tensordict: TensorDictBase) -> None:
        """ Performs the ESC update step to learn the optimal target diversity (k_hat). """
        # This function is identical to the one in HetControlMlpEsc
        reward_key = ("next", self.agent_group, "reward")
        if reward_key not in tensordict.keys(include_nested=True):
            return
        with torch.no_grad():
            reward = tensordict.get(reward_key)
            if reward.abs().sum() < 1e-6:
                return
            
            esc_updates = self._calculate_esc_update(reward, self.s_reward, self.last_dither)
            self.s_reward.copy_(esc_updates["s_reward_new"])
            self.k_hat.add_(esc_updates["k_hat_update"])
            
            log_data = {
                "reward_raw": reward, "J_mean": esc_updates["J_mean"], "s_reward": self.s_reward,
                "grad_estimate": esc_updates["gradient_estimate"],
                "k_hat_update": esc_updates["k_hat_update"], "k_hat": self.k_hat,
            }
            self._log_to_tensordict(tensordict, "esc_learning", log_data)
    
    def _calculate_esc_update(self, reward: torch.Tensor, s_reward_old: torch.Tensor, last_dither: torch.Tensor) -> Dict[str, torch.Tensor]:
        # This function is identical to the one in HetControlMlpEsc
        j_mean = reward.mean()
        s_reward_new = s_reward_old * (1 - self.esc_tau) + j_mean * self.esc_tau
        gradient_estimate = s_reward_new * last_dither
        k_hat_update = self.esc_gain * gradient_estimate
        return {
            "J_mean": j_mean, "s_reward_new": s_reward_new,
            "gradient_estimate": gradient_estimate, "k_hat_update": k_hat_update,
        }

    def _get_dither_signal(self) -> torch.Tensor:
        """ Generates the sinusoidal dither signal for ESC. """
        with torch.no_grad():
            self.time_step.add_(1)
            t = self.time_step.to(dtype=torch.float32)
            current_dither = self.esc_amplitude * torch.sin(self.esc_frequency * t)
            self.last_dither.copy_(current_dither)
            return current_dither

    def estimate_snd(self, obs: torch.Tensor) -> torch.Tensor:
        """ Estimates the current behavioral diversity (SND). """
        # This function is taken from HetControlMlpEmpirical
        with torch.no_grad():
            agent_actions = []
            for agent_net in self.agent_mlps.agent_networks:
                agent_outputs = agent_net(obs)
                agent_actions.append(agent_outputs)

            distance = compute_behavioral_distance(agent_actions=agent_actions, just_mean=True).mean().unsqueeze(-1)
        
        # Soft update of the estimated SND
        if self.estimated_snd.isnan().any():
            self.estimated_snd.copy_(distance)
            return distance
        else:
            smoothed_distance = (1 - self.snd_tau) * self.estimated_snd + self.snd_tau * distance
            self.estimated_snd.copy_(smoothed_distance)
            return smoothed_distance

    # Helper functions for logging, processing, and device management remain largely the same
    def _log_to_tensordict(self, td: TensorDictBase, namespace: str, data: Dict[str, torch.Tensor]):
        agent_group_shape = td.get_item_shape(self.agent_group)
        for key, tensor in data.items():
            tensor_to_log = tensor.expand(agent_group_shape) if tensor.numel() == 1 and len(agent_group_shape) > 0 else tensor
            td.set((self.agent_group, namespace, key), tensor_to_log)

    def _log_outputs(self, td: TensorDictBase, out: torch.Tensor, scaling_ratio: torch.Tensor, dither: torch.Tensor, distance: torch.Tensor):
        loc_to_normalize = out.chunk(2, dim=-1)[0] if self.probabilistic else out
        out_loc_norm = overflowing_logits_norm(loc_to_normalize, self.action_spec[self.agent_group, "action"])
        log_data = {
            "logits": out, "scaling_ratio": scaling_ratio, "dither": dither,
            "out_loc_norm": out_loc_norm, "measured_snd": distance, "target_snd": self.k_hat
        }
        self._log_to_tensordict(td, "output", log_data)

    def process_shared_out(self, logits: torch.Tensor) -> torch.Tensor:
        if not self.probabilistic and self.process_shared:
            return squash(logits, action_spec=self.action_spec[self.agent_group, "action"], clamp=False)
        elif self.probabilistic:
            loc, scale = self.scale_extractor(logits)
            if self.process_shared:
                loc = squash(loc, action_spec=self.action_spec[self.agent_group, "action"], clamp=False)
            return torch.cat([loc, scale], dim=-1)
        return logits

    def _ensure_device(self, target: torch.device) -> None:
        if self._current_device != target:
            self.to(target)
            self._current_device = target

# --- The New Config Class ---
@dataclass
class HetControlMlpEscSndConfig(ModelConfig):
    activation_class: Type[nn.Module] = MISSING
    num_cells: Sequence[int] = MISSING
    # ESC params
    esc_gain: float = MISSING
    esc_amplitude: float = MISSING
    esc_frequency: float = MISSING
    initial_k: float = MISSING       # Initial target diversity
    tau: float = MISSING             # Reward smoothing tau
    # SND params
    snd_tau: float = MISSING         # SND measurement smoothing tau
    # General params
    process_shared: bool = MISSING
    probabilistic: Optional[bool] = MISSING
    scale_mapping: Optional[str] = MISSING

    @staticmethod
    def associated_class():
        return HetControlMlpEscSnd