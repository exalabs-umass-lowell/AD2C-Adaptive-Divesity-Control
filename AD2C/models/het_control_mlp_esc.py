from __future__ import annotations
from dataclasses import dataclass, MISSING
from typing import Optional, Sequence, Tuple, Type

import torch
from tensordict import TensorDictBase
from tensordict.nn import NormalParamExtractor
from torch import nn
from torchrl.modules import MultiAgentMLP

from benchmarl.models.common import Model, ModelConfig
from AD2C.utils import overflowing_logits_norm
from .utils import squash


class HetControlMlpEsc(Model):
    def __init__(
        self,
        activation_class: Type[nn.Module],
        num_cells: Sequence[int],
        esc_gain: float,
        esc_amplitude: float,
        esc_frequency: float,
        initial_k: float,
        tau: float,
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

        # ESC hyperparameters
        self.esc_gain = float(esc_gain)
        self.esc_amplitude = float(esc_amplitude)
        self.esc_frequency = float(esc_frequency)
        self.tau = float(tau)

        # --- ESC state (buffers created on self.device) ---
        self.register_buffer("k_hat", torch.tensor([initial_k], device=self.device, dtype=torch.float))
        self.register_buffer("time_step", torch.tensor(0, device=self.device, dtype=torch.long))
        self.register_buffer("last_dither", torch.tensor([0.0], device=self.device, dtype=torch.float))
        self.register_buffer("s_reward", torch.tensor(0.0, device=self.device, dtype=torch.float))  # EMA(J)

        self.scale_extractor = (
            NormalParamExtractor(scale_mapping=scale_mapping) if scale_mapping is not None else None
        )

        self.input_features = self.input_leaf_spec.shape[-1]
        self.output_features = self.output_leaf_spec.shape[-1]

        # Build submodules DIRECTLY on model device
        self.shared_mlp = MultiAgentMLP(
            n_agent_inputs=self.input_features,
            n_agent_outputs=self.output_features,
            n_agents=self.n_agents,
            centralised=False,
            share_params=True,
            activation_class=self.activation_class,
            num_cells=self.num_cells,
            device=self.device,  # <<< important
        )
        agent_outputs = self.output_features // 2 if self.probabilistic else self.output_features
        self.agent_mlps = MultiAgentMLP(
            n_agent_inputs=self.input_features,
            n_agent_outputs=agent_outputs,
            n_agents=self.n_agents,
            centralised=False,
            share_params=False,
            activation_class=self.activation_class,
            num_cells=self.num_cells,
            device=self.device,  # <<< important
        )

        # track where the module actually lives (for lazy moves)
        self._current_device = torch.device(self.device)

    # --------- optional safety: lazy device move when seeing first input ----------
    def _ensure_device(self, target: torch.device) -> None:
        if self._current_device != target:
            super().to(target)  # move whole module (submodules + buffers)
            self._current_device = target

    # ------------------------ Checks ------------------------
    def _perform_checks(self):
        super()._perform_checks()
        if self.centralised or not self.input_has_agent_dim:
            raise ValueError(f"{self.__class__.__name__} can only be used for policies")
        if self.input_has_agent_dim and self.input_leaf_spec.shape[-2] != self.n_agents:
            raise ValueError("If the MLP input has the agent dimension, the second to last spec dim must be n_agents")
        if self.output_has_agent_dim and self.output_leaf_spec.shape[-2] != self.n_agents:
            raise ValueError("If the MLP output has the agent dimension, the second to last spec dim must be n_agents")

    # ------------------------ ESC core ------------------------
    def _update_esc_controller(self, tensordict: TensorDictBase) -> None:
        reward_key = ("next", self.agent_group, "reward")
        
        # ========== THIS IS THE CORRECTED LINE ==========
        if reward_key not in tensordict.keys(include_nested=True):
            return
        # ================================================

        with torch.no_grad():
            reward = tensordict.get(reward_key)
            
            J = reward.mean()

            print(J)

            # EMA(J)
            self.s_reward.mul_(1 - self.tau).add_(J * self.tau)

            # gradient estimate and update
            gradient_estimate = self.s_reward * self.last_dither
            k_hat_update = self.esc_gain * gradient_estimate
            self.k_hat.add_(k_hat_update)

            # expand to agent-group batch shape for logging
            agent_group_shape = tensordict.get_item_shape(self.agent_group)
            rdev, rdtype = reward.device, reward.dtype


            # print(f'J : {J}; estinated Gradeant: {gradient_estimate}; K_hat Update : {k_hat_update}')
            tensordict.set((self.agent_group, "esc_gain"),
                           torch.as_tensor(self.esc_gain, device=rdev, dtype=rdtype).expand(agent_group_shape))
            tensordict.set((self.agent_group, "esc_reward_J"),
                           torch.as_tensor(self.s_reward, device=rdev, dtype=rdtype).expand(agent_group_shape))
            tensordict.set((self.agent_group, "esc_J_raw"),
                           torch.as_tensor(J, device=rdev, dtype=rdtype).expand(agent_group_shape))
            tensordict.set((self.agent_group, "esc_grad_estimate"),
                           torch.as_tensor(gradient_estimate, device=rdev, dtype=rdtype).expand(agent_group_shape))
            tensordict.set((self.agent_group, "esc_k_hat_update"),
                           torch.as_tensor(k_hat_update, device=rdev, dtype=rdtype).expand(agent_group_shape))
            tensordict.set((self.agent_group, "k_hat_new"),
                           self.k_hat.to(device=rdev, dtype=rdtype).expand(agent_group_shape))

    def _get_esc_scaling_ratio(self) -> Tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            self.time_step.add_(1)
            t = self.time_step.to(dtype=torch.float32)
            current_dither = self.esc_amplitude * torch.sin(self.esc_frequency * t)
            scaling_ratio = self.k_hat + current_dither
            self.last_dither.copy_(current_dither)
        return scaling_ratio, current_dither

    # ------------------------ Forward ------------------------
    def _forward(self, tensordict: TensorDictBase, **kwargs) -> TensorDictBase:
        agent_index = kwargs.get("agent_index", None)

        # Observations
        input = tensordict.get(self.in_key)  # [*batch, n_agents, n_features]

        # Ensure the whole module is on the same device as inputs (fixes CPU/CUDA mismatch)
        self._ensure_device(input.device)

        # Shared and agent outputs
        shared_out = self.shared_mlp(input)
        agent_out = (
            self.agent_mlps(input) if agent_index is None
            else self.agent_mlps.agent_networks[agent_index](input)
        )

        # Optional squashing / param extraction
        shared_out = self.process_shared_out(shared_out)

        # ESC update + control signal
        self._update_esc_controller(tensordict)
        scaling_ratio, current_dither = self._get_esc_scaling_ratio()

        # Combine shared + hetero heads using ESC scaling
        if self.probabilistic:
            shared_loc, shared_scale = shared_out.chunk(2, -1)
            agent_loc = shared_loc + agent_out * scaling_ratio
            out_loc_norm = overflowing_logits_norm(agent_loc, self.action_spec[self.agent_group, "action"])
            out = torch.cat([agent_loc, shared_scale], dim=-1)
        else:
            out = shared_out + scaling_ratio * agent_out
            out_loc_norm = overflowing_logits_norm(out, self.action_spec[self.agent_group, "action"])

        # Logging
        tensordict.set((self.agent_group, "k_hat"),
                       self.k_hat.expand(tensordict.get_item_shape(self.agent_group)))
        tensordict.set((self.agent_group, "esc_dither"),
                       current_dither.expand(tensordict.get_item_shape(self.agent_group)))
        tensordict.set((self.agent_group, "scaling_ratio"), scaling_ratio.expand_as(out))
        tensordict.set((self.agent_group, "logits"), out)
        tensordict.set((self.agent_group, "out_loc_norm"), out_loc_norm)

        tensordict.set(self.out_key, out)
        return tensordict

    def process_shared_out(self, logits: torch.Tensor):
        if not self.probabilistic and self.process_shared:
            return squash(logits, action_spec=self.action_spec[self.agent_group, "action"], clamp=False)
        elif self.probabilistic:
            loc, scale = self.scale_extractor(logits)  # type: ignore[arg-type]
            if self.process_shared:
                loc = squash(loc, action_spec=self.action_spec[self.agent_group, "action"], clamp=False)
            return torch.cat([loc, scale], dim=-1)
        else:
            return logits


@dataclass
class HetControlMlpEscConfig(ModelConfig):
    activation_class: Type[nn.Module] = MISSING
    num_cells: Sequence[int] = MISSING
    esc_gain: float = MISSING
    esc_amplitude: float = MISSING
    esc_frequency: float = MISSING
    initial_k: float = MISSING
    tau: float = MISSING
    process_shared: bool = MISSING
    probabilistic: Optional[bool] = MISSING
    scale_mapping: Optional[str] = MISSING

    @staticmethod
    def associated_class():
        return HetControlMlpEsc