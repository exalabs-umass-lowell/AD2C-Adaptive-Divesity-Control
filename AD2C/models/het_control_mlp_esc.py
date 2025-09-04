from __future__ import annotations

from dataclasses import dataclass, MISSING
from typing import Type, Sequence, Optional, Tuple

import torch
from tensordict import TensorDictBase
from tensordict.nn import NormalParamExtractor
from torch import nn
from torchrl.modules import MultiAgentMLP

from benchmarl.models.common import Model, ModelConfig
from AD2C.snd import compute_behavioral_distance
from AD2C.utils import overflowing_logits_norm
from .utils import squash


class HetControlMlpEsc(Model):
    def __init__(
        self,
        activation_class: Type[nn.Module],
        num_cells: Sequence[int],
        probabilistic: bool,
        scale_mapping: Optional[str],
        process_shared: bool,
        esc_gain: float,
        esc_amplitude: float,
        esc_frequency: float,
        initial_k: float,
        **kwargs,
    ):
        """Policy model with Extremum Seeking Control (ESC) for diversity."""
        env_spec = kwargs["env_spec"]
        
        super().__init__(**kwargs)

        self.num_cells = num_cells
        self.activation_class = activation_class
        self.probabilistic = probabilistic
        self.scale_mapping = scale_mapping
        self.process_shared = process_shared
        self.esc_gain = esc_gain
        self.esc_amplitude = esc_amplitude
        self.esc_frequency = esc_frequency
        self.tau = 0.005
        
        # --- ESC State Buffers ---
        self.register_buffer("k_hat", torch.tensor([initial_k], device=self.device, dtype=torch.float))
        self.register_buffer("time_step", torch.tensor(0, device=self.device, dtype=torch.int))
        self.register_buffer("last_dither", torch.tensor([0.0], device=self.device, dtype=torch.float))
        
        reward_shape = env_spec[self.agent_group, "reward"].shape
        self.register_buffer("ema_reward", torch.zeros(*reward_shape, device=self.device))
                
        self.scale_extractor = (
            NormalParamExtractor(scale_mapping=scale_mapping)
            if scale_mapping is not None
            else None
        )

        self.input_features = self.input_leaf_spec.shape[-1]
        self.output_features = self.output_leaf_spec.shape[-1]

        self.shared_mlp = MultiAgentMLP(
            n_agent_inputs=self.input_features,
            n_agent_outputs=self.output_features,
            n_agents=self.n_agents,
            centralised=False,
            share_params=True,
            device=self.device,
            activation_class=self.activation_class,
            num_cells=self.num_cells,
        )

        agent_outputs = self.output_features // 2 if self.probabilistic else self.output_features
        self.agent_mlps = MultiAgentMLP(
            n_agent_inputs=self.input_features,
            n_agent_outputs=agent_outputs,
            n_agents=self.n_agents,
            centralised=False,
            share_params=False,
            device=self.device,
            activation_class=self.activation_class,
            num_cells=self.num_cells,
        )

    def _perform_checks(self):
        super()._perform_checks()
        if self.centralised or not self.input_has_agent_dim:
            raise ValueError(f"{self.__class__.__name__} can only be used for policies")
        if self.input_has_agent_dim and self.input_leaf_spec.shape[-2] != self.n_agents:
            raise ValueError("Input agent dimension mismatch")
        if self.output_has_agent_dim and self.output_leaf_spec.shape[-2] != self.n_agents:
            raise ValueError("Output agent dimension mismatch")

    def _update_esc_controller(self, tensordict: TensorDictBase) -> None:
        """
        Updates the ESC controller's k_hat based on the reward signal.
        This method is only called during training.
        """
        reward_key = ("next", "agents", "reward")
        # This check ensures we only update during the training phase
        if reward_key not in tensordict.keys(include_nested=True) or not torch.is_grad_enabled():
            return

        with torch.no_grad():
            # Get the reward tensor and its shape, which will be our reference
            reward = tensordict.get(reward_key)
            target_shape = reward.shape

            # J is the objective function, here approximated by the mean reward per agent
            J_per_agent = reward.mean(dim=0)
            
            # Expand the agent-wise average reward across the batch dimension
            # This ensures its shape is compatible for element-wise operations
            self.ema_reward.mul_(1 - self.tau).add_(J_per_agent * self.tau)
            J_expanded = self.ema_reward.unsqueeze(0).expand(target_shape)        
            # J_expanded = J_per_agent.unsqueeze(0).expand(target_shape)
            
            # Calculate gradient estimate and update k_hat
            # The gradient is estimated by multiplying the objective by the dither signal
            gradient_estimate = J_expanded * self.last_dither
            k_hat_update = self.esc_gain * gradient_estimate
            
            # Update our estimate of the optimal k by taking a step along the gradient
            self.k_hat.add_(k_hat_update.mean())

            # Log ESC-related metrics to the tensordict for monitoring
            # All logged tensors are expanded to the same `target_shape` for consistency
            logs = {
                (self.agent_group, "esc_reward_J"): J_expanded,
                (self.agent_group, "esc_grad_estimate"): gradient_estimate,
                (self.agent_group, "esc_k_hat_update"): k_hat_update,
                (self.agent_group, "esc_gain"): torch.tensor(
                    self.esc_gain, device=self.device
                ).expand(target_shape),
                (self.agent_group, "k_hat_new"): self.k_hat.expand(target_shape),
            }
            tensordict.update(logs)

    def _get_esc_scaling_ratio(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generates the oscillating dither signal and computes the final scaling ratio.
        """
        with torch.no_grad():
            self.time_step += 1
            current_dither = self.esc_amplitude * torch.sin(self.esc_frequency * self.time_step.float())
            scaling_ratio = self.k_hat + current_dither
            self.last_dither.copy_(current_dither)
        return scaling_ratio, current_dither

    def _get_policy_output(
        self, input_obs: torch.Tensor, scaling_ratio: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Performs the forward pass of the policy networks (shared and agent-specific)
        and combines their outputs using the provided scaling ratio.
        """
        shared_out = self.process_shared_out(self.shared_mlp(input_obs))
        agent_out = self.agent_mlps(input_obs)

        if self.probabilistic:
            shared_loc, shared_scale = shared_out.chunk(2, -1)
            agent_loc = shared_loc + agent_out * scaling_ratio
            agent_scale = shared_scale
            out = torch.cat([agent_loc, agent_scale], dim=-1)
            out_loc_norm = overflowing_logits_norm(agent_loc, self.action_spec[self.agent_group, "action"])
        else:
            out = shared_out + agent_out * scaling_ratio
            out_loc_norm = overflowing_logits_norm(out, self.action_spec[self.agent_group, "action"])
        
        return out, out_loc_norm

    def _forward(self, tensordict: TensorDictBase, **kwargs) -> TensorDictBase:
        """
        Orchestrates the model's forward pass.
        1. Updates the ESC controller if in a training step.
        2. Gets the latest scaling ratio from the ESC controller.
        3. Computes the final policy output (action logits).
        4. Logs all relevant data to the tensordict.
        """
        # 1. Update ESC controller state (only runs during training)
        self._update_esc_controller(tensordict)

        # 2. Get the control signal from the ESC
        scaling_ratio, current_dither = self._get_esc_scaling_ratio()
        
        # 3. Get the final policy output from the networks
        input_obs = tensordict.get(self.in_key)
        out, out_loc_norm = self._get_policy_output(input_obs, scaling_ratio)
        
        # 4. Log all data and set the final output
        shape_out = out.shape
        agent_group_shape = tensordict.get_item_shape(self.agent_group)
        logs = {
            (self.agent_group, "k_hat"): self.k_hat.expand(agent_group_shape),
            (self.agent_group, "scaling_ratio"): scaling_ratio.expand(shape_out),
            (self.agent_group, "esc_dither"): current_dither.expand(agent_group_shape),
            (self.agent_group, "logits"): out,
            (self.agent_group, "out_loc_norm"): out_loc_norm,
            self.out_key: out,
        }
        tensordict.update(logs)

        return tensordict

    def process_shared_out(self, logits: torch.Tensor):
        if not self.probabilistic and self.process_shared:
            return squash(logits, action_spec=self.action_spec[self.agent_group, "action"], clamp=False)
        elif self.probabilistic:
            loc, scale = self.scale_extractor(logits)
            if self.process_shared:
                loc = squash(loc, action_spec=self.action_spec[self.agent_group, "action"], clamp=False)
            return torch.cat([loc, scale], dim=-1)
        else:
            return logits

@dataclass
class HetControlMlpEscConfig(ModelConfig):
    """Config for the new ESC model."""
    activation_class: Type[nn.Module] = MISSING
    num_cells: Sequence[int] = MISSING
    esc_gain: float = MISSING
    esc_amplitude: float = MISSING
    esc_frequency: float = MISSING
    initial_k: float = MISSING
    probabilistic: Optional[bool] = MISSING
    scale_mapping: Optional[str] = MISSING
    process_shared: bool = MISSING

    @staticmethod
    def associated_class():
        return HetControlMlpEsc