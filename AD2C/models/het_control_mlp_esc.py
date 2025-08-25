from __future__ import annotations

from dataclasses import dataclass, MISSING
from typing import Type, Sequence, Optional

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
        """Policy model with Extremum Seeking Control (ESC) for diversity.

        Args:
            activation_class (Type[nn.Module]): activation class to be used.
            num_cells (int or Sequence[int]): number of cells of every layer.
            probabilistic (bool): Whether the model has stochastic actions or not.
            scale_mapping (str, optional): Type of mapping for the std_dev.
            process_shared (bool): Whether to squash the shared policy component.
            esc_gain (float): The learning rate/gain (K) for the ESC integrator.
            esc_amplitude (float): The amplitude (a) of the dither signal.
            esc_frequency (float): The frequency (ω) of the dither signal.
            initial_k (float): The initial value for the diversity scaling factor.
        """
        super().__init__(**kwargs)

        self.num_cells = num_cells
        self.activation_class = activation_class
        self.probabilistic = probabilistic
        self.scale_mapping = scale_mapping
        self.process_shared = process_shared
        self.esc_gain = esc_gain
        self.esc_amplitude = esc_amplitude
        self.esc_frequency = esc_frequency

        # --- ESC State Buffers ---
        self.register_buffer(
            "k_hat",
            torch.tensor([initial_k], device=self.device, dtype=torch.float),
        )  # Best estimate of the optimal scaling factor k
        self.register_buffer(
            "time_step",
            torch.tensor(0, device=self.device, dtype=torch.int),
        ) # Time step for generating dither signal
        self.register_buffer(
            "last_dither",
            torch.tensor([0.0], device=self.device, dtype=torch.float),
        ) # Stores the last dither signal for demodulation

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

        agent_outputs = (
            self.output_features // 2 if self.probabilistic else self.output_features
        )
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
            raise ValueError(
                "If the MLP input has the agent dimension,"
                " the second to last spec dimension should be the number of agents"
            )
        if (
            self.output_has_agent_dim
            and self.output_leaf_spec.shape[-2] != self.n_agents
        ):
            raise ValueError(
                "If the MLP output has the agent dimension,"
                " the second to last spec dimension should be the number of agents"
            )

    def _forward(self, tensordict: TensorDictBase, **kwargs) -> TensorDictBase:

        # --- 1. Perform ESC Update if Reward is Available ---
        reward_key = ("next", "reward")
        if reward_key in tensordict.keys(include_nested=True) and torch.is_grad_enabled():
            with torch.no_grad():
                reward = tensordict.get(reward_key)
                J = reward.mean()
                gradient_estimate = J * self.last_dither
                k_hat_update = self.esc_gain * gradient_estimate

                # --- PROPER LOGGING ---
                # Log the internal state of the ESC controller
                tensordict.set((self.agent_group, "esc_reward_J"), J.expand(tensordict.get_item_shape(self.agent_group)))
                tensordict.set((self.agent_group, "esc_grad_estimate"), gradient_estimate.expand(tensordict.get_item_shape(self.agent_group)))
                tensordict.set((self.agent_group, "esc_k_hat_update"), k_hat_update.expand(tensordict.get_item_shape(self.agent_group)))
                # --- END LOGGING ---

                self.k_hat.add_(k_hat_update)


        # --- 2. Generate New Scaling Ratio from ESC ---
        with torch.no_grad():
            self.time_step += 1
            current_dither = self.esc_amplitude * torch.sin(
                self.esc_frequency * self.time_step.float()
            )
            scaling_ratio = self.k_hat + current_dither
            self.last_dither[:] = current_dither


        # --- 3. Standard Policy Forward Pass ---
        input_obs = tensordict.get(self.in_key)
        shared_out = self.shared_mlp.forward(input_obs)
        agent_out = self.agent_mlps.forward(input_obs)
        shared_out = self.process_shared_out(shared_out)

        # --- 4. Apply ESC-derived Scaling Ratio ---
        if self.probabilistic:
            shared_loc, shared_scale = shared_out.chunk(2, -1)
            agent_loc = shared_loc + agent_out * scaling_ratio
            agent_scale = shared_scale
            out = torch.cat([agent_loc, agent_scale], dim=-1)
        else:
            out = shared_out + agent_out * scaling_ratio
        
        out_loc_norm = overflowing_logits_norm(
            out.chunk(2,-1)[0] if self.probabilistic else out,
            self.action_spec[self.agent_group, "action"],
        )

        # --- 5. Populate TensorDict with outputs and logs ---
        tensordict.set(
            (self.agent_group, "k_hat"),
            self.k_hat.expand(tensordict.get_item_shape(self.agent_group)),
        )
        tensordict.set(
            (self.agent_group, "scaling_ratio"),
            scaling_ratio.expand_as(out),
        )
        # --- PROPER LOGGING ---
        # Log the dither signal
        tensordict.set(
            (self.agent_group, "esc_dither"),
            current_dither.expand(tensordict.get_item_shape(self.agent_group))
        )
        # --- END LOGGING ---
        tensordict.set((self.agent_group, "logits"), out)
        tensordict.set((self.agent_group, "out_loc_norm"), out_loc_norm)
        tensordict.set(self.out_key, out)

        return tensordict

    def process_shared_out(self, logits: torch.Tensor):
        if not self.probabilistic and self.process_shared:
            return squash(
                logits,
                action_spec=self.action_spec[self.agent_group, "action"],
                clamp=False,
            )
        elif self.probabilistic:
            loc, scale = self.scale_extractor(logits)
            if self.process_shared:
                loc = squash(
                    loc,
                    action_spec=self.action_spec[self.agent_group, "action"],
                    clamp=False,
                )
            return torch.cat([loc, scale], dim=-1)
        else:
            return logits

@dataclass
class HetControlMlpEscConfig(ModelConfig):
    """Config for the new ESC model."""
    activation_class: Type[nn.Module] = MISSING
    num_cells: Sequence[int] = MISSING

    # ESC parameters
    esc_gain: float = MISSING
    esc_amplitude: float = MISSING
    esc_frequency: float = MISSING
    initial_k: float = MISSING

    # Standard parameters
    probabilistic: Optional[bool] = MISSING
    scale_mapping: Optional[str] = MISSING
    process_shared: bool = MISSING


    @staticmethod
    def associated_class():
        return HetControlMlpEsc