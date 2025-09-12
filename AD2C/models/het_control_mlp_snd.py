from __future__ import annotations
from dataclasses import dataclass, MISSING
from typing import Optional, Sequence, Tuple, Type, Dict

import torch
from tensordict import TensorDictBase
from tensordict.nn import NormalParamExtractor
from torch import nn
from torchrl.modules import MultiAgentMLP

from .utils import HighPassFilter, LowPassFilter

from benchmarl.models.common import Model, ModelConfig
# Make sure these imports are correct for your project structure
from AD2C.snd import compute_behavioral_distance
from AD2C.utils import overflowing_logits_norm
from .utils import squash

import numpy as np


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

        # esc_gain: float,
        # esc_amplitude: float,
        # esc_frequency: float,
        
        initial_k: float,           # This is now the initial target diversity
        # tau: float,                 # This is the ESC reward smoothing tau
        snd_tau: float,             # This is the SND measurement smoothing tau
        probabilistic: bool,
        
        bootstrap_from_desired_snd: bool,
        scale_mapping: Optional[str],
        
        sampling_period: float,
        disturbance_frequency: float,
        disturbance_magnitude: float,
        integrator_gain: float,
        high_pass_cutoff_frequency: float,
        low_pass_cutoff_frequency: float,
        use_adapter: bool,

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
        # self.esc_gain = float(esc_gain)
        # self.esc_amplitude = float(esc_amplitude)
        # self.esc_frequency = float(esc_frequency)
        
        self.dt = sampling_period
        self.disturbance_frequency = disturbance_frequency
        self.disturbance_magnitude = disturbance_magnitude
        self.integrator_gain = integrator_gain # Should be positive for reward maximization
        self.use_adapter = use_adapter

        self.k_hat_initial = initial_k # Store the initial value for the integrator
        self.register_buffer("integral", torch.tensor(0.0, device=self.device, dtype=torch.float))
        self.register_buffer("wt", torch.tensor(0.0, device=self.device, dtype=torch.float))


        self.high_pass_filter = HighPassFilter(sampling_period, high_pass_cutoff_frequency)
        self.low_pass_filter = LowPassFilter(sampling_period, low_pass_cutoff_frequency)
        
        self.register_buffer("m2", torch.tensor(0.0, device=self.device, dtype=torch.float))
        self.b2 = 0.9 # As in the paper's code
        self.epsilon = 1e-8

        # SND parameter for smoothing the diversity measurement
        self.snd_tau = snd_tau
        # self.tau = tau

        self.bootstrap_from_desired_snd = bootstrap_from_desired_snd

        self._setup_esc_and_snd_state(self.k_hat_initial, self.snd_tau)
        self._setup_networks()
        self.scale_extractor = (
            NormalParamExtractor(scale_mapping=scale_mapping) if scale_mapping is not None else None
        )
        self._current_device = torch.device(self.device)

    def _setup_esc_and_snd_state(self, initial_k: float, esc_tau: float):
        # State for learning the target diversity (k_hat) via ESC
        # self.esc_tau = float(esc_tau)
        self.register_buffer("k_hat", torch.tensor([initial_k], device=self.device, dtype=torch.float))
        self.register_buffer("time_step", torch.tensor(0, device=self.device, dtype=torch.long))
        self.register_buffer("last_dither", torch.tensor([0.0], device=self.device, dtype=torch.float))
        self.register_buffer("s_reward", torch.tensor(0.0, device=self.device, dtype=torch.float))
        
        # State for measuring diversity (SND)
        self.register_buffer(
            "estimated_snd", torch.tensor([float("nan")], device=self.device, dtype=torch.float)
        )
        self.register_buffer(
            name="desired_snd",
            tensor=torch.tensor([initial_k], device=self.device, dtype=torch.float),
        )  # Buffer for SND_{des}


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

    # def _forward(self, 
    #              tensordict: TensorDictBase,
    #                **kwargs) -> TensorDictBase:
    #     input_tensor = tensordict.get(self.in_key)
    #     self._ensure_device(input_tensor.device)

    #     # 1. Get the dither signal and target diversity for ESC
    #     current_dither = self._get_dither_signal()
    #     dithered_target_diversity = self.k_hat + current_dither

    #     # 2. Measure the raw behavioral diversity for the current step
    #     raw_distance = self.estimate_snd(input_tensor)

    #     # 3. Calculate the smoothed diversity using the previous estimate and the current raw value
    #     if self.estimated_snd.isnan().any():
    #         current_smoothed_distance = raw_distance
    #     else:
    #         current_smoothed_distance = (1 - self.snd_tau) * self.estimated_snd + self.snd_tau * raw_distance
        
    #     # 4. Calculate the scaling ratio using this newly computed smoothed diversity
    #     #    (This correctly mimics the HetControlMlpEmpirical logic)
    #     scaling_ratio = dithered_target_diversity / current_smoothed_distance.clamp(min=1e-6)

    #     # 5. Update the stored diversity estimate for the *next* forward pass
    #     with torch.no_grad():
    #         self.estimated_snd.copy_(current_smoothed_distance)

    #     # 6. Get network outputs
    #     shared_out = self.shared_mlp(input_tensor)
    #     agent_out = self.agent_mlps(input_tensor) # Use the raw agent output

    #     shared_out = self.process_shared_out(shared_out)

    #     # 7. Apply scaling directly to the raw agent output (NO normalization)
    #     if self.probabilistic:
    #         shared_loc, shared_scale = shared_out.chunk(2, -1)
    #         agent_loc = shared_loc + agent_out * scaling_ratio
    #         out = torch.cat([agent_loc, shared_scale], dim=-1)
    #     else:
    #         out = shared_out + agent_out * scaling_ratio
        
    #     # 8. Log outputs and update the tensordict
    #     self._log_outputs(tensordict, out, scaling_ratio, current_dither, raw_distance)
    #     tensordict.set(self.out_key, out)
    #     return tensordict

    
    def _forward(self, 
                tensordict: TensorDictBase,
                agent_index: int = None,
                update_estimate: bool = True,
                compute_estimate: bool = True,
                **kwargs
    ) -> TensorDictBase:
        

        input = tensordict.get(self.in_key)
        # self._ensure_device(input.device)

        # --------------------
        
        # # 1. Get the dither signal and target diversity for ESC
        # current_dither = self._get_dither_signal()
        # dithered_target_diversity = self.k_hat + current_dither

        current_dither = self.disturbance_magnitude * torch.sin(self.wt)
        target_diversity = self.k_hat + current_dither

        # # 2. Measure the raw behavioral diversity for the current step
        # raw_distance = self.estimate_snd(input)

        # # 3. Calculate the smoothed diversity using the previous estimate and the current raw value
        # if self.estimated_snd.isnan().any():
        #     current_smoothed_distance = raw_distance
        # else:
        #     current_smoothed_distance = (1 - self.snd_tau) * self.estimated_snd + self.snd_tau * raw_distance

        # # 5. Update the stored diversity estimate for the *next* forward pass
        # with torch.no_grad():
        #     self.estimated_snd.copy_(current_smoothed_distance)

        # ------------ ES Updates --------------------                                          

        shared_out = self.shared_mlp.forward(input)
        if agent_index is None:  # Gather outputs for all agents on the obs
            # tensor of shape [*batch, n_agents, n_actions], where the outputs
            # along the n_agent dimension are taken with the different agent networks
            agent_out = self.agent_mlps.forward(input)
        else:  # Gather outputs for one agent on the obs
            # tensor of shape [*batch, n_agents, n_actions], where the outputs
            # along the n_agent dimension are taken with the same (agent_index) agent network
            agent_out = self.agent_mlps.agent_networks[agent_index].forward(input)


        shared_out = self.process_shared_out(shared_out)

        if (
            self.k_hat > 0
            and torch.is_grad_enabled()  # we are training
            and compute_estimate
            and self.n_agents > 1
        ):
            # Update \widehat{SND}
            distance = self.estimate_snd(input)
            if update_estimate:
                self.estimated_snd[:] = distance.detach()
        else:
            distance = self.estimated_snd
        if self.k_hat == 0 or self.k_hat < -1:
            scaling_ratio = 0.0
        elif (
            self.k_hat == -1   # Unconstrained networks
            or distance.isnan().any()  # It is the first iteration
            or self.n_agents == 1
        ):
            scaling_ratio = 1.0
        else:  # DiCo scaling
            scaling_ratio = torch.where(
                distance != target_diversity,
                # self.k_hat / distance,
                target_diversity / distance,
                1,
            )


        if self.probabilistic:
            shared_loc, shared_scale = shared_out.chunk(2, -1)

            # DiCo scaling
            agent_loc = shared_loc + agent_out * scaling_ratio
            out_loc_norm = overflowing_logits_norm(
                agent_loc, self.action_spec[self.agent_group, "action"]
            )  # For logging
            agent_scale = shared_scale

            out = torch.cat([agent_loc, agent_scale], dim=-1)
        else:
            # DiCo scaling
            out = shared_out + scaling_ratio * agent_out
            out_loc_norm = overflowing_logits_norm(
                out, self.action_spec[self.agent_group, "action"]
            )  # For logging
        
        # 8. Log outputs and update the tensordict
        tensordict.set(
            (self.agent_group, "k_hat"),
            self.estimated_snd.expand(tensordict.get_item_shape(self.agent_group)),
        )
        tensordict.set(
            (self.agent_group, "estimated_snd"),
            self.estimated_snd.expand(tensordict.get_item_shape(self.agent_group)),
        )
        tensordict.set(
            (self.agent_group, "scaling_ratio"),
            (
                torch.tensor(scaling_ratio, device=self.device).expand_as(out)
                if not isinstance(scaling_ratio, torch.Tensor)
                else scaling_ratio.expand_as(out)
            ),
        )
        
        tensordict.set(
            (self.agent_group, "current_dither"),
            target_diversity.expand(tensordict.get_item_shape(self.agent_group)),
        )
        tensordict.set(
            (self.agent_group, "target_diversity"),
            target_diversity.expand(tensordict.get_item_shape(self.agent_group)),
        )
        tensordict.set((self.agent_group, "logits"), out)
        tensordict.set((self.agent_group, "out_loc_norm"), out_loc_norm)
        
        # self._log_outputs(tensordict, out, scaling_ratio, current_dither, raw_distance)
        
        tensordict.set(self.out_key, out)
        
        return tensordict

    # Logs Controller Values

    # def _update_esc(self, tensordict: TensorDictBase) -> None:
    #     """ Performs the ESC update step to learn the optimal target diversity (k_hat). """
    #     # This function is identical to the one in HetControlMlpEsc
    #     reward_key = ("next", self.agent_group, "reward")
    #     if reward_key not in tensordict.keys(include_nested=True):
    #         return
    #     with torch.no_grad():
    #         reward = tensordict.get(reward_key)
    #         if reward.abs().sum() < 1e-6:
    #             return
            
    #         esc_updates = self._calculate_esc_update(reward, self.s_reward, self.last_dither)
    #         self.s_reward.copy_(esc_updates["s_reward_new"])
    #         self.k_hat.add_(esc_updates["k_hat_update"])
            
    #         log_data = {
    #             "reward_raw": reward,
    #             "J_mean": esc_updates["J_mean"], 
    #             "s_reward": self.s_reward,
    #             "grad_estimate": esc_updates["gradient_estimate"],
    #             "k_hat_update": esc_updates["k_hat_update"], 
    #             "k_hat": self.k_hat,
    #         }
    #         self._log_to_tensordict(tensordict, "esc_learning", log_data)
    
    # In the HetControlMlpEscSnd class:

    def _update_esc(self, tensordict: TensorDictBase) -> None:
        """
        Performs the full ESC update from the paper to learn the optimal target diversity (k_hat).
        This now includes high-pass/low-pass filtering and the adaptive step size.
        """
        reward_key = ("next", self.agent_group, "reward")
        if reward_key not in tensordict.keys(include_nested=True):
            return

        with torch.no_grad():
            # The 'cost' is the mean reward from the environment
            cost = (tensordict.get(reward_key).mean())


            # 1. High-pass filter the cost to isolate changes
            high_pass_output = self.high_pass_filter(cost)

            # 2. Demodulate to get the raw gradient signal
            demodulation_signal = torch.sin(self.wt)
            low_pass_input = high_pass_output * demodulation_signal

            # 3. Low-pass filter to get a clean gradient estimate
            low_pass_output = self.low_pass_filter(low_pass_input)

            # 4. (Optional) Use the adapter to scale the gradient
            if self.use_adapter:
                self.m2 = self.b2 * self.m2 + (1 - self.b2) * torch.pow(low_pass_output, 2)
                m2_sqrt = torch.sqrt(self.m2)
                if m2_sqrt > 1.0:
                    gradient = low_pass_output / (m2_sqrt + self.epsilon)
                else:
                    gradient = low_pass_output * m2_sqrt
            else:
                gradient = low_pass_output

            # 5. Integrate the gradient to update the search value
            # Note: We use `self.integrator_gain` (should be positive to maximize reward)
            self.integral += self.integrator_gain * gradient * self.dt
            
            # The setpoint is the initial value plus the integral
            setpoint = self.k_hat_initial + self.integral # We need to store initial_k as self.k_hat_initial
            
            # 6. Update k_hat with the new setpoint
            self.k_hat.copy_(setpoint)

            # 7. Update the perturbation phase for the next step
            self.wt += self.disturbance_frequency * self.dt
            if self.wt > 2 * np.pi:
                self.wt -= 2 * np.pi
                
            # --- Logging ---
            log_data = {
                "reward_mean": cost, "hpf_out": high_pass_output, "lpf_out": low_pass_output,
                "gradient_final": gradient, "k_hat": self.k_hat, "integral": self.integral,
                "m2_sqrt": torch.sqrt(self.m2),
            }
            self._log_to_tensordict(tensordict, "esc_learning", log_data)

    # def _calculate_esc_update(self, reward: torch.Tensor, s_reward_old: torch.Tensor, last_dither: torch.Tensor) -> Dict[str, torch.Tensor]:
    #     # This function is identical to the one in HetControlMlpEsc
    #     j_mean = reward.mean()
    #     s_reward_new = s_reward_old * (1 - self.esc_tau) + j_mean * self.esc_tau
    #     gradient_estimate = s_reward_new * last_dither
    #     k_hat_update = self.esc_gain * gradient_estimate
    #     return {
    #         "J_mean": j_mean, "s_reward_new": s_reward_new,
    #         "gradient_estimate": gradient_estimate, "k_hat_update": k_hat_update,
    #     }

    # def _get_dither_signal(self) -> torch.Tensor:
    #     """ Generates the sinusoidal dither signal for ESC. """
    #     with torch.no_grad():
    #         self.time_step.add_(1)
    #         t = self.time_step.to(dtype=torch.float32)
    #         current_dither = self.esc_amplitude * torch.sin(self.esc_frequency * t)
    #         self.last_dither.copy_(current_dither)
    #         return current_dither
    
    # @torch.no_grad()
    def estimate_snd(self, obs: torch.Tensor):
        """
        Update \widehat{SND}
        """
        agent_actions = []
        # Gather what actions each agent would take if given the obs tensor
        for agent_net in self.agent_mlps.agent_networks:
            agent_outputs = agent_net(obs)
            agent_actions.append(agent_outputs)

        distance = (
            compute_behavioral_distance(agent_actions=agent_actions, just_mean=True)
            .mean()
            .unsqueeze(-1)
        )  # Compute the SND if these unscaled policies

        if self.estimated_snd.isnan().any():  # First iteration
            distance = self.desired_snd if self.bootstrap_from_desired_snd else distance
        else:
            # Soft update of \widehat{SND}
            distance = (1 - self.snd_tau) * self.estimated_snd + self.snd_tau * distance

        return distance

    # Helper functions for logging, processing, and device management remain largely the same
    def _log_to_tensordict(self, td: TensorDictBase, namespace: str, data: Dict[str, torch.Tensor]):
        agent_group_shape = td.get_item_shape(self.agent_group)
        for key, tensor in data.items():
            tensor_to_log = tensor.expand(agent_group_shape) if tensor.numel() == 1 and len(agent_group_shape) > 0 else tensor
            td.set((self.agent_group, namespace, key), tensor_to_log)

    # def _log_outputs(self, td: TensorDictBase, out: torch.Tensor, scaling_ratio: torch.Tensor, dither: torch.Tensor, distance: torch.Tensor):
    #     loc_to_normalize = out.chunk(2, dim=-1)[0] if self.probabilistic else out
    #     out_loc_norm = overflowing_logits_norm(loc_to_normalize, self.action_spec[self.agent_group, "action"])
    #     log_data = {
    #         "logits": out, "scaling_ratio": scaling_ratio, "dither": dither,
    #         "out_loc_norm": out_loc_norm, "actual_snd": distance, "target_snd": self.k_hat
    #     }
    #     self._log_to_tensordict(td, "output", log_data)

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

@dataclass
class HetControlMlpEscSndConfig(ModelConfig):
    activation_class: Type[nn.Module] = MISSING
    num_cells: Sequence[int] = MISSING
    
    sampling_period: float = MISSING
    disturbance_frequency: float = MISSING
    disturbance_magnitude: float = MISSING
    integrator_gain: float = MISSING
    high_pass_cutoff_frequency: float = MISSING
    low_pass_cutoff_frequency: float = MISSING

    initial_k: float = MISSING
    snd_tau: float = MISSING   

    # General params
    process_shared: bool = MISSING
    bootstrap_from_desired_snd: bool = MISSING
    probabilistic: Optional[bool] = MISSING
    scale_mapping: Optional[str] = MISSING

    use_adapter: bool = False # Default to False, can be overridden

    @staticmethod
    def associated_class():
        return HetControlMlpEscSnd