#  Copyright (c) 2024.
#  ProrokLab (https://www.proroklab.org/)
#  All rights reserved.

from typing import List, Tuple, Type, Optional

import os
import pickle
from typing import List, Dict,Any, Callable

from PIL import Image
from matplotlib import pyplot as plt
import numpy as np
import io
import torch
import torch.nn as nn
import wandb
from tensordict import TensorDictBase, TensorDict
from typing import List, Dict, Union

from benchmarl.experiment.callback import Callback
from AD2C.models.het_control_mlp_empirical import HetControlMlpEmpirical
# Make sure to import all compatible models
from AD2C.models.het_control_mlp_esc import HetControlMlpEsc
from AD2C.snd import compute_behavioral_distance
from AD2C.utils import overflowing_logits_norm

from .utils import get_het_model


# FIX 2: Define a tuple of all compatible heterogeneous models
HET_CONTROL_MODELS: Tuple[Type[nn.Module]] = (HetControlMlpEmpirical, HetControlMlpEsc)

class NormLoggerCallback(Callback):
    """Callback to log some training metrics"""
    def on_batch_collected(self, batch: TensorDictBase):
        for group in self.experiment.group_map.keys():
            # FIX: Added the new ESC controller keys to this list
            keys_to_norm = [
                # Original keys
                (group, "logits"), 
                (group, "observation"),
                (group, "out_loc_norm"), 
                (group, "estimated_snd"),
                (group, "target_diversity"), 
                (group, "scaling_ratio"),
                (group, "current_dither"),
                # New keys for the ESC model
                (group, "k_hat"),
                (group, "esc_dither"),
                (group, "esc_reward_J"),
                (group, "esc_grad_estimate"),
                (group, "esc_k_hat_update"),
            ]
            to_log = {}
            for key in keys_to_norm:
                value = batch.get(key, None)
                if value is not None:
                    # This will create log names like "collection/agents/k_hat" in W&B
                    to_log[f"collection/{'/'.join(key)}"] = torch.mean(value).item()
            
            if to_log:
                self.experiment.logger.log(to_log, step=self.experiment.n_iters_performed)
            
class TagCurriculum(Callback):
    """Tag curriculum used to freeze the green agents' policies during training"""
    def __init__(self, simple_tag_freeze_policy_after_frames, simple_tag_freeze_policy):
        super().__init__()
        self.n_frames_train = simple_tag_freeze_policy_after_frames
        self.simple_tag_freeze_policy = simple_tag_freeze_policy
        self.activated = not simple_tag_freeze_policy

    def on_setup(self):
        self.experiment.logger.log_hparams(
            simple_tag_freeze_policy_after_frames=self.n_frames_train,
            simple_tag_freeze_policy=self.simple_tag_freeze_policy,
        )
        policy = self.experiment.group_policies["agents"]
        model = get_het_model(policy)
        # Only set desired_snd if the model is the DiCo model
        if isinstance(model, HetControlMlpEmpirical):
            model.desired_snd[:] = 0

    def on_batch_collected(self, batch: TensorDictBase):
        if (self.experiment.total_frames >= self.n_frames_train and
                not self.activated and self.simple_tag_freeze_policy):
            if "agents" in self.experiment.train_group_map:
                del self.experiment.train_group_map["agents"]
                self.activated = True

class ActionSpaceLoss(Callback):
    """Loss to disincentivize actions outside of the space"""
    def __init__(self, use_action_loss, action_loss_lr):
        super().__init__()
        self.opt_dict = {}
        self.use_action_loss = use_action_loss
        self.action_loss_lr = action_loss_lr

    def on_setup(self):
        self.experiment.logger.log_hparams(
            use_action_loss=self.use_action_loss, action_loss_lr=self.action_loss_lr
        )

    def on_train_step(self, batch: TensorDictBase, group: str) -> TensorDictBase:
        if not self.use_action_loss:
            return TensorDict({}, [])
        policy = self.experiment.group_policies[group]
        model = get_het_model(policy)
        if model is None:
            return TensorDict({}, [])

        if group not in self.opt_dict:
            self.opt_dict[group] = torch.optim.Adam(model.parameters(), lr=self.action_loss_lr)
        opt = self.opt_dict[group]
        loss = self.action_space_loss(group, model, batch)
        loss_td = TensorDict({"loss_action_space": loss}, [])
        loss.backward()
        grad_norm = self.experiment._grad_clip(opt)
        loss_td.set(f"grad_norm_action_space", torch.tensor(grad_norm, device=self.experiment.config.train_device))
        opt.step()
        opt.zero_grad()
        return loss_td

    def action_space_loss(self, group, model, batch):
        logits = model._forward(
            batch.select(*model.in_keys), compute_estimate=True, update_estimate=False
        ).get(model.out_key)
        if model.probabilistic:
            logits, _ = torch.chunk(logits, 2, dim=-1)
        out_loc_norm = overflowing_logits_norm(logits, self.experiment.action_spec[group, "action"])
        max_overflowing_logits_norm = out_loc_norm.max(dim=-1)[0]
        loss = max_overflowing_logits_norm.pow(2).mean()
        return loss

class PIDHetControlUpdater:
    """Implements the AD2C PID-Optimizer logic."""
    def __init__(
        self, model: HetControlMlpEmpirical, kp: float, ki: float, kd: float,
        initial_snd: float, baseline_update_rate_alpha: float, max_update_step: float = 0.05,
    ):
        self.model = model
        self.kp, self.ki, self.kd = kp, ki, kd
        self.alpha = baseline_update_rate_alpha
        self.max_update_step = max_update_step
        self.model.desired_snd[:] = float(initial_snd)
        self._r_baseline = 0.0
        self._total_improvement = 0.0
        self._prev_improvement = 0.0
        self._is_first_step = True

    def step(self, current_performance: float, actual_snd: float) -> dict:
        """Performs one step of the PID update and returns a dictionary of metrics for logging."""
        if self._is_first_step:
            self._r_baseline = current_performance
            self._is_first_step = False
            return {
                "pid/desired_snd": self.model.desired_snd.item(),
                "pid/actual_snd": actual_snd,
                "pid/reward_baseline": self._r_baseline,
                "pid/performance_improvement": 0.0,
            }
        improvement = current_performance - self._r_baseline
        self._total_improvement += improvement
        proportional_term = self.kp * improvement
        integral_term = self.ki * self._total_improvement
        derivative_term = self.kd * (improvement - self._prev_improvement)
        update_step = proportional_term + integral_term + derivative_term
        clamped_update = max(-self.max_update_step, min(self.max_update_step, update_step))
        current_snd = self.model.desired_snd.item()
        new_snd = max(0.0, min(2.0, current_snd + clamped_update))
        self.model.desired_snd[:] = new_snd
        self._r_baseline = (1 - self.alpha) * self._r_baseline + self.alpha * current_performance
        self._prev_improvement = improvement
        return {
            "pid/actual_snd": actual_snd,
            "pid/desired_snd": new_snd,
            "pid/reward_baseline": self._r_baseline,
            "pid/current_performance": current_performance,
            "pid/performance_improvement": improvement,
            "pid/proportional_term": proportional_term,
            "pid/integral_term": integral_term,
            "pid/derivative_term": derivative_term,
            "pid/update_step": update_step,
        }

class HetControlMetricsCallback(Callback):
    """
    A unified callback to compute and log SND, model-specific metrics, and update
    the PID controller for desired SND.
    """
    def __init__(self, use_pid_controller: bool = False, kp: float = 0.0, ki: float = 0.0, kd: float = 0.0, initial_snd: float = 0.0, alpha: float = 0.1):
        super().__init__()
        self.use_pid_controller = use_pid_controller
        self.kp, self.ki, self.kd = kp, ki, kd
        self.initial_snd = initial_snd
        self.alpha = alpha
        self.pid_updater = None
        self.pid_group = None

    def on_setup(self):
        if not self.use_pid_controller:
            return
        for group, policy in self.experiment.group_policies.items():
            model_instance = get_het_model(policy)
            # Only initialize PID for the DiCo model
            if isinstance(model_instance, HetControlMlpEmpirical):
                self.pid_updater = PIDHetControlUpdater(
                    model=model_instance, kp=self.kp, ki=self.ki, kd=self.kd,
                    initial_snd=self.initial_snd, baseline_update_rate_alpha=self.alpha,
                )
                self.pid_group = group
                print(f"\nSUCCESS: PID Controller initialized for group '{group}'.\n")
                return
        if self.use_pid_controller:
            print("\nWARNING: PID Controller was enabled, but a compatible model was not found.\n")

    def on_evaluation_end(self, results: dict):
        logs_to_push = {}
        snds_per_group = {}
        if "evaluation_rollouts" not in results:
            return
        rollouts = results["evaluation_rollouts"]

        for group in self.experiment.group_map.keys():
            if len(self.experiment.group_map.get(group, [])) <= 1:
                continue

            policy = self.experiment.group_policies[group]
            model = get_het_model(policy)
            if model is None:
                continue

            key = (group, "observation")
            if key not in rollouts[0].keys(include_nested=True):
                continue
            obs = torch.cat([rollout.get(key) for rollout in rollouts], dim=0)

            # FIX 1: Replace the loop with a single vectorized call to the model
            with torch.no_grad():
                td_in = TensorDict({model.in_key: obs}, batch_size=obs.shape[:-2])
                # A single forward pass gets actions for all agents
                td_out = model._forward(td_in, agent_index=None, compute_estimate=False)
                # The output shape is [batch, n_agents, action_dim]
                agent_actions = td_out.get(model.out_key)

            # This now correctly computes SND on the vectorized output
            actual_snd = compute_behavioral_distance(agent_actions, just_mean=True)
            snds_per_group[group] = actual_snd.mean().item()
            logs_to_push[f"eval/{group}/snd_actual"] = snds_per_group[group]

            if isinstance(model, HetControlMlpEmpirical):
                logs_to_push[f"eval/{group}/desired_snd_used"] = model.desired_snd.item()
                if hasattr(model, 'estimated_snd'):
                    logs_to_push[f"eval/{group}/estimated_snd"] = model.estimated_snd.item()

        if self.pid_updater is not None:
            # FIX 3: Use a standard reward key. Check results.keys() to confirm!
            reward_key = "mean_reward"
            if reward_key in results and self.pid_group in snds_per_group:
                current_performance = results[reward_key]
                actual_snd_for_pid = snds_per_group[self.pid_group]
                pid_logs = self.pid_updater.step(
                    current_performance=current_performance,
                    actual_snd=actual_snd_for_pid
                )
                logs_to_push.update(pid_logs)
            else:
                print(f"\nWARNING: '{reward_key}' not found in results.keys() or group SND not found. PID Controller cannot update.\n")
                # Optional: print available keys for debugging
                # print("Available keys in evaluation results:", results.keys())

        if logs_to_push:
            self.experiment.logger.log(logs_to_push, step=self.experiment.n_iters_performed)