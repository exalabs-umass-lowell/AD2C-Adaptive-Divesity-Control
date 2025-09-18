# Copyright (c) 2024.
# ProrokLab (https://www.proroklab.org/)
# All rights reserved.

import hydra
import os
import torch
import numpy as np
import pickle
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from typing import List

# --- Your imports ---
from AD2C.callbacks.esc_callback import ExtremumSeekingController
from AD2C.callbacks.fixed_callbacks import *
import benchmarl.models
from benchmarl.algorithms import *
from benchmarl.environments import VmasTask
from benchmarl.experiment import Experiment
from benchmarl.hydra_config import (
    load_algorithm_config_from_hydra,
    load_experiment_config_from_hydra,
    load_task_config_from_hydra,
    load_model_config_from_hydra,
)
from AD2C.models.het_control_mlp_empirical import HetControlMlpEmpirical, HetControlMlpEmpiricalConfig
from AD2C.environments.vmas import render_callback
from AD2C.callbacks.sndESLogger import TrajectorySNDLoggerCallback
from AD2C.callbacks.performaceLoggerCallback import performaceLoggerCallback

from AD2C.callbacks.SndLogCallback import SndLoggingCallback



def setup(task_name):
    benchmarl.models.model_config_registry.update(
        {"hetcontrolmlpempirical": HetControlMlpEmpiricalConfig}
    )
    if task_name == "vmas/navigation":
        VmasTask.render_callback = render_callback

# I've removed the unused `target_snd` parameter here
def create_experiment(cfg: DictConfig, callbacks_for_run, run_name: str) -> Experiment:
    hydra_choices = HydraConfig.get().runtime.choices
    task_name = hydra_choices.task
    
    setup(task_name)
    
    algorithm_config = load_algorithm_config_from_hydra(cfg.algorithm)
    experiment_config = load_experiment_config_from_hydra(cfg.experiment)
    task_config = load_task_config_from_hydra(cfg.task, task_name)
    critic_model_config = load_model_config_from_hydra(cfg.critic_model)
    model_config = load_model_config_from_hydra(cfg.model)
    
    if "desired_snd" in cfg.model and isinstance(model_config, HetControlMlpEmpiricalConfig):
        model_config.desired_snd = float(cfg.model.desired_snd)
    
    if isinstance(algorithm_config, (MappoConfig, IppoConfig, MasacConfig, IsacConfig)):
        model_config.probabilistic = True
        model_config.scale_mapping = algorithm_config.scale_mapping
        algorithm_config.scale_mapping = "relu"
    else:
        model_config.probabilistic = False

    experiment_config.name = run_name

    return Experiment(
        task=task_config,
        algorithm_config=algorithm_config,
        model_config=model_config,
        critic_model_config=critic_model_config,
        seed=cfg.seed,
        config=experiment_config,
        callbacks=callbacks_for_run,
    )


@hydra.main(version_base=None, config_path="conf", config_name="navigation_ippo")
def hydra_main(cfg: DictConfig) -> None:
    hydra_choices = HydraConfig.get().runtime.choices
    algorithm_name = hydra_choices.algorithm
    
    # 1. Run a single, longer experiment to let the ESC find the optimal SND
    total_frames = 6_000_000
    
    # 2. Pick one reasonable starting SND for the controller and model
    starting_snd = 0.5   #1.0  

    print(f"\n--- RUNNING EXPERIMENT WITH EXTREMUM SEEKING CONTROLLER ---")
    
    run_cfg = OmegaConf.create(cfg)
    run_cfg.experiment.max_n_frames = total_frames
    
    # This line is crucial to prevent the TypeError on startup
    run_cfg.model.desired_snd = starting_snd
    
    run_name = f"{algorithm_name}_ESC_seeking_optimum_SND"
    
    # 3. Configure the controller
    callbacks = [
        SndLoggingCallback(
            # control_group = "agents",
            ),
        ExtremumSeekingController(
            control_group="agents",
            initial_snd=starting_snd,
            dither_magnitude=0.1,
            dither_frequency_rad_s=0.5,
            integral_gain=-0.05,  # KEY CHANGE: Use a negative gain to maximize reward
            high_pass_cutoff_rad_s=0.1,
            low_pass_cutoff_rad_s=0.1,
            sampling_period=1.0
        ),
        
        TrajectorySNDLoggerCallback(
            control_group = "agents",
        ),
        performaceLoggerCallback(
            control_group = "agents",
            initial_snd=starting_snd,
        ),
        NormLoggerCallback(),
        ActionSpaceLoss(use_action_loss=cfg.use_action_loss, action_loss_lr=cfg.action_loss_lr),
    ]
    
    print(f"\nStarting run '{run_name}' (Seed: {run_cfg.seed})")
    
    experiment = create_experiment(
        cfg=run_cfg, 
        callbacks_for_run=callbacks, 
        run_name=run_name
    )
    experiment.run()
    
    print("\n--- Experiment finished ---")


if __name__ == "__main__":
    hydra_main()