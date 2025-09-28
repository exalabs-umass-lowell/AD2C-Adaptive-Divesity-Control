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

def create_experiment(cfg: DictConfig, callbacks_for_run) -> Experiment:
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
    """
    Defines and runs multiple experiment suites directly from the code.
    """
    hydra_choices = HydraConfig.get().runtime.choices
    algorithm_name = hydra_choices.algorithm

    # ===================================================================
    # STEP 1: Define all your experiment suites in this list
    # This is your main control panel for running experiments.
    # esc_single_run and fixed_sweep
    # ===================================================================
    experiment_suites = [
        # {
        #     "suite_name": "Simple Navigation Sweep",
        #     "experiment_type": "esc_single_run",
        #     "task_overrides": {
        #         "n_agents": 3,
        #         "agents_with_same_goal": 3,
        #     },
        #     "snd_values": [0.0, 0.3, 0.5, 0.8, 1.2],
        # },
        # {
        #     "suite_name": "Complex Navigation Sweep",
        #     "experiment_type": "esc_single_run",
        #     "task_overrides": {
        #         "n_agents": 3,
        #         "agents_with_same_goal": 2,
        #     },
        #     "snd_values": [0.0, 0.3, 0.5, 0.8, 1.2],
        # },
        {
            "suite_name": "ESC Run on Simple Task",
            "experiment_type": "esc_single_run",
            "task_overrides": {
                "n_agents": 3,
                "agents_with_same_goal": 1,
            },
            "snd_values": [0.0, 0.3, 0.5, 0.8, 1.2],
        },
    ]

    # ===================================================================
    # STEP 2: The script loops through your suites and runs them
    # ===================================================================
    for suite in experiment_suites:
        print(f"\n{'='*25} RUNNING SUITE: {suite['suite_name']} {'='*25}")

        for initial_snd in suite["snd_values"]:
            # Create a fresh copy of the base config for this specific run
            run_cfg = OmegaConf.create(cfg)

            # Unlock the config to allow modifications
            OmegaConf.set_struct(run_cfg, False)

            # A) Apply the task/environment overrides for this suite
            for key, value in suite["task_overrides"].items():
                run_cfg.task[key] = value

            # B) Set the specific SND value for this run
            # run_cfg.model.initial_snd = initial_snd
            run_cfg.model.desired_snd = initial_snd

            # Lock the config again now that we're done modifying it
            OmegaConf.set_struct(run_cfg, True)
            
            # C) Build the correct callbacks based on the experiment type
            experiment_type = suite["experiment_type"]
            if experiment_type == "fixed_sweep":
                # run_name = f"{run_cfg.task.config_overrides.scenario_name}_{algorithm_name}_fixedSND_{initial_snd:.2f}"
                callbacks = [
                    SndLoggingCallback(),
                    performaceLoggerCallback(control_group="agents", initial_snd=initial_snd),
                    NormLoggerCallback(),
                    ActionSpaceLoss(use_action_loss=cfg.use_action_loss, action_loss_lr=cfg.action_loss_lr),
                ]
            elif experiment_type == "esc_single_run":
                # run_name = f"{run_cfg.task.config_overrides.scenario_name}_{algorithm_name}_ESC_startSND_{initial_snd:.2f}"
                callbacks = [
                    SndLoggingCallback(),
                    ExtremumSeekingController(
                        control_group="agents",
                        initial_snd=initial_snd,
                        dither_magnitude=0.25,
                        dither_frequency_rad_s=0.5,
                        integral_gain=-0.05,
                        high_pass_cutoff_rad_s=0.1,
                        low_pass_cutoff_rad_s=0.1,
                        sampling_period=1.0
                    ),
                    TrajectorySNDLoggerCallback(control_group="agents"),
                    performaceLoggerCallback(control_group="agents", initial_snd=initial_snd),
                    NormLoggerCallback(),
                    ActionSpaceLoss(use_action_loss=cfg.use_action_loss, action_loss_lr=cfg.action_loss_lr),
                ]
            else:
                raise ValueError(f"Unknown experiment_type: '{experiment_type}'")

            # D) Create and run the experiment with the modified config
            print(f"\nStarting {suite['suite_name']} run (Seed: {run_cfg.seed})")
            
            experiment = create_experiment(
                cfg=run_cfg,
                callbacks_for_run=callbacks,
                # run_name=run_name
            )
            experiment.run()
            print(f"\n--- finished {suite['suite_name']} ---")

    print("\n--- All experiment suites finished ---")

if __name__ == "__main__":
    hydra_main()