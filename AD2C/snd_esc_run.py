# Copyright (c) 2024.
# ProrokLab (https://www.proroklab.org/)
# All rights reserved.

import hydra
import os
import torch
import numpy as np
import pickle
from hydra.utils import HydraConfig
from omegaconf import DictConfig, OmegaConf
from typing import List

from AD2C.callbacks.pIControllerCallback import pIControllerCallback
from AD2C.callbacks.SndCallback import SndCallback as SndCallbackClass
from AD2C.callbacks.SimpleProportionalController import SimpleProportionalController
from AD2C.callbacks.clusterSndCallback import clusterSndCallback
from AD2C.callbacks.fixed_callbacks import *
from AD2C.callbacks.clusterLogger import TrajectoryLoggerCallback
from AD2C.callbacks.sndESLogger import TrajectorySNDLoggerCallback
from AD2C.callbacks.performaceLoggerCallback import performaceLoggerCallback

import benchmarl.models
from benchmarl.algorithms import *
from benchmarl.environments import VmasTask
from benchmarl.experiment import Experiment
from benchmarl.experiment.callback import Callback
from benchmarl.hydra_config import (
    load_algorithm_config_from_hydra,
    load_experiment_config_from_hydra,
    load_task_config_from_hydra,
    load_model_config_from_hydra,
)
from AD2C.models.het_control_mlp_empirical import HetControlMlpEmpirical, HetControlMlpEmpiricalConfig
from AD2C.models.het_control_mlp_esc import HetControlMlpEsc, HetControlMlpEscConfig
from AD2C.models.het_control_mlp_snd import HetControlMlpEscSnd, HetControlMlpEscSndConfig
# from AD2C.callback123 import *
from AD2C.callbacks.SndLogCallback import SndLoggingCallback
from AD2C.environments.vmas import render_callback
from AD2C.callbacks.escLoggerCallback import EscLoggerCallback


def setup(task_name):
    benchmarl.models.model_config_registry.update(
        {
            # "hetcontrolmlpempirical": HetControlMlpEmpiricalConfig,
            "hetcontrolmlpsnd": HetControlMlpEscSndConfig
        }
    )
    if task_name == "vmas/navigation":
        VmasTask.render_callback = render_callback

def create_experiment(cfg: DictConfig, callbacks_for_run: List[Callback], run_name: str) -> Experiment:
    hydra_choices = HydraConfig.get().runtime.choices
    task_name = hydra_choices.task
    algorithm_name = hydra_choices.algorithm
    
    setup(task_name)
    
    # Load configs from Hydra and apply logic
    algorithm_config = load_algorithm_config_from_hydra(cfg.algorithm)
    experiment_config = load_experiment_config_from_hydra(cfg.experiment)
    task_config = load_task_config_from_hydra(cfg.task, task_name)
    critic_model_config = load_model_config_from_hydra(cfg.critic_model)
    model_config = load_model_config_from_hydra(cfg.model)
    
    if "desired_snd" in cfg.model and isinstance(model_config, HetControlMlpEscSndConfig):
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


@hydra.main(version_base=None, config_path="conf", config_name="navigation_ippo_snd")
def hydra_main(cfg: DictConfig) -> None:
    hydra_choices = HydraConfig.get().runtime.choices
    task_name = hydra_choices.task
    algorithm_name = hydra_choices.algorithm

    # For the ESC model, you typically run one experiment, not a sweep
    total_frames = 3_000_000
    
    base_save_folder = HydraConfig.get().run.dir

    experiment_cfg = OmegaConf.create(cfg)
    
    run_save_folder = os.path.join(base_save_folder, "esc_run")
    os.makedirs(run_save_folder, exist_ok=True)
    
    experiment_cfg.experiment.save_folder = run_save_folder
    experiment_cfg.experiment.max_n_frames = total_frames
    experiment_cfg.seed = cfg.seed
    
    # The run name for the ESC model
    run_name = f"{algorithm_name}_esc_controller"
            
    # Callbacks for the experiment
    callbacks = [
        SndLoggingCallback(
            # control_group = "agents",
            ),
        # EscLoggerCallback(
        #     # control_group = "agents",
        # ),
        # TrajectorySNDLoggerCallback(
        #     control_group = "agents",
        # ),
        # TrajectoryDataLogger(
        #     save_path="/home/svarp/Desktop/Projects/AD2C/Saved Run Tables"
        #     ),
        performaceLoggerCallback(
            control_group = "agents",
            initial_snd=cfg.model.initial_k,

        ),
        NormLoggerCallback(),
        ActionSpaceLoss(use_action_loss=cfg.use_action_loss, action_loss_lr=cfg.action_loss_lr),
    ]
    
    print(f"\nRunning experiment with ESC model (Seed: {experiment_cfg.seed})")
    
    # Create and run the experiment
    experiment = create_experiment(
        cfg=experiment_cfg, 
        callbacks_for_run=callbacks, 
        run_name=run_name
    )
    experiment.run()


if __name__ == "__main__":
    hydra_main()