# Copyright (c) 2024.
# ProrokLab (https://www.proroklab.org/)
# All rights reserved.

import hydra
import os
import torch
import numpy as np
import pickle # Added for saving/loading results
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from typing import List

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
from AD2C.callback import *
from AD2C.environments.vmas import render_callback


def setup(task_name):
    benchmarl.models.model_config_registry.update(
        {
            "hetcontrolmlpempirical": HetControlMlpEmpiricalConfig,
        }
    )
    if task_name == "vmas/navigation":
        VmasTask.render_callback = render_callback

def create_experiment(cfg: DictConfig, callbacks_for_run) -> Experiment:
    hydra_choices = HydraConfig.get().runtime.choices
    task_name = hydra_choices.task
    # algorithm_name = hydra_choices.algorithm
    
    setup(task_name)
    
    # Load configs from Hydra and apply logic
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
    hydra_choices = HydraConfig.get().runtime.choices
    task_name = hydra_choices.task
    
    
    # Define SND arms and other parameters
    # snd_arms = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    snd_arms = [1,2,3]
    exploration_frames = 6_000_000
    final_frames = 6_000_000
    
    base_save_folder = HydraConfig.get().run.dir

    # --- Phase 1: Exploration ---
    print("\n--- PHASE 1: EXPLORATION (Running short experiments) ---")
    exploration_parent_folder = os.path.join(base_save_folder, "exploration_runs")
    
    for i, snd in enumerate(snd_arms):
        exploration_cfg = OmegaConf.create(cfg)
        
        run_save_folder = os.path.join(exploration_parent_folder, f"snd_{snd}")
        os.makedirs(run_save_folder, exist_ok=True)
        
        exploration_cfg.experiment.save_folder = run_save_folder
        exploration_cfg.experiment.max_n_frames = exploration_frames
        exploration_cfg.seed = cfg.seed
        
        exploration_cfg.model.desired_snd = snd
                
        callbacks = [
            # EloEvaluationCallback(
            #     snd=snd, 
            #     run_name=f"snd_{snd}", 
            #     control_group="agents",
            #     save_folder=run_save_folder # Pass the folder path here
            # ),
            
            # SndCallback(
            #     control_group="agents",
            #     proportional_gain=0.2,
            #     initial_snd=1,  # 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0
            # ),

            # gradientBaseSndCallback(
            #     control_group = "agents",
            #     initial_snd = snd,
            #     learning_rate = 0.01,
            #     alpha = 0.01,
            # ),
        
            clusterSndCallback(
                control_group="agents",
                initial_snd=snd,  # 0.0, 0.1,
            ),

            # pIControllerCallback(
            #     control_group="agents",
            #     proportional_gain=0.2,
            #     initial_snd=snd,  # 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0
            # ),

            # SimpleProportionalController(
            #     control_group="agents",
            #     proportional_gain=0.2,
            #     initial_snd=snd,  # 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0
            # ),

            NormLoggerCallback(),
            ActionSpaceLoss(use_action_loss=cfg.use_action_loss, action_loss_lr=cfg.action_loss_lr),
        ]
        
        print(f"\nRunning exploration for SND: {snd} (Seed: {exploration_cfg.seed})")
        experiment = create_experiment(cfg=exploration_cfg, callbacks_for_run=callbacks)
        experiment.run()
        
    # --- Phase 2: Ranking ---
    # print("\n--- PHASE 2: RANKING (Finding the best SND) ---")
    # best_snd = find_best_snd(exploration_parent_folder)
    # if best_snd is None:
    #     print("Could not determine best SND. Aborting final training.")
    #     return
    
    # --- Phase 3: Final Training ---
    # final_training_cfg = OmegaConf.create(cfg)
    
    # final_run_folder = os.path.join(base_save_folder, "final_run")
    # os.makedirs(final_run_folder, exist_ok=True)
    
    # final_training_cfg.experiment.save_folder = final_run_folder
    # final_training_cfg.experiment.max_n_frames = final_frames
    # final_training_cfg.seed = cfg.seed + len(snd_arms)
    
    # final_callbacks = [
    #     SndCallback(),
    #     SimpleProportionalController(
    #         control_group="agents",
    #         proportional_gain=0.1,
    #         initial_snd=best_snd,
    #     ),
    #     NormLoggerCallback(),
    #     ActionSpaceLoss(use_action_loss=cfg.use_action_loss, action_loss_lr=cfg.action_loss_lr),
    #     SndSetterCallback(control_group="agents", snd=best_snd)
    # ]
    
    # experiment = create_experiment(cfg=final_training_cfg, callbacks_for_run=final_callbacks)
    # experiment.run()

if __name__ == "__main__":
    hydra_main()