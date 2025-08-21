# main.py

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

from AD2C.callbacks.pIControllerCallback import *
from AD2C.callbacks.SndCallback import SndCallback as SndCallbackClass
from AD2C.callbacks.SimpleProportionalController import SimpleProportionalController
from AD2C.callbacks.clusterSndCallback import clusterSndCallback
from AD2C.callbacks.fixed_callbacks import *

# --- New Callback for Data Logging ---
from benchmarl.experiment.callback import Callback

class TrajectoryDataLogger(Callback):
    def __init__(self, save_path):
        self.episode_rewards = []
        self.episode_snds = []
        self.episode_numbers = []
        self.target_snds = []
        self.save_path = save_path

    def on_episode_end(self, algorithm, **kwargs):
        current_episode = algorithm.total_episodes
        current_reward = torch.mean(algorithm.reward_buffer).item()
        
        # Ensure the model has these attributes
        current_snd = algorithm.model.snd.item()
        target_snd = algorithm.model.desired_snd
        
        self.episode_numbers.append(current_episode)
        self.episode_rewards.append(current_reward)
        self.episode_snds.append(current_snd)
        self.target_snds.append(target_snd)

    def on_experiment_end(self, logger, **kwargs):
        data = {
            'episodes': self.episode_numbers,
            'returns': self.episode_rewards,
            'actual_snd': self.episode_snds,
            'target_snd': self.target_snds
        }
        with open(self.save_path, 'wb') as f:
            pickle.dump(data, f)
        print(f"Trajectory data saved to {self.save_path}")

# --- End of New Callback ---

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
from AD2C.callback123 import *
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
    hydra_choices = HydraConfig.get().runtime.choices
    task_name = hydra_choices.task
    
    snd_arms = [0.5, 0.8, 1, 2, 3]
    exploration_frames = 600_000
    
    base_save_folder = HydraConfig.get().run.dir

    print("\n--- PHASE 1: EXPLORATION (Running short experiments) ---")
    exploration_parent_folder = os.path.join(base_save_folder, "exploration_runs")
    
    for i, snd in enumerate(snd_arms):
        exploration_cfg = OmegaConf.create(cfg)
        
        run_save_folder = os.path.join(exploration_parent_folder, f"snd_{snd}")
        os.makedirs(run_save_folder, exist_ok=True)
        
        exploration_cfg.experiment.save_folder = run_save_folder
        exploration_cfg.experiment.max_n_frames = exploration_frames
        exploration_cfg.seed = cfg.seed + i
        
        exploration_cfg.model.desired_snd = snd
                
        plotting_data_path = os.path.join(run_save_folder, "trajectory_data.pkl")
                
        callbacks = [
            clusterSndCallback(
                control_group="agents",
                initial_snd=snd,
            ),
            NormLoggerCallback(),
            ActionSpaceLoss(use_action_loss=cfg.use_action_loss, action_loss_lr=cfg.action_loss_lr),
            TrajectoryDataLogger(save_path=plotting_data_path),
        ]
        
        print(f"\nRunning exploration for SND: {snd} (Seed: {exploration_cfg.seed})")
        experiment = create_experiment(cfg=exploration_cfg, callbacks_for_run=callbacks)
        experiment.run()
        
    print("\n--- All exploration runs completed. You can now run the plotting script. ---")

if __name__ == "__main__":
    hydra_main()