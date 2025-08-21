# plotting_callback.py

import torch
import os
import pickle
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