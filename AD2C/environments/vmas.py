#  Copyright (c) 2024.
#  ProrokLab (https://www.proroklab.org/)
#  All rights reserved.

import torch
from tensordict import TensorDictBase
from torchrl.envs import EnvBase

from AD2C.callbacks.utils import get_het_model
from AD2C.snd import compute_behavioral_distance


def render_callback(experiment, env: EnvBase, data: TensorDictBase):
    """
    Render callback that visualizes the diversity among all agents.
    
    This version is fully vectorized for high performance and is robust to complex
    policy structures like PID controllers.
    """
    policy = experiment.group_policies["agents"]

    # 1. Reliably find the underlying heterogeneous model, even if wrapped
    model = get_het_model(policy)
    env_index = 0

    def snd(pos):
        """
        Given a position, this function returns the SND of the policies in that observation
        """
        obs = env.scenario.observation_from_pos(
            torch.tensor(pos, device=model.device), env_index=env_index
        )  # Get the observation in the scenarip, given the position
        obs = obs.view(-1, env.n_agents, obs.shape[-1]).to(torch.float)
        obs_td = TensorDict(
            {"agents": TensorDict({"observation": obs}, obs.shape[:2])}, obs.shape[:1]
        )

        agent_actions = []
        # For each agent, get the policy output in this observation
        for i in range(model.n_agents):
            agent_actions.append(
                model._forward(obs_td, agent_index=i).get(model.out_key)
            )
        # Compute the SND using the agents' output
        distance = compute_behavioral_distance(
            agent_actions,
            just_mean=False,
        )
        # Calculate the mean for the batch and ensure correct shape
        distance = distance_matrix.mean(dim=[-1, -2]).view(-1, 1)

        return distance

    # Render the environment with the fast, vectorized SND visualization
    return env.render(
        mode="rgb_array",
        visualize_when_rgb=False,
        plot_position_function=snd,
        plot_position_function_range=1.5,
        plot_position_function_cmap_alpha=0.5,
        env_index=env_index,
        plot_position_function_precision=0.05,
        plot_position_function_cmap_range=[0.0, 1.0],
    )