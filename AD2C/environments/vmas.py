#  Copyright (c) 2024.
#  ProrokLab (https://www.proroklab.org/)
#  All rights reserved.

import torch
from tensordict import TensorDictBase
from torchrl.envs import EnvBase

# Using the more robust helper function to find the correct model instance
from AD2C.callback import get_het_model
from AD2C.snd import compute_behavioral_distance


def render_callback(experiment, env: EnvBase, data: TensorDictBase):
    """
    Render callback that visualizes the diversity among all agents.
    
    """
    policy = experiment.group_policies["agents"]

    # 1. Reliably find the underlying heterogeneous model, even if wrapped
    model = get_het_model(policy)
    if model is None:
        # If no compatible model is found, just render the environment without the heatmap
        return env.render(mode="rgb_array")

    env_index = 0  # We will render the first parallel environment

    # 2. Select all agents for comparison (agent_groups logic removed)
    agent_indices_to_compare = list(range(model.n_agents))

    # Safety check: Cannot compute diversity with less than 2 agents
    if len(agent_indices_to_compare) < 2:
        return env.render(mode="rgb_array", env_index=env_index)

    # 3. Directly access the raw neural networks for the selected agents
    selected_agent_nets = [
        model.agent_mlps.agent_networks[i] for i in agent_indices_to_compare
    ]

    def snd(pos):
        pos_tensor = torch.tensor(pos, device=model.device, dtype=torch.float)
        # print("Input positions shape:", pos_tensor.shape)  # Expect [N, 2]
        obs_batch = env.scenario.observation_from_pos(pos_tensor, env_index=env_index)

        with torch.no_grad():
            agent_actions = [net(obs_batch) for net in selected_agent_nets]

        # Stack actions per agent: shape [num_agents, N, action_dim]
        actions_tensor = torch.stack(agent_actions)  # [num_agents, N, action_dim]

        # Compute pairwise behavioral distance for each position (broadcasting over N)
        # compute_behavioral_distance should be modified to output [N] or [N, 1]
        distance_matrix = compute_behavioral_distance(
            agent_actions,
            just_mean=False,  # Let it return full matrix per position
        )  # shape [N, num_agent_pairs] or similar

        # Now compute mean diversity for each position (across agent pairs)
        # If shape is [N, M], do mean over second dimension
        distance_per_position = distance_matrix.mean(dim=1).view(-1, 1)  # [N, 1]

        # print("Output distance shape:", distance_per_position.shape)  # Should match input N

        return distance_per_position



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