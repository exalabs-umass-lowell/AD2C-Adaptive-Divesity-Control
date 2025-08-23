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
<<<<<<< Updated upstream
    env_index = 0

=======

    env_index = 0  # We will render the first parallel environment
        
>>>>>>> Stashed changes
    def snd(pos):
        """
        Given a position, this function returns the SND of the policies in that observation
        """
<<<<<<< Updated upstream
        obs = env.scenario.observation_from_pos(
            torch.tensor(pos, device=model.device), env_index=env_index
        )  # Get the observation in the scenarip, given the position
        obs = obs.view(-1, env.n_agents, obs.shape[-1]).to(torch.float)
=======
        pos_tensor = torch.from_numpy(pos).to(torch.float).to(model.device)
        
        # Get observations for the ENTIRE batch of positions in one go
        # This should return a tensor of shape (num_positions, features)
        obs = env.scenario.observation_from_pos(pos_tensor, env_index=env_index)

        # obs is likely of shape (num_positions, feat_size)
        num_pos = obs.shape[0]
        feat_size = obs.shape[1]
        
        # We need to reshape this 2D tensor to a 3D tensor to match the expected
        # input for the policy, which is (num_positions, n_agents, feat_size)
        # The `unsqueeze` operation adds a dimension for the number of agents.
        # Note: This implies the same observation is used for all agents for a given position.
        obs = obs.unsqueeze(1).repeat(1, env.n_agents, 1)

        obs = obs.to(torch.float)
>>>>>>> Stashed changes
        obs_td = TensorDict(
            {"agents": TensorDict({"observation": obs}, obs.shape[:2])}, obs.shape[:1]
        )

<<<<<<< Updated upstream
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

=======
        agent_actions = [model._forward(obs_td, agent_index=i).get(model.out_key) for i in range(env.n_agents)]

        distance = compute_behavioral_distance(agent_actions, just_mean=False)
        
        # Check shape of distance here. It should be (num_pos, n_agents)
        print(f"distance shape before mean: {distance.shape}")

        # Corrected line: Average over *agents* (dim=1)
        distance_mean = distance.mean(dim=1) 

        assert distance_mean.shape[0] == num_pos, f"Mismatch distance {distance_mean.shape[0]} vs pos {num_pos}"

        return distance_mean.view(-1)
    
>>>>>>> Stashed changes
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