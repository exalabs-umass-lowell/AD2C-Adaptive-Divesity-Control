#  Copyright (c) 2024.
#  ProrokLab (https://www.proroklab.org/)
#  All rights reserved.

import torch
import numpy as np
from tensordict import TensorDictBase, TensorDict
from torchrl.envs import EnvBase

from AD2C.callbacks.utils import get_het_model
from AD2C.snd import compute_behavioral_distance


def render_callback(experiment, env: EnvBase, data: TensorDictBase):
    """
    Render callback that visualizes diversity (SND) across positions.
    """
    policy = experiment.group_policies.get("agents")
    if not policy:
        # If there's no policy for the 'agents' group, do nothing.
        return env.render(mode="rgb_array")

    model = get_het_model(policy)

    # FIX 1: Add a check to ensure the model was found before using it.
    if model is None:
        # If no compatible model is found, render the environment normally
        # without the diversity overlay.
        return env.render(mode="rgb_array")

    env_index = 0

    # Resolve device
    device = next(model.parameters()).device
    n_agents = env.n_agents

    def snd(pos: np.ndarray):
        """
        Computes the SND for a given set of grid positions.
        pos: np.ndarray with shape (N, 2)
        return: np.ndarray with shape (N,)
        """
        pos_t = torch.as_tensor(pos, dtype=torch.float32, device=device)
        N = pos_t.shape[0]

        # Build observations for all positions
        obs_raw = env.scenario.observation_from_pos(pos_t, env_index=env_index)
        obs = obs_raw.view(-1, n_agents, obs_raw.shape[-1]).to(torch.float32)
        M = obs.shape[0]

        if M == 0:
            return np.zeros(N)

        # Create a TensorDict for the batch of observations
        obs_td = TensorDict(
            {"agents": {"observation": obs}},
            batch_size=[M],
            device=device,
        )

        with torch.no_grad():
            # FIX 2: Perform a single, vectorized forward pass to get all actions
            td_out = model._forward(obs_td, agent_index=None, compute_estimate=False)
            # The output shape is [batch, n_agents, action_dim]
            agent_actions = td_out.get(("agents", model.out_key))

        # Compute SND for each joint observation in the batch -> shape (M,)
        distances = compute_behavioral_distance(agent_actions, just_mean=True)

        # FIX 3: Simplify the reshaping logic.
        # Each of the M joint observations corresponds to n_agents individual positions.
        # We repeat each SND value n_agents times to match the input position shape.
        distances_expanded = distances.repeat_interleave(n_agents)

        # Final guard for edge cases where N might not be M * n_agents
        if distances_expanded.shape[0] != N:
            distances_expanded = torch.cat([distances_expanded, torch.zeros(N - distances_expanded.shape[0], device=device)])

        return distances_expanded.cpu().numpy()

    # Render the environment with the SND heatmap overlay
    return env.render(
        mode="rgb_array",
        visualize_when_rgb=True, # Ensure visualization is on
        plot_position_function=snd,
        plot_position_function_range=1.5,
        plot_position_function_cmap_alpha=0.5,
        env_index=env_index,
        plot_position_function_precision=0.05,
        plot_position_function_cmap_range=[0.0, 1.0],
    )