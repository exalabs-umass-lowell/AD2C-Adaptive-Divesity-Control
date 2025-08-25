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
        return env.render(mode="rgb_array")

    model = get_het_model(policy)

    if model is None:
        # If no compatible model is found, render the environment normally.
        return env.render(mode="rgb_array")

    env_index = 0
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

        obs_raw = env.scenario.observation_from_pos(pos_t, env_index=env_index)
        obs = obs_raw.view(-1, n_agents, obs_raw.shape[-1]).to(torch.float32)
        M = obs.shape[0]

        if M == 0:
            return np.zeros(N)

        obs_td = TensorDict(
            {"agents": {"observation": obs}},
            batch_size=[M],
            device=device,
        )

        with torch.no_grad():
            td_out = model._forward(obs_td, agent_index=None, compute_estimate=False)
            agent_actions = td_out.get(model.out_key)

        distances = compute_behavioral_distance(agent_actions, just_mean=True)
        distances_expanded = distances.repeat_interleave(n_agents)

        # --- START FIX ---
        # This logic robustly handles cases where distances_expanded is either
        # smaller or larger than N, preventing the negative dimension error.
        
        # 1. Create a correctly-sized tensor of zeros.
        final_distances = torch.zeros(N, device=device)
        
        # 2. Determine how many elements to copy over.
        num_to_copy = min(N, distances_expanded.shape[0])
        
        # 3. Copy the computed distances, effectively truncating if too large
        #    or leaving zero-padding if too small.
        final_distances[:num_to_copy] = distances_expanded[:num_to_copy]
        
        return final_distances.cpu().numpy()
        # --- END FIX ---

    # Render the environment with the SND heatmap overlay
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
