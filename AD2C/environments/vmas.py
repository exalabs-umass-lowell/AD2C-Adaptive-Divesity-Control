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

    VMAS's renderer passes a batch of positions with shape (N, 2) and expects
    a return array of shape (N,). We compute one SND per joint observation
    (shape M), then expand it back to per-position length N.
    """
    policy = experiment.group_policies["agents"]
    model = get_het_model(policy)
    env_index = 0

    # Resolve device
    try:
        device = model.device
    except AttributeError:
        device = next(model.parameters()).device

    n_agents = env.n_agents

    def snd(pos):
        """
        pos: np.ndarray or torch.Tensor with shape (N, 2) or (2,)
        return: np.ndarray with shape (N,)
        """
        # Ensure tensor, batch-first
        pos_t = torch.as_tensor(pos, dtype=torch.float32, device=device)
        if pos_t.dim() == 1:
            pos_t = pos_t.unsqueeze(0)  # (1, 2)
        N = pos_t.shape[0]

        # Build observations for all positions (VMAS groups positions into joint obs)
        obs_raw = env.scenario.observation_from_pos(pos_t, env_index=env_index)
        # Reshape to (M, n_agents, obs_dim)
        obs = obs_raw.view(-1, n_agents, obs_raw.shape[-1]).to(torch.float32)
        M = obs.shape[0]

        # Create TensorDict for batch of joint observations
        obs_td = TensorDict(
            {"agents": TensorDict({"observation": obs}, batch_size=[M, n_agents])},
            batch_size=[M],
            device=device,
        )

        # Forward all agents on the same batch
        agent_actions = []
        for i in range(model.n_agents):
            out_td = model._forward(obs_td, agent_index=i)
            agent_actions.append(out_td.get(model.out_key))  # shape (M, action_dim)

        # Compute SND per joint observation -> shape (M,)
        distances = compute_behavioral_distance(agent_actions, just_mean=True)  # (M,)

        # Expand to per-position length N
        # Typically, N == M * n_agents (one pos per agent). Use integer factor to be safe.
        if M > 0:
            factor = N // M
        else:
            factor = 1  # degenerate but avoids div-by-zero

        # If factor is not an integer multiple of n_agents for some scenario,
        # repeat_interleave with computed factor to match N.
        distances_expanded = distances.repeat_interleave(factor)  # shape (M * factor,)

        # Final guard: if still mismatched (due to any scenario-specific packing), resize safely.
        if distances_expanded.shape[0] != N:
            # Try to match exactly by trimming or repeating as needed.
            if distances_expanded.shape[0] > N:
                distances_expanded = distances_expanded[:N]
            else:
                # Pad by repeating last value
                pad = N - distances_expanded.shape[0]
                distances_expanded = torch.cat(
                    [distances_expanded, distances_expanded[-1:].repeat(pad)]
                )
        # print("N positions:", N, "M obs:", M, "factor:", factor, flush=True)
        
        # Return numpy array of shape (N,)
        return distances_expanded.detach().cpu().numpy()

    # Render with SND overlay
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
