#  Copyright (c) 2024.
#  ProrokLab (https://www.proroklab.org/)
#  All rights reserved.


import torch
from tensordict import TensorDictBase, TensorDict
from torchrl.envs import EnvBase

from AD2C.callbacks.utils import get_het_model
from AD2C.snd import compute_behavioral_distance


def render_callback(experiment, env: EnvBase, data: TensorDictBase):
    """
    Render callback used in the Multi-Agent Navigation scenario to visualize the
    diversity distribution under the evaluation rendering.

    """
    policy = experiment.group_policies["agents"]
    model = get_het_model(policy)
    env_index = 0

    def snd(pos):
        """
        Given a position, this function returns the SND of the policies in that observation
        """
        obs = env.scenario.observation_from_pos(
            torch.tensor(pos, device=model.device), env_index=env_index
        )
        obs = obs.view(-1, env.n_agents, obs.shape[-1]).to(torch.float)
        obs_td = TensorDict(
            {"agents": TensorDict({"observation": obs}, obs.shape[:2])}, obs.shape[:1]
        )

        agent_actions = []
        for i in range(model.n_agents):
            agent_actions.append(
                model._forward(obs_td, agent_index=i).get(model.out_key)
            )
        
        # This function now correctly returns a tensor of shape [batch_size, n_pairs].
        pairwise_distances = compute_behavioral_distance(
            agent_actions,
            just_mean=True,
        )
        
        # We can now simply take the mean across the pairs.
        avg_distance = pairwise_distances.mean(dim=-1)

        # Reshape to the [batch_size, 1] column vector required by the renderer.
        return avg_distance.view(-1, 1)

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