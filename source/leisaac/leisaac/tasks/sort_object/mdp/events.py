from __future__ import annotations

import torch
from isaaclab.assets import RigidObject
from isaaclab.envs import ManagerBasedEnv
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg


def reset_object_to_discrete_location(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg,
    locations: list[tuple[float, float, float]],
    position_noise_std: float = 0.005,
) -> None:
    """Reset an object to one of several predefined env-local positions with small Gaussian noise.

    Each reset episode picks one location uniformly at random and adds independent
    Gaussian noise in x/y/z to prevent the policy from memorising exact coordinates.
    The object's default orientation (from the USD spawn) is preserved.

    Args:
        locations: List of (x, y, z) positions in the env-local frame.
        position_noise_std: Std-dev of the additive Gaussian noise in metres.
    """
    asset: RigidObject = env.scene[asset_cfg.name]

    default_state = asset.data.default_root_state[env_ids].clone()

    locs = torch.tensor(locations, dtype=torch.float32, device=env.device)
    chosen = locs[torch.randint(0, len(locations), (len(env_ids),), device=env.device)]

    noise = torch.randn(len(env_ids), 3, device=env.device) * position_noise_std

    # position in world frame; orientation kept from USD default
    default_state[:, :3] = env.scene.env_origins[env_ids] + chosen + noise
    default_state[:, 7:] = 0.0  # zero linear + angular velocity

    asset.write_root_state_to_sim(default_state, env_ids=env_ids)


def discrete_location_term(
    name: str,
    locations: list[tuple[float, float, float]],
    position_noise_std: float = 0.005,
) -> EventTerm:
    """Convenience factory for reset_object_to_discrete_location as an EventTerm."""
    return EventTerm(
        func=reset_object_to_discrete_location,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg(name),
            "locations": locations,
            "position_noise_std": position_noise_std,
        },
    )
