from __future__ import annotations

import torch
from isaaclab.assets import RigidObject
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg


def object_in_box(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg,
    box_cfg: SceneEntityCfg,
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    height_threshold: float,
) -> torch.Tensor:
    """True when the object lies within the horizontal footprint of the box and below a z threshold.
    Uses box entity lookup — only reliable for entities declared as class-level fields in the
    scene config (e.g. box2).  For USD-sourced entities added dynamically (e.g. box) use
    object_in_box_at_position instead.
    """
    box: RigidObject = env.scene[box_cfg.name]
    box_x = box.data.root_pos_w[:, 0] - env.scene.env_origins[:, 0]
    box_y = box.data.root_pos_w[:, 1] - env.scene.env_origins[:, 1]

    obj: RigidObject = env.scene[object_cfg.name]
    obj_x = obj.data.root_pos_w[:, 0] - env.scene.env_origins[:, 0]
    obj_y = obj.data.root_pos_w[:, 1] - env.scene.env_origins[:, 1]
    obj_z = obj.data.root_pos_w[:, 2] - env.scene.env_origins[:, 2]

    in_x = (obj_x > box_x + x_range[0]) & (obj_x < box_x + x_range[1])
    in_y = (obj_y > box_y + y_range[0]) & (obj_y < box_y + y_range[1])
    below_height = obj_z < height_threshold

    return in_x & in_y & below_height


def object_in_box_at_position(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg,
    box_x: float,
    box_y: float,
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    height_threshold: float,
) -> torch.Tensor:
    """True when the object is within x/y range of a fixed env-local box centre and below height_threshold.
    Uses hardcoded box coordinates instead of a SceneEntityCfg lookup, so it works reliably even
    for USD-sourced entities that are added to the scene config dynamically.
    """
    obj: RigidObject = env.scene[object_cfg.name]
    obj_x = obj.data.root_pos_w[:, 0] - env.scene.env_origins[:, 0]
    obj_y = obj.data.root_pos_w[:, 1] - env.scene.env_origins[:, 1]
    obj_z = obj.data.root_pos_w[:, 2] - env.scene.env_origins[:, 2]

    in_x = (obj_x > box_x + x_range[0]) & (obj_x < box_x + x_range[1])
    in_y = (obj_y > box_y + y_range[0]) & (obj_y < box_y + y_range[1])

    return in_x & in_y & (obj_z < height_threshold)


def object_dropped_elsewhere(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg,
    box_a_x: float,
    box_a_y: float,
    box_b_x: float,
    box_b_y: float,
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    height_threshold: float,
    lift_threshold: float = 0.15,
) -> torch.Tensor:
    """True when the object was lifted above lift_threshold and then came to rest
    outside the footprints of both boxes (i.e. dropped on the table or fell off it).
    Uses a per-episode 'lifted' flag (shared key with rewards.py) so the spawn
    position on the table never triggers a false positive at episode start.
    Box positions are passed as fixed coordinates to avoid SceneEntityCfg resolution issues.
    """
    obj: RigidObject = env.scene[object_cfg.name]
    obj_z = obj.data.root_pos_w[:, 2] - env.scene.env_origins[:, 2]

    flag_key = f"_sort_lifted_{object_cfg.name}"
    if not hasattr(env, flag_key) or getattr(env, flag_key).shape[0] != env.num_envs:
        setattr(env, flag_key, torch.zeros(env.num_envs, dtype=torch.bool, device=env.device))
    lifted: torch.Tensor = getattr(env, flag_key)
    lifted[env.episode_length_buf <= 1] = False
    lifted |= obj_z > lift_threshold

    in_a = object_in_box_at_position(env, object_cfg, box_a_x, box_a_y, x_range, y_range, height_threshold)
    in_b = object_in_box_at_position(env, object_cfg, box_b_x, box_b_y, x_range, y_range, height_threshold)

    # Require the object to be at rest so that actively holding or lowering the
    # object through this z band (while still grasped) doesn't fire prematurely.
    speed = obj.data.root_lin_vel_w.norm(dim=-1)
    at_rest = speed < 0.05  # m/s

    return lifted & (obj_z < height_threshold) & at_rest & ~in_a & ~in_b
