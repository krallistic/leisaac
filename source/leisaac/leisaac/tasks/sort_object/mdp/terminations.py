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
    object_name: str,
    box_x: float,
    box_y: float,
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    height_threshold: float,
    post_term_delay: int = 0,
) -> torch.Tensor:
    """True when the object is within x/y range of a fixed env-local box centre and below height_threshold.
    Accepts the object entity name as a plain str (not SceneEntityCfg) so that no manager-side
    resolution takes place — the scene lookup happens at call time and works for all entity types,
    including those added dynamically to the scene config (USD-sourced or procedural).
    If post_term_delay > 0, the condition must hold for that many consecutive steps before firing.
    """
    obj: RigidObject = env.scene[object_name]
    obj_x = obj.data.root_pos_w[:, 0] - env.scene.env_origins[:, 0]
    obj_y = obj.data.root_pos_w[:, 1] - env.scene.env_origins[:, 1]
    obj_z = obj.data.root_pos_w[:, 2] - env.scene.env_origins[:, 2]

    in_x = (obj_x > box_x + x_range[0]) & (obj_x < box_x + x_range[1])
    in_y = (obj_y > box_y + y_range[0]) & (obj_y < box_y + y_range[1])
    in_box = in_x & in_y & (obj_z < height_threshold)

    if post_term_delay == 0:
        return in_box

    flag_key = f"_inbox_delay_{object_name}_{box_x:.3f}_{box_y:.3f}"
    if not hasattr(env, flag_key) or getattr(env, flag_key).shape[0] != env.num_envs:
        setattr(env, flag_key, torch.zeros(env.num_envs, dtype=torch.long, device=env.device))
    ctr: torch.Tensor = getattr(env, flag_key)
    ctr[env.episode_length_buf <= 1] = 0  # reset at episode start
    ctr[in_box] += 1
    ctr[~in_box] = 0  # reset if object leaves the box
    return ctr >= post_term_delay


def object_dropped_elsewhere(
    env: ManagerBasedRLEnv,
    object_name: str,
    box_a_x: float,
    box_a_y: float,
    box_b_x: float,
    box_b_y: float,
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    height_threshold: float,
    lift_threshold: float = 0.15,
    post_term_delay: int = 0,
) -> torch.Tensor:
    """True when the object was lifted above lift_threshold and then came to rest
    outside the footprints of both boxes (i.e. dropped on the table or fell off it).
    Uses a per-episode 'lifted' flag so the spawn position on the table never triggers
    a false positive at episode start.  object_name is a plain str to avoid SceneEntityCfg
    resolution failures for dynamically-added entities.
    If post_term_delay > 0, the condition must hold for that many consecutive steps before firing.
    """
    obj: RigidObject = env.scene[object_name]
    obj_z = obj.data.root_pos_w[:, 2] - env.scene.env_origins[:, 2]

    flag_key = f"_sort_lifted_{object_name}"
    if not hasattr(env, flag_key) or getattr(env, flag_key).shape[0] != env.num_envs:
        setattr(env, flag_key, torch.zeros(env.num_envs, dtype=torch.bool, device=env.device))
    lifted: torch.Tensor = getattr(env, flag_key)
    lifted[env.episode_length_buf <= 1] = False
    lifted |= obj_z > lift_threshold

    # Use undelayed box checks so the ~in_a / ~in_b exclusion is always immediate.
    in_a = object_in_box_at_position(env, object_name, box_a_x, box_a_y, x_range, y_range, height_threshold)
    in_b = object_in_box_at_position(env, object_name, box_b_x, box_b_y, x_range, y_range, height_threshold)

    # Require the object to be at rest so that actively holding or lowering the
    # object through this z band (while still grasped) doesn't fire prematurely.
    speed = obj.data.root_lin_vel_w.norm(dim=-1)
    at_rest = speed < 0.05  # m/s

    dropped = lifted & (obj_z < height_threshold) & at_rest & ~in_a & ~in_b

    if post_term_delay == 0:
        return dropped

    flag_key2 = f"_drop_delay_{object_name}"
    if not hasattr(env, flag_key2) or getattr(env, flag_key2).shape[0] != env.num_envs:
        setattr(env, flag_key2, torch.zeros(env.num_envs, dtype=torch.long, device=env.device))
    ctr: torch.Tensor = getattr(env, flag_key2)
    ctr[env.episode_length_buf <= 1] = 0  # reset at episode start
    ctr[dropped] += 1
    ctr[~dropped] = 0  # reset if condition no longer holds (e.g. object picked up again)
    return ctr >= post_term_delay
