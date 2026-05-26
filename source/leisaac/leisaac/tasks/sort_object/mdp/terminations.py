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
    """True when the object lies within the horizontal footprint of the box and below a z threshold."""
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
