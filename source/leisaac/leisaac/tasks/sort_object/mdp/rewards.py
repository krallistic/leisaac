from __future__ import annotations

import torch
from isaaclab.assets import RigidObject
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg

from .terminations import object_in_box


def placement_reward(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg,
    correct_box_cfg: SceneEntityCfg,
    wrong_box_cfg: SceneEntityCfg,
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    height_threshold: float,
    lift_threshold: float = 0.15,
) -> torch.Tensor:
    """Sparse placement reward (fires once at the terminal step):
        3 = object placed in the correct box
        2 = object placed in the wrong box
        1 = object dropped outside both boxes (after being lifted)
        0 = not yet placed / episode timed out
    Prints to console whenever a non-zero reward is assigned.
    """
    obj: RigidObject = env.scene[object_cfg.name]
    obj_z = obj.data.root_pos_w[:, 2] - env.scene.env_origins[:, 2]

    # Per-episode lifted tracker (shared key with terminations.py)
    flag_key = f"_sort_lifted_{object_cfg.name}"
    if not hasattr(env, flag_key) or getattr(env, flag_key).shape[0] != env.num_envs:
        setattr(env, flag_key, torch.zeros(env.num_envs, dtype=torch.bool, device=env.device))
    lifted: torch.Tensor = getattr(env, flag_key)
    lifted[env.episode_length_buf <= 1] = False
    lifted |= obj_z > lift_threshold

    in_correct = object_in_box(env, object_cfg, correct_box_cfg, x_range, y_range, height_threshold)
    in_wrong = object_in_box(env, object_cfg, wrong_box_cfg, x_range, y_range, height_threshold)

    # Only classify as "dropped elsewhere" when the object is at rest, so that
    # actively lowering the object while still grasped doesn't fire prematurely.
    speed = obj.data.root_lin_vel_w.norm(dim=-1)
    at_rest = speed < 0.05  # m/s
    dropped_elsewhere = lifted & (obj_z < height_threshold) & at_rest & ~in_correct & ~in_wrong

    reward = torch.zeros(env.num_envs, device=env.device)
    reward[dropped_elsewhere] = 1.0
    reward[in_wrong] = 2.0
    reward[in_correct] = 3.0  # correct takes priority if boxes overlap

    for r_val, label in ((3, "correct box"), (2, "wrong box"), (1, "dropped elsewhere")):
        count = int((reward == r_val).sum().item())
        if count > 0:
            print(f"[SortObject] Reward {r_val} ({label}) — {count} env(s)")

    return reward
