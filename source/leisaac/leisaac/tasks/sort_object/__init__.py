import gymnasium as gym

# Base registration (cube/red default) — kept for backward compatibility.
gym.register(
    id="LeIsaac-SO101-SortObject-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.sort_object_env_cfg:SortObjectCubeRedEnvCfg",
    },
)

# ── Cube variants ─────────────────────────────────────────────────────────────
gym.register(
    id="LeIsaac-SO101-SortObject-CubeRed-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": f"{__name__}.sort_object_env_cfg:SortObjectCubeRedEnvCfg"},
)
gym.register(
    id="LeIsaac-SO101-SortObject-CubeGreen-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": f"{__name__}.sort_object_env_cfg:SortObjectCubeGreenEnvCfg"},
)
gym.register(
    id="LeIsaac-SO101-SortObject-CubeYellow-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": f"{__name__}.sort_object_env_cfg:SortObjectCubeYellowEnvCfg"},
)

# ── Rectangle variants ────────────────────────────────────────────────────────
gym.register(
    id="LeIsaac-SO101-SortObject-RectangleRed-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": f"{__name__}.sort_object_env_cfg:SortObjectRectangleRedEnvCfg"},
)
gym.register(
    id="LeIsaac-SO101-SortObject-RectangleBlue-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": f"{__name__}.sort_object_env_cfg:SortObjectRectangleBlueEnvCfg"},
)
gym.register(
    id="LeIsaac-SO101-SortObject-RectangleGreen-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": f"{__name__}.sort_object_env_cfg:SortObjectRectangleGreenEnvCfg"},
)
gym.register(
    id="LeIsaac-SO101-SortObject-RectangleYellow-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": f"{__name__}.sort_object_env_cfg:SortObjectRectangleYellowEnvCfg"},
)

# ── Cylinder variants ─────────────────────────────────────────────────────────
gym.register(
    id="LeIsaac-SO101-SortObject-CylinderRed-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": f"{__name__}.sort_object_env_cfg:SortObjectCylinderRedEnvCfg"},
)
gym.register(
    id="LeIsaac-SO101-SortObject-CylinderBlue-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": f"{__name__}.sort_object_env_cfg:SortObjectCylinderBlueEnvCfg"},
)
gym.register(
    id="LeIsaac-SO101-SortObject-CylinderGreen-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": f"{__name__}.sort_object_env_cfg:SortObjectCylinderGreenEnvCfg"},
)
