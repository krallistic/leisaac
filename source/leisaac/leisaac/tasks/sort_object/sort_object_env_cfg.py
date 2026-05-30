from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from leisaac.assets.scenes.sort_object import SORT_OBJECT_SCENE_CFG, SORT_OBJECT_USD_PATH
from leisaac.utils.domain_randomization import domain_randomization
from leisaac.utils.general_assets import parse_usd_and_create_subassets

from ..template import (
    SingleArmObservationsCfg,
    SingleArmTaskEnvCfg,
    SingleArmTaskSceneCfg,
    SingleArmTerminationsCfg,
)
from ..template.single_arm_env_cfg import SingleArmRewardsCfg
from . import mdp

_BOX_USD_PATH = str(Path(SORT_OBJECT_USD_PATH).parent / "box" / "box.usd")

# Five candidate spawn positions in env-local frame (x, y, z).
# Table surface z ≈ 0.04 m; cube rests at z ≈ 0.062 m.
# Box A (box)  centre: (0.478, -0.460) footprint ±0.075 m → x ∈ [0.403, 0.553], y ∈ [-0.535, -0.385]
# Box B (box2) centre: (0.478, -0.308) footprint ±0.075 m → x ∈ [0.403, 0.553], y ∈ [-0.383, -0.233]
#   Gap between boxes: 2 mm (almost touching).
# All locations below are to the left of both boxes (x < 0.40), clustered tightly
# around (0.28, -0.39) — the midpoint between the two boxes in y.
# Distances from robot base (0.35, -0.64): Box A ≈ 0.22 m, Box B ≈ 0.36 m, spawns ≈ 0.26 m.
_OBJECT_SPAWN_LOCATIONS: list[tuple[float, float, float]] = [
    (0.28, -0.39, 0.062),  # centre
    (0.25, -0.37, 0.062),  # left, forward
    (0.25, -0.43, 0.062),  # left, back
    (0.31, -0.35, 0.062),  # right, forward
    (0.31, -0.45, 0.062),  # right, back
]

# Where to park the cube when it isn't the active object (non-cube variants).
# Placed far outside the robot workspace so it never interferes.
_CUBE_PARKED_LOCATION: tuple[float, float, float] = (0.0, 5.0, 1.0)

# RGB diffuse colors for procedurally-spawned shapes (rectangle, cylinder).
_COLOR_RGB: dict[str, tuple[float, float, float]] = {
    "red":    (0.8, 0.1, 0.1),
    "green":  (0.1, 0.7, 0.1),
    "blue":   (0.1, 0.3, 0.9),
    "yellow": (0.9, 0.8, 0.0),
}

# All valid (shape, color) combinations and their pre-computed targets.
# Used both for the sorting rule and for the startup print.
_SORTING_TABLE: list[tuple[str, str, str]] = [
    ("cube",      "red",    "box"),
    ("cube",      "green",  "box"),
    ("cube",      "yellow", "box2"),
    ("rectangle", "red",    "box2"),
    ("rectangle", "blue",   "box2"),
    ("rectangle", "green",  "box2"),
    ("rectangle", "yellow", "box2"),
    ("cylinder",  "red",    "box2"),
    ("cylinder",  "blue",   "box"),
    ("cylinder",  "green",  "box2"),
]


def _sorting_target(shape: str, color: str) -> str:
    """Return 'box' (Area A) or 'box2' (Area B) for the given object."""
    for s, c, target in _SORTING_TABLE:
        if s == shape and c == color:
            return target
    raise ValueError(f"Unknown (shape, color) combination: ({shape!r}, {color!r})")


def _make_procedural_object_cfg(shape: str, color: str) -> RigidObjectCfg:
    """Return a RigidObjectCfg for a procedurally-spawned rectangle or cylinder.

    The object is placed at the first spawn location as its default state;
    the reset event (discrete_location_term) moves it to a random location
    each episode.
    """
    visual = sim_utils.PreviewSurfaceCfg(diffuse_color=_COLOR_RGB[color])
    rigid  = sim_utils.RigidBodyPropertiesCfg()
    mass   = sim_utils.MassPropertiesCfg(mass=0.05)
    coll   = sim_utils.CollisionPropertiesCfg()

    if shape == "rectangle":
        spawn_cfg = sim_utils.CuboidCfg(
            size=(0.08, 0.05, 0.04),
            visual_material=visual,
            rigid_props=rigid,
            mass_props=mass,
            collision_props=coll,
        )
    elif shape == "cylinder":
        spawn_cfg = sim_utils.CylinderCfg(
            radius=0.025,
            height=0.05,
            visual_material=visual,
            rigid_props=rigid,
            mass_props=mass,
            collision_props=coll,
        )
    else:
        raise ValueError(f"No procedural spawner for shape {shape!r}")

    return RigidObjectCfg(
        prim_path=f"{{ENV_REGEX_NS}}/{shape}",
        spawn=spawn_cfg,
        init_state=RigidObjectCfg.InitialStateCfg(pos=_OBJECT_SPAWN_LOCATIONS[0]),
    )


def _print_startup_info(shape: str, color: str, target_box: str) -> None:
    sep = "=" * 58
    target_label = "Area A  (prim: box)" if target_box == "box" else "Area B  (prim: box2)"
    print(f"\n[SortObject] {sep}")
    print(f"[SortObject]  Active object : {shape} ({color})")
    print(f"[SortObject]  Target box    : {target_label}")
    print(f"[SortObject]")
    print(f"[SortObject]  Full sorting rule:")
    print(f"[SortObject]  {'Shape':<12}  {'Color':<10}  Target")
    print(f"[SortObject]  {'----------':<12}  {'----------':<10}  ------")
    for s, c, t in _SORTING_TABLE:
        tgt = "Area A" if t == "box" else "Area B"
        marker = "  <-- active" if (s == shape and c == color) else ""
        print(f"[SortObject]  {s:<12}  {c:<10}  {tgt}{marker}")
    print(f"[SortObject]")
    print(f"[SortObject]  Pickup locations (5, noise std = 0.001 m):")
    for i, loc in enumerate(_OBJECT_SPAWN_LOCATIONS, 1):
        print(f"[SortObject]    {i}. x={loc[0]:.3f}  y={loc[1]:.3f}  z={loc[2]:.3f}")
    print(f"[SortObject] {sep}\n")


@configclass
class SortObjectSceneCfg(SingleArmTaskSceneCfg):
    """Scene configuration for the sort-object task."""

    scene: AssetBaseCfg = SORT_OBJECT_SCENE_CFG.replace(prim_path="{ENV_REGEX_NS}/Scene")

    # Area B: spawned from the same box USD at a new prim path.
    # Area A (box) is extracted from scene.usd by parse_usd_and_create_subassets;
    # its init_state.pos is overridden in SortObjectEnvCfg.__post_init__ to (0.478, -0.460).
    # Box A centre: (0.478, -0.460); Box B centre: (0.478, -0.308) → ~2 mm gap (almost touching).
    box2: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/box2",
        spawn=sim_utils.UsdFileCfg(usd_path=_BOX_USD_PATH),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.478, -0.308, 0.046),  # shifted -0.15 m in y (was -0.158); ≈ 0.36 m from robot
        ),
    )


# Box A and Box B env-local centre positions (x, y).
# These match the init_state.pos values set in SortObjectEnvCfg.__post_init__.
_BOX_A_POS = (0.478, -0.460)   # Area A  (prim: box,  loaded from scene.usd)
_BOX_B_POS = (0.478, -0.308)   # Area B  (prim: box2, explicit RigidObjectCfg)


@configclass
class SortObjectRewardsCfg(SingleArmRewardsCfg):
    """Sparse placement reward for the sort-object task.

    Box positions and object_cfg are replaced in SortObjectEnvCfg.__post_init__
    based on object_shape and object_color.
    """

    placement: RewTerm = RewTerm(
        func=mdp.placement_reward,
        weight=1.0,
        params={
            "object_name":     "cube",            # replaced in __post_init__
            "correct_box_x":   _BOX_A_POS[0],    # replaced in __post_init__
            "correct_box_y":   _BOX_A_POS[1],    # replaced in __post_init__
            "wrong_box_x":     _BOX_B_POS[0],    # replaced in __post_init__
            "wrong_box_y":     _BOX_B_POS[1],    # replaced in __post_init__
            "x_range":         (-0.065, 0.065),
            "y_range":         (-0.065, 0.065),
            "height_threshold": 0.08,
        },
    )


@configclass
class TerminationsCfg(SingleArmTerminationsCfg):
    """Terminates on correct-box, wrong-box, dropped-elsewhere, or timeout.

    All box positions and object_name are replaced in SortObjectEnvCfg.__post_init__.
    Functions take object_name as a plain str (not SceneEntityCfg) so that no
    manager-side entity resolution takes place — the scene lookup happens at call time
    and works for dynamically-added entities (USD-sourced and procedural shapes).
    """

    correct_box = DoneTerm(
        func=mdp.object_in_box_at_position,
        params={
            "object_name":      "cube",           # replaced in __post_init__
            "box_x":            _BOX_A_POS[0],    # replaced in __post_init__
            "box_y":            _BOX_A_POS[1],    # replaced in __post_init__
            # box footprint ±0.075 m; ±0.065 requires a reasonably centred placement
            "x_range":          (-0.065, 0.065),
            "y_range":          (-0.065, 0.065),
            # box top ≈ 0.046 + 0.050 = 0.096 m; object resting inside ≈ 0.062 m
            "height_threshold": 0.08,
            "post_term_delay":  60,
        },
    )

    wrong_box = DoneTerm(
        func=mdp.object_in_box_at_position,
        params={
            "object_name":      "cube",           # replaced in __post_init__
            "box_x":            _BOX_B_POS[0],    # replaced in __post_init__
            "box_y":            _BOX_B_POS[1],    # replaced in __post_init__
            "x_range":          (-0.065, 0.065),
            "y_range":          (-0.065, 0.065),
            "height_threshold": 0.08,
            "post_term_delay":  60,
        },
    )

    dropped_elsewhere = DoneTerm(
        func=mdp.object_dropped_elsewhere,
        params={
            "object_name":      "cube",           # replaced in __post_init__
            "box_a_x":          _BOX_A_POS[0],
            "box_a_y":          _BOX_A_POS[1],
            "box_b_x":          _BOX_B_POS[0],
            "box_b_y":          _BOX_B_POS[1],
            "x_range":          (-0.065, 0.065),
            "y_range":          (-0.065, 0.065),
            "height_threshold": 0.08,
            "post_term_delay":  60,
        },
    )


@configclass
class SortObjectEnvCfg(SingleArmTaskEnvCfg):
    """Configuration for the sort-object environment.

    object_shape and object_color identify the active object variant.
    The correct target box is derived automatically via the sorting rule.
    Instantiate a shape/color-specific subclass (or set these fields) rather
    than editing this base class directly.
    """

    scene: SortObjectSceneCfg = SortObjectSceneCfg(env_spacing=8.0)

    observations: SingleArmObservationsCfg = SingleArmObservationsCfg()

    rewards: SortObjectRewardsCfg = SortObjectRewardsCfg()

    terminations: TerminationsCfg = TerminationsCfg()

    object_shape: str = "cube"
    object_color: str = "red"

    task_description: str = "Pick up the object and place it in the correct box."

    def apply_object_variant(self, shape: str, color: str) -> None:
        """Re-apply sorting logic after __post_init__ when shape/color is overridden via CLI.

        Only updates the parts that depend on shape/color; does not re-parse the USD or
        re-run domain_randomization from scratch.
        """
        self.object_shape = shape
        self.object_color = color

        target_box  = _sorting_target(shape, color)
        correct_pos = _BOX_A_POS if target_box == "box" else _BOX_B_POS
        wrong_pos   = _BOX_B_POS if target_box == "box" else _BOX_A_POS

        self.terminations.correct_box.params["object_name"] = shape
        self.terminations.correct_box.params["box_x"]       = correct_pos[0]
        self.terminations.correct_box.params["box_y"]       = correct_pos[1]
        self.terminations.wrong_box.params["object_name"]   = shape
        self.terminations.wrong_box.params["box_x"]         = wrong_pos[0]
        self.terminations.wrong_box.params["box_y"]         = wrong_pos[1]
        self.terminations.dropped_elsewhere.params["object_name"] = shape

        self.rewards.placement.params["object_name"]    = shape
        self.rewards.placement.params["correct_box_x"]  = correct_pos[0]
        self.rewards.placement.params["correct_box_y"]  = correct_pos[1]
        self.rewards.placement.params["wrong_box_x"]    = wrong_pos[0]
        self.rewards.placement.params["wrong_box_y"]    = wrong_pos[1]

        # Update the object-reset event term (domain_randomize_0) to the new prim name.
        if hasattr(self.events, "domain_randomize_0"):
            self.events.domain_randomize_0.params["asset_cfg"] = SceneEntityCfg(shape)

        _print_startup_info(shape, color, target_box)

    def __post_init__(self) -> None:
        super().__post_init__()

        self.episode_length_s = 30.0

        self.viewer.eye = (-0.2, -1.0, 0.5)
        self.viewer.lookat = (0.6, 0.0, -0.2)

        self.scene.robot.init_state.pos = (0.35, -0.64, 0.01)

        # Always load the full scene.usd (cube is baked in and can't be excluded
        # at load time). Registering scene.cube lets us control its position.
        parse_usd_and_create_subassets(SORT_OBJECT_USD_PATH, self)

        # Shift Box A 0.15 m closer to the robot in y (USD bakes it at y ≈ -0.310).
        # Keep the z from the USD so it still rests on the table surface.
        _box_z = self.scene.box.init_state.pos[2]
        self.scene.box.init_state.pos = (0.478, -0.460, _box_z)

        if self.object_shape != "cube":
            # Add the procedural shape alongside the already-registered cube.
            setattr(self.scene, self.object_shape,
                    _make_procedural_object_cfg(self.object_shape, self.object_color))

        # Wire terminations and rewards to the correct/wrong box positions and object prim.
        # Fixed-coordinate lookups are used to avoid SceneEntityCfg resolution failures
        # for USD-sourced entities (box) that are added to the scene config dynamically.
        target_box  = _sorting_target(self.object_shape, self.object_color)
        correct_pos = _BOX_A_POS if target_box == "box" else _BOX_B_POS
        wrong_pos   = _BOX_B_POS if target_box == "box" else _BOX_A_POS

        self.terminations.correct_box.params["object_name"] = self.object_shape
        self.terminations.correct_box.params["box_x"]      = correct_pos[0]
        self.terminations.correct_box.params["box_y"]      = correct_pos[1]
        self.terminations.wrong_box.params["object_name"]   = self.object_shape
        self.terminations.wrong_box.params["box_x"]         = wrong_pos[0]
        self.terminations.wrong_box.params["box_y"]         = wrong_pos[1]
        self.terminations.dropped_elsewhere.params["object_name"] = self.object_shape

        self.rewards.placement.params["object_name"]    = self.object_shape
        self.rewards.placement.params["correct_box_x"]  = correct_pos[0]
        self.rewards.placement.params["correct_box_y"]  = correct_pos[1]
        self.rewards.placement.params["wrong_box_x"]    = wrong_pos[0]
        self.rewards.placement.params["wrong_box_y"]    = wrong_pos[1]

        _print_startup_info(self.object_shape, self.object_color, target_box)

        reset_events = [
            mdp.discrete_location_term(
                self.object_shape,
                locations=_OBJECT_SPAWN_LOCATIONS,
                position_noise_std=0.001,
            ),
        ]
        if self.object_shape != "cube":
            # Park the cube far outside the workspace so it never interferes.
            reset_events.append(
                mdp.discrete_location_term(
                    "cube",
                    locations=[_CUBE_PARKED_LOCATION],
                    position_noise_std=0.0,
                )
            )
        domain_randomization(self, random_options=reset_events)


# ── Per-variant subclasses ────────────────────────────────────────────────────
# Pass --task=LeIsaac-SO101-SortObject-<Shape><Color>-v0 on the CLI to select.
# Only cube variants have a USD in assets/scenes/sort_object/ right now;
# rectangle and cylinder variants are registered for future use.

@configclass
class SortObjectCubeRedEnvCfg(SortObjectEnvCfg):
    object_shape: str = "cube"
    object_color: str = "red"

@configclass
class SortObjectCubeGreenEnvCfg(SortObjectEnvCfg):
    object_shape: str = "cube"
    object_color: str = "green"

@configclass
class SortObjectCubeYellowEnvCfg(SortObjectEnvCfg):
    object_shape: str = "cube"
    object_color: str = "yellow"

@configclass
class SortObjectRectangleRedEnvCfg(SortObjectEnvCfg):
    object_shape: str = "rectangle"
    object_color: str = "red"

@configclass
class SortObjectRectangleBlueEnvCfg(SortObjectEnvCfg):
    object_shape: str = "rectangle"
    object_color: str = "blue"

@configclass
class SortObjectRectangleGreenEnvCfg(SortObjectEnvCfg):
    object_shape: str = "rectangle"
    object_color: str = "green"

@configclass
class SortObjectRectangleYellowEnvCfg(SortObjectEnvCfg):
    object_shape: str = "rectangle"
    object_color: str = "yellow"

@configclass
class SortObjectCylinderRedEnvCfg(SortObjectEnvCfg):
    object_shape: str = "cylinder"
    object_color: str = "red"

@configclass
class SortObjectCylinderBlueEnvCfg(SortObjectEnvCfg):
    object_shape: str = "cylinder"
    object_color: str = "blue"

@configclass
class SortObjectCylinderGreenEnvCfg(SortObjectEnvCfg):
    object_shape: str = "cylinder"
    object_color: str = "green"
