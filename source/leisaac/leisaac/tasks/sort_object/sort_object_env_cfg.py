from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from leisaac.assets.scenes.sort_object import SORT_OBJECT_SCENE_CFG, SORT_OBJECT_USD_PATH
from leisaac.utils.domain_randomization import domain_randomization, randomize_object_uniform
from leisaac.utils.general_assets import parse_usd_and_create_subassets

from ..template import (
    SingleArmObservationsCfg,
    SingleArmTaskEnvCfg,
    SingleArmTaskSceneCfg,
    SingleArmTerminationsCfg,
)
from . import mdp

_BOX_USD_PATH = str(Path(SORT_OBJECT_USD_PATH).parent / "box" / "box.usd")

# Five candidate spawn positions in env-local frame (x, y, z).
# Table surface z ≈ 0.04 m; cube rests at z ≈ 0.062 m.
# Box A (box)  centre: (0.478, -0.310) footprint ±0.075 m → x ∈ [0.403, 0.553], y ∈ [-0.385, -0.235]
# Box B (box2) centre: (0.478, -0.120) footprint ±0.075 m → x ∈ [0.403, 0.553], y ∈ [-0.195, -0.045]
# All locations below fall outside both footprints.
_OBJECT_SPAWN_LOCATIONS: list[tuple[float, float, float]] = [
    (0.28, -0.31, 0.062),  # left of both boxes
    (0.25, -0.18, 0.062),  # left, between the two boxes in y
    (0.20, -0.42, 0.062),  # left, behind Box A
    (0.35, -0.15, 0.062),  # left, in front of Box B
    (0.35, -0.48, 0.062),  # left, well behind Box A
]

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
    print(f"[SortObject]  Pickup locations (5, noise std = 0.005 m):")
    for i, loc in enumerate(_OBJECT_SPAWN_LOCATIONS, 1):
        print(f"[SortObject]    {i}. x={loc[0]:.3f}  y={loc[1]:.3f}  z={loc[2]:.3f}")
    print(f"[SortObject] {sep}\n")


@configclass
class SortObjectSceneCfg(SingleArmTaskSceneCfg):
    """Scene configuration for the sort-object task."""

    scene: AssetBaseCfg = SORT_OBJECT_SCENE_CFG.replace(prim_path="{ENV_REGEX_NS}/Scene")

    # Area B: spawned from the same box USD at a new prim path.
    # Area A (box) is extracted from scene.usd by parse_usd_and_create_subassets.
    # Box A centre: (0.478, -0.310); Box B at (0.478, -0.120) → ~4 cm gap.
    box2: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/box2",
        spawn=sim_utils.UsdFileCfg(usd_path=_BOX_USD_PATH),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.478, -0.120, 0.046),
        ),
    )


@configclass
class TerminationsCfg(SingleArmTerminationsCfg):
    """Terminates successfully when the object is placed in the correct target box.

    Both box_cfg and object_cfg are overridden in SortObjectEnvCfg.__post_init__
    based on object_shape and object_color.
    """

    success = DoneTerm(
        func=mdp.object_in_box,
        params={
            "object_cfg": SceneEntityCfg("cube"),  # replaced in __post_init__
            "box_cfg":    SceneEntityCfg("box"),    # replaced in __post_init__
            # box footprint ±0.075 m; ±0.065 requires a reasonably centred placement
            "x_range": (-0.065, 0.065),
            "y_range": (-0.065, 0.065),
            # box top ≈ 0.046 + 0.050 = 0.096 m; object resting inside ≈ 0.062 m
            "height_threshold": 0.10,
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

        target_box = _sorting_target(shape, color)
        self.terminations.success.params["box_cfg"]    = SceneEntityCfg(target_box)
        self.terminations.success.params["object_cfg"] = SceneEntityCfg(shape)

        # Update the object-reset event term (domain_randomize_0) to the new prim name.
        if hasattr(self.events, "domain_randomize_0"):
            self.events.domain_randomize_0.params["asset_cfg"] = SceneEntityCfg(shape)

        _print_startup_info(shape, color, target_box)

    def __post_init__(self) -> None:
        super().__post_init__()

        self.viewer.eye = (-0.2, -1.0, 0.5)
        self.viewer.lookat = (0.6, 0.0, -0.2)

        self.scene.robot.init_state.pos = (0.35, -0.64, 0.01)

        # Populates scene.cube, scene.box (and scene.counter_right_main_group) from scene.usd.
        parse_usd_and_create_subassets(SORT_OBJECT_USD_PATH, self)

        # Wire the termination to the correct target box and object prim.
        target_box = _sorting_target(self.object_shape, self.object_color)
        self.terminations.success.params["box_cfg"]    = SceneEntityCfg(target_box)
        self.terminations.success.params["object_cfg"] = SceneEntityCfg(self.object_shape)

        _print_startup_info(self.object_shape, self.object_color, target_box)

        domain_randomization(
            self,
            random_options=[
                # Object spawns at one of five fixed positions + small noise each episode.
                mdp.discrete_location_term(
                    self.object_shape,
                    locations=_OBJECT_SPAWN_LOCATIONS,
                    position_noise_std=0.005,
                ),
                # Both boxes get a small uniform jitter so the robot cannot memorise
                # exact target positions.
                randomize_object_uniform(
                    "box",
                    pose_range={"x": (-0.02, 0.02), "y": (-0.02, 0.02), "z": (0.0, 0.0)},
                ),
                randomize_object_uniform(
                    "box2",
                    pose_range={"x": (-0.02, 0.02), "y": (-0.02, 0.02), "z": (0.0, 0.0)},
                ),
            ],
        )


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
