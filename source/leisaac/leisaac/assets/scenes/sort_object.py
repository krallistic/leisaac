from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from leisaac.utils.constant import ASSETS_ROOT

SCENES_ROOT = Path(ASSETS_ROOT) / "scenes"

SORT_OBJECT_USD_PATH = str(SCENES_ROOT / "sort_object" / "scene.usd")

SORT_OBJECT_SCENE_CFG = AssetBaseCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=SORT_OBJECT_USD_PATH,
    )
)
