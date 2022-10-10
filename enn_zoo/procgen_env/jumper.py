from typing import Dict, List, Optional

from enn_zoo.procgen_env.base_env import BaseEnv
from enn_zoo.procgen_env.deserializer import ByteBuffer

# b->write_int(jump_count);
# b->write_int(jump_delta);
# b->write_int(jump_time);
# b->write_bool(has_support);
# b->write_bool(facing_right);
# b->write_int(wall_theme);
# b->write_float(compass_dim);
JUMPER_FEATS: List[str] = [
    "jump_count",
    "jump_delta",
    "jump_time",
    "has_support",
    "facing_right",
    "wall_theme",
    "compass_dim",
]


class Jumper(BaseEnv):
    def __init__(self, distribution_mode: str = "hard") -> None:
        super().__init__("jumper", distribution_mode)

    def _global_feats(self) -> List[str]:
        return JUMPER_FEATS

    def deserialize_global_feats(self, data: ByteBuffer) -> List[float]:
        return [
            float(data.read_int()),
            float(data.read_int()),
            float(data.read_int()),
            float(data.read_int()),
            float(data.read_int()),
            float(data.read_int()),
            data.read_float(),
        ]

    def _entity_types(self) -> Dict[int, str]:
        # const int GOAL = 1;
        # const int SPIKE = 2;
        # const int CAVEWALL = 6;
        # const int CAVEWALL_TOP = 7;
        return {
            1: "Goal",
            2: "Spike",
            59: "???",
        }

    def _tile_types(self) -> Optional[Dict[int, str]]:
        return {
            6: "CaveWall",
            7: "CaveWallTop",
            100: "Empty",
        }
