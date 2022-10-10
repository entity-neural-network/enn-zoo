from typing import Dict, List, Optional

from enn_zoo.procgen_env.base_env import BaseEnv
from enn_zoo.procgen_env.deserializer import ByteBuffer

# b->write_bool(has_support);
# b->write_bool(facing_right);
# b->write_int(last_fire_time);
# b->write_int(wall_theme);
# b->write_float(gravity);
# b->write_float(air_control);
# b->write_float(jump_charge);
# b->write_float(jump_charge_inc);
NINJA_FEATS: List[str] = [
    "has_support",
    "facing_right",
    "last_fire_time",
    "wall_theme",
    "gravity",
    "air_control",
    "jump_charge",
    "jump_charge_inc",
]


class Ninja(BaseEnv):
    def __init__(self, distribution_mode: str = "hard") -> None:
        super().__init__("ninja", distribution_mode)

    def _global_feats(self) -> List[str]:
        return NINJA_FEATS

    def deserialize_global_feats(self, data: ByteBuffer) -> List[float]:
        return [
            float(data.read_int()),
            float(data.read_int()),
            float(data.read_int()),
            float(data.read_int()),
            data.read_float(),
            data.read_float(),
            data.read_float(),
            data.read_float(),
        ]

    def _entity_types(self) -> Dict[int, str]:
        # const int GOAL = 1;
        # const int BOMB = 6;
        # const int THROWING_STAR = 7;
        # const int PLAYER_JUMP = 9;
        # const int PLAYER_RIGHT1 = 12;
        # const int PLAYER_RIGHT2 = 13;
        # const int FIRE = 14;
        # const int WALL_MID = 20;

        return {
            1: "Goal",
            7: "ThrowingStar",
            54: "???",
        }

    def _tile_types(self) -> Optional[Dict[int, str]]:
        return {
            6: "Bomb",
            9: "PlayerJump",
            12: "PlayerRight1",
            13: "PlayerRight2",
            14: "Fire",
            20: "Wall",
            100: "Empty",
        }
