from typing import Dict, List

from enn_zoo.procgen_env.base_env import BaseEnv
from enn_zoo.procgen_env.deserializer import ByteBuffer

# b->write_float(min_dim);
# b->write_float(bullet_vscale);
# b->write_int(last_fire_time);
FRUITBOT_FEATS: List[str] = [
    "min_dim",
    "bullet_vscale",
    "last_fire_time",
]


class FruitBot(BaseEnv):
    def __init__(self, distribution_mode: str = "hard") -> None:
        super().__init__("fruitbot", distribution_mode)

    def _global_feats(self) -> List[str]:
        return FRUITBOT_FEATS

    def deserialize_global_feats(self, data: ByteBuffer) -> List[float]:
        return [
            data.read_float(),
            data.read_float(),
            float(data.read_int()),
        ]

    def _entity_types(self) -> Dict[int, str]:
        # const int BARRIER = 1;
        # const int OUT_OF_BOUNDS_WALL = 2;
        # const int PLAYER_BULLET = 3;
        # const int BAD_OBJ = 4;
        # const int GOOD_OBJ = 7;
        # const int LOCKED_DOOR = 10;
        # const int LOCK = 11;
        # const int PRESENT = 12;
        return {
            1: "Barrier",
            2: "OutOfBoundsWall",
            3: "PlayerBullet",
            4: "BadObj",
            7: "GoodObj",
            10: "LockedDoor",
            11: "Lock",
            12: "Present",
        }
