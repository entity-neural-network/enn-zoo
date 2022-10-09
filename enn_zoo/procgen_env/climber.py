from typing import Dict, List, Optional

from enn_zoo.procgen_env.base_env import BaseEnv
from enn_zoo.procgen_env.deserializer import ByteBuffer

# b->write_bool(has_support);
# b->write_bool(facing_right);
# b->write_int(coin_quota);
# b->write_int(coins_collected);
# b->write_int(wall_theme);
# b->write_float(gravity);
# b->write_float(air_control);
CLIMBER_FEATS: List[str] = [
    "has_support",
    "facing_right",
    "coin_quota",
    "coins_collected",
    "wall_theme",
    "gravity",
    "air_control",
]


class Climber(BaseEnv):
    def __init__(self, distribution_mode: str = "hard") -> None:
        super().__init__("climber", distribution_mode)

    def _global_feats(self) -> List[str]:
        return CLIMBER_FEATS

    def deserialize_global_feats(self, data: ByteBuffer) -> List[float]:
        return [
            float(data.read_int()),
            float(data.read_int()),
            float(data.read_int()),
            float(data.read_int()),
            float(data.read_int()),
            data.read_float(),
            data.read_float(),
        ]

    def _entity_types(self) -> Dict[int, str]:
        # const int COIN = 1;
        # const int ENEMY = 5;
        # const int ENEMY1 = 6;
        # const int ENEMY2 = 7;

        # const int PLAYER_JUMP = 9;
        # const int PLAYER_RIGHT1 = 12;
        # const int PLAYER_RIGHT2 = 13;

        # const int WALL_MID = 15;
        # const int WALL_TOP = 16;
        # const int ENEMY_BARRIER = 19;
        return {
            1: "Coin",
            5: "Enemy",
            6: "Enemy1",
            7: "Enemy2",
        }

    def _tile_types(self) -> Optional[Dict[int, str]]:
        return {
            15: "WallMid",
            16: "WallTop",
            19: "EnemyBarrier",
            100: "Empty",
        }
