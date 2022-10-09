from typing import Dict, List

from enn_zoo.procgen_env.base_env import BaseEnv
from enn_zoo.procgen_env.deserializer import ByteBuffer

# b->write_float(min_dim);
# b->write_float(hard_min_dim);
# b->write_float(ball_vscale);
# b->write_float(ball_r);
# b->write_int(last_fire_time);
# b->write_int(num_enemies);
# b->write_int(enemy_fire_delay);
DODGEBALL_FEATS: List[str] = [
    "min_dim",
    "hard_min_dim",
    "ball_vscale",
    "ball_r",
    "last_fire_time",
    "num_enemies",
    "enemy_fire_delay",
]


class Dodgeball(BaseEnv):
    def __init__(self, distribution_mode: str = "hard") -> None:
        super().__init__("dodgeball", distribution_mode)

    def _global_feats(self) -> List[str]:
        return DODGEBALL_FEATS

    def deserialize_global_feats(self, data: ByteBuffer) -> List[float]:
        return [
            data.read_float(),
            data.read_float(),
            data.read_float(),
            data.read_float(),
            float(data.read_int()),
            float(data.read_int()),
            float(data.read_int()),
        ]

    def _entity_types(self) -> Dict[int, str]:
        # const int LAVA_WALL = 1;
        # const int PLAYER_BALL = 3;
        # const int ENEMY = 4;
        # const int DOOR = 5;
        # const int ENEMY_BALL = 6;
        # const int DOOR_OPEN = 7;
        # const int DUST_CLOUD = 8;

        # const int OOB_WALL = 10;
        return {
            1: "LavaWall",
            3: "PlayerBall",
            4: "Enemy",
            5: "Door",
            6: "EnemyBall",
            7: "DoorOpen",
            8: "DustCloud",
            10: "OOBWall",
        }
