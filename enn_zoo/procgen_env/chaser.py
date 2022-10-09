from typing import Dict, List, Optional

from enn_zoo.procgen_env.base_env import BaseEnv
from enn_zoo.procgen_env.deserializer import ByteBuffer

# b->write_vector_int(free_cells);
# b->write_vector_bool(is_space_vec);
# b->write_int(eat_timeout);
# b->write_int(egg_timeout);
# b->write_int(eat_time);
# b->write_int(total_enemies);
# b->write_int(total_orbs);
# b->write_int(orbs_collected);
# b->write_int(maze_dim);
CHASER_FEATS: List[str] = [
    # "free_cells",
    # "is_space_vec",
    "eat_timeout",
    "egg_timeout",
    "eat_time",
    "total_enemies",
    "total_orbs",
    "orbs_collected",
    "maze_dim",
]


class Chaser(BaseEnv):
    def __init__(self, distribution_mode: str = "hard") -> None:
        super().__init__("chaser", distribution_mode)
        self.chunk_width = 4

    def _global_feats(self) -> List[str]:
        return CHASER_FEATS

    def deserialize_global_feats(self, data: ByteBuffer) -> List[float]:
        data.read_int_array()
        data.read_int_array()
        return [
            float(data.read_int()),
            float(data.read_int()),
            float(data.read_int()),
            float(data.read_int()),
            float(data.read_int()),
            float(data.read_int()),
            float(data.read_int()),
        ]

    def _entity_types(self) -> Dict[int, str]:
        # const int LARGE_ORB = 2;
        # const int ENEMY_WEAK = 3;
        # const int ENEMY_EGG = 4;
        # const int MAZE_WALL = 5;
        # const int ENEMY = 6;
        # const int ENEMY2 = 7;
        # const int ENEMY3 = 8;

        # const int MARKER = 1001;
        # const int ORB = 1002;
        return {
            2: "LargeOrb",
            3: "EnemyWeak",
            4: "EnemyEgg",
            6: "Enemy",
            7: "Enemy2",
            8: "Enemy3",
            1001: "Marker",
        }

    def _tile_types(self) -> Optional[Dict[int, str]]:
        return {
            5: "MazeWall",
            100: "Empty",
            1002: "Orb",
        }
