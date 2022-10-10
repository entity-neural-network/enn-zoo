from typing import Dict, List, Optional

from enn_zoo.procgen_env.base_env import BaseEnv
from enn_zoo.procgen_env.deserializer import ByteBuffer

# b->write_int(maze_dim);
# b->write_int(world_dim);
MAZE_FEATS: List[str] = [
    "maze_dim",
    "world_dim",
]


class Maze(BaseEnv):
    def __init__(self, distribution_mode: str = "hard") -> None:
        super().__init__("maze", distribution_mode)

    def _global_feats(self) -> List[str]:
        return MAZE_FEATS

    def deserialize_global_feats(self, data: ByteBuffer) -> List[float]:
        return [
            float(data.read_int()),
            float(data.read_int()),
        ]

    def _entity_types(self) -> Dict[int, str]:
        # const int GOAL = 2;
        return {}

    def _tile_types(self) -> Optional[Dict[int, str]]:
        return {
            2: "Goal",
            51: "Wall",
            100: "Empty",
        }
