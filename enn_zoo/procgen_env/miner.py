from typing import Dict, List, Optional

from enn_zoo.procgen_env.base_env import BaseEnv
from enn_zoo.procgen_env.deserializer import ByteBuffer

# b->write_int(diamonds_remaining);
MINER_FEATS: List[str] = [
    "diamonds_remaining",
]


class Miner(BaseEnv):
    def __init__(self, distribution_mode: str = "hard") -> None:
        super().__init__("miner", distribution_mode)

    def _global_feats(self) -> List[str]:
        return MINER_FEATS

    def deserialize_global_feats(self, data: ByteBuffer) -> List[float]:
        return [
            float(data.read_int()),
        ]

    def _entity_types(self) -> Dict[int, str]:
        # const int BOULDER = 1;
        # const int DIAMOND = 2;
        # const int MOVING_BOULDER = 3;
        # const int MOVING_DIAMOND = 4;
        # const int ENEMY = 5;
        # const int EXIT = 6;
        # const int DIRT = 9;
        return {
            5: "Enemy",
            6: "Exit",
        }

    def _tile_types(self) -> Optional[Dict[int, str]]:
        return {
            1: "Boulder",
            2: "Diamond",
            3: "MovingBoulder",
            4: "MovingDiamond",
            9: "Dirt",
            100: "Empty",
        }
