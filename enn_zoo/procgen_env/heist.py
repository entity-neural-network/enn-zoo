from typing import Dict, List, Optional

from enn_zoo.procgen_env.base_env import BaseEnv
from enn_zoo.procgen_env.deserializer import ByteBuffer

# b->write_int(num_keys);
# b->write_int(world_dim);
# b->write_vector_bool(has_keys);
HEIST_FEATS: List[str] = [
    "num_keys",
    "world_dim",
    # "has_keys",
]


class Heist(BaseEnv):
    def __init__(self, distribution_mode: str = "hard") -> None:
        super().__init__("heist", distribution_mode)

    def _global_feats(self) -> List[str]:
        return HEIST_FEATS

    def deserialize_global_feats(self, data: ByteBuffer) -> List[float]:
        return [
            float(data.read_int()),
            float(data.read_int()),
        ]

    def _entity_types(self) -> Dict[int, str]:
        # const int LOCKED_DOOR = 1;
        # const int KEY = 2;
        # const int EXIT = 9;
        # const int KEY_ON_RING = 11;
        return {
            1: "LockedDoor",
            2: "Key",
            9: "Exit",
            11: "KeyOnRing",
        }

    def _tile_types(self) -> Optional[Dict[int, str]]:
        return {
            51: "Wall",
            100: "Empty",
        }
