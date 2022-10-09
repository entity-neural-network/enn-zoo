from typing import Dict, List, Optional

from enn_zoo.procgen_env.base_env import BaseEnv
from enn_zoo.procgen_env.deserializer import ByteBuffer

CAVE_FLYER_FEATS: List[str] = []


class CaveFlyer(BaseEnv):
    def __init__(self, distribution_mode: str = "hard") -> None:
        super().__init__("caveflyer", distribution_mode)

    def _global_feats(self) -> List[str]:
        return CAVE_FLYER_FEATS

    def deserialize_global_feats(self, data: ByteBuffer) -> List[float]:
        return []

    def _entity_types(self) -> Dict[int, str]:
        return {
            1: "Goal",
            2: "Obstacle",
            3: "Target",
            4: "PlayerBullet",
            5: "Enemy",
            9: "Exhaust",
            54: "???",
            1003: "Marker",
        }

    def _tile_types(self) -> Optional[Dict[int, str]]:
        return {
            8: "CaveWall",
            100: "Empty",
        }
