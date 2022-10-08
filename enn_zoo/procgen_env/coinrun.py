from typing import Dict, List

from enn_zoo.procgen_env.base_env import BaseEnv
from enn_zoo.procgen_env.deserializer import ByteBuffer

# b->write_float(last_agent_y);
# b->write_int(wall_theme);
# b->write_bool(has_support);
# b->write_bool(facing_right);
# b->write_bool(is_on_crate);
# b->write_float(gravity);
# b->write_float(air_control);

# TODO: should all of these be exposed to policy?
COINRUN_FEATS = [
    "last_agent_y",
    "wall_theme",
    "has_support",
    "facing_right",
    "is_on_crate",
    "gravity",
    "air_control",
]

class CoinRun(BaseEnv):
    def __init__(self, distribution_mode: str = "hard") -> None:
        super().__init__("coinrun", distribution_mode)

    def _global_feats(self) -> List[str]:
        return COINRUN_FEATS

    def deserialize_global_feats(self, data: ByteBuffer) -> List[float]:
        feats = [
            data.read_float(),
            float(data.read_int()),
            float(data.read_int()),
            float(data.read_int()),
            float(data.read_int()),
            data.read_float(),
            data.read_float(),
        ]
        return feats

    def _entity_types(self) -> Dict[int, str]:
        # const int GOAL = 1;
        # const int SAW = 2;
        # const int SAW2 = 3;
        # const int ENEMY = 5;
        # const int ENEMY1 = 6;
        # const int ENEMY2 = 7;
        return {
            1: "Goal",
            2: "Saw",
            3: "Saw2",
            5: "Enemy",
            6: "Enemy1",
            7: "Enemy2",
            20: "Crate",
            59: "???",
        }
