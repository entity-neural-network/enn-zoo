from abc import abstractmethod
from typing import Dict, List, Mapping, Optional

import numpy as np
from entity_gym.env import *
from procgen import ProcgenGym3Env

from enn_zoo.procgen_env.deserializer import ByteBuffer
from enn_zoo.procgen_env.fast_deserializer import MinimalProcgenState

ENTITY_FEATS = [
    "x",
    "y",
    "vx",
    "vy",
    "rx",
    "ry",
    "type",
    "image_type",
    "image_theme",
    "render_z",
    "will_erase",
    "collides_with_entities",
    "collision_margin",
    "rotation",
    "vrot",
    "is_reflected",
    "fire_time",
    "spawn_time",
    "life_time",
    "expire_time",
    "use_abs_coords",
    "friction",
    "smart_step",
    "avoids_collisions",
    "auto_erase",
    "alpha",
    "health",
    "theta",
    "grow_rate",
    "alpha_decay",
    "climber_spawn_x",
]


class BaseEnv(Environment):
    def __init__(self, env_name: str, distribution_mode: str) -> None:
        self.env = ProcgenGym3Env(
            num=1,
            env_name=env_name,
            start_level=0,
            num_levels=0,
            distribution_mode=distribution_mode,
        )
        self._init_tiles()

    @abstractmethod
    def _global_feats(self) -> List[str]:
        pass

    @abstractmethod
    def deserialize_global_feats(self, data: ByteBuffer) -> List[float]:
        pass

    @abstractmethod
    def _entity_types(self) -> Dict[int, str]:
        pass

    def _tile_types(self) -> Optional[Dict[int, str]]:
        return None

    def _init_tiles(self):
        tile_types = self._tile_types()
        if tile_types is None:
            return
        self._tile_idx_to_id = list(sorted(tile_types.keys()))

    def obs_space(self) -> ObsSpace:
        entities = {
            "Player": Entity(features=ENTITY_FEATS + self._global_feats()),
            **{
                entity_type: Entity(features=ENTITY_FEATS + self._global_feats())
                for entity_type in self._entity_types().values()
            },
        }
        if self._tile_types() is not None:
            entities["Tiles"] = Entity(
                features=["x", "y"]
                + [
                    f"{x},{y}={tiletype}"
                    for x in range(5)
                    for y in range(5)
                    for tiletype in self._tile_types().values()
                ]
            )
        return ObsSpace(entities=entities)

    def action_space(self) -> Dict[ActionName, ActionSpace]:
        return {
            "act": CategoricalActionSpace(
                [
                    "left-down",
                    "left",
                    "left-up",
                    "down",
                    "none",
                    "up",
                    "right-down",
                    "right",
                    "right-up",
                    "d",
                    "a",
                    "w",
                    "s",
                    "q",
                    "e",
                ]
            )
        }

    def reset(self) -> Observation:
        return self.observe()

    def observe(self) -> Observation:
        states = self.env.callmethod("get_state")
        data = ByteBuffer(states[0])
        state = MinimalProcgenState.from_bytes(data)

        global_feats = np.array(
            self.deserialize_global_feats(data), dtype=np.float32
        ).reshape(1, -1)
        player_feats = state.entities[state.entities[:, 6] == 0.0]
        entities = {
            "Player": (
                np.concatenate([player_feats, global_feats], axis=1),
                [0],
            )
        }
        x = player_feats[0, 0]
        y = player_feats[0, 1]
        total = 1
        for type_id, name in self._entity_types().items():
            feats = state.entities[state.entities[:, 6] == type_id]
            if feats.shape[0] > 0:
                feats = np.concatenate(
                    [feats, global_feats.repeat(feats.shape[0], axis=0)],
                    axis=1,
                )
                total += feats.shape[0]
            else:
                feats = np.zeros(
                    (0, feats.shape[1] + global_feats.shape[1]), dtype=np.float32
                )
            entities[name] = feats
        if self._tile_types() is not None:
            center_x = round(x - 0.5)
            center_y = round(y - 0.5)
            if center_x < 12:
                center_x = 12
            if center_x + 12 > state.grid_width:
                center_x = state.grid_width - 12
            if center_y < 12:
                center_y = 12
            if center_y + 12 > state.grid_height:
                center_y = state.grid_height - 12
            tilemap = np.zeros((25, 25, len(self._tile_idx_to_id)), dtype=np.float32)
            subgrid = state.grid[
                center_x - 12 : center_x + 13, center_y - 12 : center_y + 13
            ]
            for idx, id in enumerate(self._tile_idx_to_id):
                tilemap[subgrid == id, idx] = 1.0
            assert (
                tilemap.sum() == tilemap.shape[0] * tilemap.shape[1]
            ), "Not all tiles are accounted for, does _tile_types() contain all ids?"
            tilemap = (
                tilemap.reshape(5, 5, 5, 5, len(self._tile_idx_to_id))
                .transpose(0, 2, 1, 3, 4)
                .reshape(25, 25 * len(self._tile_idx_to_id))
            )
            xs = (
                (np.arange(5, dtype=np.float32) * 5 + 2.5 + center_x - 12)
                .reshape(1, 5, 1)
                .repeat(5, axis=0)
            ).reshape(25, 1)
            ys = (
                (np.arange(5, dtype=np.float32) * 5 + 2.5 + center_x - 12)
                .reshape(5, 1, 1)
                .repeat(5, axis=1)
            ).reshape(25, 1)
            tile_entities = np.concatenate([xs, ys, tilemap], axis=1)
            entities["Tiles"] = tile_entities
        assert (
            total == state.entities.shape[0]
        ), f"Not all entities were accounted for: self._entity_types.keys()={self._entity_types().keys()}, state.entities[:, 6]={state.entities[:, 6]}"
        return Observation(
            entities=entities,
            actions={"act": CategoricalActionMask(actor_types=["Player"])},
            done=state.step_data.done == 1,
            reward=state.step_data.reward,
        )

    def act(self, actions: Mapping[ActionName, Action]) -> Observation:
        act = actions["act"]
        assert isinstance(act, CategoricalAction)
        self.env.act(act.indices)
        return self.observe()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--get-state", action="store_true")
    parser.add_argument("--parse-state", action="store_true")
    parser.add_argument("--num-envs", type=int, default=1)
    args = parser.parse_args()

    env = ProcgenGym3Env(
        num=args.num_envs,
        env_name="bigfish",
        start_level=0,
        num_levels=0,
        distribution_mode="easy",
    )

    import time

    trials = 3
    samples = 50000 // args.num_envs
    for trial in range(trials):
        print(f"Trial {trial}")
        start = time.time()
        for i in range(samples):
            env.act(np.zeros(args.num_envs, dtype=np.int32))
            if args.get_state:
                states = env.callmethod("get_state")
                if args.parse_state:
                    data = ByteBuffer(states[0])
                    state = MinimalProcgenState.from_bytes(data)
        print(
            f"Trial {trial} took {time.time() - start:.2f} sec. Throughput: {samples * args.num_envs / (time.time() - start):.0f} samples/sec"
        )
