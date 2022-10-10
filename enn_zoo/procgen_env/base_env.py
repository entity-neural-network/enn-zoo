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
        self.chunk_width = 5

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

    def _init_tiles(self) -> None:
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
        tile_types = self._tile_types()
        if tile_types is not None:
            entities["Tiles"] = Entity(
                features=["x", "y"]
                + [
                    f"{x},{y}={tiletype}"
                    for x in range(self.chunk_width)
                    for y in range(self.chunk_width)
                    for tiletype in tile_types.values()
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
        chunk_rows = min(5, state.grid_width // self.chunk_width)
        if self._tile_types() is not None:
            center_x = round(x - 0.5)
            center_y = round(y - 0.5)
            offset = self.chunk_width * chunk_rows // 2
            offset2 = offset if self.chunk_width * chunk_rows % 2 == 0 else offset + 1
            if center_x < offset:
                center_x = offset
            if center_x + offset2 >= state.grid_width:
                center_x = state.grid_width - offset2
            if center_y < offset:
                center_y = offset
            if center_y + offset2 > state.grid_height:
                center_y = state.grid_height - offset2
            tilemap = np.zeros(
                (
                    chunk_rows * self.chunk_width,
                    chunk_rows * self.chunk_width,
                    len(self._tile_idx_to_id),
                ),
                dtype=np.float32,
            )
            subgrid = state.grid[
                center_x - offset : center_x + offset2,
                center_y - offset : center_y + offset2,
            ]
            for idx, id in enumerate(self._tile_idx_to_id):
                tilemap[subgrid == id, idx] = 1.0
            assert (
                tilemap.sum() == tilemap.shape[0] * tilemap.shape[1]
            ), "Not all tiles are accounted for, does `self._tile_types()` contain all ids that are present in `subgrid`?"
            tilemap = (
                tilemap.reshape(
                    chunk_rows,
                    self.chunk_width,
                    chunk_rows,
                    self.chunk_width,
                    len(self._tile_idx_to_id),
                )
                .transpose(0, 2, 1, 3, 4)
                .reshape(
                    chunk_rows * chunk_rows,
                    self.chunk_width * self.chunk_width * len(self._tile_idx_to_id),
                )
            )
            xs = (
                (
                    np.arange(chunk_rows, dtype=np.float32) * self.chunk_width
                    + self.chunk_width / 2
                    + center_x
                    - offset
                )
                .reshape(1, chunk_rows, 1)
                .repeat(chunk_rows, axis=0)
            ).reshape(chunk_rows * chunk_rows, 1)
            ys = (
                (
                    np.arange(chunk_rows, dtype=np.float32) * self.chunk_width
                    + self.chunk_width / 2
                    + center_x
                    - offset
                )
                .reshape(chunk_rows, 1, 1)
                .repeat(chunk_rows, axis=1)
            ).reshape(chunk_rows * chunk_rows, 1)
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
