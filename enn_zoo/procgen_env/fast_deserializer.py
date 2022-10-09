from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from enn_zoo.procgen_env.deserializer import ByteBuffer, Entity, ProcgenGame, StepData


@dataclass
class MinimalProcgenState:
    step_data: StepData
    entities: npt.NDArray[np.float32]
    grid_size: int
    grid_width: int
    grid_height: int
    grid: npt.NDArray[np.int32]

    @classmethod
    def from_bytes(cls, data: ByteBuffer) -> "MinimalProcgenState":
        step_data = ProcgenGame.just_step_data(data)
        grid_size = data.read_int()
        entities = Entity.array_from_bytes(data)
        # use_procgen_background, background_index, bg_tile_ratio, bg_pct_x, char_dim, last_move_action, move_action, special_action, mixrate, maxspeed, max_jump, action_vx, action_vy, action_vrot, center_x, center_y, random_agent_start, has_useful_vel_info, step_rand_int
        data.offset += 4 * 19
        # asset_rand_gen
        data.offset += 4
        data.read_array(elem_size=1)
        # main_width, main_height, out_of_bounds_object, unit, view_dim, x_off, y_off, visibility, min_visibility
        data.offset += 4 * 9
        # grid
        w = data.read_int()
        h = data.read_int()
        grid = data.read_int_array().reshape((w, h))
        size = grid.shape[0] * grid.shape[1]
        assert w * h == size == grid_size
        return cls(
            step_data=step_data,
            entities=entities,
            grid_size=grid_size,
            grid_width=w,
            grid_height=h,
            grid=grid,
        )
