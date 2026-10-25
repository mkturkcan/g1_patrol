"""Planning routes on the map SLAM builds while the robot drives (numpy only; used by ``planner_node.py``).

The map is an occupancy grid (``nav_msgs/OccupancyGrid``: 0-100 occupied, -1 unknown) that grows as the robot
explores. A route from the robot to the goal is planned with the planner of the training routes
(:func:`g1_patrol.planning.best_path`): it keeps the largest of 0.6, 0.5 and 0.4 m from obstacles where that costs
little detour, the robot's half width where it must. Unknown cells count as free, so that routes lead into unexplored
space towards a goal; they are planned again as the map fills in. The grid is cut to the area around the robot and
the goal, with ``margin`` m around both, and coarsened to ``resolution`` (any occupied cell makes the coarse cell
occupied).
"""

from __future__ import annotations

import math

import numpy as np

from g1_patrol.planning import best_path, occupancy_planner


def plan_on_grid(grid: np.ndarray, origin: tuple[float, float], map_res: float, start, goal, half_width: float = 0.29,
                 resolution: float = 0.1, margin: float = 6.0, occupied: int = 50, free_radius: float = 0.7,
                 reach: float = 0.6) -> list | None:
    """Route (points) from ``start`` to ``goal`` (map frame) on the occupancy ``grid`` (rows along y, columns along
    x, as in ``OccupancyGrid.data``) whose cell (0, 0) has its corner at ``origin``; None if there is none."""
    rows, cols = grid.shape
    # the planning window: robot and goal with a margin, on the coarse grid aligned with the map's cells
    lo = np.minimum(start, goal) - margin
    hi = np.maximum(start, goal) + margin
    f = max(1, int(round(resolution / map_res)))
    res = f * map_res
    c0 = int(math.floor((lo[0] - origin[0]) / res))
    r0 = int(math.floor((lo[1] - origin[1]) / res))
    nc = int(math.ceil((hi[0] - origin[0]) / res)) - c0
    nr = int(math.ceil((hi[1] - origin[1]) / res)) - r0
    occ = np.zeros((nr, nc), dtype=bool)  # outside the map: unknown, counted as free
    # the map's fine cells inside the window, max-pooled into coarse cells
    fr0, fc0 = r0 * f, c0 * f
    a0, a1 = max(fr0, 0), min(fr0 + nr * f, rows)
    b0, b1 = max(fc0, 0), min(fc0 + nc * f, cols)
    if a1 > a0 and b1 > b0:
        fine = np.zeros((nr * f, nc * f), dtype=bool)
        fine[a0 - fr0:a1 - fr0, b0 - fc0:b1 - fc0] = grid[a0:a1, b0:b1] >= occupied
        occ = fine.reshape(nr, f, nc, f).any(axis=(1, 3))
    x0, y0 = origin[0] + c0 * res, origin[1] + r0 * res
    plan = occupancy_planner(occ.T, x0, y0, res, free_around=[tuple(start)], free_radius=free_radius)
    return best_path(plan, tuple(map(float, start)), tuple(map(float, goal)), half_width, reach, max_points=10**6)
