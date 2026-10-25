"""Tests for planning on the SLAM map (no ROS needed)."""

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "deploy"))

from b2w.global_planner import plan_on_grid  # noqa: E402

RES = 0.05


def _lab():
    """A 20 x 12 m room at 5 cm (rows along y), a wall at x = 10 m with a 1.0 m door at y = 3-4 m, a bench."""
    grid = np.zeros((240, 400), dtype=np.int8)
    grid[[0, -1], :] = 100
    grid[:, [0, -1]] = 100
    grid[:, 199:201] = 100
    grid[60:80, 199:201] = 0  # door from y = 3 to 4 m
    grid[150:170, 60:140] = 100  # bench
    grid[200:, 300:] = -1  # an unexplored corner
    return grid


def _clearance(grid, path, origin):
    occ = np.argwhere(grid >= 50) * RES + RES / 2
    pts = np.concatenate([np.linspace(p, q, 50) for p, q in zip(path[:-1], path[1:])])
    d = np.min(np.hypot(pts[:, None, 0] - origin[0] - occ[None, :, 1], pts[:, None, 1] - origin[1] - occ[None, :, 0]), axis=1)
    return d


def test_route_goes_through_the_door_with_clearance():
    grid, origin = _lab(), (0.0, 0.0)
    t = time.perf_counter()
    path = plan_on_grid(grid, origin, RES, np.array([3.0, 9.0]), np.array([16.0, 2.0]))
    took = time.perf_counter() - t
    assert path is not None and np.allclose(path[0], (3.0, 9.0)) and np.allclose(path[-1], (16.0, 2.0))
    crossing = [p for p, q in zip(path[:-1], path[1:]) if (p[0] - 10) * (q[0] - 10) <= 0][0]
    d = _clearance(grid, path, origin)
    assert d.min() > 0.29 - 0.08  # keeps the robot's half width (a coarse cell of slack)
    assert 3.0 < np.interp(10.0, [crossing[0], path[path.index(crossing) + 1][0]],
                           [crossing[1], path[path.index(crossing) + 1][1]]) < 4.0  # through the door
    assert took < 2.0


def test_unknown_space_and_a_goal_beyond_the_map_are_planned_into():
    grid, origin = _lab(), (0.0, 0.0)
    path = plan_on_grid(grid, origin, RES, np.array([12.0, 6.0]), np.array([18.5, 11.0]))  # into the unexplored corner
    assert path is not None
    path = plan_on_grid(grid, origin, RES, np.array([12.0, 6.0]), np.array([25.0, 6.0]))  # beyond the map's edge
    assert path is not None and np.allclose(path[-1], (25.0, 6.0))


def test_noise_at_the_robot_does_not_trap_it_and_a_closed_room_has_no_route():
    grid, origin = _lab(), (0.0, 0.0)
    grid[178:182, 58:62] = 100  # an occupied cell where the robot stands (itself, or noise)
    assert plan_on_grid(grid, origin, RES, np.array([3.0, 9.0]), np.array([5.0, 3.0])) is not None
    grid[60:80, 199:201] = 100  # the door closed
    assert plan_on_grid(grid, origin, RES, np.array([3.0, 9.0]), np.array([16.0, 2.0]), margin=3.0) is None
