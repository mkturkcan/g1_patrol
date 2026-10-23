"""Tests for route planning on the tiles and route following (no simulator needed)."""

import math

import numpy as np
import pytest
import torch

from g1_patrol.route import path_at, path_project, polyline, remaining, waypoints
from g1_patrol.tasks.navigation.nyc.indoor import Planner
from g1_patrol.tasks.navigation.nyc.kinds import ALL_BUILDERS, LAB_KINDS, TRAINING_KINDS
from g1_patrol.tasks.navigation.nyc.routing import COARSE_DETOUR, MAX_ROUTE_POINTS, length, plan_routes, without_small

HALF_WIDTH = 0.29
"""The B2-W's half width (m)."""
U_TURN = [(0.0, 0.0), (8.0, 0.0), (8.0, 1.0), (0.0, 1.0)]
"""A route that comes back 1 m beside itself (as around the end of a wall), 17 m long."""


def _route(points):
    return polyline(torch.tensor([points], dtype=torch.float32))


def test_place_follows_the_robot_and_does_not_jump_across():
    """Driving along a U-turn route, the place grows steadily; on the way out it does not jump to the way back, 1 m
    beside it (the window around the last place)."""
    pts, cum = _route(U_TURN)
    s = torch.zeros(1)
    for x in np.linspace(0.0, 4.0, 17):
        s = path_project(pts, cum, torch.tensor([[x, 0.6]]), s)  # nearer the way back than the way out
        assert float(s) == pytest.approx(x, abs=1e-5)
    # without the window, the closest point is on the way back
    assert float(path_project(pts, cum, torch.tensor([[1.0, 0.6]]))) > 9.0


def test_remaining_distance_is_along_the_route_plus_the_offset():
    pts, cum = _route(U_TURN)
    s = torch.tensor([1.0])
    on = remaining(pts, cum, torch.tensor([[1.0, 0.0]]), s)
    off = remaining(pts, cum, torch.tensor([[1.0, -0.5]]), s)
    assert float(on) == pytest.approx(17.0 - 1.0)
    assert float(off) == pytest.approx(16.5)


def test_waypoints_are_ahead_in_the_heading_frame_and_stop_at_the_goal():
    pts, cum = _route(U_TURN)
    look = torch.tensor([1.0, 2.5])
    x, yaw = torch.tensor([[7.0, 0.0]]), torch.tensor([math.pi / 2])  # facing +y at 7 m along the way out
    w = waypoints(pts, cum, torch.tensor([7.0]), x, yaw, look)[0]
    # 1 m ahead: the corner (8, 0) -> 1 m to the robot's right; 2.5 m ahead: (8, 1) + 0.5 m back -> (7.5, 1)
    assert torch.allclose(w[0], torch.tensor([0.0, -1.0]), atol=1e-5)
    assert torch.allclose(w[1], torch.tensor([1.0, -0.5]), atol=1e-5)
    w = waypoints(pts, cum, torch.tensor([16.0]), torch.tensor([[0.5, 1.0]]), torch.tensor([math.pi]), look)[0]
    assert torch.allclose(w, torch.tensor([[0.5, 0.0], [0.5, 0.0]]), atol=1e-5)  # both at the goal


def _layouts(kinds, seeds=range(4), difficulties=(0.0, 0.5, 1.0)):
    for kind in kinds:
        for d in difficulties:
            for seed in seeds:
                yield kind, d, seed, ALL_BUILDERS[kind](d, np.random.default_rng(seed), None)


@pytest.mark.parametrize("kind", TRAINING_KINDS + LAB_KINDS)
def test_planned_routes_keep_clear_and_end_at_the_goal(kind):
    """Every drivable route gets a plan from its start to its goal that keeps the robot's half width from structures
    (but for the last hop to a goal close to one), and a coarse plan that is not much shorter. (Tiles without a
    drivable route, some construction sites, are built again from another seed by the terrain generator.)"""
    drivable = 0
    for _, d, seed, lay in _layouts([kind]):
        keep = plan_routes(lay, HALF_WIDTH)
        drivable += bool(keep)
        plan = Planner(lay, z_max=1.0)
        for k in keep:
            sx, sy, gx, gy = lay.routes[k]
            for points in (lay.route_paths[k], lay.route_paths_coarse[k]):
                assert 2 <= len(points) <= MAX_ROUTE_POINTS
                assert np.allclose(points[0], (sx, sy)) and np.allclose(points[-1], (gx, gy))
            points = lay.route_paths[k]
            route, cum = _route(points)
            # points of the route every 5 cm, but the last 0.45 m, keep the half width (one grid cell of slack)
            s = torch.arange(0.0, float(cum[0, -1]) - 0.45, 0.05)
            xy, _ = path_at(route.expand(len(s), -1, -1), cum.expand(len(s), -1), s)
            clear = np.array([plan.clearance(p) for p in xy.numpy()])
            assert clear.min() > HALF_WIDTH - 0.1, f"{kind} d={d} seed={seed} route {k}: {clear.min():.2f} m"
            assert length(points) <= COARSE_DETOUR[0] * length(lay.route_paths_coarse[k]) + COARSE_DETOUR[1] + 1e-6
    assert drivable >= 9, f"{kind}: {drivable} of 12 tiles drivable"


def test_coarse_plans_leave_out_only_small_structures():
    lay = ALL_BUILDERS["makerlab"](1.0, np.random.default_rng(4), None)
    small = without_small(lay)
    assert len(small.boxes) < len(lay.boxes) or len(small.circles) < len(lay.circles)
    walls = [b for b in lay.boxes if max(b[3], b[4]) > 2.0]
    assert all(w in small.boxes for w in walls)
