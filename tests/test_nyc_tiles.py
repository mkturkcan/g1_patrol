"""Tests for the New York street tiles (pure numpy, no simulator needed)."""

import math
from collections import deque

import numpy as np
import pytest

from g1_patrol.tasks.navigation.nyc.tiles import (
    BUILDERS,
    HALF_L,
    HALF_W,
    MAX_BOXES,
    MAX_CIRCLES,
    MAX_FORBIDDEN,
    MAX_PATH_POINTS,
    MAX_PATHS,
    MAX_ROUTES,
    MAX_STANDING,
    OVERHEAD_Z,
    TILE_SIZE,
)

ROBOT_CLEARANCE = 0.3
"""Robot radius (0.25 m) plus a little room for walking sway."""
RES = 0.05


def _free_space(lay) -> np.ndarray:
    """Cells (x, y) the robot's center can occupy: away from structures it could hit and out of deep holes."""
    xs = np.arange(-TILE_SIZE[0] / 2, TILE_SIZE[0] / 2, RES) + RES / 2
    ys = np.arange(-TILE_SIZE[1] / 2, TILE_SIZE[1] / 2, RES) + RES / 2
    px, py = np.meshgrid(xs, ys, indexing="ij")
    free = np.ones(px.shape, dtype=bool)
    for x, y, r, z0, _ in lay.circles:
        if z0 < OVERHEAD_Z:
            free &= np.hypot(px - x, py - y) > r + ROBOT_CLEARANCE
    for cx, cy, yaw, hx, hy, z0, _ in lay.boxes:
        if z0 < OVERHEAD_Z:
            c, s = math.cos(yaw), math.sin(yaw)
            lx = c * (px - cx) + s * (py - cy)
            ly = -s * (px - cx) + c * (py - cy)
            dx = np.maximum(np.abs(lx) - hx, 0.0)
            dy = np.maximum(np.abs(ly) - hy, 0.0)
            free &= np.hypot(dx, dy) > ROBOT_CLEARANCE
    # stairwells and other drops
    hi = np.clip(((px + TILE_SIZE[0] / 2) / 0.1).round().astype(int), 0, lay.heights.shape[0] - 1)
    hj = np.clip(((py + TILE_SIZE[1] / 2) / 0.1).round().astype(int), 0, lay.heights.shape[1] - 1)
    free &= lay.heights[hi, hj] > -0.3
    return free


def _cell(free: np.ndarray, x: float, y: float) -> tuple[int, int]:
    i = int(np.clip((x + TILE_SIZE[0] / 2) / RES, 0, free.shape[0] - 1))
    j = int(np.clip((y + TILE_SIZE[1] / 2) / RES, 0, free.shape[1] - 1))
    return i, j


def _connected(free: np.ndarray, start: tuple[int, int], goal: tuple[int, int]) -> bool:
    seen = np.zeros_like(free)
    queue = deque([start])
    seen[start] = True
    while queue:
        i, j = queue.popleft()
        if (i, j) == goal:
            return True
        for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            a, b = i + di, j + dj
            if 0 <= a < free.shape[0] and 0 <= b < free.shape[1] and free[a, b] and not seen[a, b]:
                seen[a, b] = True
                queue.append((a, b))
    return False


@pytest.mark.parametrize("kind", list(BUILDERS))
def test_tiles_fit_the_tensor_caps(kind):
    for d in np.linspace(0.0, 1.0, 5):
        for seed in range(5):
            lay = BUILDERS[kind](float(d), np.random.default_rng(seed))
            assert 1 <= len(lay.routes) <= MAX_ROUTES
            assert 1 <= len(lay.paths) <= MAX_PATHS
            assert all(2 <= len(points) <= MAX_PATH_POINTS for points, *_ in lay.paths)
            assert len(lay.circles) <= MAX_CIRCLES
            assert len(lay.boxes) <= MAX_BOXES
            assert len(lay.standing) <= MAX_STANDING
            assert len(lay.forbidden) <= MAX_FORBIDDEN


@pytest.mark.parametrize("kind", list(BUILDERS))
def test_routes_are_passable(kind):
    """Every route must have a path at least 2 x 0.3 m wide, up to the hardest difficulty."""
    for d in (0.0, 0.5, 0.8, 1.0):
        for seed in range(6):
            lay = BUILDERS[kind](d, np.random.default_rng(seed))
            free = _free_space(lay)
            for sx, sy, gx, gy in lay.routes:
                start, goal = _cell(free, sx, sy), _cell(free, gx, gy)
                assert free[start], f"{kind} d={d} seed={seed}: route start ({sx:.1f}, {sy:.1f}) is blocked"
                assert free[goal], f"{kind} d={d} seed={seed}: route goal ({gx:.1f}, {gy:.1f}) is blocked"
                assert _connected(free, start, goal), f"{kind} d={d} seed={seed}: no passage from start to goal"


@pytest.mark.parametrize("kind", list(BUILDERS))
def test_pedestrians_enter_and_leave_out_of_view(kind):
    """People never appear or vanish on the tile: every path ends beyond the tile's ends and starts there, behind a
    door or at the foot of a subway stairwell."""
    for d in (0.0, 0.5, 1.0):
        for seed in range(6):
            lay = BUILDERS[kind](d, np.random.default_rng(seed))
            stairs = [item[1:] for item in lay.decor if item[0] == "stairs"]
            for points, *_ in lay.paths:
                (sx, sy), (ex, ey) = points[0], points[-1]
                assert abs(ex) > HALF_L + 1.0, f"{kind} d={d} seed={seed}: path ends on the tile at ({ex:.1f}, {ey:.1f})"
                off_tile = abs(sx) > HALF_L + 1.0
                behind_door = sy > HALF_W
                in_stairwell = any(x0 <= sx <= x1 and y0 <= sy <= y1 for x0, x1, y0, y1 in stairs)
                assert off_tile or behind_door or in_stairwell, f"{kind} d={d} seed={seed}: path starts at ({sx:.1f}, {sy:.1f})"
