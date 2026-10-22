"""Tests for the indoor tiles (pure numpy, no simulator needed)."""

import math

import numpy as np
import pytest

from g1_patrol.tasks.navigation.nyc.indoor import IN_L, IN_W, INDOOR_BUILDERS, ROBOT_HALF_WIDTH, Planner
from g1_patrol.tasks.navigation.nyc.kinds import ALL_BUILDERS, ALL_KINDS
from g1_patrol.tasks.navigation.nyc.makerlab import LAB_BUILDERS
from g1_patrol.tasks.navigation.nyc.tiles import (
    HALF_L,
    HALF_W,
    KINDS,
    MAX_BOXES,
    MAX_CIRCLES,
    MAX_FORBIDDEN,
    MAX_PATH_POINTS,
    MAX_PATHS,
    MAX_ROUTES,
    MAX_STANDING,
)

B2W_HALF = (0.56, 0.29)
"""Half length and half width (m) of the B2-W's footprint."""
DIFFICULTIES = (0.0, 0.3, 0.6, 1.0)
SEEDS = range(6)


BUILDERS_UNDER_TEST = {**INDOOR_BUILDERS, **LAB_BUILDERS}


def _layouts(kind):
    for d in DIFFICULTIES:
        for seed in SEEDS:
            yield d, seed, BUILDERS_UNDER_TEST[kind](d, np.random.default_rng(seed))


def test_street_kinds_keep_their_indices():
    assert ALL_KINDS[: len(KINDS)] == KINDS
    assert set(ALL_BUILDERS) | {"twin"} == set(ALL_KINDS)


@pytest.mark.parametrize("kind", list(BUILDERS_UNDER_TEST))
def test_tiles_fit_the_tensor_caps(kind):
    for d, seed, lay in _layouts(kind):
        where = f"{kind} d={d} seed={seed}"
        assert 1 <= len(lay.routes) <= MAX_ROUTES, where
        assert len(lay.route_headings) == len(lay.routes), where
        assert 2 <= len(lay.paths) <= MAX_PATHS, where
        assert all(2 <= len(points) <= MAX_PATH_POINTS for points, *_ in lay.paths), where
        assert len(lay.circles) <= MAX_CIRCLES, where
        assert len(lay.boxes) <= MAX_BOXES, where
        assert len(lay.standing) <= MAX_STANDING, where
        assert len(lay.forbidden) <= MAX_FORBIDDEN, where


@pytest.mark.parametrize("kind", list(BUILDERS_UNDER_TEST))
def test_routes_are_passable_for_the_b2w(kind):
    """A 0.58 m wide robot can drive every route, up to the hardest difficulty."""
    for d, seed, lay in _layouts(kind):
        plan = Planner(lay, z_max=1.0)
        for sx, sy, gx, gy in lay.routes:
            path = plan.path(plan.distances((sx, sy), ROBOT_HALF_WIDTH), (sx, sy), (gx, gy), ROBOT_HALF_WIDTH)
            assert path is not None, f"{kind} d={d} seed={seed}: no passage from ({sx:.1f}, {sy:.1f}) to ({gx:.1f}, {gy:.1f})"


def _footprint_clearance(plan: Planner, x: float, y: float, heading: float) -> float:
    """Smallest clearance of points sampled over the B2-W's footprint at (x, y, heading)."""
    hx, hy = B2W_HALF
    c, s = math.cos(heading), math.sin(heading)
    us, vs = np.meshgrid(np.linspace(-hx, hx, 23), np.linspace(-hy, hy, 13), indexing="ij")
    return min(plan.clearance((x + c * u - s * v, y + s * u + c * v)) for u, v in zip(us.ravel(), vs.ravel()))


@pytest.mark.parametrize("kind", list(BUILDERS_UNDER_TEST))
def test_robot_starts_clear_along_its_route(kind):
    """The footprint at the route's start heading (within the spawn noise of +-0.15 rad) touches nothing."""
    for d, seed, lay in _layouts(kind):
        plan = Planner(lay, z_max=1.0)
        for (sx, sy, _, _), heading in zip(lay.routes, lay.route_headings):
            for dh in (-0.15, 0.0, 0.15):
                gap = _footprint_clearance(plan, sx, sy, heading + dh)
                assert gap > 0.05, f"{kind} d={d} seed={seed}: start ({sx:.1f}, {sy:.1f}) heading {heading + dh:.2f}: {gap:.2f}"


@pytest.mark.parametrize("kind", list(BUILDERS_UNDER_TEST))
def test_pedestrians_enter_and_leave_out_of_view(kind):
    """Every path starts and ends behind a wall (a door), beyond the tile's ends (corridors) or in a stairwell."""
    for d, seed, lay in _layouts(kind):
        stairs = [item[1:] for item in lay.decor if item[0] == "stairs"]
        for points, *_ in lay.paths:
            for x, y in (points[0], points[-1]):
                hidden = abs(x) > IN_L + 0.1 or abs(y) > IN_W + 0.1
                in_stairwell = any(x0 <= x <= x1 and y0 <= y <= y1 for x0, x1, y0, y1 in stairs)
                assert hidden or in_stairwell, f"{kind} d={d} seed={seed}: path end ({x:.1f}, {y:.1f}) in view"


@pytest.mark.parametrize("kind", list(BUILDERS_UNDER_TEST))
def test_paths_keep_clear_of_furniture(kind):
    """Inside the room, people keep clear of structures (by 0.2 m on the planner's 0.1 m grid, i.e. about their
    radius), except while coming through a door (the first meters of a path) or walking on stairs."""
    for d, seed, lay in _layouts(kind):
        plan = Planner(lay)
        stairs = [item[1:] for item in lay.decor if item[0] == "stairs"]
        for points, _, _, free_length in lay.paths:
            pts = np.asarray(points)
            seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
            cum = np.concatenate([[0.0], np.cumsum(seg)])
            for s in np.arange(free_length, cum[-1], 0.1):
                k = min(int(np.searchsorted(cum, s, side="right")) - 1, len(seg) - 1)
                p = pts[k] + (s - cum[k]) / max(seg[k], 1e-6) * (pts[k + 1] - pts[k])
                on_stairs = any(x0 - 0.2 <= p[0] <= x1 + 0.2 and y0 <= p[1] <= y1 for x0, x1, y0, y1 in stairs)
                if abs(p[0]) < IN_L - 0.3 and abs(p[1]) < IN_W - 0.3 and not on_stairs:
                    assert plan.clearance(p) > 0.2, f"{kind} d={d} seed={seed}: path at ({p[0]:.1f}, {p[1]:.1f})"


@pytest.mark.parametrize("kind", list(BUILDERS_UNDER_TEST))
def test_drops_are_forbidden(kind):
    """Every place the floor drops by more than a step (stairwell, loading dock) lies in a forbidden area."""
    for d, seed, lay in _layouts(kind):
        nx, ny = lay.heights.shape
        xs = -HALF_L + np.arange(nx) * 0.1
        ys = -HALF_W + np.arange(ny) * 0.1
        px, py = np.meshgrid(xs, ys, indexing="ij")
        inside = np.zeros_like(lay.heights, dtype=bool)
        for x0, x1, y0, y1 in lay.forbidden:
            inside |= (px >= x0 - 0.1) & (px <= x1 + 0.1) & (py >= y0 - 0.1) & (py <= y1 + 0.1)
        assert not ((lay.heights < -0.25) & ~inside).any(), f"{kind} d={d} seed={seed}"


def test_example_twin_builds_and_is_drivable():
    """The example lab twin shipped for the robot builds: people come and go through its door, its routes are
    drivable by the B2-W."""
    from pathlib import Path

    import yaml

    from g1_patrol.tasks.navigation.nyc.indoor import passable_routes
    from g1_patrol.tasks.navigation.nyc.twin import build_twin

    spec = yaml.safe_load((Path(__file__).resolve().parents[1] / "deploy" / "b2w" / "lab_twin_example.yaml").read_text())
    for d in (0.0, 1.0):
        lay = build_twin(spec, d, np.random.default_rng(0))
        assert len(lay.paths) >= 2 and len(lay.routes) == len(spec["routes"]) == len(lay.route_headings)
        assert passable_routes(lay, B2W_HALF[1]) == list(range(len(lay.routes)))
        for points, *_ in lay.paths:
            for x, y in (points[0], points[-1]):
                assert abs(x) > IN_L + 0.1 or abs(y) > 4.0, "people must enter and leave through the door"
