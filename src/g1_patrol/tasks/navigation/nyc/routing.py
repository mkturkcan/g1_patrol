"""Planned robot routes: the way from a route's start to its goal that a robot of a given width can drive, as the
polyline its navigation policy follows. In simulation this is what a global planner's path is on the robot (e.g. the
``/plan`` of Nav2's planner server); see :mod:`g1_patrol.route` for following it.

Each route gets two plans on the clearance map of the finished tile (:class:`.indoor.Planner`):

* the plan: of the ways that keep one of a few margins from structures, the one with the largest margin that is not
  much longer than the shortest way -- about what a planner on an inflated costmap returns;
* the coarse plan: the same without small structures (chairs, carts, bins, cones, posts: things that move, or that a
  map made earlier would not show), which the robot then has to get around by itself. Where leaving them out would
  shorten the way by much (a small thing closes a passage), the coarse plan is the plan.
"""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np

from .indoor import Planner
from .tiles import TileLayout

MAX_ROUTE_POINTS = 16
MARGINS = (0.6, 0.5, 0.4)
"""Clearances (m) the plans try to keep from structures, the largest first; the robot's half width is the last resort."""
DETOUR = (1.15, 0.5)
"""A plan with a larger margin is taken if it is at most this much longer (factor, m) than the shortest way."""
SMALL_SIZE = 0.9
"""Structures whose longest side (diameter) is shorter than this (m) are left out of the coarse plans."""
COARSE_DETOUR = (1.25, 1.5)
"""The coarse plan is used only if the plan is at most this much longer (factor, m)."""


def length(points) -> float:
    return float(sum(math.hypot(q[0] - p[0], q[1] - p[1]) for p, q in zip(points[:-1], points[1:])))


def _plan(plan: Planner, start, goal, margin: float, reach: float) -> list | None:
    """Way from ``start`` that keeps ``margin`` to ``goal``, or to the free cell nearest to it within ``reach`` and
    from there straight to it; None if there is none."""
    dist = plan.distances(start, margin)
    gi, gj = plan._cell(goal)
    if dist[gi, gj] >= 0:
        return plan.path(dist, start, goal, margin)
    if not hasattr(plan, "_grid"):
        plan._grid = np.meshgrid(plan.xs, plan.ys, indexing="ij")
    gap = np.where(dist >= 0, np.hypot(plan._grid[0] - goal[0], plan._grid[1] - goal[1]), np.inf)
    i, j = np.unravel_index(int(np.argmin(gap)), gap.shape)
    if not gap[i, j] < reach:
        return None
    return plan.path(dist, start, (float(plan.xs[i]), float(plan.ys[j])), margin) + [tuple(goal)]


def _best(plan: Planner, start, goal, half_width: float, reach: float) -> list | None:
    shortest = _plan(plan, start, goal, half_width, reach)
    if shortest is None:
        return None
    limit = DETOUR[0] * length(shortest) + DETOUR[1]
    for margin in MARGINS:
        if margin <= half_width:
            break
        points = _plan(plan, start, goal, margin, reach)
        if points is not None and length(points) <= limit and len(points) <= MAX_ROUTE_POINTS:
            return points
    return shortest if len(shortest) <= MAX_ROUTE_POINTS else None


def without_small(lay: TileLayout) -> TileLayout:
    """The layout without its small structures (see :data:`SMALL_SIZE`)."""
    circles = [c for c in lay.circles if 2 * c[2] >= SMALL_SIZE]
    boxes = [b for b in lay.boxes if 2 * max(b[3], b[4]) >= SMALL_SIZE]
    return replace(lay, circles=circles, boxes=boxes)


def plan_routes(lay: TileLayout, half_width: float, overhead: float = 1.0, reach: float = 0.4) -> list[int]:
    """Plan the routes of ``lay`` for a robot of half width ``half_width`` (structures lower than ``overhead``, drops
    and forbidden areas are in its way): sets ``lay.route_paths`` and ``lay.route_paths_coarse`` and returns the
    indices of the routes it can drive (there is a way from the start to within ``reach`` of the goal); the paths of
    the others are None."""
    plan, coarse = Planner(lay, z_max=overhead), Planner(without_small(lay), z_max=overhead)
    lay.route_paths, lay.route_paths_coarse, ok = [], [], []
    for k, (sx, sy, gx, gy) in enumerate(lay.routes):
        points = _best(plan, (sx, sy), (gx, gy), half_width, reach)
        rough = None if points is None else _best(coarse, (sx, sy), (gx, gy), half_width, reach)
        if rough is None or length(points) > COARSE_DETOUR[0] * length(rough) + COARSE_DETOUR[1]:
            rough = points
        lay.route_paths.append(points)
        lay.route_paths_coarse.append(rough)
        if points is not None:
            ok.append(k)
    return ok
