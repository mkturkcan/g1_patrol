"""Route planning on a clearance map (numpy only; shared by the simulation and the robot).

The routes the B2-W's policy is trained to follow are planned on the clearance map of each simulated tile
(:mod:`g1_patrol.tasks.navigation.nyc.routing`); on the robot the same planner runs on the occupancy grid that SLAM
builds while it drives (``deploy/b2w/planner_node.py``), so that the paths the policy follows there are of the kind
it learned to follow: the shortest way that keeps the largest of a few margins from structures where that costs
little detour, pulled taut into a few straight legs.

:class:`ClearancePlanner` holds the clearance map -- the distance (m) from each cell to the nearest structure, -1 for
cells that must not be entered -- and plans on it: breadth-first search over the cells that keep a margin, then the
grid path pulled taut (each waypoint as far along as stays clear).
"""

from __future__ import annotations

import math

import numpy as np

MAX_ROUTE_POINTS = 16
MARGINS = (0.6, 0.5, 0.4)
"""Clearances (m) the plans try to keep from structures, the largest first; the robot's half width is the last resort."""
DETOUR = (1.15, 0.5)
"""A plan with a larger margin is taken if it is at most this much longer (factor, m) than the shortest way."""


class ClearancePlanner:
    """Planner on a clearance map ``clear`` (cells along x, cells along y) with cell centres ``xs`` and ``ys``; the
    map's corner is at (``x0``, ``y0``) and its cells are ``res`` m wide."""

    def __init__(self, clear: np.ndarray, xs: np.ndarray, ys: np.ndarray, x0: float, y0: float, res: float):
        self.clear, self.xs, self.ys, self.x0, self.y0, self.res = clear, xs, ys, x0, y0, res
        self._cache = {}

    def _cell(self, p) -> tuple[int, int]:
        i = int(np.clip(np.floor((p[0] - self.x0) / self.res), 0, len(self.xs) - 1))
        j = int(np.clip(np.floor((p[1] - self.y0) / self.res), 0, len(self.ys) - 1))
        return i, j

    def clearance(self, p) -> float:
        return float(self.clear[self._cell(p)])

    def segment_free(self, p, q, margin: float) -> bool:
        n = max(int(math.hypot(q[0] - p[0], q[1] - p[1]) / (self.res / 2)), 1)
        t = np.linspace(0.0, 1.0, n + 1)
        i = np.clip(np.floor((p[0] + t * (q[0] - p[0]) - self.x0) / self.res).astype(int), 0, len(self.xs) - 1)
        j = np.clip(np.floor((p[1] + t * (q[1] - p[1]) - self.y0) / self.res).astype(int), 0, len(self.ys) - 1)
        return bool((self.clear[i, j] > margin).all())

    def distances(self, start, margin: float) -> np.ndarray:
        """Grid steps (8-connected) from ``start`` through cells with more than ``margin`` clearance; -1: unreachable."""
        key = (round(start[0], 3), round(start[1], 3), margin)
        if key in self._cache:
            return self._cache[key]
        free = self.clear > margin
        dist = np.full(free.shape, -1, dtype=np.int32)
        i, j = self._cell(start)
        if not free[i, j]:
            return dist
        dist[i, j] = 0
        frontier = np.zeros_like(free)
        frontier[i, j] = True
        k = 0
        while frontier.any():
            k += 1
            grown = frontier.copy()
            grown[1:] |= frontier[:-1]
            grown[:-1] |= frontier[1:]
            g2 = grown.copy()
            g2[:, 1:] |= grown[:, :-1]
            g2[:, :-1] |= grown[:, 1:]
            frontier = g2 & free & (dist < 0)
            dist[frontier] = k
        self._cache[key] = dist
        return dist

    def path(self, dist: np.ndarray, start, goal, margin: float) -> list | None:
        """Shortest any-angle path from ``start`` (the source of ``dist``) to ``goal``: the grid path, then pulled
        taut (each waypoint as far along as stays clear). None if ``goal`` is unreachable."""
        gi, gj = self._cell(goal)
        if dist[gi, gj] < 0:
            return None
        cells = [(gi, gj)]
        i, j = gi, gj
        while dist[i, j] > 0:
            best = None
            for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)):
                a, b = i + di, j + dj
                if 0 <= a < dist.shape[0] and 0 <= b < dist.shape[1] and dist[a, b] == dist[i, j] - 1:
                    best = (a, b) if best is None or abs(di) + abs(dj) < 2 else best
            i, j = best
            cells.append((i, j))
        pts = [tuple(start)] + [(float(self.xs[a]), float(self.ys[b])) for a, b in reversed(cells[1:-1])] + [tuple(goal)]
        return self.pull(pts, margin)

    def footprint_clearance(self, p, heading: float, half: tuple[float, float] = (0.56, 0.29)) -> float:
        """Smallest clearance over a rectangular footprint (half sizes ``half``, default the B2-W's) at ``p``."""
        c, s = math.cos(heading), math.sin(heading)
        u, v = np.meshgrid(np.linspace(-half[0], half[0], 23), np.linspace(-half[1], half[1], 13), indexing="ij")
        x, y = p[0] + c * u - s * v, p[1] + s * u + c * v
        i = np.clip(np.floor((x - self.x0) / self.res).astype(int), 0, len(self.xs) - 1)
        j = np.clip(np.floor((y - self.y0) / self.res).astype(int), 0, len(self.ys) - 1)
        return float(self.clear[i, j].min())

    def visible(self, p, pts, margin: float) -> np.ndarray:
        """Which of the points ``pts`` can be reached from ``p`` in a straight line that keeps ``margin``."""
        q = np.asarray(pts, dtype=float).reshape(-1, 2)
        n = int(np.hypot(q[:, 0] - p[0], q[:, 1] - p[1]).max() / (self.res / 4)) + 2
        t = np.linspace(0.0, 1.0, n)[None]
        i = np.clip(np.floor((p[0] + t * (q[:, :1] - p[0]) - self.x0) / self.res).astype(int), 0, len(self.xs) - 1)
        j = np.clip(np.floor((p[1] + t * (q[:, 1:] - p[1]) - self.y0) / self.res).astype(int), 0, len(self.ys) - 1)
        return (self.clear[i, j] > margin).all(axis=1)

    def pull(self, pts: list, margin: float) -> list:
        """Pull a polyline taut: from each waypoint on, straight to the farthest later point in view."""
        out, k = [pts[0]], 0
        while k < len(pts) - 1:
            seen = np.nonzero(self.visible(pts[k], pts[k + 1 :], margin))[0]
            k = k + 1 + (int(seen.max()) if len(seen) else 0)
            out.append(pts[k])
        return out


def length(points) -> float:
    return float(sum(math.hypot(q[0] - p[0], q[1] - p[1]) for p, q in zip(points[:-1], points[1:])))


def plan_to(plan: ClearancePlanner, start, goal, margin: float, reach: float) -> list | None:
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


def best_path(plan: ClearancePlanner, start, goal, half_width: float, reach: float, margins=MARGINS, detour=DETOUR,
              max_points: int = MAX_ROUTE_POINTS) -> list | None:
    """The shortest way that keeps ``half_width``, or, of the ways that keep one of ``margins`` (largest first), the
    first that is at most ``detour`` (factor, m) longer; None if there is no way (or only one of more than
    ``max_points`` points)."""
    shortest = plan_to(plan, start, goal, half_width, reach)
    if shortest is None:
        return None
    limit = detour[0] * length(shortest) + detour[1]
    for margin in margins:
        if margin <= half_width:
            break
        points = plan_to(plan, start, goal, margin, reach)
        if points is not None and length(points) <= limit and len(points) <= max_points:
            return points
    return shortest if len(shortest) <= max_points else None


def occupancy_planner(occupied: np.ndarray, x0: float, y0: float, res: float, blocked: np.ndarray | None = None,
                      free_around=None, free_radius: float = 0.0) -> ClearancePlanner:
    """Planner on an occupancy grid: ``occupied`` (cells along x, cells along y) marks obstacles, ``blocked`` cells
    that must not be entered; cells within ``free_radius`` of the points ``free_around`` count as free (the robot's
    own place, where the map may show the robot itself or noise)."""
    from scipy.ndimage import distance_transform_edt

    occ = np.asarray(occupied, dtype=bool).copy()
    xs = x0 + res / 2 + res * np.arange(occ.shape[0])
    ys = y0 + res / 2 + res * np.arange(occ.shape[1])
    if free_around is not None and free_radius > 0:
        px, py = np.meshgrid(xs, ys, indexing="ij")
        for fx, fy in free_around:
            occ &= np.hypot(px - fx, py - fy) > free_radius
    # distance from each cell centre to the nearest obstacle cell's edge
    clear = distance_transform_edt(~occ) * res - res / 2 if occ.any() else np.full(occ.shape, 1e3)
    clear = np.where(occ, 0.0, clear)
    if blocked is not None:
        clear = np.where(blocked, -1.0, clear)
    return ClearancePlanner(clear, xs, ys, x0, y0, res)
