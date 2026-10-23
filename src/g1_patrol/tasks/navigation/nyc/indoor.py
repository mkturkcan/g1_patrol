"""Procedural indoor tiles: labs, corridors, doorways between rooms, offices, warehouses and lobbies.

Built like the street tiles (:mod:`.tiles`): one :class:`TileLayout` per 16 x 10 m tile at a difficulty in [0, 1],
with structures (primitives and meshes), robot routes, pedestrian paths and standing spots. Rooms are closed by walls
and people enter and leave them through doors, so that nobody appears or vanishes in view (paths start and end behind
a door); corridors continue beyond the tile's ends like streets. Difficulty narrows aisles, corridors and doors and
adds clutter: chairs pulled out, boxes and pallets on the floor, carts, open door leaves, people standing at benches.

Pedestrian paths and robot routes are planned on a clearance map of the finished layout (:class:`Planner`), so that
they keep to aisles and pass through doors whatever the furniture. The only areas the robot must not enter are drops
it can perceive (a loading dock, a stairwell); there are no keep-out zones marked by paint alone, which its sensors
could not see.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np

from .tiles import HALF_L, HALF_W, MAX_PATH_POINTS, MAX_ROUTES, OFF, TileLayout, _lerp, _new_layout

WALL_T, WALL_H = 0.15, 2.7
IN_L, IN_W = HALF_L - WALL_T, HALF_W - WALL_T
"""Half sizes of a room that fills the tile, inside its walls."""
BEHIND = 1.0
"""Paths start and end this far (m) behind a door."""
PATH_HALF_WIDTH = 0.1
"""Lateral spread (m) of people around an indoor path."""
PATH_MARGIN = 0.32
"""Clearance (m) of indoor paths from structures: a person at the widest offset keeps clear."""
ROBOT_HALF_WIDTH = 0.29
"""Routes are planned for a robot this wide (half, m): the B2-W with a little room to spare."""
ROUTE_MARGIN = ROBOT_HALF_WIDTH + 0.06
END_CLEARANCE = 0.5
"""Route ends keep this much clearance (m), so that the robot can stand there along its route."""


def _door(x: float, y: float, nx: float, ny: float, width: float) -> tuple:
    """A door: center of the opening on the room side of the wall, inward normal, width."""
    return (float(x), float(y), float(nx), float(ny), float(width))


def _outside(door: tuple, depth: float = BEHIND) -> tuple[float, float]:
    x, y, nx, ny, _ = door
    return (x - nx * (WALL_T + depth), y - ny * (WALL_T + depth))


def _inside(door: tuple, depth: float = 0.8) -> tuple[float, float]:
    x, y, nx, ny, _ = door
    return (x + nx * depth, y + ny * depth)


def _wall_x(lay: TileLayout, y: float, x0: float, x1: float, openings=(), height: float = WALL_H, mat: str = "wall",
            thickness: float = WALL_T):
    """A wall along x (center line ``y``) from ``x0`` to ``x1`` with openings (x center, width)."""
    a = x0
    for c0, c1 in sorted((xc - w / 2, xc + w / 2) for xc, w in openings) + [(x1, x1)]:
        if c0 - a > 0.05:
            lay.box((a + c0) / 2, y, 0.0, c0 - a, thickness, height, mat=mat)
        a = max(a, c1)


def _wall_y(lay: TileLayout, x: float, y0: float, y1: float, openings=(), height: float = WALL_H, mat: str = "wall",
            thickness: float = WALL_T):
    """A wall along y (center line ``x``) from ``y0`` to ``y1`` with openings (y center, width)."""
    a = y0
    for c0, c1 in sorted((yc - w / 2, yc + w / 2) for yc, w in openings) + [(y1, y1)]:
        if c0 - a > 0.05:
            lay.box(x, (a + c0) / 2, 0.0, thickness, c0 - a, height, mat=mat)
        a = max(a, c1)


def _room(lay: TileLayout, top=(), bottom=(), left=(), right=(), open_ends: bool = False) -> list:
    """Walls along the tile's edges with door openings ((x or y center, width) per side); returns the doors."""
    yt, xr = HALF_W - WALL_T / 2, HALF_L - WALL_T / 2
    _wall_x(lay, yt, -HALF_L, HALF_L, top)
    _wall_x(lay, -yt, -HALF_L, HALF_L, bottom)
    if not open_ends:
        _wall_y(lay, -xr, -IN_W, IN_W, left)
        _wall_y(lay, xr, -IN_W, IN_W, right)
    doors = [_door(x, IN_W, 0, -1, w) for x, w in top] + [_door(x, -IN_W, 0, 1, w) for x, w in bottom]
    doors += [_door(-IN_L, y, 1, 0, w) for y, w in left] + [_door(IN_L, y, -1, 0, w) for y, w in right]
    return doors


def _door_leaf(lay: TileLayout, door: tuple, side: float, into: bool = True):
    """The leaf of an open door: hinged at one edge of the opening (``side`` = +-1 along the wall) and opened by 90
    degrees, into the room (``into``) or out of it."""
    x, y, nx, ny, w = door
    tx, ty = -ny, nx
    sign = 1.0 if into else -1.0
    hx, hy = x + side * (w / 2 - 0.03) * tx, y + side * (w / 2 - 0.03) * ty
    if not into:
        hx, hy = hx - nx * WALL_T, hy - ny * WALL_T
    lay.box(hx + sign * nx * w / 2, hy + sign * ny * w / 2, math.atan2(ny, nx), w, 0.05, 2.1, mat="door")


def _spread(rng: np.random.Generator, count: int, lo: float, hi: float, gap: float) -> list[float]:
    """``count`` sorted positions in [lo, hi] at least ``gap`` apart (fewer if they do not fit)."""
    for _ in range(50):
        xs = np.sort(rng.uniform(lo, hi, size=count))
        if count < 2 or np.min(np.diff(xs)) >= gap:
            return [float(x) for x in xs]
    return [float(x) for x in np.linspace(lo, hi, count)] if (hi - lo) >= gap * (count - 1) else [float((lo + hi) / 2)]


def _floor(lay: TileLayout, mat: str):
    lay.paint_rect(-HALF_L, HALF_L, -HALF_W, HALF_W, mat)


def _box_extent(cx, cy, yaw, hx, hy) -> tuple[float, float, float, float]:
    c, s = abs(math.cos(yaw)), abs(math.sin(yaw))
    ex, ey = c * hx + s * hy, s * hx + c * hy
    return cx - ex, cx + ex, cy - ey, cy + ey


class Planner:
    """Clearance map of a layout (numpy only): distance from each 0.1 m cell to the nearest structure below
    ``z_max`` (drops count as structures), for planning pedestrian paths and robot routes."""

    def __init__(self, lay: TileLayout, z_max: float = 2.0, res: float = 0.1):
        self.lay, self.res = lay, res
        self.xs = np.arange(-HALF_L + res / 2, HALF_L, res)
        self.ys = np.arange(-HALF_W + res / 2, HALF_W, res)
        px, py = np.meshgrid(self.xs, self.ys, indexing="ij")
        clear = np.full(px.shape, 1e3)
        for x, y, r, z0, _ in lay.circles:
            if z0 < z_max:
                clear = np.minimum(clear, np.hypot(px - x, py - y) - r)
        for cx, cy, yaw, hx, hy, z0, _ in lay.boxes:
            if z0 < z_max:
                c, s = math.cos(yaw), math.sin(yaw)
                lx, ly = c * (px - cx) + s * (py - cy), -s * (px - cx) + c * (py - cy)
                clear = np.minimum(clear, np.hypot(np.maximum(np.abs(lx) - hx, 0.0), np.maximum(np.abs(ly) - hy, 0.0)))
        hi = np.clip(np.round((px + HALF_L) / 0.1).astype(int), 0, lay.heights.shape[0] - 1)
        hj = np.clip(np.round((py + HALF_W) / 0.1).astype(int), 0, lay.heights.shape[1] - 1)
        drop = lay.heights[hi, hj] < -0.25
        for x0, x1, y0, y1 in lay.forbidden:
            drop |= (px > x0) & (px < x1) & (py > y0) & (py < y1)
        self.clear = np.where(drop, -1.0, clear)
        self._cache = {}

    def _cell(self, p) -> tuple[int, int]:
        i = int(np.clip(np.floor((p[0] + HALF_L) / self.res), 0, len(self.xs) - 1))
        j = int(np.clip(np.floor((p[1] + HALF_W) / self.res), 0, len(self.ys) - 1))
        return i, j

    def clearance(self, p) -> float:
        return float(self.clear[self._cell(p)])

    def segment_free(self, p, q, margin: float) -> bool:
        n = max(int(math.hypot(q[0] - p[0], q[1] - p[1]) / (self.res / 2)), 1)
        t = np.linspace(0.0, 1.0, n + 1)
        i = np.clip(np.floor((p[0] + t * (q[0] - p[0]) + HALF_L) / self.res).astype(int), 0, len(self.xs) - 1)
        j = np.clip(np.floor((p[1] + t * (q[1] - p[1]) + HALF_W) / self.res).astype(int), 0, len(self.ys) - 1)
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
        i = np.clip(np.floor((x + HALF_L) / self.res).astype(int), 0, len(self.xs) - 1)
        j = np.clip(np.floor((y + HALF_W) / self.res).astype(int), 0, len(self.ys) - 1)
        return float(self.clear[i, j].min())

    def visible(self, p, pts, margin: float) -> np.ndarray:
        """Which of the points ``pts`` can be reached from ``p`` in a straight line that keeps ``margin``."""
        q = np.asarray(pts, dtype=float).reshape(-1, 2)
        n = int(np.hypot(q[:, 0] - p[0], q[:, 1] - p[1]).max() / (self.res / 4)) + 2
        t = np.linspace(0.0, 1.0, n)[None]
        i = np.clip(np.floor((p[0] + t * (q[:, :1] - p[0]) + HALF_L) / self.res).astype(int), 0, len(self.xs) - 1)
        j = np.clip(np.floor((p[1] + t * (q[:, 1:] - p[1]) + HALF_W) / self.res).astype(int), 0, len(self.ys) - 1)
        return (self.clear[i, j] > margin).all(axis=1)

    def pull(self, pts: list, margin: float) -> list:
        """Pull a polyline taut: from each waypoint on, straight to the farthest later point in view."""
        out, k = [pts[0]], 0
        while k < len(pts) - 1:
            seen = np.nonzero(self.visible(pts[k], pts[k + 1 :], margin))[0]
            k = k + 1 + (int(seen.max()) if len(seen) else 0)
            out.append(pts[k])
        return out


def _paths_between(lay: TileLayout, plan: Planner, ends: list, rng: np.random.Generator, count: int, weight: float = 1.0,
                   via: list | None = None, pairs: list | None = None):
    """Up to ``count`` pedestrian paths between pairs of ``ends`` (each a polyline from behind a door or from off the
    tile to a point inside: a path starts with one end and finishes with another, reversed), optionally through one
    of the points ``via``; ``pairs`` restricts the pairs of ends (indices)."""
    pairs = [(a, b) for a in range(len(ends)) for b in range(len(ends)) if a != b] if pairs is None else list(pairs)
    rng.shuffle(pairs)
    made = 0
    for a, b in pairs:
        if made >= count:
            break
        ea, eb = ends[a], ends[b]
        # through a random point of ``via`` (spreading people over the aisles), else the shortest way
        for mid in ([via[rng.integers(len(via))]] if via else []) + [None]:
            legs = []
            for p, q in ((ea[-1], mid), (mid, eb[-1])) if mid is not None else ((ea[-1], eb[-1]),):
                leg = plan.path(plan.distances(p, PATH_MARGIN), p, q, PATH_MARGIN)
                if leg is None:
                    break
                legs.append(leg)
            else:
                inner = plan.pull(legs[0] + [pt for leg in legs[1:] for pt in leg[1:]], PATH_MARGIN)
                points = list(ea[:-1]) + inner + list(reversed(eb[:-1]))
                if len(points) <= MAX_PATH_POINTS:
                    lay.path(points, PATH_HALF_WIDTH, weight, free_length=_free_length(ea))
                    made += 1
                    break
    return made


def _free_length(end: list) -> float:
    """Length of an entry polyline (from behind the door to inside): structures do not push people along it."""
    return float(sum(math.hypot(q[0] - p[0], q[1] - p[1]) for p, q in zip(end[:-1], end[1:])))


def _door_end(door: tuple, depth: float = 0.8) -> list:
    return [_outside(door), _inside(door, depth)]


def _routes(lay: TileLayout, plan: Planner, candidates: list, rng: np.random.Generator, count: int = MAX_ROUTES,
            min_length: float = 5.0):
    """Robot routes between pairs of ``candidates`` that are clear, at least ``min_length`` apart and connected for
    the robot; the start heading follows the first leg of the route."""
    ok = [p for p in candidates if plan.clearance(p) > max(END_CLEARANCE, ROUTE_MARGIN)]
    pairs = [(a, b) for a in range(len(ok)) for b in range(len(ok))
             if a != b and math.hypot(ok[a][0] - ok[b][0], ok[a][1] - ok[b][1]) >= min_length]
    rng.shuffle(pairs)
    for a, b in pairs:
        if len(lay.routes) >= count:
            break
        s, g = ok[a], ok[b]
        if any(r[:2] == s and r[2:] == g for r in lay.routes):
            continue
        leg = plan.path(plan.distances(s, ROUTE_MARGIN), s, g, ROUTE_MARGIN)
        if leg is None:
            continue
        heading = math.atan2(leg[1][1] - s[1], leg[1][0] - s[0])
        if min(plan.footprint_clearance(s, heading + dh) for dh in (-0.15, 0.0, 0.15)) < 0.12:
            continue
        lay.routes.append((float(s[0]), float(s[1]), float(g[0]), float(g[1])))
        lay.route_headings.append(heading)


def _largest_gap(lo: float, hi: float, spans: list) -> float:
    """Widest stretch of [lo, hi] not covered by the intervals ``spans``."""
    best, a = 0.0, lo
    for s0, s1 in sorted(spans):
        best = max(best, s0 - a)
        a = max(a, s1)
    return max(best, hi - a)


class _Bands:
    """Aisles (bands along x or y) that must stay passable for the robot: a rectangle may be placed only if, across
    every band it intrudes on, a stretch at least ``min_free`` wide stays free of it and of everything placed before
    within ``reach`` along the band."""

    def __init__(self, bands: list, min_free: float, reach: float = 0.7, doors: list = ()):
        self.bands, self.min_free, self.reach = bands, min_free, reach
        self.placed: list = []
        # nothing in front of a door (1.6 m deep, 0.4 m wider than the door on each side)
        self.blocked = []
        for x, y, nx, ny, w in doors:
            ax, ay = x + nx * 0.8, y + ny * 0.8
            hx, hy = (w / 2 + 0.4, 0.8) if nx == 0 else (0.8, w / 2 + 0.4)
            self.blocked.append((ax - hx, ax + hx, ay - hy, ay + hy))

    def fits(self, rect) -> bool:
        x0, x1, y0, y1 = rect
        if any(x1 > b0 and x0 < b1 and y1 > c0 and y0 < c1 for b0, b1, c0, c1 in self.blocked):
            return False
        for bx0, bx1, by0, by1, axis in self.bands:
            if x1 <= bx0 or x0 >= bx1 or y1 <= by0 or y0 >= by1:
                continue
            if axis == "x":
                near = [q for q in self.placed if q[0] < x1 + self.reach and q[1] > x0 - self.reach]
                spans = [(max(q[2], by0), min(q[3], by1)) for q in near + [rect] if q[2] < by1 and q[3] > by0]
                free = _largest_gap(by0, by1, spans)
            else:
                near = [q for q in self.placed if q[2] < y1 + self.reach and q[3] > y0 - self.reach]
                spans = [(max(q[0], bx0), min(q[1], bx1)) for q in near + [rect] if q[0] < bx1 and q[1] > bx0]
                free = _largest_gap(bx0, bx1, spans)
            if free < self.min_free:
                return False
        return True

    def place(self, rect) -> bool:
        """Record ``rect`` if it fits; returns whether it does."""
        if not self.fits(rect):
            return False
        self.placed.append(rect)
        return True


"""
Builders.
"""


def build_lab(d: float, rng: np.random.Generator, civic_rng: np.random.Generator | None = None) -> TileLayout:
    """A laboratory: wall benches, island benches and tables in rows, chairs and stools, equipment, carts and gas
    cylinders; people work at the benches and come and go through the doors."""
    lay = _new_layout("lab", d)
    _floor(lay, "floor_vinyl")
    aisle = _lerp(1.9, 1.0, d) + rng.uniform(0.0, 0.15)
    cross = _lerp(1.9, 1.05, d) + rng.uniform(0.0, 0.15)
    end_aisle = _lerp(1.9, 1.1, d)
    door_w = rng.uniform(0.9, 1.1)
    top = _spread(rng, 2, -6.0, 6.0, 3.0)
    bottom = _spread(rng, 1, -6.0, 6.0, 3.0)
    doors = _room(lay, top=[(x, door_w) for x in top], bottom=[(x, door_w) for x in bottom])
    bench_d = 0.75
    bottom_bench = rng.uniform() < 0.7
    # wall benches (lab casework) with gaps in front of the doors
    for y_face, door_xs, deep in ((IN_W, top, bench_d), (-IN_W, bottom, bench_d if bottom_bench else 0.45)):
        sign = 1.0 if y_face > 0 else -1.0
        mat, height = ("bench", 0.9) if deep == bench_d else ("shelf", 2.0)
        cuts = [(x, door_w + 1.0) for x in door_xs]
        a = -IN_L
        for c0, c1 in sorted((x - w / 2, x + w / 2) for x, w in cuts) + [(IN_L, IN_L)]:
            if c0 - a > 0.6:
                lay.box((a + c0) / 2, y_face - sign * deep / 2, 0.0, c0 - a, deep, height, mat=mat)
            a = max(a, c1)
    y_lo, y_hi = -IN_W + (bench_d if bottom_bench else 0.45), IN_W - bench_d
    # island rows between the wall benches: double benches, then a single one if it fits, the rest a wider main aisle
    island = float(rng.choice([1.2, 1.5]))
    span = y_hi - y_lo
    k = max(int((span - aisle) // (island + aisle)), 0)
    depths = [island] * k
    if span - k * (island + aisle) - aisle >= 0.75 + aisle:
        depths.append(0.75)
    rng.shuffle(depths)
    extra = span - sum(depths) - (len(depths) + 1) * aisle
    aisles = [aisle] * (len(depths) + 1)
    aisles[int(rng.integers(len(aisles)))] += max(extra, 0.0)
    x_lo, x_hi = -IN_L + 0.9 + end_aisle, IN_L - 0.9 - end_aisle
    n_cross = 1 + int(rng.uniform() < d)
    cuts = _spread(rng, n_cross, x_lo + 2.5, x_hi - 2.5, cross + 2.5)
    rows, y = [], y_lo + aisles[0]
    for depth, gap in zip(depths, aisles[1:]):
        rows.append((y, y + depth))
        y += depth + gap
    aisle_ys = [y_lo + aisles[0] / 2] + [r1 + g / 2 for (_, r1), g in zip(rows, aisles[1:])]
    seats = []
    for r0, r1 in rows:
        a = x_lo
        for c0, c1 in [(x - cross / 2, x + cross / 2) for x in cuts] + [(x_hi, x_hi)]:
            if c0 - a > 1.2:
                cx, cy = (a + c0) / 2, (r0 + r1) / 2
                if rng.uniform() < 0.55:
                    lay.box(cx, cy, 0.0, c0 - a, r1 - r0, 0.9, mat="bench")
                else:
                    lay.table(cx, cy, 0.0, c0 - a, r1 - r0, 0.75, mat="table")
                seats.append((a, c0, r0, r1))
            a = c1
    # gas cylinders in a corner, equipment against the end walls (fume hoods, shelves, cabinets)
    corner_x, corner_y = float(rng.choice([-1.0, 1.0])), float(rng.choice([-1.0, 1.0]))
    cy0 = y_hi - 0.2 if corner_y > 0 else y_lo + 0.2
    for i in range(int(rng.integers(2, 5))):
        lay.circle(corner_x * (IN_L - 0.17 - 0.27 * (i % 2)), cy0 - corner_y * 0.27 * (i // 2), 0.12, 1.45, mat="gas_cylinder")
    for x_face in (-IN_L, IN_L):
        sign = 1.0 if x_face < 0 else -1.0
        lo, hi = y_lo + 1.0, y_hi - 1.0
        if sign == -corner_x:  # this end has the cylinders
            lo, hi = (lo, hi - 0.8) if corner_y > 0 else (lo + 0.8, hi)
        for yc in _spread(rng, int(rng.integers(1, 3)), lo, hi, 2.6):
            kind = rng.choice(["hood", "shelf", "cabinet"])
            deep, wide, high, mat = {"hood": (0.9, 1.8, 2.3, "fume_hood"), "shelf": (0.45, 1.6, 2.0, "shelf"),
                                     "cabinet": (0.6, 1.0, 1.9, "cabinet")}[kind]
            lay.box(x_face + sign * deep / 2, yc, 0.0, deep, wide, high, mat=mat)
    # bands that must stay passable for the robot: every aisle and cross aisle
    bands = [(x_lo - end_aisle, x_hi + end_aisle, ay - w / 2, ay + w / 2, "x") for ay, w in zip(aisle_ys, aisles)]
    bands += [(x - cross / 2, x + cross / 2, y_lo, y_hi, "y") for x in cuts]
    bands += [(x_lo - end_aisle, x_lo, y_lo, y_hi, "y"), (x_hi, x_hi + end_aisle, y_lo, y_hi, "y")]
    keep = _Bands(bands, 0.62 + _lerp(0.4, 0.2, d), doors=doors)
    # chairs and stools at the island benches, pushed in or pulled out
    pulled = _lerp(0.25, 0.7, d)
    for x0, x1, r0, r1 in seats:
        for face, sign in ((r0, -1.0), (r1, 1.0)):
            for x in np.arange(x0 + 0.5, x1 - 0.3, rng.uniform(0.9, 1.3)):
                if rng.uniform() > _lerp(0.4, 0.8, d):
                    continue
                r = float(rng.choice([0.2, 0.28]))
                out = rng.uniform(0.15, 0.4) if rng.uniform() < pulled else 0.02
                cy = face + sign * (r + out - 0.1)
                if keep.place((x - r, x + r, cy - r, cy + r)):
                    lay.circle(float(x), cy, r, 0.65 if r < 0.25 else 0.95, mat="chair")
    # carts and boxes on the floor in the aisles (low clutter the lidar sees late)
    for _ in range(int(_lerp(1, 6, d))):
        i = int(rng.integers(len(aisle_ys)))
        ay, w = aisle_ys[i], aisles[i]
        side = float(rng.choice([-1.0, 1.0]))
        if rng.uniform() < 0.3:
            sx, sy, h, mat = 0.9, 0.55, 1.0, "cart"
        else:
            sx, sy, h, mat = rng.uniform(0.3, 0.6), rng.uniform(0.3, 0.5), rng.uniform(0.15, 0.45), "carton"
        x = rng.uniform(x_lo, x_hi)
        cy = ay + side * (w / 2 - sy / 2 - 0.02)
        if abs(x) < IN_L - 1.0 and keep.place((x - sx / 2, x + sx / 2, cy - sy / 2, cy + sy / 2)):
            lay.box(x, cy, 0.0, sx, sy, h, mat=mat)
    if rng.uniform() < 0.5 and max(aisles) > 2.2:  # a robot arm on a pedestal in the wide aisle
        i = int(np.argmax(aisles))
        x = rng.uniform(x_lo + 1.0, x_hi - 1.0)
        y = aisle_ys[i] + float(rng.choice([-1, 1])) * (aisles[i] / 2 - 0.45)
        if min(abs(x - c) for c in cuts) > cross / 2 + 0.6 and keep.place((x - 0.3, x + 0.3, y - 0.3, y + 0.3)):
            lay.circle(x, y, 0.3, 1.1, mat="pedestal")
    # cable protectors across aisles (a low bump)
    for _ in range(int(_lerp(0, 3, d))):
        x = rng.uniform(x_lo, x_hi)
        lay.add_rect(x - 0.12, x + 0.12, y_lo, y_hi, 0.025)
        lay.paint_rect(x - 0.12, x + 0.12, y_lo, y_hi, "cable_cover")

    plan = Planner(lay)
    ends = [_door_end(door) for door in doors]
    # people working at the benches; paths pass them (they leave along the nearest one)
    spots = []
    for x0, x1, r0, r1 in seats:
        for face, sign in ((r0, -1.0), (r1, 1.0)):
            p = (float(rng.uniform(x0 + 0.4, x1 - 0.4)), face + sign * 0.36)
            if plan.clearance(p) > 0.28:
                spots.append(p)
    rng.shuffle(spots)
    spots = spots[: int(_lerp(3, 8, d))]
    for p in spots:
        lay.standing_spot(*p)
    lanes = [(p[0], ay) for p in spots for ay in aisle_ys if abs(ay - p[1]) < 1.0] or None
    _paths_between(lay, plan, ends, rng, count=6, weight=1.0, via=lanes)
    _paths_between(lay, plan, ends, rng, count=6, weight=1.0, via=[(x, ay) for x in cuts for ay in aisle_ys])
    junctions = [(x, ay) for x in [x_lo - end_aisle / 2, *cuts, x_hi + end_aisle / 2] for ay in aisle_ys]
    _routes(lay, plan, junctions + [_inside(door, 1.2) for door in doors], rng)
    lay.crowd = (2, 8)
    return lay


def build_corridor(d: float, rng: np.random.Generator, civic_rng: np.random.Generator | None = None) -> TileLayout:
    """A corridor between two rooms: doors (some open into the corridor), benches, bins, carts and water fountains
    along the walls; people walk along it and in and out of the rooms. The corridor continues beyond the tile's
    ends."""
    lay = _new_layout("corridor", d)
    _floor(lay, "floor_tiles")
    width = _lerp(2.6, 1.5, d) + rng.uniform(0.0, 0.2)
    yc = rng.uniform(-0.6, 0.6)
    y0, y1 = yc - width / 2, yc + width / 2
    door_w = rng.uniform(0.9, 1.0)
    doors_top = _spread(rng, int(rng.integers(2, 4)), -6.5, 6.5, 2.8)
    doors_bottom = _spread(rng, int(rng.integers(2, 4)), -6.5, 6.5, 2.8)
    _wall_x(lay, y1 + WALL_T / 2, -HALF_L, HALF_L, [(x, door_w) for x in doors_top])
    _wall_x(lay, y0 - WALL_T / 2, -HALF_L, HALF_L, [(x, door_w) for x in doors_bottom])
    # rooms beyond the corridor walls: outer doors along the tile's long edges, partitions between rooms
    out_top = _spread(rng, 2, -6.5, 6.5, 4.0)
    out_bottom = _spread(rng, 2, -6.5, 6.5, 4.0)
    outer = _room(lay, top=[(x, 1.0) for x in out_top], bottom=[(x, 1.0) for x in out_bottom], open_ends=True)
    for ya, yb in ((y1 + WALL_T, IN_W), (-IN_W, y0 - WALL_T)):
        for x in (-HALF_L + WALL_T / 2, HALF_L - WALL_T / 2):
            _wall_y(lay, x, ya, yb)
    corridor_doors = [_door(x, y1, 0, -1, door_w) for x in doors_top] + [_door(x, y0, 0, 1, door_w) for x in doors_bottom]
    room_doors = [_door(x, y1 + WALL_T, 0, 1, door_w) for x in doors_top] + [_door(x, y0 - WALL_T, 0, -1, door_w) for x in doors_bottom]
    # door leaves: into the rooms, or into the corridor where it is wide enough
    for cd, rd in zip(corridor_doors, room_doors):
        if rng.uniform() < _lerp(0.3, 0.8, d):
            if width - door_w >= 1.3 and rng.uniform() < 0.5:
                _door_leaf(lay, cd, float(rng.choice([-1.0, 1.0])))
            else:
                _door_leaf(lay, rd, float(rng.choice([-1.0, 1.0])))
    # furniture in the rooms: desks and shelves against the outer walls, away from the doors' axes
    for ya, yb, sign, xs in ((y1 + WALL_T, IN_W, 1.0, doors_top + out_top), (-IN_W, y0 - WALL_T, -1.0, doors_bottom + out_bottom)):
        depth = yb - ya
        for _ in range(int(rng.integers(2, 5))):
            x = rng.uniform(-IN_L + 0.8, IN_L - 0.8)
            if min(abs(x - dx) for dx in xs) < 1.6:
                continue
            if rng.uniform() < 0.5:
                lay.table(x, (yb if sign > 0 else ya) - sign * 0.4, 0.0, 1.4, 0.7, 0.75, mat="desk")
            else:
                lay.box(x, (yb if sign > 0 else ya) - sign * 0.25, 0.0, 1.2, 0.45, 1.9, mat="shelf")
            if depth > 3.0 and rng.uniform() < 0.5:
                lay.circle(x + rng.uniform(-0.4, 0.4), (yb if sign > 0 else ya) - sign * 1.1, 0.3, 1.0, mat="chair")
    # along the corridor walls between doors: bins, benches, carts, fountains, plants, a wet-floor sign
    keep = _Bands([(-HALF_L - 2.0, HALF_L + 2.0, y0, y1, "x")], 0.62 + _lerp(0.45, 0.25, d), doors=corridor_doors)
    for _ in range(int(_lerp(2, 8, d))):
        side = float(rng.choice([-1.0, 1.0]))
        face = y1 if side > 0 else y0
        x = rng.uniform(-HALF_L + 0.5, HALF_L - 0.5)
        if min(abs(x - dx) for dx in (doors_top if side > 0 else doors_bottom)) < door_w / 2 + 0.7:
            continue
        kind = rng.choice(["bin", "bench", "cart", "fountain", "plant", "sign"])
        if kind in ("bin", "plant", "sign"):
            r, h, mat = {"bin": (0.22, 0.75, "bin"), "plant": (0.3, 1.2, "planter"), "sign": (0.2, 0.6, "wet_floor_sign")}[kind]
            cy = face - side * (r + 0.02) if kind != "sign" else rng.uniform(y0 + 0.3, y1 - 0.3)
            if keep.place((x - r, x + r, cy - r, cy + r)):
                lay.circle(x, cy, r, h, mat=mat)
        else:
            sx, sy, h, mat = {"bench": (1.4, 0.42, 0.45, "bench_seat"), "cart": (1.0, 0.6, 1.0, "cart"),
                              "fountain": (0.45, 0.35, 0.95, "fountain")}[kind]
            cy = face - side * (sy / 2 + 0.01)
            if keep.place((x - sx / 2, x + sx / 2, cy - sy / 2, cy + sy / 2)):
                lay.box(x, cy, 0.0, sx, sy, h, mat=mat)

    plan = Planner(lay)
    lane = min(0.3, width / 2 - 0.45)
    ends = [[(-OFF, yc - lane), (-HALF_L + 0.5, yc - lane)], [(OFF, yc + lane), (HALF_L - 0.5, yc + lane)]]
    ends += [_door_end(door, 0.9) for door in outer]
    _paths_between(lay, plan, ends, rng, count=2, weight=2.0, pairs=[(0, 1), (1, 0)])  # along the corridor, keeping right
    _paths_between(lay, plan, ends, rng, count=10, weight=1.0)
    for _ in range(int(_lerp(1, 4, d))):  # people talking in the corridor or at a door
        p = (rng.uniform(-6.0, 6.0), float(rng.choice([y0 + 0.35, y1 - 0.35])))
        if plan.clearance(p) > 0.28:
            lay.standing_spot(*p)
    for door in room_doors[:4]:
        p = _inside(door, 1.4)
        if plan.clearance(p) > 0.4:
            lay.standing_spot(*p)
    candidates = [(-HALF_L + 0.8, yc), (HALF_L - 0.8, yc)] + [_inside(door, 1.4) for door in room_doors]
    _routes(lay, plan, candidates, rng)
    lay.crowd = (3, 12)
    return lay


def build_doorway(d: float, rng: np.random.Generator, civic_rng: np.random.Generator | None = None) -> TileLayout:
    """Two rooms joined by a doorway (or two): people come through it both ways and the robot has to wait its turn."""
    lay = _new_layout("doorway", d)
    _floor(lay, str(rng.choice(["floor_tiles", "floor_vinyl", "carpet"])))
    xp = rng.uniform(-1.5, 1.5)
    width = _lerp(1.4, 0.92, d)
    count = 2 if rng.uniform() > d else 1
    openings = _spread(rng, count, -IN_W + width / 2 + 0.9, IN_W - width / 2 - 0.9, width + 1.5)
    _wall_y(lay, xp, -IN_W, IN_W, [(y, width) for y in openings])
    door_w = 1.0
    # two doors per room: one in the top wall, the other in the bottom wall or the end wall
    top, bottom, left, right = [], [], [], []
    for lo, hi, end in ((-IN_L + 1.0, xp - 1.5, left), (xp + 1.5, IN_L - 1.0, right)):
        top.append((rng.uniform(lo, hi), door_w))
        if rng.uniform() < 0.5:
            bottom.append((rng.uniform(lo, hi), door_w))
        else:
            end.append((rng.uniform(-2.5, 2.5), door_w))
    doors = _room(lay, top=top, bottom=bottom, left=left, right=right)
    gates = [_door(xp - WALL_T / 2, y, -1, 0, width) for y in openings]  # partition openings, seen from the left room
    if rng.uniform() < _lerp(0.2, 0.7, d):  # the leaf of the (first) opening, open into either room
        g = gates[0] if rng.uniform() < 0.5 else _door(xp + WALL_T / 2, gates[0][1], 1, 0, width)
        _door_leaf(lay, g, float(rng.choice([-1.0, 1.0])))
    # furniture along the walls, clear of the approaches to the openings and doors
    approach = [(xp - 2.5, xp + 2.5, y - width / 2 - 0.6, y + width / 2 + 0.6) for y in openings]
    approach += [(x - 1.0, x + 1.0, IN_W - 1.6, IN_W) for x, _ in top] + [(x - 1.0, x + 1.0, -IN_W, -IN_W + 1.6) for x, _ in bottom]
    approach += [(-IN_L, -IN_L + 1.6, y - 1.0, y + 1.0) for y, _ in left] + [(IN_L - 1.6, IN_L, y - 1.0, y + 1.0) for y, _ in right]

    def clear_of_approaches(rect):
        x0, x1, y0, y1 = rect
        return all(x1 <= a0 or x0 >= a1 or y1 <= b0 or y0 >= b1 for a0, a1, b0, b1 in approach)

    for _ in range(int(_lerp(6, 14, d))):
        kind = rng.choice(["table", "shelf", "cabinet", "chair", "carton", "plant"])
        side = float(rng.choice([-1.0, 1.0]))
        if kind in ("table", "shelf", "cabinet", "carton"):
            sx, sy, h, mat = {"table": (1.6, 0.8, 0.75, "table"), "shelf": (1.4, 0.45, 2.0, "shelf"),
                              "cabinet": (0.9, 0.6, 1.2, "cabinet"), "carton": (0.5, 0.4, 0.4, "carton")}[kind]
            x = rng.uniform(-IN_L + sx / 2, IN_L - sx / 2)
            y = side * (IN_W - sy / 2 - 0.02) if rng.uniform() < 0.7 else rng.uniform(-IN_W + 1.5, IN_W - 1.5)
            rect = (x - sx / 2, x + sx / 2, y - sy / 2, y + sy / 2)
            if abs(x - xp) > sx / 2 + 0.2 and clear_of_approaches(rect):
                if kind == "table":
                    lay.table(x, y, 0.0, sx, sy, h)
                else:
                    lay.box(x, y, 0.0, sx, sy, h, mat=mat)
        else:
            r, h, mat = (0.28, 0.95, "chair") if kind == "chair" else (0.3, 1.2, "planter")
            x, y = rng.uniform(-IN_L + 0.4, IN_L - 0.4), rng.uniform(-IN_W + 0.4, IN_W - 0.4)
            if abs(x - xp) > r + 0.2 and clear_of_approaches((x - r, x + r, y - r, y + r)):
                lay.circle(x, y, r, h, mat=mat)

    plan = Planner(lay)
    ends = [_door_end(door) for door in doors]
    left_of = [_inside(door)[0] < xp for door in doors]
    cross = [(a, b) for a in range(len(ends)) for b in range(len(ends)) if left_of[a] != left_of[b]]
    same = [(a, b) for a in range(len(ends)) for b in range(len(ends)) if a != b and left_of[a] == left_of[b]]
    _paths_between(lay, plan, ends, rng, count=10, weight=1.5, via=[(xp, y) for y in openings], pairs=cross)
    _paths_between(lay, plan, ends, rng, count=4, weight=0.5, pairs=same)
    for y in openings:  # someone waiting beside the doorway
        p = (xp + float(rng.choice([-1.0, 1.0])) * 1.2, y + float(rng.choice([-1.0, 1.0])) * (width / 2 + 0.5))
        if plan.clearance(p) > 0.3:
            lay.standing_spot(*p)
    left_spots = [(x, y) for x in np.linspace(-IN_L + 1.2, xp - 2.0, 4) for y in np.linspace(-IN_W + 1.2, IN_W - 1.2, 4)]
    right_spots = [(x, y) for x in np.linspace(xp + 2.0, IN_L - 1.2, 4) for y in np.linspace(-IN_W + 1.2, IN_W - 1.2, 4)]
    rng.shuffle(left_spots)
    rng.shuffle(right_spots)
    for a, b in zip(left_spots[:6], right_spots[:6]):  # across the partition, both ways
        if len(lay.routes) >= MAX_ROUTES:
            break
        _routes(lay, plan, [a, b] if rng.uniform() < 0.5 else [b, a], rng, count=len(lay.routes) + 1)
    lay.crowd = (2, 10)
    return lay


def build_office(d: float, rng: np.random.Generator, civic_rng: np.random.Generator | None = None) -> TileLayout:
    """An open-plan office: rows of cubicles (partitions, desks, chairs) along a main aisle, cross aisles between the
    blocks, printers, plants and columns; people sit, stand at cubicle entrances and walk the aisles."""
    lay = _new_layout("office", d)
    _floor(lay, "carpet")
    main = _lerp(1.8, 1.05, d) + rng.uniform(0.0, 0.1)
    cross = _lerp(1.6, 1.0, d) + rng.uniform(0.0, 0.1)
    ym = rng.uniform(-0.5, 0.5)
    cube = min(rng.uniform(2.1, 2.5), IN_W - 1.25 - main / 2 - abs(ym))
    part_h = rng.uniform(1.15, 1.6)
    door_w = 1.0
    top = [(x, door_w) for x in _spread(rng, 2, -6.5, 6.5, 4.0)]
    bottom = [(x, door_w) for x in _spread(rng, 1, -6.5, 6.5, 4.0)]
    left = [(ym, door_w)]
    right = [(ym, door_w)] if rng.uniform() < 0.5 else []
    doors = _room(lay, top=top, bottom=bottom, left=left, right=right)
    # cubicle rows above and below the main aisle; blocks of 2-3 cubicles with cross aisles between them
    x_lo, x_hi = -IN_L + 1.3, IN_L - 1.3
    blocks, x = [], x_lo
    while x + 2 * cube <= x_hi:
        n = int(rng.integers(2, 4))
        n = min(n, int((x_hi - x) // cube))
        blocks.append((x, x + n * cube, n))
        x += n * cube + cross
    t = 0.06
    entrances = []
    for sign in (1.0, -1.0):
        f = ym + sign * main / 2  # front line (towards the main aisle)
        b = f + sign * cube  # back line
        for bx0, bx1, n in blocks:
            _wall_x(lay, b - sign * t / 2, bx0, bx1, height=part_h, mat="partition", thickness=t)
            for i in range(n + 1):
                _wall_y(lay, bx0 + i * cube, min(f, b), max(f, b), height=part_h, mat="partition", thickness=t)
            for i in range(n):
                cx0 = bx0 + i * cube
                opening = rng.uniform(cx0 + 0.6, cx0 + cube - 0.6)
                _wall_x(lay, f + sign * t / 2, cx0, cx0 + cube, [(opening, 0.85)], height=part_h, mat="partition", thickness=t)
                lay.table(cx0 + cube / 2, b - sign * 0.38, 0.0, cube - 0.2, 0.7, 0.74, mat="desk")
                if rng.uniform() < 0.8:
                    lay.circle(cx0 + cube / 2 + rng.uniform(-0.4, 0.4), b - sign * rng.uniform(1.0, 1.3), 0.3, 1.0, mat="chair")
                entrances.append((opening, f - sign * 0.35))
    # plants in the corners, a column in a wide main aisle
    for sx in (-1.0, 1.0):
        for sy in (-1.0, 1.0):
            if rng.uniform() < 0.5:
                lay.circle(sx * (IN_L - 0.35), sy * (IN_W - 0.35), 0.3, 1.3, mat="planter")
    if main > 1.5 and rng.uniform() < 0.5:
        lay.box(rng.uniform(-3.0, 3.0), ym + float(rng.choice([-1, 1])) * (main / 2 - 0.3), 0.0, 0.5, 0.5, WALL_H, mat="column")

    plan = Planner(lay)
    ends = [_door_end(door) for door in doors]
    _paths_between(lay, plan, ends, rng, count=12, weight=1.0)
    rng.shuffle(entrances)
    for p in entrances[: int(_lerp(2, 7, d))]:
        if plan.clearance(p) > 0.28:
            lay.standing_spot(*p)
    aisle_y = [ym, (ym + main / 2 + cube + IN_W) / 2, (ym - main / 2 - cube - IN_W) / 2]
    cross_x = [-IN_L + 0.65] + [b[1] + cross / 2 for b in blocks[:-1]] + [IN_L - 0.65]
    _routes(lay, plan, [(x, y) for x in cross_x for y in aisle_y], rng)
    lay.crowd = (2, 8)
    return lay


def build_warehouse(d: float, rng: np.random.Generator, civic_rng: np.random.Generator | None = None) -> TileLayout:
    """A warehouse: rows of racks, pallets and stacked goods in the aisles, a pallet jack, people picking at the racks,
    and an open loading dock (a 1.2 m drop) along one side."""
    lay = _new_layout("warehouse", d)
    _floor(lay, "floor_concrete")
    aisle = _lerp(2.6, 1.3, d) + rng.uniform(0.0, 0.2)
    cross = _lerp(2.6, 1.4, d)
    dock = rng.uniform() < 0.8
    y_dock = -IN_W + 1.6  # floor edge; the truck pit lies below it
    door_w = 1.2
    top = [(x, door_w) for x in _spread(rng, 2, -6.5, 6.5, 4.0)]
    left, right = [(rng.uniform(0.0, 3.0), door_w)], [(rng.uniform(0.0, 3.0), door_w)]
    doors = _room(lay, top=top, bottom=[], left=left, right=right)
    if dock:
        lay.add_rect(-IN_L, IN_L, -IN_W, y_dock - 0.05, -1.2)
        lay.paint_rect(-IN_L, IN_L, y_dock - 0.05, y_dock + 0.15, "hazard_stripe")
        lay.forbidden.append((-HALF_L, HALF_L, -HALF_W, y_dock))
    floor_lo = (y_dock + 1.0) if dock else -IN_W + 0.6
    # racks along x, back-to-back pairs in the middle; a cross aisle splits each row
    rack_d = rng.uniform(0.9, 1.1)
    xc = rng.uniform(-2.0, 2.0)
    y = IN_W - rack_d / 2 - 0.05  # a single row against the top wall, between its doors
    rows = [(y, rack_d, True)]
    y -= rack_d / 2 + aisle
    while y - 2 * rack_d - aisle >= floor_lo:
        rows.append((y - rack_d, 2 * rack_d, False))
        y -= 2 * rack_d + aisle
    rack_h = rng.uniform(2.2, 3.0)
    aisle_ys = []
    for yc, depth, wall in rows:
        for x0, x1 in ((-IN_L + 1.6, xc - cross / 2), (xc + cross / 2, IN_L - 1.6)):
            if wall:  # leave the top doors free
                a = x0
                for c0, c1 in sorted((x - 1.2, x + 1.2) for x, _ in top) + [(x1, x1)]:
                    if c0 - a > 1.0:
                        lay.box((a + min(c0, x1)) / 2, yc, 0.0, min(c0, x1) - a, depth, rack_h, mat="rack")
                    a = max(a, c1)
            elif x1 - x0 > 1.0:
                lay.box((x0 + x1) / 2, yc, 0.0, x1 - x0, depth, rack_h, mat="rack")
        aisle_ys.append(yc - depth / 2 - aisle / 2)
    bands = [(-IN_L, IN_L, ay - aisle / 2, ay + aisle / 2, "x") for ay in aisle_ys]
    bands += [(xc - cross / 2, xc + cross / 2, -IN_W, IN_W, "y"), (-IN_L, -IN_L + 1.6, -IN_W, IN_W, "y"),
              (IN_L - 1.6, IN_L, -IN_W, IN_W, "y")]
    keep = _Bands(bands, 0.62 + _lerp(0.5, 0.25, d), doors=doors)
    # pallets (low), stacked goods and a pallet jack along the rack faces
    for _ in range(int(_lerp(2, 9, d))):
        ay = aisle_ys[int(rng.integers(len(aisle_ys)))]
        side = float(rng.choice([-1.0, 1.0]))
        kind = rng.choice(["pallet", "stack", "jack", "carton"])
        sx, sy, h, mat = {"pallet": (1.2, 0.8, 0.15, "pallet"), "stack": (1.2, 0.8, rng.uniform(0.5, 1.4), "pallet_stack"),
                          "jack": (1.5, 0.55, 0.2, "pallet_jack"), "carton": (0.6, 0.4, rng.uniform(0.2, 0.5), "carton")}[kind]
        x = rng.uniform(-IN_L + 2.0, IN_L - 2.0)
        cy = ay + side * (aisle / 2 - sy / 2 - 0.03)
        if keep.place((x - sx / 2, x + sx / 2, cy - sy / 2, cy + sy / 2)):
            lay.box(x, cy, 0.0, sx, sy, h, mat=mat)
            if kind == "jack":
                lay.circle(x + 0.75 * float(rng.choice([-1, 1])), cy, 0.03, 1.15, mat="pallet_jack")

    plan = Planner(lay)
    ends = [_door_end(door, 1.0) for door in doors]
    _paths_between(lay, plan, ends, rng, count=10, weight=1.0, via=[(x, ay) for x in (xc,) for ay in aisle_ys])
    for _ in range(int(_lerp(2, 6, d))):  # pickers at the racks
        ay = aisle_ys[int(rng.integers(len(aisle_ys)))]
        p = (rng.uniform(-IN_L + 2.0, IN_L - 2.0), ay + float(rng.choice([-1.0, 1.0])) * (aisle / 2 - 0.35))
        if plan.clearance(p) > 0.28:
            lay.standing_spot(*p)
    candidates = [(x, ay) for x in (-IN_L + 0.8, xc, IN_L - 0.8) for ay in aisle_ys]
    if dock:  # along the dock edge, where a slip ends in the pit
        candidates += [(-IN_L + 1.2, y_dock + 0.75), (IN_L - 1.2, y_dock + 0.75)]
    _routes(lay, plan, candidates, rng)
    lay.crowd = (1, 6)
    return lay


def build_lobby(d: float, rng: np.random.Generator, civic_rng: np.random.Generator | None = None) -> TileLayout:
    """A building lobby: columns, a reception desk, seating, planters, a queue between stanchions and an open
    stairwell down to the floor below; busy crossing traffic."""
    lay = _new_layout("lobby", d)
    _floor(lay, "floor_stone")
    door_w = 1.2
    top = [(x, door_w) for x in _spread(rng, 2, -6.0, 6.0, 4.0)]
    bottom = [(x, door_w) for x in _spread(rng, 2, -6.0, 6.0, 4.0)]
    entrance = [(rng.uniform(-2.0, 2.0), 2.4)]
    doors = _room(lay, top=top, bottom=bottom, left=entrance, right=[(rng.uniform(-2.0, 2.0), door_w)])
    # columns on a grid
    spacing = rng.uniform(4.0, 5.5)
    round_cols = rng.uniform() < 0.5
    for x in np.arange(-IN_L + spacing / 2 + 0.6, IN_L - 1.0, spacing):
        for y in (-IN_W / 2 - 0.2, IN_W / 2 + 0.2):
            if round_cols:
                lay.circle(float(x), y, 0.3, WALL_H, mat="column")
            else:
                lay.box(float(x), y, 0.0, 0.6, 0.6, WALL_H, mat="column")
    # stairwell down (railed on three sides, open towards its top step)
    stairs = None
    if rng.uniform() < 0.7:
        length, depth = 3.2, 1.6
        sx, sy = rng.uniform(-4.0, 4.0), float(rng.choice([-1.0, 1.0])) * rng.uniform(0.3, 1.2)
        x0, x1, y0, y1 = sx - length / 2, sx + length / 2, sy - depth / 2, sy + depth / 2
        top_at_x1 = rng.uniform() < 0.5
        for ax, ay, bx, by in ((x0, y0, x1, y0), (x0, y1, x1, y1), (x0, y0, x0, y1) if top_at_x1 else (x1, y0, x1, y1)):
            lay.segment_box(ax, ay, bx, by, 0.05, 1.0, mat="railing")
        steps = int(length / 0.28)
        for k in range(steps):  # descending away from the open end
            xa = (x1 - (k + 1) * 0.28) if top_at_x1 else (x0 + k * 0.28)
            lay.set_rect(xa, xa + 0.28, y0 + 0.08, y1 - 0.08, -0.17 * (k + 1))
        lay.paint_rect(x0 + 0.05, x1 - 0.05, y0 + 0.05, y1 - 0.05, "stairwell")
        lay.forbidden.append((x0 - 0.05, x1 + 0.05, y0, y1))
        stairs = (x0, x1, y0, y1, top_at_x1)
        lay.decor.append(("stairs", x0 + 0.05, x1 - 0.05, y0 + 0.05, y1 - 0.05))
    # reception desk, seating groups, planters, stanchions
    for _ in range(40):
        rx, ry = rng.uniform(-4.0, 4.0), float(rng.choice([-1.0, 1.0])) * (IN_W - 0.9)
        if min(abs(rx - x) for x, _ in (top if ry > 0 else bottom)) > 2.5:
            lay.box(rx, ry, 0.0, 3.0, 0.8, 1.1, mat="reception")
            break
    for _ in range(int(_lerp(3, 7, d))):
        x, y = rng.uniform(-IN_L + 1.5, IN_L - 1.5), rng.uniform(-IN_W + 1.2, IN_W - 1.2)
        kind = rng.choice(["sofa", "armchair", "coffee", "planter"])
        sx, sy, h = {"sofa": (2.0, 0.85, 0.8), "armchair": (0.8, 0.8, 0.8), "coffee": (1.0, 0.6, 0.45), "planter": (1.0, 1.0, 0.8)}[kind]
        if stairs is not None and not (x + sx / 2 + 1.0 < stairs[0] or x - sx / 2 - 1.0 > stairs[1] or y + sy / 2 + 1.0 < stairs[2] or y - sy / 2 - 1.0 > stairs[3]):
            continue
        if kind == "coffee":
            lay.table(x, y, 0.0, sx, sy, h, mat="coffee_table")
        else:
            lay.box(x, y, float(rng.uniform(0, math.pi)) if kind == "planter" else 0.0, sx, sy, h, mat=kind)
    if rng.uniform() < _lerp(0.3, 0.8, d):  # a queue lane between stanchions and belts
        qx, qy = rng.uniform(-3.0, 3.0), float(rng.choice([-1.0, 1.0])) * rng.uniform(1.5, 2.8)
        for dy in (-0.55, 0.55):
            posts = np.linspace(qx - 2.0, qx + 2.0, 4)
            for px in posts:
                lay.circle(float(px), qy + dy, 0.15, 0.95, mat="stanchion")
            lay.segment_box(posts[0], qy + dy, posts[-1], qy + dy, 0.03, 0.06, z0=0.85, mat="stanchion_belt")

    plan = Planner(lay)
    ends = [_door_end(door, 1.0) for door in doors]
    if stairs is not None:  # people come up the stairs and go down them
        x0, x1, y0, y1, top_at_x1 = stairs
        yc = (y0 + y1) / 2
        bottom_pt, top_pt = ((x0 + 0.3, yc), (x1 + 0.6, yc)) if top_at_x1 else ((x1 - 0.3, yc), (x0 - 0.6, yc))
        ends.append([bottom_pt, top_pt])
    _paths_between(lay, plan, ends, rng, count=14, weight=1.0)
    spots = [(rng.uniform(-6.0, 6.0), rng.uniform(-4.0, 4.0)) for _ in range(30)]
    for p in [p for p in spots if plan.clearance(p) > 0.5][: int(_lerp(2, 7, d))]:
        lay.standing_spot(*p)
    edge = [(x, y) for x in (-IN_L + 1.0, -3.0, 3.0, IN_L - 1.0) for y in (-IN_W + 1.0, 0.0, IN_W - 1.0)]
    _routes(lay, plan, edge, rng, min_length=6.0)
    lay.crowd = (4, 14)
    return lay


INDOOR_BUILDERS: dict[str, Callable[..., TileLayout]] = {
    "lab": build_lab,
    "corridor": build_corridor,
    "doorway": build_doorway,
    "office": build_office,
    "warehouse": build_warehouse,
    "lobby": build_lobby,
}

INDOOR_KINDS = list(INDOOR_BUILDERS)
