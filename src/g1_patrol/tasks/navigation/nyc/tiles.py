"""Procedural New York street tiles.

Each tile is ``TILE_SIZE`` (16 m along the street x 10 m across) and is built by one scenario builder at a difficulty
in [0, 1]. A builder fills a :class:`TileLayout` with

* a ground height field (sidewalks, curbs, curb cuts, potholes, broken pavement, steel plates),
* static structures, each kept both as a mesh and as a 2D primitive for collision checks and privileged observations:
  circles (poles, hydrants, trash cans, posts) and oriented boxes (walls, fences, barriers, planters, rails),
* scenario semantics: robot routes, pedestrian paths (polylines from where people enter the scene - off the tile, a
  door, a subway stairwell - to where they leave it, off the tile), spots where people stand, and street areas the
  robot should not walk in,
* looks, for rendering only: a material name per structure, painted ground regions (asphalt, sidewalk, zebra stripes,
  steel plates, ...) and visual-only decor (trees). They never change the geometry the robot trains on.

Coordinates are tile-local, centered on the tile, x along the street; heights are meters above street level.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

TILE_SIZE = (16.0, 10.0)
HF_RESOLUTION = 0.1
OVERHEAD_Z = 2.0
"""Structures whose bottom is higher than this (scaffold decks, beams) do not block walking."""

# padded per-tile capacities of the layout tensors
MAX_CIRCLES, MAX_BOXES, MAX_ROUTES, MAX_PATHS, MAX_STANDING, MAX_FORBIDDEN = 64, 80, 4, 16, 12, 4
MAX_PATH_POINTS = 8
MAX_BASKETS, MAX_PILES, MAX_QUEUES, MAX_CROSSINGS = 6, 2, 2, 1
"""Civic tiles: litter baskets, curbside trash-bag piles, queues and signalized crossings."""
PATH_ROW = 2 * MAX_PATH_POINTS + 4
"""Width of a padded path row: point coordinates, number of points, half width, weight, free length."""
OFF_TILE = 45.0
"""Paths along the street start and end this far (m) beyond the tile's ends, so people walk into and out of view."""


@dataclass
class TileLayout:
    kind: str
    difficulty: float
    heights: np.ndarray  # (nx + 1, ny + 1) ground heights at grid points [m]
    circles: list = field(default_factory=list)  # (x, y, radius, z0, z1)
    boxes: list = field(default_factory=list)  # (cx, cy, yaw, half_x, half_y, z0, z1)
    routes: list = field(default_factory=list)  # (start_x, start_y, goal_x, goal_y)
    paths: list = field(default_factory=list)  # (points, half_width, weight, free_length), walked first to last
    standing: list = field(default_factory=list)  # (x, y)
    forbidden: list = field(default_factory=list)  # (x0, x1, y0, y1) street areas
    facades: list = field(default_factory=list)  # (y, facing): building fronts along the tile's long edges, facing +-y
    crowd: tuple = (4, 12)  # number of pedestrians at difficulty 0 and 1
    # civic tiles (litter, trash bags, queues, signals); empty otherwise
    civic: bool = False
    piles: list = field(default_factory=list)  # (x0, x1, y0, y1): where trash bags are set out at the curb
    queues: list = field(default_factory=list)  # (x, y, ux, uy, count): head of a line of people and its direction
    crossings: list = field(default_factory=list)  # (x0, x1, y0, y1): crosswalks with pedestrian signals
    # looks (rendering only)
    circle_mats: list = field(default_factory=list)  # material name per circle
    box_mats: list = field(default_factory=list)  # material name per box
    paint: list = field(default_factory=list)  # (x0, x1, y0, y1, material) ground regions, painted in order
    decor: list = field(default_factory=list)  # visual-only extras: ("tree", x, y)

    # -- geometry helpers ---------------------------------------------------------------------------------------------

    def _index(self, x: float, y: float) -> tuple[int, int]:
        i = int(round((x + TILE_SIZE[0] / 2) / HF_RESOLUTION))
        j = int(round((y + TILE_SIZE[1] / 2) / HF_RESOLUTION))
        return min(max(i, 0), self.heights.shape[0] - 1), min(max(j, 0), self.heights.shape[1] - 1)

    def ground(self, x: float, y: float) -> float:
        return float(self.heights[self._index(x, y)])

    def set_rect(self, x0: float, x1: float, y0: float, y1: float, height: float):
        i0, j0 = self._index(x0, y0)
        i1, j1 = self._index(x1, y1)
        self.heights[i0 : i1 + 1, j0 : j1 + 1] = height

    def add_rect(self, x0: float, x1: float, y0: float, y1: float, delta: float):
        i0, j0 = self._index(x0, y0)
        i1, j1 = self._index(x1, y1)
        self.heights[i0 : i1 + 1, j0 : j1 + 1] += delta

    def ramp_y(self, x0: float, x1: float, y0: float, y1: float, h0: float, h1: float):
        """Linear ramp in y from height h0 at y0 to h1 at y1 (curb cuts)."""
        i0, j0 = self._index(x0, min(y0, y1))
        i1, j1 = self._index(x1, max(y0, y1))
        for j in range(j0, j1 + 1):
            y = -TILE_SIZE[1] / 2 + j * HF_RESOLUTION
            t = (y - y0) / (y1 - y0)
            self.heights[i0 : i1 + 1, j] = h0 + (h1 - h0) * min(max(t, 0.0), 1.0)

    def noise(self, x0: float, x1: float, y0: float, y1: float, amplitude: float, rng: np.random.Generator):
        if amplitude <= 0:
            return
        i0, j0 = self._index(x0, y0)
        i1, j1 = self._index(x1, y1)
        shape = (i1 - i0 + 1, j1 - j0 + 1)
        # paving-slab scale bumps: random values on a coarse grid, repeated to 0.3 m cells
        coarse = rng.uniform(-amplitude, amplitude, size=(shape[0] // 3 + 1, shape[1] // 3 + 1))
        self.heights[i0 : i1 + 1, j0 : j1 + 1] += np.kron(coarse, np.ones((3, 3)))[: shape[0], : shape[1]]

    def potholes(self, x0, x1, y0, y1, count: int, depth: float, rng: np.random.Generator):
        for _ in range(count):
            w, l = rng.uniform(0.3, 0.9, size=2)
            cx, cy = rng.uniform(x0 + w, x1 - w), rng.uniform(y0 + l, y1 - l)
            self.add_rect(cx - w / 2, cx + w / 2, cy - l / 2, cy + l / 2, -rng.uniform(0.5, 1.0) * depth)
            self.paint_rect(cx - w / 2, cx + w / 2, cy - l / 2, cy + l / 2, "pothole")

    def paint_rect(self, x0: float, x1: float, y0: float, y1: float, material: str):
        self.paint.append((x0, x1, y0, y1, material))

    # -- structures ---------------------------------------------------------------------------------------------------

    def circle(self, x: float, y: float, radius: float, height: float, z0: float = 0.0, mat: str = "pole"):
        self.circles.append((x, y, radius, z0, z0 + height))
        self.circle_mats.append(mat)

    def box(self, cx, cy, yaw, length: float, width: float, height: float, z0: float = 0.0, mat: str = "concrete"):
        self.boxes.append((cx, cy, yaw, length / 2, width / 2, z0, z0 + height))
        self.box_mats.append(mat)

    def segment_box(self, ax, ay, bx, by, width: float, height: float, z0: float = 0.0, mat: str = "concrete"):
        length = math.hypot(bx - ax, by - ay)
        self.box((ax + bx) / 2, (ay + by) / 2, math.atan2(by - ay, bx - ax), length, width, height, z0, mat)

    def low_fence(self, x0: float, x1: float, y0: float, y1: float, height: float = 0.45):
        """Tree-pit guard: thin rails around a rectangle, with posts at the corners."""
        t = 0.03
        for ax, ay, bx, by in ((x0, y0, x1, y0), (x1, y0, x1, y1), (x1, y1, x0, y1), (x0, y1, x0, y0)):
            self.segment_box(ax, ay, bx, by, t, 0.04, z0=height - 0.04, mat="tree_guard")  # top rail
            self.segment_box(ax, ay, bx, by, t, 0.04, z0=0.06, mat="tree_guard")  # bottom rail
        for x, y in ((x0, y0), (x1, y0), (x1, y1), (x0, y1)):
            self.circle(x, y, 0.02, height, mat="tree_guard")

    def bike_rack(self, x: float, y: float, yaw: float):
        """Inverted-U rack: two posts and a top bar (0.9 m tall, 0.7 m wide)."""
        dx, dy = 0.35 * math.cos(yaw), 0.35 * math.sin(yaw)
        self.circle(x - dx, y - dy, 0.025, 0.9, mat="bike_rack")
        self.circle(x + dx, y + dy, 0.025, 0.9, mat="bike_rack")
        self.segment_box(x - dx, y - dy, x + dx, y + dy, 0.05, 0.05, z0=0.85, mat="bike_rack")

    def path(self, points, half_width: float, weight: float, free_length: float = 0.0):
        """A pedestrian path from where people enter the scene to where they leave it. Walkers keep a random lateral
        offset of up to ``half_width``; structures do not push them during the first ``free_length`` meters (out of
        a doorway or up a stairwell)."""
        assert 2 <= len(points) <= MAX_PATH_POINTS
        self.paths.append(([(float(x), float(y)) for x, y in points], float(half_width), float(weight), float(free_length)))

    def path_rows(self) -> list:
        """Paths as rows of width ``PATH_ROW`` (points padded with the last point)."""
        rows = []
        for points, half_width, weight, free_length in self.paths:
            padded = points + [points[-1]] * (MAX_PATH_POINTS - len(points))
            rows.append([c for point in padded for c in point] + [len(points), half_width, weight, free_length])
        return rows

    def standing_spot(self, x: float, y: float):
        self.standing.append((x, y))

    # -- meshes -------------------------------------------------------------------------------------------------------

    # trimesh (and with it scipy/OpenBLAS) is imported lazily: OpenBLAS threads loaded before the simulator starts
    # can crash Kit when it forks during start-up

    def ground_mesh(self):
        import trimesh

        from isaaclab.terrains.height_field.utils import convert_height_field_to_mesh

        vertical_scale = 0.005
        hf = np.round(self.heights / vertical_scale).astype(np.int16)
        vertices, triangles = convert_height_field_to_mesh(hf, HF_RESOLUTION, vertical_scale, slope_threshold=0.75)
        vertices[:, 0] -= TILE_SIZE[0] / 2
        vertices[:, 1] -= TILE_SIZE[1] / 2
        return trimesh.Trimesh(vertices=vertices, faces=triangles)

    def baskets(self) -> list:
        """Litter baskets as (x, y, radius, height of the rim above the tile's zero)."""
        return [(x, y, r, self.ground(x, y) + z1) for (x, y, r, _, z1), mat in zip(self.circles, self.circle_mats)
                if mat == "litter_basket"]

    def structure_mesh(self):
        import trimesh

        meshes = [m for _, m in self._structure_parts()]
        return trimesh.util.concatenate(meshes) if meshes else None

    def structure_meshes_by_material(self) -> dict:
        """The structures as one mesh per material (rendering)."""
        import trimesh

        groups: dict[str, list] = {}
        for mat, m in self._structure_parts(sections=24):
            groups.setdefault(mat, []).append(m)
        return {mat: trimesh.util.concatenate(ms) for mat, ms in groups.items()}

    def _structure_parts(self, sections: int = 12):
        import trimesh

        for (x, y, r, z0, z1), mat in zip(self.circles, self.circle_mats):
            if mat == "litter_basket" and self.civic:  # open top: a wall and a floor, so that dropped trash falls in
                wall = trimesh.creation.annulus(r_min=r - BASKET_WALL, r_max=r, height=z1 - z0, sections=max(sections, 24))
                wall.apply_translation((x, y, self.ground(x, y) + (z0 + z1) / 2))
                floor = trimesh.creation.cylinder(radius=r - BASKET_WALL, height=0.04, sections=max(sections, 24))
                floor.apply_translation((x, y, self.ground(x, y) + z0 + 0.02))
                yield mat, trimesh.util.concatenate([wall, floor])
                continue
            m = trimesh.creation.cylinder(radius=r, height=z1 - z0, sections=sections)
            m.apply_translation((x, y, self.ground(x, y) + (z0 + z1) / 2))
            yield mat, m
        for (cx, cy, yaw, hx, hy, z0, z1), mat in zip(self.boxes, self.box_mats):
            m = trimesh.creation.box(extents=(2 * hx, 2 * hy, z1 - z0))
            m.apply_transform(trimesh.transformations.rotation_matrix(yaw, (0, 0, 1)))
            m.apply_translation((cx, cy, self.ground(cx, cy) + (z0 + z1) / 2))
            yield mat, m

    def ground_materials(self) -> np.ndarray:
        """Material name per height-field cell (nx, ny), for rendering: raised areas are sidewalk, the rest asphalt,
        then the painted regions in order."""
        nx, ny = self.heights.shape[0] - 1, self.heights.shape[1] - 1
        cell_h = 0.25 * (self.heights[:-1, :-1] + self.heights[1:, :-1] + self.heights[:-1, 1:] + self.heights[1:, 1:])
        mats = np.where(cell_h > 0.05, "sidewalk", "asphalt").astype(object)
        xs = -TILE_SIZE[0] / 2 + (np.arange(nx) + 0.5) * HF_RESOLUTION
        ys = -TILE_SIZE[1] / 2 + (np.arange(ny) + 0.5) * HF_RESOLUTION
        for x0, x1, y0, y1, mat in self.paint:
            ix = (xs >= x0) & (xs <= x1)
            iy = (ys >= y0) & (ys <= y1)
            mats[np.ix_(ix, iy)] = mat
        return mats


def _new_layout(kind: str, difficulty: float, civic: bool = False) -> TileLayout:
    nx = int(round(TILE_SIZE[0] / HF_RESOLUTION))
    ny = int(round(TILE_SIZE[1] / HF_RESOLUTION))
    return TileLayout(kind=kind, difficulty=difficulty, heights=np.zeros((nx + 1, ny + 1)), civic=civic)


def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


HALF_L, HALF_W = TILE_SIZE[0] / 2, TILE_SIZE[1] / 2
OFF = HALF_L + OFF_TILE
"""x of the off-tile ends of paths along the street."""
BUILDING_Y = HALF_W - 0.6
"""Facade line; the building occupies y > BUILDING_Y."""
BASKET_RADIUS, BASKET_HEIGHT, BASKET_WALL = 0.3, 0.95, 0.02
"""New York litter basket (open top): the robot drops collected trash into these."""
EDGE_WALL = 0.3
"""Depth of the building fronts along both long edges of open tiles (crosswalk, plaza)."""
ROUTE_Y = HALF_W - EDGE_WALL - 0.8
"""Ends of the crossing routes: in the middle of the sidewalk's free band between the bollards at the curb and the
building fronts."""


def _edge_facades(lay: TileLayout):
    """Building fronts along both long edges of an open tile, so that the walkable area ends where the city around it
    begins (the robot perceives them and stops at them, as at any other facade)."""
    for side in (1.0, -1.0):
        lay.box(0.0, side * (HALF_W - EDGE_WALL / 2), 0.0, 2 * HALF_L, EDGE_WALL, 4.0, mat="facade")
        lay.facades.append((side * (HALF_W - EDGE_WALL), -side))


def _sidewalk_street(lay: TileLayout, d: float, width: float, rng: np.random.Generator, doors: int = 2) -> float:
    """Street (y < curb) with potholes, a raised sidewalk up to the facade, and a facade with door recesses.

    Returns the curb line y.
    """
    curb_y = BUILDING_Y - width
    curb_h = _lerp(0.10, 0.18, d)
    lay.set_rect(-HALF_L, HALF_L, curb_y, HALF_W, curb_h)
    lay.noise(-HALF_L, HALF_L, curb_y + 0.2, BUILDING_Y, _lerp(0.0, 0.025, d), rng)
    lay.noise(-HALF_L, HALF_L, -HALF_W, curb_y - 0.1, _lerp(0.005, 0.03, d), rng)
    lay.potholes(-HALF_L, HALF_L, -HALF_W, curb_y - 0.2, int(_lerp(2, 14, d)), _lerp(0.04, 0.12, d), rng)
    lay.forbidden.append((-HALF_L, HALF_L, -HALF_W, curb_y))
    lay.facades.append((BUILDING_Y, -1.0))
    lay.paint_rect(-HALF_L, HALF_L, curb_y - 0.05, curb_y + 0.3, "curb")
    if curb_y - 2.4 > -HALF_W:  # parking lane line
        lay.paint_rect(-HALF_L, HALF_L, curb_y - 2.5, curb_y - 2.38, "stripe_white")
    # facade with recessed doors (people step out of them; the robot can wait in them)
    door_xs = sorted(rng.uniform(-HALF_L + 2.0, HALF_L - 2.0, size=doors))
    edges = [-HALF_L]
    for x in door_xs:
        edges += [x - 0.8, x + 0.8]
    edges.append(HALF_L)
    for a, b in zip(edges[0::2], edges[1::2]):
        if b - a > 0.1:
            lay.box((a + b) / 2, BUILDING_Y + 0.3, 0.0, b - a, 0.6, 4.0, mat="facade")
    for x in door_xs:
        lay.box(x, HALF_W - 0.1, 0.0, 1.6, 0.2, 4.0, mat="storefront")  # door at the back of the recess
        lay.standing_spot(x, BUILDING_Y + 0.15)
        # people come out of the door (from behind it) and join the foot traffic along the sidewalk
        direction, mid = float(rng.choice([-1.0, 1.0])), curb_y + width / 2
        lay.path([(x, HALF_W + 0.4), (x, BUILDING_Y + 0.2), (x + 3.0 * direction, mid), (direction * OFF, mid)], 0.3, 0.5,
                 free_length=1.2)
    return curb_y


def _street_furniture(lay: TileLayout, d: float, curb_y: float, rng: np.random.Generator, x_ranges=None):
    """Poles, hydrants, trash cans, newsboxes, bike racks and fenced tree pits along the curb."""
    x_ranges = x_ranges or [(-HALF_L + 1.0, HALF_L - 1.0)]
    count = int(_lerp(3, 10, d))
    y = curb_y + 0.55
    for _ in range(count):
        lo, hi = x_ranges[rng.integers(len(x_ranges))]
        x = rng.uniform(lo, hi)
        kind = rng.choice(["pole", "pole", "hydrant", "can", "newsbox", "rack", "tree", "meter"])
        if kind == "pole":
            lay.circle(x, y, rng.uniform(0.04, 0.08), 3.5, mat="pole")
        elif kind == "meter":
            lay.circle(x, y, 0.05, 1.3, mat="meter")
        elif kind == "hydrant":
            lay.circle(x, y, 0.13, 0.75, mat="hydrant")
        elif kind == "can":
            lay.circle(x, y + 0.1, 0.3, 0.95, mat="litter_basket")
        elif kind == "newsbox":
            lay.box(x, y + 0.1, 0.0, 0.5, 0.5, 1.1, mat="newsbox")
        elif kind == "rack":
            lay.bike_rack(x, y + 0.2, 0.0)
        else:
            lay.low_fence(x - 0.9, x + 0.9, curb_y + 0.15, curb_y + 1.15)
            lay.paint_rect(x - 0.9, x + 0.9, curb_y + 0.15, curb_y + 1.15, "tree_pit")
            lay.decor.append(("tree", x, curb_y + 0.65))


def _place_basket(lay: TileLayout, rng: np.random.Generator, x_ranges, y_range, clearance: float = 0.35) -> bool:
    """Put a litter basket at a random spot in ``x_ranges`` x ``y_range`` that keeps ``clearance`` from the other
    structures and 1.3 m from the robot's route ends. Returns whether a spot was found."""
    ends = [p for r in lay.routes for p in (r[:2], r[2:])]
    for _ in range(40):
        lo, hi = x_ranges[rng.integers(len(x_ranges))]
        x, y = rng.uniform(lo, hi), rng.uniform(*y_range)
        gap = min([math.hypot(x - cx, y - cy) - r for cx, cy, r, z0, _ in lay.circles if z0 < 1.0] + [9.0])
        for cx, cy, yaw, hx, hy, z0, _ in lay.boxes:
            if z0 >= 1.0:
                continue
            c, s_ = math.cos(yaw), math.sin(yaw)
            lx, ly = c * (x - cx) + s_ * (y - cy), -s_ * (x - cx) + c * (y - cy)
            gap = min(gap, math.hypot(max(abs(lx) - hx, 0.0), max(abs(ly) - hy, 0.0)))
        if gap - BASKET_RADIUS < clearance or any(math.hypot(x - ex, y - ey) < 1.3 for ex, ey in ends):
            continue
        lay.circle(x, y, BASKET_RADIUS, BASKET_HEIGHT, mat="litter_basket")
        return True
    return False


def _structure_gap(lay: TileLayout, x: float, y: float, half_x: float = 0.0, half_y: float = 0.0) -> float:
    """Gap between an axis-aligned rectangle (a point for zero half sizes) and the nearest structure below 1 m."""
    gap = 9.0
    for cx, cy, r, z0, _ in lay.circles:
        if z0 < 1.0:
            gap = min(gap, math.hypot(max(abs(x - cx) - half_x, 0.0), max(abs(y - cy) - half_y, 0.0)) - r)
    for cx, cy, yaw, hx, hy, z0, _ in lay.boxes:
        if z0 >= 1.0:
            continue
        # conservative: the box's axis-aligned bounds
        c, s_ = abs(math.cos(yaw)), abs(math.sin(yaw))
        ex, ey = c * hx + s_ * hy, s_ * hx + c * hy
        gap = min(gap, math.hypot(max(abs(x - cx) - half_x - ex, 0.0), max(abs(y - cy) - half_y - ey, 0.0)))
    return gap


def _place_pile(lay: TileLayout, rng: np.random.Generator, x_ranges, curb_edge: float, inward: float) -> bool:
    """Trash bags set out for collection: a pile against the curb (``curb_edge``, the sidewalk side lies towards
    ``inward`` = +-1 in y), kept 0.25 m from other structures and 1.5 m from the robot's route ends. The pile is a low
    structure; the zone beside it (along the curb and towards the sidewalk) is where the robot sets bags down."""
    ends = [p for r in lay.routes for p in (r[:2], r[2:])]
    for _ in range(60):
        length, depth, height = rng.uniform(0.8, 1.4), rng.uniform(0.45, 0.6), rng.uniform(0.35, 0.55)
        lo, hi = x_ranges[rng.integers(len(x_ranges))]
        if hi - lo < length + 0.4:
            continue
        x = rng.uniform(lo + length / 2 + 0.2, hi - length / 2 - 0.2)
        y = curb_edge + inward * depth / 2
        if _structure_gap(lay, x, y, length / 2, depth / 2) < 0.25:
            continue
        if any(math.hypot(x - ex, y - ey) < 1.5 + length / 2 for ex, ey in ends):
            continue
        lay.box(x, y, 0.0, length, depth, height, mat="trash_bags")
        # drop zone: beside the pile along the curb and in front of it, 0.6 m wide
        y_in = curb_edge + inward * (depth + 0.6)
        lay.piles.append((x - length / 2 - 0.6, x + length / 2 + 0.6, min(curb_edge, y_in), max(curb_edge, y_in)))
        lay.decor.append(("bag_pile", x, y, length, depth, height))
        return True
    return False


def _sidewalk_paths(lay: TileLayout, curb_y: float, top_y: float, weight: float = 1.0):
    """Two-way foot traffic along the sidewalk between ``curb_y`` and ``top_y``, from and to beyond the tile."""
    width = top_y - curb_y
    for frac, direction in ((0.3, 1.0), (0.7, -1.0)):
        y = curb_y + frac * width
        lay.path([(-direction * OFF, y), (direction * OFF, y)], 0.25 * width, weight)


def _along_routes(lay: TileLayout, y: float):
    lay.routes += [(-HALF_L + 1.0, y, HALF_L - 1.0, y), (HALF_L - 1.0, y, -HALF_L + 1.0, y)]


def build_sidewalk(d: float, rng: np.random.Generator, civic_rng: np.random.Generator | None = None) -> TileLayout:
    """A busy sidewalk: two-way foot traffic, street furniture, tree pits, people stepping out of doors."""
    lay = _new_layout("sidewalk", d, civic_rng is not None)
    width = _lerp(5.0, 2.6, d)
    curb_y = _sidewalk_street(lay, d, width, rng, doors=int(rng.integers(1, 4)))
    _street_furniture(lay, d, curb_y, rng)
    _sidewalk_paths(lay, curb_y + 1.0, BUILDING_Y)
    _along_routes(lay, (curb_y + 1.0 + BUILDING_Y) / 2)
    for _ in range(int(_lerp(0, 3, d))):  # people chatting or on the phone in the middle of the sidewalk
        lay.standing_spot(rng.uniform(-5, 5), rng.uniform(curb_y + 1.2, BUILDING_Y - 0.4))
    lay.crowd = (4, 12)
    if civic_rng is not None:
        _place_basket(lay, civic_rng, [(-HALF_L + 1.0, HALF_L - 1.0)], (curb_y + 0.6, curb_y + 0.7))
        _place_pile(lay, civic_rng, [(-HALF_L + 1.0, HALF_L - 1.0)], curb_y + 0.3, 1.0)
        if civic_rng.uniform() < 0.5:  # a line out of a shop door, along the facade
            door = float(civic_rng.uniform(-HALF_L + 3.0, HALF_L - 3.0))
            direction = float(civic_rng.choice([-1.0, 1.0]))
            lay.queues.append((door, BUILDING_Y - 0.35, direction, 0.0, int(civic_rng.integers(3, 7))))
        else:  # a bus stop line along the curb
            stop = float(civic_rng.uniform(-HALF_L + 3.0, HALF_L - 3.0))
            direction = float(civic_rng.choice([-1.0, 1.0]))
            lay.queues.append((stop, curb_y + 1.0, direction, 0.0, int(civic_rng.integers(3, 6))))
    return lay


def build_scaffold(d: float, rng: np.random.Generator, civic_rng: np.random.Generator | None = None) -> TileLayout:
    """A sidewalk shed: a tunnel of thin poles and rails over the sidewalk; oncoming people force the robot to give way."""
    lay = _new_layout("scaffold", d, civic_rng is not None)
    walkway = _lerp(2.4, 1.25, d)
    inner_posts = rng.uniform() < 0.5
    width = walkway + 0.7 + (0.35 if inner_posts else 0.0)
    curb_y = _sidewalk_street(lay, d, width, rng, doors=int(rng.integers(1, 3)))
    post_y = curb_y + 0.35
    spacing = rng.uniform(2.0, 2.6)
    xs = np.arange(-HALF_L + 0.5, HALF_L, spacing)
    for x in xs:  # curb-side posts, with guard rails at 1 m and 2 m closing the tunnel off from the street
        lay.circle(x, post_y, 0.045, 3.0, mat="scaffold_steel")
        lay.box(x, (post_y + BUILDING_Y) / 2, 0.0, 0.08, BUILDING_Y - post_y, 0.12, z0=2.85, mat="scaffold_steel")
        if inner_posts:
            lay.circle(x, BUILDING_Y - 0.35, 0.045, 3.0, mat="scaffold_steel")
    for a, b in zip(xs[:-1], xs[1:]):
        for z in (0.95, 1.95):
            lay.segment_box(a, post_y, b, post_y, 0.04, 0.05, z0=z, mat="scaffold_steel")
    deck_w = BUILDING_Y - post_y + 0.3
    lay.box(0.0, (post_y + BUILDING_Y) / 2, 0.0, 2 * HALF_L, deck_w, 0.06, z0=3.0, mat="shed_plywood")  # deck
    for _ in range(int(_lerp(0, 2, d))):  # material stacked against the posts (keeps >= 0.9 m of the walkway clear)
        x = rng.choice(xs)
        lay.box(x + 0.5, post_y + 0.2, 0.0, 0.8, 0.3, 0.6, mat="lumber")
    _sidewalk_paths(lay, post_y + 0.2, BUILDING_Y - (0.35 if inner_posts else 0.0) - 0.1, weight=1.5)
    _along_routes(lay, (post_y + BUILDING_Y) / 2)
    lay.crowd = (2, 9)
    return lay


def build_crosswalk(d: float, rng: np.random.Generator, civic_rng: np.random.Generator | None = None) -> TileLayout:
    """Crossing the street: curbs or curb cuts, potholes, waves of people crossing towards the robot."""
    lay = _new_layout("crosswalk", d, civic_rng is not None)
    curb_h = _lerp(0.10, 0.18, d)
    walk = 2.6  # sidewalk depth on each side
    lay.set_rect(-HALF_L, HALF_L, HALF_W - walk, HALF_W, curb_h)
    lay.set_rect(-HALF_L, HALF_L, -HALF_W, -HALF_W + walk, curb_h)
    street = (-HALF_W + walk, HALF_W - walk)
    lay.noise(-HALF_L, HALF_L, street[0] + 0.1, street[1] - 0.1, _lerp(0.005, 0.03, d), rng)
    lay.potholes(-HALF_L, HALF_L, street[0] + 0.2, street[1] - 0.2, int(_lerp(2, 12, d)), _lerp(0.04, 0.12, d), rng)
    cx = rng.uniform(-2.0, 2.0)
    half = 1.6
    if rng.uniform() > d:  # curb cuts (ramps) on easier tiles, plain curbs on harder ones
        lay.ramp_y(cx - half, cx + half, street[1], street[1] + 1.2, 0.0, curb_h)
        lay.ramp_y(cx - half, cx + half, street[0], street[0] - 1.2, 0.0, curb_h)
    lay.forbidden += [(-HALF_L, cx - half, street[0], street[1]), (cx + half, HALF_L, street[0], street[1])]
    _edge_facades(lay)
    # looks: curbs, high-visibility zebra bars across the street, stop lines and a yellow center line
    for y in street:
        lay.paint_rect(-HALF_L, HALF_L, y - 0.15, y + 0.15, "curb")
    y = street[0] + 0.3
    while y + 0.5 < street[1] - 0.2:
        lay.paint_rect(cx - half, cx + half, y, y + 0.5, "stripe_white")
        y += 1.0
    for x0, x1 in ((cx - half - 1.4, cx - half - 1.0), (cx + half + 1.0, cx + half + 1.4)):
        lay.paint_rect(x0, x1, street[0] + 0.2, street[1] - 0.2, "stripe_white")
    for x0, x1 in ((-HALF_L, cx - half - 1.6), (cx + half + 1.6, HALF_L)):
        lay.paint_rect(x0, x1, -0.16, -0.04, "stripe_yellow")
        lay.paint_rect(x0, x1, 0.04, 0.16, "stripe_yellow")
    # signal poles, signs and bollards at the corners
    for sy in (street[1] + 0.4, street[0] - 0.4):
        lay.circle(cx - half - 0.6, sy, 0.1, 4.0, mat="signal_pole")
        lay.circle(cx + half + 0.6 + rng.uniform(0, 2), sy, 0.05, 3.0, mat="pole")
        for _ in range(int(_lerp(0, 3, d))):
            lay.circle(rng.uniform(-HALF_L + 1, HALF_L - 1), sy + np.sign(sy) * 0.3, 0.1, 0.9, mat="bollard")
    # waiting crowds at the curb, crossing flows both ways, sidewalk flows
    for sy in (street[1] + 0.5, street[0] - 0.5):
        for _ in range(2):
            lay.standing_spot(cx + rng.uniform(-half, half), sy + np.sign(sy) * rng.uniform(0, 0.8))
    walk_y = HALF_W - walk / 2
    for y in (walk_y, -walk_y):
        lay.path([(-OFF, y), (OFF, y)], 0.6, 0.6)
        lay.path([(OFF, y), (-OFF, y)], 0.6, 0.6)
    # crossing: along one sidewalk to the crosswalk, across the street, and on along the other sidewalk
    for y in (walk_y, -walk_y):
        for from_x in (-OFF, OFF):
            for to_x in (-OFF, OFF):
                lay.path([(from_x, y), (cx, y), (cx, -y), (to_x, -y)], half * 0.8, 0.5)
    lay.routes += [
        (cx + rng.uniform(-1, 1), -ROUTE_Y, cx + rng.uniform(-1, 1), ROUTE_Y),
        (cx + rng.uniform(-1, 1), ROUTE_Y, cx + rng.uniform(-1, 1), -ROUTE_Y),
    ]
    lay.crowd = (4, 16)
    if civic_rng is not None:
        beside = [(-HALF_L + 1.0, cx - half - 2.5), (cx + half + 2.5, HALF_L - 1.0)]  # corners, beside the crossing
        _place_basket(lay, civic_rng, beside, (street[1] + 0.7, street[1] + 0.8))
        _place_basket(lay, civic_rng, beside, (street[0] - 0.8, street[0] - 0.7))
        side = float(civic_rng.choice([-1.0, 1.0]))  # bags at one corner, against the curb
        _place_pile(lay, civic_rng, beside, street[1] + 0.3 if side > 0 else street[0] - 0.3, -side)
        lay.crossings.append((cx - half, cx + half, street[0], street[1]))
    return lay


def build_construction(d: float, rng: np.random.Generator, civic_rng: np.random.Generator | None = None) -> TileLayout:
    """A construction zone: barriers squeeze the sidewalk into a chicane over steel plates and broken pavement.

    The clear walkway (and the crossover between fence and barriers) narrows from 1.8 m to 1.0 m: one person or the
    robot at a time on hard tiles, so the robot has to wait for oncoming people to come through.
    """
    lay = _new_layout("construction", d, civic_rng is not None)
    width = _lerp(4.5, 3.4, d)
    curb_y = _sidewalk_street(lay, d, width, rng, doors=1)
    clear = _lerp(1.8, 1.0, d)
    barrier = 0.6
    zone = (-3.5, 3.5)
    lay.noise(zone[0], zone[1], curb_y + 0.2, BUILDING_Y, _lerp(0.01, 0.04, d), rng)
    for _ in range(int(_lerp(1, 4, d))):  # steel plates
        x = rng.uniform(zone[0], zone[1])
        lay.add_rect(x - 0.6, x + 0.6, curb_y + 0.3, curb_y + 1.5, 0.025)
        lay.paint_rect(x - 0.6, x + 0.6, curb_y + 0.3, curb_y + 1.5, "steel_plate")
    # chicane: a fence from the facade side, then a run of jersey barriers from the curb side
    fence_y = curb_y + clear
    bx = -0.5 + clear + barrier / 2
    by = BUILDING_Y - clear - barrier / 2
    lay.segment_box(zone[0], BUILDING_Y, zone[0], fence_y, 0.05, 1.2, mat="construction_fence")
    lay.segment_box(zone[0], fence_y, -0.5, fence_y, 0.05, 1.2, mat="construction_fence")
    lay.segment_box(bx, curb_y + 0.1, bx, by, barrier, 0.8, mat="jersey_barrier")
    lay.segment_box(bx, by, zone[1], by, barrier, 0.8, mat="jersey_barrier")
    # cones and workers inside the two work zones (behind the fence, behind the barriers)
    work = [(zone[0] + 0.3, -0.8, fence_y + 0.3, BUILDING_Y - 0.3), (bx + 0.6, zone[1] - 0.3, curb_y + 0.3, by - 0.6)]
    for i in range(int(_lerp(2, 6, d))):
        x0, x1, y0, y1 = work[i % 2]
        lay.circle(rng.uniform(x0, x1), rng.uniform(y0, y1), 0.15, 0.5, mat="cone")
    for _ in range(int(_lerp(1, 3, d))):
        x0, x1, y0, y1 = work[0]
        lay.standing_spot(rng.uniform(x0, x1), rng.uniform(y0, y1))
    _street_furniture(lay, d, curb_y, rng, x_ranges=[(-HALF_L + 1.0, zone[0] - 1.0), (zone[1] + 1.0, HALF_L - 1.0)])
    # foot traffic threads the chicane
    cross_x = -0.5 + clear / 2
    path = [(-OFF, curb_y + width / 2), (zone[0] - 1.0, curb_y + clear / 2), (cross_x, curb_y + clear / 2),
            (cross_x, BUILDING_Y - clear / 2), (zone[1] + 1.0, BUILDING_Y - clear / 2), (OFF, curb_y + width / 2)]
    lay.path(path, 0.25, 2.5)
    lay.path(path[::-1], 0.25, 2.5)
    _along_routes(lay, curb_y + width / 2)
    lay.crowd = (2, 8)
    if civic_rng is not None:
        outside = [(-HALF_L + 1.0, zone[0] - 1.0), (zone[1] + 1.0, HALF_L - 1.0)]
        _place_basket(lay, civic_rng, outside, (curb_y + 0.6, curb_y + 0.7))
        _place_pile(lay, civic_rng, outside, curb_y + 0.3, 1.0)
    return lay


def build_bottleneck(d: float, rng: np.random.Generator, civic_rng: np.random.Generator | None = None) -> TileLayout:
    """A doorway in a partition or a subway entrance: a stream of people comes through; the robot must wait its turn."""
    lay = _new_layout("bottleneck", d, civic_rng is not None)
    width = 4.6
    curb_y = _sidewalk_street(lay, d, width, rng, doors=1)
    opening = _lerp(1.8, 0.9, d)
    if rng.uniform() < 0.5:
        # partition across the sidewalk with one opening
        oy = rng.uniform(curb_y + opening / 2 + 0.2, BUILDING_Y - opening / 2 - 0.2)
        lay.segment_box(0.0, curb_y + 0.05, 0.0, oy - opening / 2, 0.2, 2.5, mat="partition")
        lay.segment_box(0.0, oy + opening / 2, 0.0, BUILDING_Y, 0.2, 2.5, mat="partition")
        lay.path([(OFF, oy), (-OFF, oy)], opening * 0.3, 3.0)
        lay.path([(-OFF, oy), (OFF, oy)], opening * 0.3, 1.0)
        lay.standing_spot(1.5, oy + rng.uniform(-1, 1))
    else:
        # subway entrance: railed stair opening in the middle of the sidewalk, people emerging from below
        length, depth = 4.0, 1.7
        ey = (curb_y + BUILDING_Y) / 2 + rng.uniform(-0.3, 0.3)
        side = (width - depth) / 2
        if side > opening:  # shift it so one side passage has the desired width
            ey = curb_y + opening + depth / 2
        x0, x1, y0, y1 = -length / 2, length / 2, ey - depth / 2, ey + depth / 2
        for ax, ay, bx, by in ((x0, y0, x1, y0), (x0, y1, x1, y1), (x0, y0, x0, y1)):
            lay.segment_box(ax, ay, bx, by, 0.06, 1.0, mat="subway_rail")
        lay.add_rect(x0 + 0.1, x1 - 0.1, y0 + 0.1, y1 - 0.1, -1.0)  # stairwell (fenced off)
        lay.paint_rect(x0 + 0.1, x1 - 0.1, y0 + 0.1, y1 - 0.1, "stairwell")
        # people come up the stairs (which rise towards the open end at x1) and walk off along the sidewalk
        bottom, top, side = (x0 + 0.5, ey), (x1 + 0.6, ey), curb_y + opening / 2
        lay.path([bottom, top, (x1 + 0.6, side), (-OFF, side)], 0.3, 3.0, free_length=x1 - x0)
        lay.path([bottom, top, (OFF, BUILDING_Y - 0.6)], 0.3, 1.5, free_length=x1 - x0)
        lay.path([(-OFF, side), (OFF, side)], 0.25, 1.0)
        lay.decor.append(("stairs", x0 + 0.1, x1 - 0.1, y0 + 0.1, y1 - 0.1))
    _sidewalk_paths(lay, curb_y + 0.3, BUILDING_Y, weight=0.7)
    _along_routes(lay, (curb_y + BUILDING_Y) / 2)
    lay.crowd = (3, 12)
    if civic_rng is not None:
        _place_basket(lay, civic_rng, [(-HALF_L + 1.0, -3.0), (3.0, HALF_L - 1.0)], (curb_y + 0.6, curb_y + 0.7))
        _place_pile(lay, civic_rng, [(-HALF_L + 1.0, -3.0), (3.0, HALF_L - 1.0)], curb_y + 0.3, 1.0)
    return lay


def build_plaza(d: float, rng: np.random.Generator, civic_rng: np.random.Generator | None = None) -> TileLayout:
    """A pedestrian plaza (Times Square): dense criss-crossing crowds, tourists standing around, planters and kiosks."""
    lay = _new_layout("plaza", d, civic_rng is not None)
    _edge_facades(lay)
    lay.noise(-HALF_L, HALF_L, -HALF_W, HALF_W, _lerp(0.0, 0.02, d), rng)
    lay.paint_rect(-HALF_L, HALF_L, -HALF_W, HALF_W, "plaza")
    edge = [(x, y) for x in (-7.5, 7.5) for y in np.linspace(-4, 4, 3)] + [
        (x, y) for y in (-4.0, 4.0) for x in np.linspace(-6, 6, 4)
    ]
    for _ in range(2):
        a, b = rng.choice(len(edge), size=2, replace=False)
        (sx, sy), (gx, gy) = edge[a], edge[b]
        lay.routes.append((sx * 0.9, sy * 0.9, gx * 0.9, gy * 0.9))
    ends = np.array([p for r in lay.routes for p in (r[:2], r[2:])])

    def spot(half_size: float, xr=(-6.0, 6.0), yr=(-3.5, 3.5)) -> tuple[float, float]:
        """A random position whose object stays clear of the robot's start and goal positions."""
        for _ in range(30):
            x, y = rng.uniform(*xr), rng.uniform(*yr)
            if np.min(np.hypot(ends[:, 0] - x, ends[:, 1] - y)) > half_size + 1.2:
                break
        return x, y

    for _ in range(int(_lerp(3, 8, d))):  # planters
        lay.box(*spot(0.85), rng.uniform(0, math.pi), 1.2, 1.2, 0.6, mat="planter")
    for _ in range(int(_lerp(2, 6, d))):  # cafe tables
        lay.box(*spot(0.5), 0.0, 0.7, 0.7, 0.75, mat="cafe_table")
    for _ in range(int(_lerp(2, 6, d))):  # poles and signs
        lay.circle(*spot(0.06, (-7, 7), (-4.5, 4.5)), 0.06, 3.5, mat="pole")
    kiosk = None
    if rng.uniform() < 0.6:
        kiosk = spot(1.25, (-4, 4), (-2, 2))
        lay.box(*kiosk, 0.0, 2.0, 1.5, 2.5, mat="kiosk")
    # foot traffic enters and leaves along the street (the plaza continues beyond the tile's ends): five paths cross
    # the plaza through a random point, three come in and go back out on the same side through two points
    ys = np.linspace(-4.0, 4.0, 5)
    for k in range(8):
        side = float(rng.choice([-1.0, 1.0]))
        start = (side * OFF, float(rng.choice(ys)))
        if k < 5:
            middle = [(rng.uniform(-4.0, 4.0), rng.uniform(-3.5, 3.5))]
            end = (-side * OFF, float(rng.choice(ys)))
        else:
            middle = [(rng.uniform(-4.0, 4.0), rng.uniform(-3.5, -1.0)), (rng.uniform(-4.0, 4.0), rng.uniform(1.0, 3.5))]
            middle = middle if rng.uniform() < 0.5 else middle[::-1]
            end = (side * OFF, float(rng.choice(ys)))
        lay.path([start, *middle, end], 0.6, 1.0)
    for _ in range(int(_lerp(2, 6, d))):
        lay.standing_spot(*spot(0.3))
    lay.crowd = (6, 20)
    if civic_rng is not None:
        for sy in (1.0, -1.0):  # litter baskets against the building fronts
            _place_basket(lay, civic_rng, [(-6.0, 6.0)], (sy * 4.0 - 0.05, sy * 4.0 + 0.05))
        if kiosk is not None:  # a line at the kiosk's window, away from it along the plaza
            kx, ky = kiosk
            direction = float(civic_rng.choice([-1.0, 1.0]))
            lay.queues.append((kx + direction * 1.4, ky, direction, 0.0, int(civic_rng.integers(3, 7))))
    return lay


BUILDERS: dict[str, Callable[..., TileLayout]] = {
    "sidewalk": build_sidewalk,
    "scaffold": build_scaffold,
    "crosswalk": build_crosswalk,
    "construction": build_construction,
    "bottleneck": build_bottleneck,
    "plaza": build_plaza,
}

KINDS = list(BUILDERS)
