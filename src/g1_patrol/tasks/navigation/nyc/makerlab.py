"""Procedural tiles of the lab the B2-W is deployed in: a fabrication and testing hall (photographs in ``lab.zip``).

A section of the hall, 16 x 10 m: a main aisle marked by painted lines runs along the tile and continues beyond its
ends; on both sides drawer-cabinet benches, tall test machines and open racks, steel bins and buckets at the aisle's
edge, rolling carts, pallets with boxes and buckets, a column at the aisle line; on one side an open workstation area
with office chairs and camera tripods, reached through a painted cross aisle. People walk the aisle and the cross
aisle, stand or sit at the benches and machines.

Things with thin legs (tripods, office chairs, carts, open racks, scaffolds) are kept out of as a whole for collisions
(nobody walks or drives between a tripod's legs), while the sensors see what is there: legs, shelves, seats. The floor
lines are paint only; the robot does not perceive them and is not asked to keep to them.
"""

from __future__ import annotations

import math

import numpy as np

from .indoor import IN_L, IN_W, MAX_ROUTES, Planner, _Bands, _door_end, _paths_between, _routes, _room
from .tiles import HALF_L, HALF_W, OFF, TileLayout, _lerp, _new_layout

LINE_W = 0.1
"""Width of the painted aisle lines (m)."""


def _hide(lay: TileLayout):
    """Keep the last box as a collision primitive without a mesh of its own."""
    lay.unmeshed.add(("box", len(lay.boxes) - 1))


def tripod(lay: TileLayout, x: float, y: float, rng: np.random.Generator, height: float = 1.45):
    """A camera tripod: three thin legs spread 0.4-0.5 m around (x, y), a camera on top. Kept out of as a whole."""
    spread = rng.uniform(0.38, 0.5)
    lay.circle(x, y, spread + 0.03, height + 0.15, mat="tripod")
    lay.unmeshed.add(("circle", len(lay.circles) - 1))
    turn = rng.uniform(0, 2 * math.pi)
    for k in range(3):  # each leg as four short segments stepping in towards the head
        a = turn + k * 2 * math.pi / 3
        for seg in range(4):
            z0, z1 = height * seg / 4, height * (seg + 1) / 4
            r = spread * (1.0 - (seg + 0.5) / 4)
            lay.mesh_parts.append(("circle", x + r * math.cos(a), y + r * math.sin(a), 0.015, z0, z1, "tripod"))
    lay.mesh_box(x, y, turn, 0.18, 0.12, 0.15, z0=height, mat="camera")


def office_chair(lay: TileLayout, x: float, y: float, yaw: float):
    """A rolling office chair: five-star base, column, seat, backrest (kept out of within 0.33 m)."""
    lay.circle(x, y, 0.33, 1.0, mat="chair")
    lay.unmeshed.add(("circle", len(lay.circles) - 1))
    for k in range(5):
        a = yaw + k * 2 * math.pi / 5
        lay.mesh_box(x + 0.15 * math.cos(a), y + 0.15 * math.sin(a), a, 0.3, 0.04, 0.05, z0=0.04, mat="chair")
    lay.mesh_parts.append(("circle", x, y, 0.03, 0.09, 0.45, "chair"))
    lay.mesh_parts.append(("circle", x, y, 0.24, 0.45, 0.52, "chair"))
    c, s = math.cos(yaw), math.sin(yaw)
    lay.mesh_box(x - 0.22 * c, y - 0.22 * s, yaw + math.pi / 2, 0.42, 0.05, 0.45, z0=0.55, mat="chair")


def cart(lay: TileLayout, x: float, y: float, yaw: float, length: float = 1.0, width: float = 0.6, height: float = 0.9):
    """A rolling cart: a top and a bottom shelf on four posts; loaded with boxes or toolboxes."""
    lay.box(x, y, yaw, length, width, height, mat="cart")
    _hide(lay)
    lay.mesh_box(x, y, yaw, length, width, 0.04, z0=height - 0.04, mat="cart")
    lay.mesh_box(x, y, yaw, length, width, 0.04, z0=0.15, mat="cart")
    c, s = math.cos(yaw), math.sin(yaw)
    for sx in (-1, 1):
        for sy in (-1, 1):
            lx, ly = sx * (length / 2 - 0.03), sy * (width / 2 - 0.03)
            lay.mesh_parts.append(("circle", x + c * lx - s * ly, y + s * lx + c * ly, 0.015, 0.0, height, "cart"))


def open_rack(lay: TileLayout, x: float, y: float, yaw: float, length: float, depth: float, height: float,
              rng: np.random.Generator):
    """An open metal rack: four posts and shelves, some loaded (the space between the shelves stays open)."""
    lay.box(x, y, yaw, length, depth, height, mat="rack")
    _hide(lay)
    c, s = math.cos(yaw), math.sin(yaw)
    for sx in (-1, 1):
        for sy in (-1, 1):
            lx, ly = sx * (length / 2 - 0.03), sy * (depth / 2 - 0.03)
            lay.mesh_parts.append(("circle", x + c * lx - s * ly, y + s * lx + c * ly, 0.02, 0.0, height, "rack"))
    for z in np.arange(0.12, height, rng.uniform(0.35, 0.5)):
        lay.mesh_box(x, y, yaw, length, depth, 0.03, z0=float(z), mat="rack")
        if rng.uniform() < 0.6:  # bins and boxes on the shelf
            lay.mesh_box(x, y, yaw, length * rng.uniform(0.3, 0.9), depth * 0.8, rng.uniform(0.1, 0.25), z0=float(z) + 0.03,
                         mat="carton")


def scaffold(lay: TileLayout, x: float, y: float, yaw: float):
    """A rolling scaffold tower: thin frames, a platform at 1 m, wheels."""
    length, width, height = 1.8, 0.75, 1.9
    lay.box(x, y, yaw, length, width, height, mat="scaffold")
    _hide(lay)
    c, s = math.cos(yaw), math.sin(yaw)
    for sx in (-1, 1):
        for sy in (-1, 1):
            lx, ly = sx * (length / 2 - 0.03), sy * (width / 2 - 0.03)
            lay.mesh_parts.append(("circle", x + c * lx - s * ly, y + s * lx + c * ly, 0.025, 0.0, height, "scaffold"))
    for z in (0.1, 1.0):
        lay.mesh_box(x, y, yaw, length, width, 0.05, z0=z, mat="scaffold")


def _bench(lay: TileLayout, x0: float, x1: float, y_face: float, side: float, depth: float = 0.75):
    """Drawer-cabinet benches (0.9 m) along x from x0 to x1, their front at ``y_face``, extending towards ``side``."""
    lay.box((x0 + x1) / 2, y_face + side * depth / 2, 0.0, x1 - x0, depth, 0.92, mat="bench")


def build_makerlab(d: float, rng: np.random.Generator, civic_rng: np.random.Generator | None = None) -> TileLayout:
    """A section of the lab hall (see the module)."""
    lay = _new_layout("makerlab", d)
    lay.paint_rect(-HALF_L, HALF_L, -HALF_W, HALF_W, "floor_concrete_polished")
    aisle = _lerp(2.6, 1.8, d) + rng.uniform(-0.1, 0.15)
    yc = rng.uniform(-1.2, 1.2)  # the aisle's centre line
    y0, y1 = yc - aisle / 2, yc + aisle / 2
    for y in (y0, y1):
        lay.paint_rect(-HALF_L, HALF_L, y - LINE_W / 2, y + LINE_W / 2, "floor_line")
    # the open workstation area: on the wider side of the aisle, reached through a painted cross aisle; the hall's
    # outer walls along the tile's long edges have a door on that side (and sometimes on the other); the aisle runs
    # through the tile's ends
    open_side = 1.0 if (IN_W - y1) > (y0 + IN_W) else -1.0
    open_doors = [(float(rng.uniform(-6.0, 6.0)), 1.0)]
    other_doors = [(float(rng.uniform(-6.0, 6.0)), 1.0)] if rng.uniform() < 0.4 else []
    top, bottom = (open_doors, other_doors) if open_side > 0 else (other_doors, open_doors)
    side_doors = _room(lay, top=top, bottom=bottom, open_ends=True)
    edge = y1 if open_side > 0 else y0
    far = IN_W if open_side > 0 else -IN_W
    cross_x = rng.uniform(-3.0, 3.0)
    cross_w = rng.uniform(1.6, 2.2)
    lay.paint_rect(cross_x - cross_w / 2, cross_x + cross_w / 2, min(edge, far), max(edge, far), "floor_line_zone")
    keep = _Bands([(-HALF_L - 1.0, HALF_L + 1.0, y0, y1, "x"),
                   (cross_x - cross_w / 2, cross_x + cross_w / 2, min(edge, far), max(edge, far), "y")],
                  0.62 + _lerp(0.5, 0.2, d), doors=side_doors)

    # the closed side: benches, machines, racks right behind the aisle line, broken by gaps
    closed = -open_side
    face = y0 if closed < 0 else y1
    back = -IN_W if closed < 0 else IN_W
    x = -HALF_L
    standing = []
    while x < HALF_L - 1.0:
        kind = rng.choice(["bench", "bench", "machine", "rack", "gap"])
        length = {"bench": rng.uniform(2.0, 3.5), "machine": rng.uniform(0.8, 1.4), "rack": rng.uniform(1.0, 1.8),
                  "gap": rng.uniform(0.6, 1.2)}[kind]
        x1 = min(x + length, HALF_L)
        setback = rng.uniform(0.05, 0.3)
        f = face + closed * setback
        if kind == "bench":
            _bench(lay, x, x1, f, closed)
            if rng.uniform() < 0.6:  # someone working there, on a chair or standing
                px = rng.uniform(x + 0.4, x1 - 0.4)
                cy = f - closed * 0.35
                if rng.uniform() < 0.5 and keep.place((px - 0.33, px + 0.33, cy - 0.33, cy + 0.33)):
                    office_chair(lay, px, cy, math.pi / 2 * closed)
                standing.append((px, f - closed * 0.4))
        elif kind == "machine":
            lay.box((x + x1) / 2, f + closed * 0.45, 0.0, x1 - x, 0.9, rng.uniform(1.8, 2.6), mat="machine")
            if rng.uniform() < 0.4:
                standing.append(((x + x1) / 2, f - closed * 0.4))
        elif kind == "rack":
            open_rack(lay, (x + x1) / 2, f + closed * 0.3, 0.0, x1 - x, 0.6, rng.uniform(1.5, 2.1), rng)
        x = x1 + rng.uniform(0.0, 0.3)
    # behind the front row, up to the wall: more benches and machines (seen through the gaps)
    for _ in range(int(rng.integers(2, 5))):
        bx = rng.uniform(-IN_L + 1.0, IN_L - 1.0)
        by = (face + closed * 1.6 + back) / 2
        if abs(back - (face + closed * 1.6)) > 1.0:
            lay.box(bx, by, 0.0, rng.uniform(1.0, 2.5), rng.uniform(0.6, 1.0), rng.uniform(0.9, 2.2), mat="machine")
    # the aisle's edges: bins, buckets, a column, a pallet, carts parked along the line (narrowing it)
    for _ in range(int(_lerp(3, 9, d))):
        side = float(rng.choice([-1.0, 1.0]))
        line = y1 if side > 0 else y0
        px = rng.uniform(-HALF_L + 0.5, HALF_L - 0.5)
        kind = rng.choice(["bin", "bin", "bucket", "barrel", "pallet", "cart", "toolboxes", "column"])
        if kind in ("bin", "bucket", "barrel", "column"):
            r, h, mat = {"bin": (0.19, 0.8, "steel_bin"), "bucket": (0.16, 0.4, "bucket"), "barrel": (0.29, 0.9, "barrel"),
                         "column": (0.12, 2.7, "column")}[kind]
            py = line + side * (r - rng.uniform(0.0, 0.15 if d > 0.5 else 0.05))
            if keep.place((px - r, px + r, py - r, py + r)):
                lay.circle(px, py, r, h, mat=mat)
        elif kind == "pallet":  # a pallet with boxes and buckets: low at its edge, higher in the middle
            sx, sy = 1.2, 1.0
            py = line + side * (sy / 2 + rng.uniform(-0.1, 0.1))
            if keep.place((px - sx / 2, px + sx / 2, py - sy / 2, py + sy / 2)):
                lay.box(px, py, 0.0, sx, sy, 0.15, mat="pallet")
                lay.box(px + rng.uniform(-0.2, 0.2), py, 0.0, 0.8, 0.6, rng.uniform(0.3, 0.8), z0=0.15, mat="carton")
        elif kind == "cart":
            py = line + side * (0.3 + rng.uniform(-0.05, 0.1))
            if keep.place((px - 0.5, px + 0.5, py - 0.3, py + 0.3)):
                cart(lay, px, py, 0.0)
        else:  # stacked toolboxes on a dolly
            py = line + side * 0.25
            if keep.place((px - 0.35, px + 0.35, py - 0.25, py + 0.25)):
                lay.box(px, py, 0.0, 0.7, 0.45, rng.uniform(0.5, 1.0), mat="toolboxes")
    # the open side: workstations (benches with chairs), tripods, a scaffold, carts; the cross aisle stays clear
    o0, o1 = (y1 + 0.4, IN_W) if open_side > 0 else (-IN_W, y0 - 0.4)
    for _ in range(int(rng.integers(2, 4))):  # workstation benches against the far wall or free-standing
        wx = rng.uniform(-IN_L + 1.5, IN_L - 1.5)
        if abs(wx - cross_x) < cross_w / 2 + 1.6:
            continue
        wl = rng.uniform(1.6, 2.6)
        wy = far - open_side * 0.4 if rng.uniform() < 0.6 else rng.uniform(o0 + 1.2, o1 - 1.2)
        if keep.place((wx - wl / 2, wx + wl / 2, wy - 0.4, wy + 0.4)):
            lay.box(wx, wy, 0.0, wl, 0.75, 0.92, mat="bench")
            for cx in np.linspace(wx - wl / 3, wx + wl / 3, int(rng.integers(1, 3))):
                cy = wy - open_side * 0.75
                if keep.place((cx - 0.33, cx + 0.33, cy - 0.33, cy + 0.33)):
                    office_chair(lay, float(cx), cy, -math.pi / 2 * open_side)
                    if rng.uniform() < 0.5:
                        standing.append((float(cx), cy))
    for _ in range(int(_lerp(1, 4, d))):  # tripods in the open floor, sometimes near the aisle line
        tx, ty = rng.uniform(-IN_L + 1.0, IN_L - 1.0), rng.uniform(o0 + 0.6, o1 - 0.6)
        if keep.place((tx - 0.5, tx + 0.5, ty - 0.5, ty + 0.5)):
            tripod(lay, tx, ty, rng)
    if rng.uniform() < 0.4:
        sx, sy = rng.uniform(-IN_L + 1.5, IN_L - 1.5), rng.uniform(o0 + 0.6, o1 - 0.6)
        if keep.place((sx - 0.95, sx + 0.95, sy - 0.45, sy + 0.45)):
            scaffold(lay, sx, sy, 0.0)
    for _ in range(int(_lerp(0, 3, d))):  # boxes and buckets on the floor of the open area
        bx, by = rng.uniform(-IN_L + 0.5, IN_L - 0.5), rng.uniform(o0 + 0.3, o1 - 0.3)
        sx, sy, h = rng.uniform(0.3, 0.6), rng.uniform(0.3, 0.5), rng.uniform(0.2, 0.5)
        if keep.place((bx - sx / 2, bx + sx / 2, by - sy / 2, by + sy / 2)):
            lay.box(bx, by, rng.uniform(0, math.pi), sx, sy, h, mat="carton")

    plan = Planner(lay)

    def lane(x: float, prefer: float) -> float:
        """The clearest point across the aisle at x, preferring the side ``prefer`` (+-1: people keep right)."""
        ys = np.linspace(y0 + 0.3, y1 - 0.3, 12)
        score = [plan.clearance((x, y)) + 0.1 * prefer * (y - yc) for y in ys]
        return float(ys[int(np.argmax(score))])

    left, right = lane(-HALF_L + 0.5, -1.0), lane(HALF_L - 0.5, 1.0)
    ends = [[(-OFF, left), (-HALF_L + 0.5, left)], [(OFF, right), (HALF_L - 0.5, right)]]
    ends += [_door_end(door) for door in side_doors]
    _paths_between(lay, plan, ends, rng, count=2, weight=2.5, pairs=[(0, 1), (1, 0)])  # along the aisle, keeping right
    spots_open = [(float(x), float(y)) for x in np.linspace(-6.5, 6.5, 8) for y in np.linspace(o0 + 0.5, o1 - 0.5, 3)]
    spots_open = [p for p in spots_open if plan.clearance(p) > 0.6]
    rng.shuffle(spots_open)
    _paths_between(lay, plan, ends, rng, count=6, weight=1.0, pairs=[(a, b) for a in range(len(ends)) for b in range(len(ends))
                                                                      if a != b and (a >= 2 or b >= 2)])  # aisle <-> doors
    _paths_between(lay, plan, ends, rng, count=4, weight=0.7, via=[(cross_x, (edge + far) / 2)] + spots_open[:4],
                   pairs=[(0, 1), (1, 0)])  # along the aisle with a detour into the workstation area
    for p in standing[:10]:
        if plan.clearance(p) > 0.28:
            lay.standing_spot(*p)
    # robot routes: down the aisle, into the workstation area, across
    candidates = [(-HALF_L + 0.8, yc), (HALF_L - 0.8, yc), (cross_x, yc)] + spots_open[:6]
    _routes(lay, plan, candidates, rng, min_length=5.0)
    if len(lay.routes) < MAX_ROUTES:  # the aisle end to end, both ways
        _routes(lay, plan, [(-HALF_L + 0.8, yc), (HALF_L - 0.8, yc)], rng, min_length=5.0)
    lay.crowd = (2, 9)
    return lay


LAB_BUILDERS = {"makerlab": build_makerlab}
