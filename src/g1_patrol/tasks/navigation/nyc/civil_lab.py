"""Procedural tiles of a modern civil engineering laboratory (structural and materials testing), furnished with NVIDIA
SimReady props (:mod:`.simready`, CC-BY 4.0).

A section of the lab hall, 16 x 10 m. A main aisle marked by painted lines runs along the tile and continues beyond its
ends. On one side stands the test row, right behind the aisle line: load frames (two columns and a crosshead over a
base, a specimen in the grips, a control pedestal with a screen), environmental chambers, a line of grey actuator
columns, workbenches with chairs and stools, tool trolleys; behind it, against the wall, shelving, cabinets, lockers
and safety stations. On the other side lies the strong floor (anchor inserts on a 1 m grid), reached through a painted
cross aisle: beams on supports, loaded pallets, drums, carts, hand trucks, a scissor lift table, a rolling scaffold,
ladders, camera tripods and work lights, desks with screens; racks and shelving line its far wall. Things stand along
the aisle lines: bins, pails, drums, hand trucks, carts, and a round column. People walk the aisle and come and go
through the doors, stand or sit at the machines, benches and desks.

As in :mod:`.makerlab`, every prop is kept out of as a whole (its bounding box, or the circle around a round one) while
the sensors see its mesh: legs, shelves, handles, the space under a bench top.
"""

from __future__ import annotations

import math

import numpy as np

from .indoor import IN_L, IN_W, MAX_ROUTES, WALL_H, Planner, _Bands, _door_end, _paths_between, _routes, _room
from .makerlab import LINE_W, _hide, tripod
from .simready import catalog
from .tiles import HALF_L, MAX_BOXES, MAX_CIRCLES, OFF, TileLayout, _lerp, _new_layout

FRONT = {
    # direction (rad, in the prop's frame) of its front: where one works, sits or reaches in; for shelving and things
    # that stand against a wall, the axis across it
    "furniture_workbench_standard_d01": 0.0,
    "furniture_workbench_standard_e01": 0.0,
    "furniture_workbench_modular_a01": math.pi,
    "furniture_workbench_modular_b01": math.pi,
    "desk_a01": 0.0,
    "chair_office_amf_black_a01": -math.pi / 2,
    "chair_office_fabric_manager_a01": -math.pi / 2,
    "stool_mobile_padded_vinyl_a01": -math.pi / 2,
    "storage_locker_double_bench_b01": -math.pi / 2,
    "cabinet_file_a01": math.pi,
    "storage_shelf_basic_a01": 0.0,
    "storage_shelf_basic_b01": 0.0,
    "storage_shelf_basic_c01": 0.0,
    "rack_bulk_storage_a03": math.pi / 2,
    "rack_composition_a04": math.pi / 2,
    "rack_composition_a05": math.pi / 2,
    "ladder_extention_fiberglass_a03": math.pi / 2,
    "computer_monitor_office_a01": -math.pi / 2,
    "equipment_airCompressor_a01": -math.pi / 2,
    "safety_eyewashStation_stand_selfContained_a01": 0.0,
}
BENCHES = ["furniture_workbench_standard_d01", "furniture_workbench_standard_e01", "furniture_workbench_modular_a01",
           "furniture_workbench_modular_b01"]
SEATS = ["chair_office_amf_black_a01", "chair_office_fabric_manager_a01", "stool_mobile_padded_vinyl_a01"]
CARTS = ["cart_mobile_a01", "cart_mobile_b01", "cart_mobile_c01", "cart_mobile_d01", "cart_mobile_e01",
         "cart_mobile_f01", "cart_computer_a01", "cart_luggage_platform_a01", "equipment_tools_trolley_a01"]
HAND_TRUCKS = ["handTruck_convertible_steel_orange_a01", "handTruck_convertible_steel_red_a01"]
CRATES = ["crate_blue_a01", "crate_black_a02", "crate_gray_a08"]
BOXES = ["box_cardboard_a01", "box_cardboard_b02", "box_cardboard_c03"]
PAILS = [("pail_plastic_green_a02", 1.7), ("pail_plastic_red_a02", 1.7), ("pail_closedHead_blue_a01", 1.3)]
"""Pails and the scale that makes them 5-gallon buckets."""
DRUMS = [("drum_steel_a01", 1.0), ("drum_plastic_blue_a01", 2.1)]
"""Drums and the scale that makes them 55-gallon drums."""
WALL_ITEMS = ["storage_shelf_basic_a01", "storage_shelf_basic_b01", "storage_shelf_basic_c01", "rack_bulk_storage_a03",
              "rack_composition_a04", "rack_composition_a05", "storage_locker_double_bench_b01", "cabinet_file_a01",
              "safety_eyewashStation_stand_selfContained_a01", "fireExtinguisher_stand_a01",
              "ladder_extention_fiberglass_a03", "trashCan_wheeled_green_a01", "equipment_airCompressor_a01",
              "equipment_3d_printer_a01"]
WALL_WEIGHTS = [3, 3, 3, 2, 1, 1, 2, 2, 1, 1, 1, 1, 1, 1]
INSERT_PITCH = 1.0
"""Spacing (m) of the strong floor's anchor inserts."""
FLOOR_GAP = 0.9
"""Least gap (m) between things on the strong floor and anything else: passages the robot (0.58 m wide) can turn in."""


def _extent(name: str, facing: float, scale: float = 1.0) -> tuple[float, float]:
    """Half extents along x and y of a prop's box with its front facing ``facing``."""
    lo, hi = catalog()[name]["min"], catalog()[name]["max"]
    hx, hy = scale * (hi[0] - lo[0]) / 2, scale * (hi[1] - lo[1]) / 2
    yaw = facing - FRONT.get(name, 0.0)
    c, s = abs(math.cos(yaw)), abs(math.sin(yaw))
    return c * hx + s * hy, s * hx + c * hy


def _height(name: str, scale: float = 1.0) -> float:
    return scale * (catalog()[name]["max"][2] - catalog()[name]["min"][2])


def _room_for(lay: TileLayout, boxes: int = 1, circles: int = 0) -> bool:
    """Whether the layout can take more primitives (the last box is reserved for a fallen person)."""
    return len(lay.boxes) + boxes <= MAX_BOXES - 2 and len(lay.circles) + circles <= MAX_CIRCLES - 1


class _Floor:
    """The floor plan so far: things must not overlap (by more than an allowance) and must keep the aisles passable
    and the doors clear (:class:`.indoor._Bands`)."""

    def __init__(self, lay: TileLayout, keep: _Bands):
        self.lay, self.keep, self.taken = lay, keep, []

    def take(self, rect, overlap: float = 0.0, gap: float = 0.0) -> bool:
        """Claim the rectangle (x0, x1, y0, y1) if it overlaps nothing taken by more than ``overlap`` (m), keeps
        ``gap`` (m) from everything taken, and keeps the aisles passable; returns whether it was claimed."""
        x0, x1, y0, y1 = rect[0] - gap, rect[1] + gap, rect[2] - gap, rect[3] + gap
        o = overlap
        if any(x1 - o > a0 and x0 + o < a1 and y1 - o > b0 and y0 + o < b1 for a0, a1, b0, b1 in self.taken):
            return False
        if not self.keep.place(rect):
            return False
        self.taken.append(rect)
        return True

    def put(self, name: str, x: float, y: float, facing: float = 0.0, z0: float = 0.0, scale: float = 1.0,
            round_: bool = False, overlap: float = 0.0, gap: float = 0.0) -> bool:
        """Place a prop centered at (x, y), its front facing ``facing``, if its box can be claimed and there is room
        for its primitive; returns whether it was placed."""
        if not _room_for(self.lay, boxes=0 if round_ else 1, circles=1 if round_ else 0):
            return False
        ex, ey = _extent(name, facing, scale)
        if not self.take((x - ex, x + ex, y - ey, y + ey), overlap, gap):
            return False
        self.lay.prop(name, x, y, facing - FRONT.get(name, 0.0), z0=z0, scale=scale, round_=round_)
        return True


def _raise_box(lay: TileLayout, index: int, top: float):
    """Raise the top of box ``index`` (a pallet's primitive covering its load)."""
    cx, cy, yaw, hx, hy, z0, z1 = lay.boxes[index]
    lay.boxes[index] = (cx, cy, yaw, hx, hy, z0, max(z1, top))


"""
Structures of the lab.
"""


def load_frame(lay: TileLayout, x: float, face: float, side: float, rng: np.random.Generator,
               standing: list) -> float:
    """A universal testing machine, its front at y = ``face`` (it extends towards ``side``): a base, two columns, a
    crosshead, a top beam, a specimen in the grips; a control pedestal with a screen at its aisle end. Kept out of as a
    whole. Returns its length along x."""
    w, depth, h = rng.uniform(0.9, 1.3), rng.uniform(0.6, 0.85), rng.uniform(2.4, 3.3)
    cx, cy = x + w / 2, face + side * depth / 2
    lay.box(cx, cy, 0.0, w, depth, h, mat="load_frame")
    _hide(lay)
    lay.mesh_box(cx, cy, 0.0, w, depth, 0.8, mat="load_frame")
    for sx in (-1.0, 1.0):
        lay.mesh_parts.append(("circle", cx + sx * (w / 2 - 0.12), cy, 0.06, 0.8, h - 0.3, "steel_polished"))
    head = rng.uniform(1.3, h - 0.8)
    lay.mesh_box(cx, cy, 0.0, w - 0.02, depth * 0.5, 0.22, z0=head, mat="load_frame")
    lay.mesh_box(cx, cy, 0.0, w - 0.02, depth * 0.6, 0.3, z0=h - 0.3, mat="load_frame")
    lay.mesh_box(cx, cy, 0.0, 0.14, 0.14, 0.2, z0=0.8, mat="steel")  # lower grip
    lay.mesh_box(cx, cy, 0.0, 0.14, 0.14, 0.2, z0=head - 0.2, mat="steel")  # upper grip
    lay.mesh_box(cx, cy, 0.0, 0.05, 0.05, head - 1.2, z0=1.0, mat="specimen")
    # the control pedestal with its screen
    px, py = x + w + 0.3, face + side * 0.3
    lay.box(px, py, 0.0, 0.5, 0.45, 1.1, mat="cabinet")
    lay.prop("computer_monitor_office_a01", px, py, -side * math.pi / 2 - FRONT["computer_monitor_office_a01"],
             z0=1.1, collide=False)
    if rng.uniform() < 0.5:
        standing.append((px, face - side * 0.45))
    return w + 0.6


def chamber(lay: TileLayout, x: float, face: float, side: float, rng: np.random.Generator) -> float:
    """An environmental chamber (or a furnace): a grey box with a door panel and a control box on its front."""
    w, depth, h = rng.uniform(0.9, 1.4), rng.uniform(0.85, 1.1), rng.uniform(1.7, 2.1)
    cx = x + w / 2
    lay.box(cx, face + side * depth / 2, 0.0, w, depth, h, mat="chamber")
    lay.mesh_box(cx - 0.1 * w, face, 0.0, 0.65 * w, 0.04, 0.6 * h, z0=0.35, mat="chamber_door")
    lay.mesh_box(cx + 0.38 * w, face, 0.0, 0.16 * w, 0.06, 0.25, z0=1.1, mat="cabinet")
    return w


def actuators(lay: TileLayout, x: float, y: float, rng: np.random.Generator) -> float:
    """A line of grey actuator columns (3-4 steel tubes, floor to 3-4 m) along x."""
    n, r, h = int(rng.integers(3, 5)), rng.uniform(0.13, 0.17), rng.uniform(3.2, 3.8)
    gap = 2 * r + rng.uniform(0.12, 0.25)
    for k in range(n):
        lay.circle(x + r + k * gap, y, r, h, mat="actuator")
    return (n - 1) * gap + 2 * r


def beam_specimen(lay: TileLayout, x: float, y: float, yaw: float, rng: np.random.Generator) -> tuple[float, float]:
    """A concrete beam on two steel supports (kept out of as a whole). Returns its half length and width."""
    length, width, height, support = (rng.uniform(2.0, 3.4), rng.uniform(0.25, 0.4), rng.uniform(0.3, 0.5),
                                      rng.uniform(0.25, 0.4))
    lay.box(x, y, yaw, length, max(width, 0.35), support + height, mat="concrete_specimen")
    _hide(lay)
    c, s = math.cos(yaw), math.sin(yaw)
    for t in (-1.0, 1.0):
        d = t * (length / 2 - 0.3)
        lay.mesh_box(x + c * d, y + s * d, yaw, 0.3, 0.35, support, mat="steel")
    lay.mesh_box(x, y, yaw, length, width, height, z0=support, mat="concrete_specimen")
    return length / 2, max(width, 0.35) / 2


def pallet_load(lay: TileLayout, x: float, y: float, yaw: float, rng: np.random.Generator):
    """A pallet with a load: stacked crates or boxes, a drum, buckets or concrete cylinders; its primitive covers the
    load."""
    lay.prop("pallet_a01", x, y, yaw)
    index = len(lay.boxes) - 1
    base = _height("pallet_a01")
    c, s = math.cos(yaw), math.sin(yaw)
    kind = rng.choice(["crates", "boxes", "drum", "pails", "cylinders"])
    top = base
    if kind in ("crates", "boxes"):
        for slot in (-1, 1):
            if rng.uniform() < 0.25:
                continue
            z = base
            for _ in range(int(rng.integers(1, 4 if kind == "crates" else 3))):
                name = rng.choice(CRATES if kind == "crates" else BOXES)
                ox = slot * 0.3
                lay.prop(name, x + c * ox, y + s * ox, yaw + (math.pi / 2 if kind == "boxes" else 0.0), z0=z,
                         collide=False)
                z += _height(name)
            top = max(top, z)
    elif kind == "drum":
        name, scale = DRUMS[int(rng.integers(len(DRUMS)))]
        lay.prop(name, x, y, rng.uniform(0, 2 * math.pi), z0=base, scale=scale, collide=False)
        top = base + _height(name, scale)
    elif kind == "pails":
        for ox, oy in ((-0.35, -0.2), (0.0, 0.2), (0.35, -0.2), (-0.35, 0.2))[: int(rng.integers(2, 5))]:
            name, scale = PAILS[int(rng.integers(len(PAILS)))]
            lay.prop(name, x + c * ox - s * oy, y + s * ox + c * oy, rng.uniform(0, 2 * math.pi), z0=base, scale=scale,
                     collide=False)
            top = max(top, base + _height(name, scale))
    else:  # concrete test cylinders (150 x 300 mm), standing in rows
        for ox in (-0.4, -0.2, 0.0, 0.2, 0.4):
            for oy in (-0.2, 0.0, 0.2):
                if rng.uniform() < 0.7:
                    lay.mesh_parts.append(("circle", x + c * ox - s * oy, y + s * ox + c * oy, 0.075, base, base + 0.3,
                                           "concrete_specimen"))
        top = base + 0.3
    _raise_box(lay, index, top)


"""
The tile.
"""


def build_civil_lab(d: float, rng: np.random.Generator, civic_rng: np.random.Generator | None = None) -> TileLayout:
    """A section of the civil engineering lab hall (see the module)."""
    lay = _new_layout("civil_lab", d)
    lay.paint_rect(-HALF_L, HALF_L, -IN_W - 0.15, IN_W + 0.15, "floor_concrete_polished")
    aisle = _lerp(2.8, 1.9, d) + rng.uniform(-0.1, 0.15)
    yc = rng.uniform(-1.2, 1.2)  # the aisle's centre line
    y0, y1 = yc - aisle / 2, yc + aisle / 2
    for y in (y0, y1):
        lay.paint_rect(-HALF_L, HALF_L, y - LINE_W / 2, y + LINE_W / 2, "floor_line_white")
    # the strong floor (the open side) on the wider side of the aisle, reached through a cross aisle from a door in its
    # far wall; sometimes a door in the wall behind the test row too
    open_side = 1.0 if (IN_W - y1) > (y0 + IN_W) else -1.0
    test = -open_side
    open_doors = [(float(rng.uniform(-6.0, 6.0)), float(rng.uniform(1.0, 1.6)))]
    other_doors = [(float(rng.uniform(-6.0, 6.0)), 1.0)] if rng.uniform() < 0.4 else []
    top, bottom = (open_doors, other_doors) if open_side > 0 else (other_doors, open_doors)
    side_doors = _room(lay, top=top, bottom=bottom, open_ends=True)
    edge = y1 if open_side > 0 else y0  # the aisle line on the open side
    face = y0 if test < 0 else y1  # the aisle line on the test side
    far = IN_W if open_side > 0 else -IN_W
    wall = -far
    cross_x = open_doors[0][0] if rng.uniform() < 0.6 else rng.uniform(-4.0, 4.0)
    cross_w = rng.uniform(1.6, 2.2)
    lay.paint_rect(cross_x - cross_w / 2, cross_x + cross_w / 2, min(edge, far), max(edge, far), "floor_line_zone")
    keep = _Bands([(-HALF_L - 1.0, HALF_L + 1.0, y0, y1, "x"),
                   (cross_x - cross_w / 2, cross_x + cross_w / 2, min(edge, far), max(edge, far), "y")],
                  0.62 + _lerp(0.5, 0.2, d), doors=side_doors)
    floor = _Floor(lay, keep)
    for ix in np.arange(-HALF_L + INSERT_PITCH / 2, HALF_L, INSERT_PITCH):  # anchor inserts (visual only)
        for iy in np.arange(min(face, far) + 0.3, max(face, far) - 0.2, INSERT_PITCH):
            lay.decor.append(("floor_insert", float(ix), float(iy)))
    standing: list = []

    # the test row, right behind the aisle line on the test side; gaps where a door opens behind it
    x = -HALF_L + rng.uniform(0.0, 0.6)
    while x < HALF_L - 0.8 and _room_for(lay, 3, 4):
        kind = rng.choice(["frame", "frame", "chamber", "bench", "bench", "actuators", "trolley", "gap"])
        f = face + test * rng.uniform(0.1, 0.4)
        length = rng.uniform(0.6, 1.2)
        if kind in ("frame", "chamber", "actuators"):
            reach = {"frame": 2.0, "chamber": 1.4, "actuators": 2.2}[kind]
            if floor.take((x, min(x + reach, HALF_L), min(f, f + test * 1.15), max(f, f + test * 1.15))):
                if kind == "frame":
                    length = load_frame(lay, x, f, test, rng, standing)
                elif kind == "chamber":
                    length = chamber(lay, x, f, test, rng)
                else:
                    length = actuators(lay, x, f + test * 0.45, rng)
                floor.taken[-1] = (x, x + length, *floor.taken[-1][2:])
        elif kind == "bench":  # one or two workbenches facing the aisle, a seat in front or someone standing there
            length = 0.0
            for _ in range(int(rng.integers(1, 3))):
                name = rng.choice(BENCHES)
                ex, ey = _extent(name, -test * math.pi / 2)
                bx = x + length + ex
                if bx + ex > HALF_L or not floor.put(name, bx, f + test * ey, -test * math.pi / 2):
                    break
                sx = bx + rng.uniform(-0.3, 0.3)
                seat, turn = rng.choice(SEATS), test * math.pi / 2 + rng.uniform(-0.6, 0.6)
                sy = f - test * (_extent(seat, turn)[1] - rng.uniform(0.05, 0.15))
                if not (rng.uniform() < 0.6 and floor.put(seat, sx, sy, turn, overlap=0.15)):
                    standing.append((sx, f - test * 0.4))
                length += 2 * ex + rng.uniform(0.0, 0.1)
        elif kind == "trolley":
            name = rng.choice(
                ["equipment_tools_trolley_a01", "cart_computer_a01", "equipment_liftTable_scissorMobile_a01"]
            )
            ex, ey = _extent(name, 0.0)
            if floor.put(name, x + ex, f + test * ey, 0.0):
                length = 2 * ex
        x += length + rng.uniform(0.1, 0.4)
    # behind the row, against the wall: shelving, cabinets, lockers, safety stations (seen through the row's gaps)
    _wall_row(floor, wall, -test, rng, count=int(rng.integers(3, 7)))
    # racks and shelving along the far wall of the strong floor, the cross aisle kept clear
    _wall_row(floor, far, -open_side, rng, count=int(rng.integers(3, 7)), avoid=(cross_x, cross_w))

    # the strong floor: specimens, loaded pallets, carts, a lift table, scaffold, ladders, lights, desks, each with a
    # passage around it (FLOOR_GAP)
    g = FLOOR_GAP
    square = [0.0, math.pi / 2, math.pi, -math.pi / 2]
    o0, o1 = (edge + 0.5, far - 0.5) if open_side > 0 else (far + 0.5, edge - 0.5)
    groups = ["beam", "pallet", "pallet", "cart", "hand_truck", "lift", "scaffold", "ladder", "desk", "drums", "light"]
    for attempt in range(3 * (int(_lerp(3, 8, d)) + int(rng.integers(0, 3)))):  # about a third of the tries fit
        if o1 - o0 < 0.6 or not _room_for(lay, 3):
            break
        kind = rng.choice(groups)
        gx, gy = rng.uniform(-IN_L + 0.8, IN_L - 0.8), rng.uniform(o0, o1)
        if kind == "beam":
            yaw = 0.0 if rng.uniform() < 0.7 else math.pi / 2
            hx, hy = (1.75, 0.2) if yaw == 0.0 else (0.2, 1.75)
            if floor.take((gx - hx, gx + hx, gy - hy, gy + hy), gap=g):
                beam_specimen(lay, gx, gy, yaw, rng)
        elif kind == "pallet":
            yaw = float(rng.choice([0.0, math.pi / 2]))
            ex, ey = _extent("pallet_a01", yaw)
            if floor.take((gx - ex, gx + ex, gy - ey, gy + ey), gap=g):
                pallet_load(lay, gx, gy, yaw, rng)
        elif kind == "cart":
            floor.put(rng.choice(CARTS), gx, gy, float(rng.choice(square)), gap=g)
        elif kind == "hand_truck":
            floor.put(rng.choice(HAND_TRUCKS), gx, gy, float(rng.choice(square)), gap=g)
        elif kind == "lift":
            floor.put(rng.choice(["equipment_liftTable_scissorMobile_a01", "vehicleSupport_rolling_dolly_a01"]), gx, gy,
                      float(rng.choice(square)), gap=g)
        elif kind == "scaffold":
            floor.put("scaffold_rolling_steel_a01", gx, gy, float(rng.choice([0.0, math.pi / 2])), gap=g)
        elif kind == "ladder":
            floor.put(rng.choice(["ladder_step_aluminium_a06", "ladder_rolling_yellow_a02"]), gx, gy,
                      float(rng.choice(square)), gap=g)
        elif kind == "desk":  # a desk with a screen, a seat at it or someone standing there
            facing = float(rng.choice([0.0, math.pi]))
            if floor.put("desk_a01", gx, gy, facing, gap=g):
                c = math.cos(facing)
                lay.prop("computer_monitor_office_a01", gx - 0.15 * c, gy,
                         facing - FRONT["computer_monitor_office_a01"], z0=_height("desk_a01"), collide=False)
                sx = gx + c * 0.7
                if not floor.put(rng.choice(SEATS), sx, gy, facing + math.pi + rng.uniform(-0.5, 0.5), overlap=0.1):
                    standing.append((gx + c * 0.75, gy))
        elif kind == "drums":
            for k in range(int(rng.integers(1, 4))):
                name, scale = DRUMS[int(rng.integers(len(DRUMS)))]
                r = max(_extent(name, 0.0, scale))
                floor.put(name, gx + k * (2 * r + 0.05), gy, rng.uniform(0, 2 * math.pi), scale=scale, round_=True,
                          gap=g if k == 0 else 0.0)
        else:  # a work light on a tripod
            floor.put("lighting_taskLight_tripodFlood_a01", gx, gy, float(rng.choice(square)), gap=g)
    for _ in range(int(_lerp(1, 4, d))):  # camera tripods around specimens and machines
        if o1 - o0 < 1.0:
            break
        tx, ty = rng.uniform(-IN_L + 1.0, IN_L - 1.0), rng.uniform(o0 + 0.5, o1 - 0.5)
        if _room_for(lay, 0, 1) and floor.take((tx - 0.5, tx + 0.5, ty - 0.5, ty + 0.5), gap=g):
            tripod(lay, tx, ty, rng)

    # along the aisle lines: a round column, bins, pails, drums, hand trucks, carts, tripods (narrowing the aisle)
    side = float(rng.choice([-1.0, 1.0]))
    cx, cy = rng.uniform(-6.0, 6.0), (y1 if side > 0 else y0) + side * 0.21  # just outside the line
    if floor.take((cx - 0.16, cx + 0.16, cy - 0.16, cy + 0.16)):
        lay.circle(cx, cy, 0.16, WALL_H, mat="column_wood")
    for _ in range(int(_lerp(2, 8, d))):
        side = float(rng.choice([-1.0, 1.0]))
        line = y1 if side > 0 else y0
        px = rng.uniform(-HALF_L + 0.5, HALF_L - 0.5)
        intrude = rng.uniform(0.0, 0.2 if d > 0.5 else 0.05)
        kind = rng.choice(["bin", "bin", "pail", "drum", "hand_truck", "cart", "tripod"])
        if kind == "bin":
            name = rng.choice(["trashCan_thin_gray_a03", "trashCan_office_gray_a03", "bin_waste_a01"])
            r = max(_extent(name, 0.0))
            floor.put(name, px, line + side * (r - intrude), rng.uniform(0, 2 * math.pi), round_=True)
        elif kind == "pail":
            for k in range(int(rng.integers(1, 3))):
                name, scale = PAILS[int(rng.integers(len(PAILS)))]
                r = max(_extent(name, 0.0, scale))
                floor.put(name, px + k * (2 * r + 0.04), line + side * (r - intrude), rng.uniform(0, 2 * math.pi),
                          scale=scale, round_=True)
        elif kind == "drum":
            name, scale = DRUMS[int(rng.integers(len(DRUMS)))]
            r = max(_extent(name, 0.0, scale))
            floor.put(name, px, line + side * (r - intrude), rng.uniform(0, 2 * math.pi), scale=scale, round_=True)
        elif kind == "hand_truck":
            name = rng.choice(HAND_TRUCKS)
            facing = float(rng.choice([0.0, math.pi / 2, math.pi, -math.pi / 2]))
            floor.put(name, px, line + side * (_extent(name, facing)[1] - intrude), facing)
        elif kind == "cart":
            name = rng.choice(["cart_mobile_b01", "cart_mobile_a01", "cart_mobile_c01", "cart_luggage_platform_a01"])
            facing = float(rng.choice([0.0, math.pi]))
            floor.put(name, px, line + side * (_extent(name, facing)[1] + 0.05 - intrude), facing)
        elif _room_for(lay, 0, 1) and floor.take(
            (px - 0.5, px + 0.5, line + side * 0.5 - 0.5, line + side * 0.5 + 0.5)
        ):
            tripod(lay, px, line + side * 0.5, rng)

    plan = Planner(lay)
    # no anchor inserts under things
    lay.decor = [item for item in lay.decor if item[0] != "floor_insert" or plan.clearance(item[1:]) > 0.1]

    def lane(x: float, prefer: float) -> float:
        """The clearest point across the aisle at x, preferring the side ``prefer`` (+-1: people keep right)."""
        ys = np.linspace(y0 + 0.3, y1 - 0.3, 12)
        score = [plan.clearance((x, y)) + 0.1 * prefer * (y - yc) for y in ys]
        return float(ys[int(np.argmax(score))])

    left, right = lane(-HALF_L + 0.5, -1.0), lane(HALF_L - 0.5, 1.0)
    ends = [[(-OFF, left), (-HALF_L + 0.5, left)], [(OFF, right), (HALF_L - 0.5, right)]]
    ends += [_door_end(door) for door in side_doors]
    _paths_between(lay, plan, ends, rng, count=2, weight=2.5, pairs=[(0, 1), (1, 0)])  # along the aisle, keeping right
    lo, hi = min(o0, o1), max(o0, o1)
    spots_open = [(float(x), float(y)) for x in np.linspace(-6.5, 6.5, 8) for y in np.linspace(lo, hi, 3)]
    spots_open = [p for p in spots_open if plan.clearance(p) > 0.8]
    rng.shuffle(spots_open)
    _paths_between(lay, plan, ends, rng, count=6, weight=1.0,  # aisle <-> doors
                   pairs=[(a, b) for a in range(len(ends)) for b in range(len(ends)) if a != b and (a >= 2 or b >= 2)])
    _paths_between(lay, plan, ends, rng, count=4, weight=0.7, via=[(cross_x, (edge + far) / 2)] + spots_open[:4],
                   pairs=[(0, 1), (1, 0)])  # along the aisle with a detour over the strong floor
    for p in standing[:10]:
        if plan.clearance(p) > 0.28:
            lay.standing_spot(*p)
    # robot routes: down the aisle, onto the strong floor, across
    candidates = [(-HALF_L + 0.8, yc), (HALF_L - 0.8, yc), (cross_x, yc)] + spots_open[:6]
    _routes(lay, plan, candidates, rng, min_length=5.0)
    if len(lay.routes) < MAX_ROUTES:  # the aisle end to end, both ways
        _routes(lay, plan, [(-HALF_L + 0.8, yc), (HALF_L - 0.8, yc)], rng, min_length=5.0)
    lay.crowd = (2, 9)
    return lay


def _wall_row(floor: _Floor, wall: float, inward: float, rng: np.random.Generator, count: int,
              avoid: tuple[float, float] | None = None):
    """Up to ``count`` things against the wall at y = ``wall`` (the room lies towards ``inward``): shelving, racks,
    lockers, cabinets, safety stations, a ladder; not in front of doors, nor across ``avoid`` (x, width)."""
    facing = math.pi / 2 if inward > 0 else -math.pi / 2
    weights = np.asarray(WALL_WEIGHTS, dtype=float) / sum(WALL_WEIGHTS)
    x = -IN_L + rng.uniform(0.0, 1.0)
    placed = 0
    while x < IN_L - 0.4 and placed < count and _room_for(floor.lay, 2):
        name = WALL_ITEMS[int(rng.choice(len(WALL_ITEMS), p=weights))]
        ex, ey = _extent(name, facing)
        cx, cy = x + ex, wall + inward * (ey + 0.02)
        if cx + ex > IN_L:
            break
        blocked = avoid is not None and abs(cx - avoid[0]) < ex + avoid[1] / 2 + 0.3
        if not blocked and floor.put(name, cx, cy, facing):
            placed += 1
            x = cx + ex + rng.uniform(0.05, 0.8)
        else:
            x += rng.uniform(0.3, 0.8)


CIVIL_LAB_BUILDERS = {"civil_lab": build_civil_lab}
