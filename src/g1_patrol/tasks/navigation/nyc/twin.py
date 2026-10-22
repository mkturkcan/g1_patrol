"""A tile from a description of a real place (a digital twin of the lab the robot will drive in).

The description is a YAML file in tile coordinates (meters from the tile's center, x along its 16 m side; the place
must fit in 16 x 10 m)::

    name: lab
    walls:                    # segments (x0, y0, x1, y1[, thickness, height]); the tile's edges are not walls
      - [-8.0, 5.0, 8.0, 5.0]
    doors:                    # where people come and go: (x, y, nx, ny, width), the opening's center on the room
      - [-3.0, 4.85, 0, -1, 1.0]   # side of the wall and the normal pointing into the room
    boxes:                    # solid furniture: (cx, cy, yaw_deg, length, width, height[, z0])
      - [0.0, 0.0, 0, 2.4, 0.9, 0.9]
    tables:                   # tables (sensors see a top on legs): (cx, cy, yaw_deg, length, width, height)
      - [3.0, -2.0, 0, 1.6, 0.8, 0.75]
    circles:                  # posts, bins, chairs: (x, y, radius, height)
      - [5.0, 1.0, 0.3, 0.9]
    drops:                    # stairwells and other drops the robot must stay out of: (x0, x1, y0, y1, depth)
      - [6.0, 7.5, -4.5, -3.0, 1.0]
    routes:                   # robot start and goal (sx, sy, gx, gy); omitted: drawn between open spots
      - [-6.0, 0.0, 6.0, 0.0]
    standing:                 # where people stand around (x, y); omitted: none
      - [2.0, 1.5]
    crowd: [1, 6]             # number of people at the lowest and highest difficulty

``scripts/b2w/occupancy_to_twin.py`` drafts such a file from a 2D occupancy map of the place (e.g. from slam_toolbox
with the robot's lidar); doors, tables and low obstacles are best added by hand. Difficulty only scales the crowd.
"""

from __future__ import annotations

import math

import numpy as np

from .indoor import MAX_ROUTES, Planner, _door_end, _paths_between, _routes, _floor
from .tiles import TileLayout, _new_layout


def build_twin(spec: dict, d: float, rng: np.random.Generator) -> TileLayout:
    lay = _new_layout("twin", d)
    _floor(lay, spec.get("floor", "floor_vinyl"))
    for w in spec.get("walls", []):
        x0, y0, x1, y1 = w[:4]
        thickness, height = (w[4] if len(w) > 4 else 0.15), (w[5] if len(w) > 5 else 2.7)
        lay.segment_box(x0, y0, x1, y1, thickness, height, mat="wall")
    for b in spec.get("boxes", []):
        cx, cy, yaw, length, width, height = b[:6]
        lay.box(cx, cy, math.radians(yaw), length, width, height, z0=b[6] if len(b) > 6 else 0.0, mat="furniture")
    for t in spec.get("tables", []):
        cx, cy, yaw, length, width, height = t[:6]
        lay.table(cx, cy, math.radians(yaw), length, width, height)
    for x, y, r, h in spec.get("circles", []):
        lay.circle(x, y, r, h, mat="post")
    for x0, x1, y0, y1, depth in spec.get("drops", []):
        lay.add_rect(x0, x1, y0, y1, -abs(depth))
        lay.forbidden.append((x0, x1, y0, y1))
        lay.paint_rect(x0, x1, y0, y1, "stairwell")
    for x, y in spec.get("standing", []):
        lay.standing_spot(x, y)
    doors = [tuple(float(v) for v in door) for door in spec.get("doors", [])]
    plan = Planner(lay)
    ends = [_door_end(door) for door in doors]
    spots = [(float(x), float(y)) for x in np.linspace(-7.0, 7.0, 15) for y in np.linspace(-4.0, 4.0, 9)]
    open_spots = [p for p in spots if plan.clearance(p) > 0.8]
    rng.shuffle(open_spots)
    if len(ends) >= 2:  # through the room from door to door, some past a random open spot
        _paths_between(lay, plan, ends, rng, count=6, weight=1.0)
        _paths_between(lay, plan, ends, rng, count=6, weight=1.0, via=open_spots[:8] or None)
    elif len(ends) == 1:  # in through the only door, around, and out again
        _paths_between(lay, plan, ends * 2, rng, count=8, weight=1.0, via=open_spots[:8] or None, pairs=[(0, 1), (1, 0)] * 4)
    if not lay.paths:  # a closed room: people walk far away, behind the walls (the crowd needs a path)
        lay.path([(-60.0, 30.0), (60.0, 30.0)], 0.1, 1.0)
    if spec.get("routes"):
        for sx, sy, gx, gy in spec["routes"][:MAX_ROUTES]:
            leg = plan.path(plan.distances((sx, sy), 0.35), (sx, sy), (gx, gy), 0.35)
            lay.routes.append((float(sx), float(sy), float(gx), float(gy)))
            lay.route_headings.append(math.atan2(leg[1][1] - sy, leg[1][0] - sx) if leg else math.atan2(gy - sy, gx - sx))
    else:  # between open spots of the place
        _routes(lay, plan, open_spots[:40], rng, min_length=4.0)
    lay.crowd = tuple(spec.get("crowd", (1, 6)))
    return lay


def load_twin(path: str) -> dict:
    import yaml

    with open(path) as f:
        return yaml.safe_load(f)
