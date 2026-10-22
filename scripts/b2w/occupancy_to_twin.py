"""Draft a digital-twin tile (:mod:`g1_patrol.tasks.navigation.nyc.twin`) from a 2D occupancy map of the real place.

Map the place with the robot's lidar (e.g. slam_toolbox, saved with ``nav2_map_server``'s map_saver: a ``.pgm`` image
and a ``.yaml`` with resolution and origin), choose the region the robot will drive in (at most 16 x 10 m, its long
side along x after ``--rotate``), and convert: occupied cells become solid boxes, merged into as few rectangles as
the tile's capacity allows. The draft knows nothing about doors, tables (open underneath) or obstacles below the
lidar's scan plane; add those by hand, together with robot routes, before evaluating the policy on the twin
(``scripts/b2w/evaluate.py --twin``).

Usage:
    python scripts/b2w/occupancy_to_twin.py --map lab.yaml --region -2 -3 13 6 --out lab_twin.yaml
"""

import argparse
import math
from pathlib import Path

import numpy as np
import yaml

MAX_BOXES = 70
"""Room left for hand-added furniture under the tile's capacity of 80 boxes."""


def read_pgm(path: Path) -> np.ndarray:
    data = path.read_bytes()
    tokens, pos = [], 0
    while len(tokens) < 4:  # magic, width, height, maxval (comments start with #)
        while data[pos : pos + 1].isspace():
            pos += 1
        if data[pos : pos + 1] == b"#":
            pos = data.index(b"\n", pos) + 1
            continue
        end = pos
        while not data[end : end + 1].isspace():
            end += 1
        tokens.append(data[pos:end])
        pos = end
    magic, w, h, maxval = tokens[0], int(tokens[1]), int(tokens[2]), int(tokens[3])
    pos += 1
    if magic == b"P5":
        img = np.frombuffer(data[pos : pos + w * h * (2 if maxval > 255 else 1)], dtype=np.uint8 if maxval <= 255 else ">u2")
    else:  # P2
        img = np.array(data[pos:].split(), dtype=np.int64)[: w * h]
    return img.reshape(h, w).astype(np.float64) / maxval


def rectangles(occ: np.ndarray) -> list[tuple[int, int, int, int]]:
    """Cover the occupied cells with rectangles (i0, i1, j0, j1), inclusive: runs along rows, merged down columns."""
    occ = occ.copy()
    rects = []
    for i in range(occ.shape[0]):
        j = 0
        while j < occ.shape[1]:
            if not occ[i, j]:
                j += 1
                continue
            j1 = j
            while j1 + 1 < occ.shape[1] and occ[i, j1 + 1]:
                j1 += 1
            i1 = i
            while i1 + 1 < occ.shape[0] and occ[i1 + 1, j : j1 + 1].all():
                i1 += 1
            occ[i : i1 + 1, j : j1 + 1] = False
            rects.append((i, i1, j, j1))
            j = j1 + 1
    return rects


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--map", required=True, help="map_server YAML")
    parser.add_argument("--region", type=float, nargs=4, required=True, metavar=("X0", "Y0", "X1", "Y1"),
                        help="region in map coordinates (after --rotate)")
    parser.add_argument("--rotate", type=float, default=0.0, help="rotate the map by this (deg) first")
    parser.add_argument("--height", type=float, default=1.8, help="height given to occupied cells (m)")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    meta = yaml.safe_load(open(args.map))
    img = read_pgm(Path(args.map).parent / meta["image"])
    p_occ = img if meta.get("negate", 0) else 1.0 - img
    occupied = p_occ > meta.get("occupied_thresh", 0.65)
    res, (ox, oy) = meta["resolution"], meta["origin"][:2]
    # cell centres in map coordinates (row 0 of the image is the top, i.e. the largest y), rotated
    h, w = occupied.shape
    ys, xs = np.nonzero(occupied)
    px, py = ox + (xs + 0.5) * res, oy + (h - ys - 0.5) * res
    c, s = math.cos(math.radians(args.rotate)), math.sin(math.radians(args.rotate))
    px, py = c * px - s * py, s * px + c * py
    x0, y0, x1, y1 = args.region
    if x1 - x0 > 16.0 + 1e-6 or y1 - y0 > 10.0 + 1e-6:
        raise SystemExit("the region must fit in 16 x 10 m (long side along x)")
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    keep = (px >= x0) & (px <= x1) & (py >= y0) & (py <= y1)
    lx, ly = px[keep] - cx, py[keep] - cy
    # rasterize at a coarser cell until the rectangles fit the tile's capacity
    for cell in (0.1, 0.15, 0.2, 0.3, 0.4):
        nx, ny = int(math.ceil(16.0 / cell)), int(math.ceil(10.0 / cell))
        grid = np.zeros((nx, ny), dtype=bool)
        grid[np.clip(((lx + 8.0) / cell).astype(int), 0, nx - 1), np.clip(((ly + 5.0) / cell).astype(int), 0, ny - 1)] = True
        rects = rectangles(grid)
        if len(rects) <= MAX_BOXES:
            break
    boxes = []
    for i0, i1, j0, j1 in rects:
        bx0, bx1, by0, by1 = -8.0 + i0 * cell, -8.0 + (i1 + 1) * cell, -5.0 + j0 * cell, -5.0 + (j1 + 1) * cell
        boxes.append([round((bx0 + bx1) / 2, 3), round((by0 + by1) / 2, 3), 0, round(bx1 - bx0, 3), round(by1 - by0, 3), args.height])
    spec = {
        "name": Path(args.out).stem,
        "source": {"map": str(args.map), "region": list(args.region), "rotate": args.rotate, "cell": cell},
        "boxes": boxes,
        "walls": [],
        "doors": [],
        "tables": [],
        "circles": [],
        "drops": [],
        "standing": [],
        "crowd": [1, 6],
    }
    with open(args.out, "w") as f:
        f.write("# Draft twin: occupied map cells as boxes. Add doors (x, y, nx, ny, width), tables, low obstacles,\n"
                "# drops and robot routes (sx, sy, gx, gy) by hand; coordinates are meters from the region's center.\n")
        yaml.safe_dump(spec, f, default_flow_style=None, sort_keys=False)
    print(f"{len(boxes)} boxes at {cell} m cells -> {args.out}")


if __name__ == "__main__":
    main()
