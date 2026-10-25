"""Check the demo's dressing for z-fighting: faces of different parts that lie in the same plane, face the same way and
overlap flicker in the renderer (which of the two is drawn changes from frame to frame).

Dresses tiles of every kind with a stand-in USD stage (no Kit app needed), collects the meshes ``nyc_look`` would add,
and reports, per tile, the pairs of parts with coplanar overlapping faces (overlap area in m^2). Exits with status 1 if
there are any.

Usage:
    python scripts/demo/check_dressing.py [--seeds 3] [--kinds sidewalk scaffold ...] [--civic]
"""

from __future__ import annotations

import argparse
import os
import sys
import types
from collections import defaultdict
from unittest import mock

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
for name in ("isaaclab", "isaaclab.sim", "pxr"):  # stand-ins: only add_mesh touches USD, and it is replaced below
    sys.modules.setdefault(name, mock.MagicMock())
# the tile layouts (numpy only) without the task packages' __init__, which registers the Isaac Lab environments
for pkg in ("g1_patrol.tasks", "g1_patrol.tasks.navigation", "g1_patrol.tasks.navigation.nyc"):
    module = types.ModuleType(pkg)
    module.__path__ = [os.path.join(ROOT, "src", *pkg.split("."))]
    sys.modules.setdefault(pkg, module)

import trimesh  # noqa: E402
from shapely.geometry import Polygon  # noqa: E402
from shapely.ops import unary_union  # noqa: E402

import nyc_look  # noqa: E402
from g1_patrol.tasks.navigation.nyc import tiles  # noqa: E402

KINDS = ("sidewalk", "scaffold", "crosswalk", "construction", "bottleneck", "plaza")


def collect(lay, rng) -> list[tuple[str, str, trimesh.Trimesh]]:
    """The meshes the dressing adds for one tile (centred at the origin), split into their parts (one per box), with
    their materials."""
    parts = []

    def add_mesh(stage, path, mesh, material):
        for k, part in enumerate(mesh.split(only_watertight=False) if len(mesh.faces) < 20000 else [mesh]):
            parts.append((f"{path.split('/Tiles/')[-1]}#{k}", material, part))

    with mock.patch.object(nyc_look, "add_mesh", add_mesh):
        nyc_look.dress_tile(mock.MagicMock(), lay, (0.0, 0.0), "t", rng, far_side=True)
    return parts


def coplanar_overlaps(parts, tol: float = 0.002, min_area: float = 1e-3) -> list[tuple[str, str, float]]:
    """Pairs of parts of different materials with faces in the same plane (within ``tol`` m), facing the same way,
    overlapping by more than ``min_area`` m^2. Parts of one material look the same where they overlap (the texture
    coordinates are projected from world coordinates), faces looking down within 0.3 m of the street level rest on
    the ground, and faces in the tile's end planes meet the next copy of the tile: none of them can flicker."""
    groups = defaultdict(list)  # (normal, plane offset bucket) -> [(part index, 2D triangle)]
    for i, (_, _, m) in enumerate(parts):
        tri = np.asarray(m.triangles)
        n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        area = np.linalg.norm(n, axis=1)
        keep = (area > 1e-9) & ~((n[:, 2] < -0.9 * area) & (tri[:, :, 2].max(axis=1) < 0.3))
        # faces in the tile's end planes meet the next copy of the tile along the street (the demo repeats tiles)
        keep &= ~(np.abs(np.abs(tri[:, :, 0]) - tiles.HALF_L) < 1e-3).all(axis=1)
        tri, n = tri[keep], n[keep] / area[keep, None]
        d = (n * tri[:, 0]).sum(1)
        for t, nn, dd in zip(tri, n, d):
            key = tuple(np.round(nn, 3)) + (int(np.floor(dd / tol)),)
            groups[key].append((i, t, nn))
    pairs = defaultdict(float)
    seen = set()
    for key, items in groups.items():
        # neighbouring offset buckets: planes within tol may straddle a bucket boundary
        near = items + groups.get(key[:3] + (key[3] + 1,), [])
        owners = {i for i, _, _ in near}
        if len(owners) < 2:
            continue
        nn = items[0][2]
        u = np.cross(nn, [0.0, 0.0, 1.0] if abs(nn[2]) < 0.9 else [1.0, 0.0, 0.0])
        u /= np.linalg.norm(u)
        v = np.cross(nn, u)
        polys = defaultdict(list)
        for i, t, _ in near:
            p = Polygon([(float(q @ u), float(q @ v)) for q in t])
            if p.is_valid and p.area > 1e-9:
                polys[i].append(p)
        shapes = {i: unary_union(ps) for i, ps in polys.items()}
        ids = sorted(shapes)
        for a in range(len(ids)):
            for b in range(a + 1, len(ids)):
                if parts[ids[a]][1] == parts[ids[b]][1]:
                    continue
                pair = (ids[a], ids[b], key)
                if pair in seen:
                    continue
                seen.add(pair)
                inter = shapes[ids[a]].intersection(shapes[ids[b]]).area
                if inter > min_area:
                    pairs[(parts[ids[a]][0], parts[ids[b]][0])] += inter
    return sorted(((a, b, area) for (a, b), area in pairs.items()), key=lambda r: -r[2])


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--kinds", nargs="+", default=list(KINDS))
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--difficulties", type=float, nargs="+", default=[0.0, 0.5, 1.0])
    parser.add_argument("--civic", action="store_true", help="civic tiles (litter baskets, bag piles, signals)")
    args = parser.parse_args()
    bad = 0
    for kind in args.kinds:
        build = getattr(tiles, f"build_{kind}")
        for d in args.difficulties:
            for seed in range(args.seeds):
                rng = np.random.default_rng(seed)
                lay = build(d, rng, np.random.default_rng(seed + 100)) if args.civic else build(d, rng)
                found = coplanar_overlaps(collect(lay, np.random.default_rng(seed)))
                status = "ok" if not found else f"{len(found)} overlapping pairs"
                print(f"{kind:12s} d={d:.1f} seed={seed}: {status}")
                for a, b, area in found[:8]:
                    print(f"    {a}  <->  {b}: {area:.3f} m^2")
                bad += bool(found)
    print(f"{bad} tiles with coplanar overlaps")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
