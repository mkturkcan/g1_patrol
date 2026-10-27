"""Lidar returns from a pedestrian as the B2-W's simulated lidar sees it: a capsule (training) against the posed
Isaac Sim characters (``scripts/b2w/bake_people_meshes.py``), by distance.

Casts the rays of the simulated RS-Helios-5515 (32 channels over -55 to 15 deg, 1 deg horizontally; mounted 0.18 m
above the base, which stands 0.63 m above the floor, the middle of its range) at a pedestrian standing at each distance,
turned to 8 headings, and counts the rays that hit it. Prints a table and, with ``--figure``, plots the returns by
distance and the points at one distance. CPU only (Open3D).

Usage:
    python scripts/figures/b2w_people_returns.py [--figure docs/b2w/media/people_returns.png]
"""

import argparse
import glob
import json
import math
import os

import numpy as np

LIDAR_HEIGHT = 0.63 + 0.17851  # base height (middle of 0.55-0.72 m) + the lidar mount
CHANNELS, VFOV = 32, (-55.0, 15.0)
CAPSULE = dict(radius=0.25, height=1.2, center=0.86)  # NycScenarioCommandCfg pedestrians: 1.7 m tall


def _rays(distance_max: float = 12.0):
    """Unit directions of the lidar's rays within 15 deg of +x (the others cannot reach a pedestrian on the x axis)."""
    v = np.radians(np.linspace(*VFOV, CHANNELS))
    h = np.radians(np.arange(-15.0, 15.0 + 1e-6, 1.0))
    vv, hh = np.meshgrid(v, h, indexing="ij")
    return np.stack([np.cos(vv) * np.cos(hh), np.cos(vv) * np.sin(hh), np.sin(vv)], -1).reshape(-1, 3)


def _capsule_mesh():
    import open3d as o3d

    r, h = CAPSULE["radius"], CAPSULE["height"]
    cyl = o3d.geometry.TriangleMesh.create_cylinder(radius=r, height=h, resolution=48)
    top = o3d.geometry.TriangleMesh.create_sphere(radius=r, resolution=24).translate((0, 0, h / 2))
    bottom = o3d.geometry.TriangleMesh.create_sphere(radius=r, resolution=24).translate((0, 0, -h / 2))
    mesh = cyl + top + bottom
    mesh.translate((0, 0, CAPSULE["center"]))
    return np.asarray(mesh.vertices), np.asarray(mesh.triangles)


def _hits(vertices, faces, distance, yaw, rays):
    """Points at which the rays from the lidar hit the mesh, standing at (distance, 0) turned by yaw."""
    import open3d as o3d

    c, s = math.cos(yaw), math.sin(yaw)
    v = vertices @ np.array([[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0]]) + np.array([distance, 0.0, 0.0])
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.core.Tensor(v.astype(np.float32)), o3d.core.Tensor(faces.astype(np.uint32)))
    origin = np.array([0.0, 0.0, LIDAR_HEIGHT], dtype=np.float32)
    query = np.concatenate([np.broadcast_to(origin, rays.shape), rays], 1).astype(np.float32)
    t = scene.cast_rays(o3d.core.Tensor(query))["t_hit"].numpy()
    hit = np.isfinite(t)
    return origin + rays[hit] * t[hit, None]


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--meshes", default=os.environ.get("G1_PATROL_PEOPLE_MESHES",
                                                           os.path.expanduser("~/.cache/g1_patrol/people_meshes")))
    parser.add_argument("--distances", type=float, nargs="+", default=[1, 1.5, 2, 3, 4, 5, 6, 8])
    parser.add_argument("--figure", default=None)
    parser.add_argument("--json", default=None)
    args = parser.parse_args()
    rays = _rays()
    yaws = np.radians(np.arange(0, 360, 45))
    people = [np.load(f) for f in sorted(glob.glob(os.path.join(args.meshes, "*.npz")))]
    assert people, f"no people meshes in {args.meshes}: run scripts/b2w/bake_people_meshes.py"
    capsule = _capsule_mesh()
    rows = []
    for d in args.distances:
        cap = np.mean([len(_hits(*capsule, d, y, rays)) for y in yaws])
        counts = np.array([[len(_hits(p["vertices"], p["faces"], d, y, rays)) for y in yaws] for p in people])
        rows.append({"distance_m": d, "capsule": float(cap), "people_mean": float(counts.mean()),
                     "people_p10": float(np.percentile(counts, 10)), "people_p90": float(np.percentile(counts, 90)),
                     "people_none": float((counts == 0).mean())})
    print(f"{'distance (m)':>12} {'capsule':>8} {'people':>8} {'p10':>5} {'p90':>5} {'no return':>9}")
    for r in rows:
        print(f"{r['distance_m']:12.1f} {r['capsule']:8.1f} {r['people_mean']:8.1f} {r['people_p10']:5.0f} "
              f"{r['people_p90']:5.0f} {100 * r['people_none']:8.1f}%")
    if args.json:
        json.dump(rows, open(args.json, "w"), indent=1)
    if args.figure:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, (ax, bx) = plt.subplots(1, 2, figsize=(9.0, 3.4), gridspec_kw={"width_ratios": [1.3, 1]})
        d = [r["distance_m"] for r in rows]
        ax.plot(d, [r["capsule"] for r in rows], "o-", color="#3b6ea5", lw=2, ms=5, label="capsule (training)")
        ax.plot(d, [r["people_mean"] for r in rows], "o-", color="#c4572a", lw=2, ms=5, label="people (mean)")
        ax.fill_between(d, [r["people_p10"] for r in rows], [r["people_p90"] for r in rows], color="#c4572a",
                        alpha=0.18, lw=0, label="people (10-90%)")
        ax.set_xlabel("distance (m)")
        ax.set_ylabel("lidar returns per scan")
        ax.set_yscale("log")
        ax.grid(alpha=0.3, lw=0.6)
        ax.legend(frameon=False, fontsize=8)
        from matplotlib.collections import PolyCollection

        at, shift = 3.0, 0.62

        def show(vertices, faces, yaw, dx, color):  # silhouette (as the lidar faces it) and the returns on it
            c, s = math.cos(yaw), math.sin(yaw)
            v = vertices @ np.array([[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0]])
            bx.add_collection(PolyCollection(v[faces][:, :, 1:] + [dx, 0.0], facecolor="#d9d9d9", edgecolor="none"))
            pts = _hits(vertices, faces, at, yaw, rays)
            bx.scatter(pts[:, 1] + dx, pts[:, 2], s=6, color=color, zorder=3)

        show(*capsule, 0.0, -shift, "#3b6ea5")
        for k, p in enumerate(people[:: max(1, len(people) // 3)][:3]):
            show(p["vertices"], p["faces"], math.pi / 2, shift * k, "#c4572a")
        bx.set_aspect("equal")
        bx.set_xlim(-shift - 0.45, 2 * shift + 0.45)
        bx.set_ylim(0, 1.9)
        bx.set_xlabel(f"returns at {at:.0f} m, seen from the lidar (m)")
        bx.grid(alpha=0.3, lw=0.6)
        fig.tight_layout()
        fig.savefig(args.figure, dpi=200)
        print("wrote", args.figure)


if __name__ == "__main__":
    main()
