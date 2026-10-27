"""Bake static meshes of the Isaac Sim People characters in walking poses, for what the lidar and the depth cameras see
of a pedestrian (:mod:`g1_patrol.tasks.navigation.config.b2w.human_targets`) instead of a capsule.

Each character of ``scripts/demo/people.py`` is posed at ``--phases`` of its walk clip (baked by
``scripts/demo/bake_people_animations.py``; a clip holds two gait cycles, and the default phases are 8 steps through the
first), skinned (UsdSkel linear blend skinning), turned to face +x with its feet on z = 0 and decimated to ``--faces``
triangles: ``<out>/<character>_w<phase>.npz`` (``vertices``, ``faces``). The characters are NVIDIA Isaac Sim assets and
are not redistributed: the meshes go to a cache (default ``~/.cache/g1_patrol/people_meshes``). Needs Isaac Sim (the
assets are read through Kit) and Open3D.

Usage (headless):
    python scripts/b2w/bake_people_meshes.py [--out ~/.cache/g1_patrol/people_meshes] [--phases 0 0.0625 ...]
"""

import argparse
import math
import os
import sys

from isaacsim import SimulationApp

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.dirname(os.path.dirname(HERE))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default=os.path.expanduser("~/.cache/g1_patrol/people_meshes"))
    parser.add_argument("--phases", type=float, nargs="+", default=[i / 16 for i in range(8)],
                        help="Phases of the clip (0-1), which holds two gait cycles.")
    parser.add_argument("--faces", type=int, default=4000)
    args = parser.parse_args()
    app = SimulationApp({"headless": True})

    import numpy as np
    import omni.usd
    import open3d as o3d
    from pxr import Usd, UsdGeom, UsdSkel, Vt

    sys.path.insert(0, os.path.join(PROJECT, "scripts", "demo"))
    from people import CHARACTERS, PEOPLE, YAW_OFFSET, _load_clip, _sample

    clips_dir = os.path.join(PROJECT, "src", "g1_patrol", "assets", "data", "people")
    os.makedirs(args.out, exist_ok=True)
    ctx = omni.usd.get_context()
    for name in CHARACTERS:
        ctx.new_stage()
        stage = ctx.get_stage()
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        UsdGeom.SetStageMetersPerUnit(stage, 1.0)
        root = stage.DefinePrim("/Character", "Xform")
        root.GetReferences().AddReference(f"{PEOPLE}/{CHARACTERS[name]}")
        for _ in range(10):  # let the asset load
            app.update()
        skel = next(p for p in Usd.PrimRange(root) if p.IsA(UsdSkel.Skeleton))
        skel_root = next(p for p in Usd.PrimRange(root) if p.IsA(UsdSkel.Root))
        clip = _load_clip(os.path.join(clips_dir, f"{name}.stand_walk_loop_in_place.skelanim.usd"))
        anim = UsdSkel.Animation.Define(stage, "/Animation")
        anim.CreateJointsAttr().Set(clip["joints"])
        anim.CreateScalesAttr().Set(clip["scales"])
        UsdSkel.BindingAPI.Apply(skel).CreateAnimationSourceRel().SetTargets([anim.GetPath()])
        for phase in args.phases:
            rot, trans = _sample(clip, phase)
            anim.CreateRotationsAttr().Set(Vt.QuatfArray.FromNumpy(rot.astype(np.float32)))
            anim.CreateTranslationsAttr().Set(Vt.Vec3fArray.FromNumpy(trans.astype(np.float32)))
            t = Usd.TimeCode.Default()
            cache = UsdSkel.Cache()
            cache.Populate(UsdSkel.Root(skel_root), Usd.TraverseInstanceProxies())
            vertices, faces = [], []
            for binding in cache.ComputeSkelBindings(UsdSkel.Root(skel_root), Usd.TraverseInstanceProxies()):
                query = cache.GetSkelQuery(binding.GetSkeleton())
                xforms = query.ComputeSkinningTransforms(t)
                to_world = np.array(UsdGeom.Xformable(binding.GetSkeleton().GetPrim()).ComputeLocalToWorldTransform(t))
                for target in binding.GetSkinningTargets():
                    mesh = UsdGeom.Mesh(target.GetPrim())
                    if not mesh or not UsdGeom.Imageable(mesh).ComputeVisibility(t) == UsdGeom.Tokens.inherited:
                        continue
                    points = Vt.Vec3fArray(mesh.GetPointsAttr().Get(t))
                    if not target.ComputeSkinnedPoints(xforms, points, t):
                        continue
                    p = np.asarray(points, dtype=np.float64)
                    p = p @ to_world[:3, :3] + to_world[3, :3]  # (USD matrices act on row vectors)
                    counts = np.asarray(mesh.GetFaceVertexCountsAttr().Get(t))
                    indices = np.asarray(mesh.GetFaceVertexIndicesAttr().Get(t))
                    tris, k = [], 0
                    for c in counts:  # fan triangulation of the polygons
                        tris += [(indices[k], indices[k + j], indices[k + j + 1]) for j in range(1, c - 1)]
                        k += c
                    faces.append(np.asarray(tris, dtype=np.int64) + sum(len(v) for v in vertices))
                    vertices.append(p)
            v, f = np.concatenate(vertices), np.concatenate(faces)
            c, s = math.cos(YAW_OFFSET), math.sin(YAW_OFFSET)  # the characters face -y: turn them to face +x
            v = v @ np.array([[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0]])
            v[:, 2] -= v[:, 2].min()
            om = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(v), o3d.utility.Vector3iVector(f))
            om = om.remove_duplicated_vertices().remove_degenerate_triangles()
            om = om.simplify_quadric_decimation(target_number_of_triangles=args.faces)
            out = os.path.join(args.out, f"{name}_w{phase:.2f}.npz")
            vv, ff = np.asarray(om.vertices, dtype=np.float32), np.asarray(om.triangles, dtype=np.int32)
            np.savez_compressed(out, vertices=vv, faces=ff)
            ext = vv.max(0) - vv.min(0)
            print(f"[bake_people_meshes] {os.path.basename(out)}: {len(ff)} triangles, {ext[0]:.2f} x {ext[1]:.2f} x "
                  f"{ext[2]:.2f} m (from {len(f)})", flush=True)
    app.close()


if __name__ == "__main__":
    main()
