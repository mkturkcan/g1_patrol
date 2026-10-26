"""Download NVIDIA SimReady props (CC-BY 4.0, Hugging Face dataset ``nvidia/simready-assets``) for the civil engineering
lab tile (:mod:`g1_patrol.tasks.navigation.nyc.civil_lab`) and convert them to the meshes its sensors see.

For each prop of ``g1_patrol.tasks.navigation.nyc.simready.PROPS``: download the package at the pinned revision into
``<root>/nvidia_simready_assets/<name>``, read its root USD (all meshes, instances resolved, in meters and Z-up), merge
the meshes, decimate them to at most ``--max_faces`` triangles and write ``<root>/processed/<name>.npz`` (vertices,
faces). ``--catalog`` (the repository's ``simready_catalog.json``) gets the bounds of every prop, from which the tile
builder places them; ``<root>/ATTRIBUTION.md`` credits the packages. Nothing of the packages is copied into the
repository, and nothing in them is executed: the USD layers are parsed as data.

``<root>`` defaults to ``$G1_PATROL_SIMREADY`` (``~/.cache/g1_patrol/simready``), where the tasks look for the meshes.

Usage:
    python -I scripts/assets/fetch_simready.py [--root DIR] [--skip_download]
"""

import argparse
import json
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "src"))

from g1_patrol.tasks.navigation.nyc.simready import (  # noqa: E402
    CATALOG_PATH,
    DATASET,
    PROPS,
    REVISION,
    default_root,
    root_usd,
)


def download(root: str, names: list[str]):
    from huggingface_hub import snapshot_download

    snapshot_download(
        DATASET,
        repo_type="dataset",
        revision=REVISION,
        local_dir=os.path.join(root, "nvidia_simready_assets"),
        allow_patterns=[f"{n}/*" for n in names] + [f"{n}/**" for n in names],
        max_workers=16,
    )


def read_meshes(path: str):
    """All render meshes of a USD file as (vertices, triangles) in meters, Z-up, in the file's root frame."""
    from pxr import Gf, Usd, UsdGeom

    stage = Usd.Stage.Open(path)
    scale = UsdGeom.GetStageMetersPerUnit(stage)
    up = Gf.Matrix4d(1.0)
    if UsdGeom.GetStageUpAxis(stage) == UsdGeom.Tokens.y:
        up = Gf.Matrix4d().SetRotate(Gf.Rotation(Gf.Vec3d(1, 0, 0), 90.0))
    cache = UsdGeom.XformCache()
    parts = []

    def add(mesh_prim, transform):
        mesh = UsdGeom.Mesh(mesh_prim)
        points = mesh.GetPointsAttr().Get()
        counts = mesh.GetFaceVertexCountsAttr().Get()
        indices = mesh.GetFaceVertexIndicesAttr().Get()
        if not points or not counts or not indices:
            return
        counts = np.asarray(counts)
        indices = np.asarray(indices)
        starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
        tris = [
            np.stack(
                [indices[s + np.zeros(c - 2, int)], indices[s + np.arange(1, c - 1)], indices[s + np.arange(2, c)]], 1
            )
            for s, c in zip(starts, counts)
            if c >= 3
        ]
        if not tris:
            return
        m = np.array(transform * up, dtype=np.float64)  # USD matrices act on row vectors
        v = np.asarray(points, dtype=np.float64)
        v = (np.c_[v, np.ones(len(v))] @ m)[:, :3] * scale
        parts.append((v, np.concatenate(tris)))

    prims = iter(Usd.PrimRange(stage.GetPseudoRoot(), Usd.TraverseInstanceProxies()))
    for prim in prims:
        imageable = UsdGeom.Imageable(prim)
        if imageable and (
            imageable.ComputePurpose() not in (UsdGeom.Tokens.default_, UsdGeom.Tokens.render)
            or imageable.ComputeVisibility() == UsdGeom.Tokens.invisible
        ):
            prims.PruneChildren()
            continue
        if prim.IsA(UsdGeom.Mesh):
            add(prim, cache.GetLocalToWorldTransform(prim))
        elif prim.IsA(UsdGeom.PointInstancer):
            instancer = UsdGeom.PointInstancer(prim)
            targets = instancer.GetPrototypesRel().GetTargets()
            ids = instancer.GetProtoIndicesAttr().Get() or []
            frames = instancer.ComputeInstanceTransformsAtTime(Usd.TimeCode.Default(), Usd.TimeCode.Default())
            parent = cache.GetLocalToWorldTransform(prim)
            prims.PruneChildren()  # the prototypes are drawn only where instanced
            for proto, frame in zip(ids, frames):
                base = stage.GetPrimAtPath(targets[proto])
                for sub in Usd.PrimRange(base, Usd.TraverseInstanceProxies()):
                    if sub.IsA(UsdGeom.Mesh):
                        local = cache.GetLocalToWorldTransform(sub) * cache.GetLocalToWorldTransform(base).GetInverse()
                        add(sub, local * frame * parent)
    if not parts:
        raise ValueError(f"no meshes in {path}")
    offsets = np.cumsum([0] + [len(v) for v, _ in parts[:-1]])
    vertices = np.concatenate([v for v, _ in parts])
    faces = np.concatenate([f + o for (_, f), o in zip(parts, offsets)])
    return vertices, faces


def simplify(vertices: np.ndarray, faces: np.ndarray, max_faces: int):
    """At most about ``max_faces`` triangles: parts smaller than 3 cm (screws, labels) dropped, quadric decimation
    (Open3D), and where many small parts keep it above that, vertex clustering on a 1.5-4 cm grid first."""
    import open3d as o3d
    import trimesh

    def decimate(mesh, target):
        o3d_mesh = o3d.geometry.TriangleMesh(
            o3d.utility.Vector3dVector(mesh.vertices), o3d.utility.Vector3iVector(mesh.faces)
        )
        o3d_mesh = o3d_mesh.simplify_quadric_decimation(target_number_of_triangles=target)
        out = trimesh.Trimesh(np.asarray(o3d_mesh.vertices), np.asarray(o3d_mesh.triangles), process=True)
        out.update_faces(out.nondegenerate_faces())
        return out

    mesh = trimesh.Trimesh(vertices, faces, process=True)  # merges the vertices split at texture seams
    mesh.update_faces(mesh.nondegenerate_faces())
    if len(mesh.faces) > max_faces and mesh.extents.max() > 0.3:
        parts = mesh.split(only_watertight=False)
        kept = [part for part in parts if part.extents.max() >= 0.03]
        if kept and len(kept) < len(parts):
            mesh = trimesh.util.concatenate(kept)
    if len(mesh.faces) > max_faces:
        mesh = decimate(mesh, max_faces)
    for voxel in (0.015, 0.025, 0.04):
        if len(mesh.faces) <= 1.5 * max_faces:
            break
        o3d_mesh = o3d.geometry.TriangleMesh(
            o3d.utility.Vector3dVector(mesh.vertices), o3d.utility.Vector3iVector(mesh.faces)
        )
        o3d_mesh = o3d_mesh.simplify_vertex_clustering(voxel_size=voxel)
        mesh = decimate(trimesh.Trimesh(np.asarray(o3d_mesh.vertices), np.asarray(o3d_mesh.triangles), process=True),
                        max_faces)
    return mesh.vertices.astype(np.float32), mesh.faces.astype(np.int32)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", default=default_root())
    parser.add_argument("--catalog", default=CATALOG_PATH)
    parser.add_argument("--max_faces", type=int, default=3000, help="triangles of a prop 1.5 m long or longer")
    parser.add_argument("--skip_download", action="store_true")
    parser.add_argument("--only", nargs="*", default=None)
    args = parser.parse_args()

    names = args.only or list(PROPS)
    if not args.skip_download:
        download(args.root, names)
    out = os.path.join(args.root, "processed")
    os.makedirs(out, exist_ok=True)
    catalog = json.load(open(args.catalog)) if os.path.exists(args.catalog) else {}
    for name in names:
        package = os.path.join(args.root, "nvidia_simready_assets", name)
        license_ = json.load(open(os.path.join(package, "com.nvidia.simready.packaging.json")))["license"]
        if license_ != "CC-BY-4.0":
            raise ValueError(f"{name}: license {license_}")
        vertices, faces = read_meshes(root_usd(name, args.root))
        lo, hi = vertices.min(0), vertices.max(0)
        budget = int(np.clip(args.max_faces * (hi - lo)[:2].max() / 1.5, 300, args.max_faces))
        v, f = simplify(vertices, faces, budget)
        np.savez_compressed(os.path.join(out, f"{name}.npz"), vertices=v, faces=f)
        catalog[name] = {
            "min": [round(float(x), 4) for x in lo],
            "max": [round(float(x), 4) for x in hi],
            "faces": int(len(f)),
            "source_faces": int(len(faces)),
            "license": license_,
        }
        print(f"{name}: {len(faces)} -> {len(f)} triangles, size {np.round(hi - lo, 3).tolist()} m", flush=True)
    with open(args.catalog, "w") as fh:
        json.dump(dict(sorted(catalog.items())), fh, indent=1)
        fh.write("\n")
    with open(os.path.join(args.root, "ATTRIBUTION.md"), "w") as fh:
        fh.write(f"Props of the civil engineering lab tile: NVIDIA SimReady assets ({DATASET}, revision {REVISION}), "
                 "licensed under CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/); meshes decimated for the "
                 "sensors.\n\n" + "".join(f"- {name}: {entry['license']}\n" for name, entry in sorted(catalog.items())))


if __name__ == "__main__":
    main()
