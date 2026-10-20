"""Import downloaded litter and trash-bag models (CC0 / CC-BY 4.0, from Poly Haven and Objaverse) into rigid-body USD
assets for the civic navigation tasks.

For each model of ``SELECTION``: load the glTF, keep the selected parts, turn it Z-up, scale it to its real size,
center it on its bounding box, and write ``<out>/<name>.usdc`` with the visual meshes (base-color texture, roughness
and metallic factors), a convex collision hull and the mass, plus the texture (``<out>/textures``). The models are added
to ``<out>/manifest.json`` (as written by ``make_nyc_trash.py``) and credited in ``<out>/ATTRIBUTION.md``.

The downloads are data only (nothing in them is executed); ``--downloads`` is the directory with ``manifest.json``
listing each model's local path, source, author and license.

Usage:
    python -I scripts/assets/import_litter_assets.py --downloads <litter_online> --out src/g1_patrol/assets/data/trash
"""

import argparse
import json
import math
import os

import numpy as np
import trimesh
from PIL import Image

# name: (download id, kind, size (m) of the longest side after import, mass (kg), parts to keep (geometry indices in
# file order, None: all), note). Litter rests with its smallest side up unless the note says "upright" (as modeled).
SELECTION = {
    # trash bags
    "bag_crumpled_black": ("garbage_bag__trashbag", "bag", 0.52, 2.4, None, ""),
    "bag_kitchen_black": ("garbage_bag__trashbag01__06b6e2b9", "bag", 0.50, 1.8, None, ""),
    "bag_scanned_black": ("garbage_bag__garbage_scan_day183__46e7a742", "bag", 0.52, 2.4, None, ""),
    "bag_tied_black": ("garbage_bag__garbage_bag_tied__bf200222", "bag", 0.42, 1.5, None, ""),
    "bag_squat_black": ("garbage_bag__garbage_bag_crumpled__e0e9fb92", "bag", 0.44, 1.4, None, ""),
    "bag_recycling_clear": ("recycling_bag__clear_trash_bag__181b8f80", "bag", 0.50, 1.0, None, ""),
    # litter
    "cup_to_go": ("coffee_cup__coffee_to_go_cup__cdb8ee22", "litter", 0.136, 0.015, None, ""),
    "cup_paper_dented": ("coffee_cup__paper_cup_unperfect__1903f387", "litter", 0.13, 0.012, None, ""),
    "can_tallboy": ("soda_can__generic_beverage_can__3d0c9c38", "litter", 0.17, 0.02, None, ""),
    "can_dented": ("soda_can__wasteland_soft_drink_can__e1dc6cd9", "litter", 0.122, 0.015, None, ""),
    "bottle_crushed": ("pet_bottle__crushed_dirty_water_bottle__731efe26", "litter", 0.21, 0.02, None, ""),
    "chip_bag_fresh": ("chip_bag__fresh_chip_bag__7116fcdf", "litter", 0.24, 0.02, None, ""),
    "takeout_box_crumpled": ("takeout__chinese_takeout_box_crumpled__a6796493", "litter", 0.2, 0.03, None, ""),
    "pizza_box_open": ("takeout__pizza_box_open_greasy__054be7a9", "litter", 0.38, 0.15, None, "upright"),
    "paper_wad": ("paper__paper_wad_scan__466d8e12", "litter", 0.07, 0.005, None, ""),
    "cigarette_pack": ("cigarette_pack__cigarette_pack", "litter", 0.103, 0.02, None, ""),
    "beer_bottle": ("glass_bottle__empty_beer_bottles__c5c009d8", "litter", 0.23, 0.2, [0], "one of the two bottles"),
    "plastic_bag_tied": ("shopping_bag__plastic_bag_dirty3__1a4bbf8e", "litter", 0.3, 0.03, None, ""),
    "newspaper_folded": ("paper__newspaper_silver_herald__84683791", "litter", 0.3, 0.15, None, ""),
}
TEXTURE_SIZE = 1024


def load_parts(path: str, keep):
    """Meshes of a glTF file in world coordinates (node transforms applied), with their base-color image (or color),
    roughness and metallic factors."""
    scene = trimesh.load(path, force="scene")
    parts = []
    for i, node in enumerate(scene.graph.nodes_geometry):
        transform, name = scene.graph[node]
        mesh = scene.geometry[name]
        if not isinstance(mesh, trimesh.Trimesh) or len(mesh.faces) == 0:
            continue
        parts.append((i, mesh, transform))
    if keep is not None:
        parts = [p for k, p in enumerate(parts) if k in keep]
    out = []
    for _, mesh, transform in parts:
        m = mesh.copy()
        m.apply_transform(transform)
        visual = m.visual
        image, color, rough, metal, uv = None, (0.6, 0.6, 0.6, 1.0), 0.6, 0.0, None
        if isinstance(visual, trimesh.visual.TextureVisuals):
            mat = visual.material
            uv = visual.uv
            if isinstance(mat, trimesh.visual.material.PBRMaterial):
                image = mat.baseColorTexture
                if mat.baseColorFactor is not None:
                    color = tuple(np.asarray(mat.baseColorFactor, dtype=float) / (255.0 if np.max(mat.baseColorFactor) > 1 else 1.0))
                rough = float(mat.roughnessFactor) if mat.roughnessFactor is not None else 0.6
                metal = float(mat.metallicFactor) if mat.metallicFactor is not None else 0.0
            elif hasattr(mat, "image"):
                image = mat.image
        elif isinstance(visual, trimesh.visual.ColorVisuals):
            c = np.asarray(visual.main_color, dtype=float) / 255.0
            color = tuple(c)
        out.append({"mesh": m, "image": image, "color": color, "roughness": rough, "metallic": metal, "uv": uv})
    return out


def write_usd(path: str, name: str, parts, mass: float, tex_dir: str):
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade, Vt

    stage = Usd.Stage.CreateNew(path)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    root = UsdGeom.Xform.Define(stage, "/trash")
    stage.SetDefaultPrim(root.GetPrim())
    UsdPhysics.MassAPI.Apply(root.GetPrim()).CreateMassAttr(mass)
    looks = UsdGeom.Scope.Define(stage, "/trash/Looks")
    _ = looks
    all_v = []
    for k, part in enumerate(parts):
        m = part["mesh"]
        mesh = UsdGeom.Mesh.Define(stage, f"/trash/part{k}")
        mesh.CreatePointsAttr(Vt.Vec3fArray([Gf.Vec3f(*map(float, p)) for p in m.vertices]))
        mesh.CreateFaceVertexCountsAttr(Vt.IntArray([3] * len(m.faces)))
        mesh.CreateFaceVertexIndicesAttr(Vt.IntArray([int(i) for i in np.asarray(m.faces).reshape(-1)]))
        mesh.CreateNormalsAttr(Vt.Vec3fArray([Gf.Vec3f(*map(float, n)) for n in m.vertex_normals]))
        mesh.SetNormalsInterpolation(UsdGeom.Tokens.vertex)
        mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
        all_v.append(np.asarray(m.vertices))
        # material
        mat = UsdShade.Material.Define(stage, f"/trash/Looks/mat{k}")
        shader = UsdShade.Shader.Define(stage, f"/trash/Looks/mat{k}/shader")
        shader.CreateIdAttr("UsdPreviewSurface")
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(float(np.clip(part["roughness"], 0.05, 1.0)))
        shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(float(np.clip(part["metallic"], 0.0, 1.0)))
        color = part["color"]
        if part["image"] is not None and part["uv"] is not None:
            img = part["image"].convert("RGB")
            if max(img.size) > TEXTURE_SIZE:
                img.thumbnail((TEXTURE_SIZE, TEXTURE_SIZE), Image.LANCZOS)
            tex_name = f"{name}_{k}.jpg"
            img.save(os.path.join(tex_dir, tex_name), quality=90)
            st = UsdGeom.PrimvarsAPI(mesh).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex)
            st.Set(Vt.Vec2fArray([Gf.Vec2f(float(u), float(v)) for u, v in part["uv"]]))
            reader = UsdShade.Shader.Define(stage, f"/trash/Looks/mat{k}/st")
            reader.CreateIdAttr("UsdPrimvarReader_float2")
            reader.CreateInput("varname", Sdf.ValueTypeNames.String).Set("st")
            tex = UsdShade.Shader.Define(stage, f"/trash/Looks/mat{k}/tex")
            tex.CreateIdAttr("UsdUVTexture")
            tex.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(f"./textures/{tex_name}")
            tex.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(reader.ConnectableAPI(), "result")
            tex.CreateInput("wrapS", Sdf.ValueTypeNames.Token).Set("repeat")
            tex.CreateInput("wrapT", Sdf.ValueTypeNames.Token).Set("repeat")
            tex.CreateInput("scale", Sdf.ValueTypeNames.Float4).Set(Gf.Vec4f(float(color[0]), float(color[1]), float(color[2]), 1.0))
            tex.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
            shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(tex.ConnectableAPI(), "rgb")
        else:
            shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*map(float, color[:3])))
        mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
        UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(mat)
    hull = trimesh.convex.convex_hull(np.concatenate(all_v))
    col = UsdGeom.Mesh.Define(stage, "/trash/collision")
    col.CreatePointsAttr(Vt.Vec3fArray([Gf.Vec3f(*map(float, p)) for p in hull.vertices]))
    col.CreateFaceVertexCountsAttr(Vt.IntArray([3] * len(hull.faces)))
    col.CreateFaceVertexIndicesAttr(Vt.IntArray([int(i) for i in np.asarray(hull.faces).reshape(-1)]))
    col.CreatePurposeAttr(UsdGeom.Tokens.guide)
    UsdPhysics.CollisionAPI.Apply(col.GetPrim())
    UsdPhysics.MeshCollisionAPI.Apply(col.GetPrim()).CreateApproximationAttr("convexHull")
    stage.GetRootLayer().Save()


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--downloads", required=True, help="Directory with the downloads' manifest.json.")
    parser.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "..", "..", "src", "g1_patrol", "assets", "data", "trash"))
    args = parser.parse_args()
    with open(os.path.join(args.downloads, "manifest.json")) as f:
        sources = {a["id"]: a for a in json.load(f)}
    tex_dir = os.path.join(args.out, "textures")
    os.makedirs(tex_dir, exist_ok=True)
    manifest_path = os.path.join(args.out, "manifest.json")
    manifest = json.load(open(manifest_path)) if os.path.exists(manifest_path) else {}
    credits = []
    y_up_to_z_up = trimesh.transformations.rotation_matrix(math.pi / 2, (1, 0, 0))
    for name, (source_id, kind, size, mass, keep, note) in SELECTION.items():
        src = sources[source_id]
        parts = load_parts(os.path.join(args.downloads, src["local_path"]), keep)
        verts = np.concatenate([p["mesh"].vertices for p in parts])
        lo, hi = verts.min(0), verts.max(0)
        scale = size / float(np.max(hi - lo))
        center = (lo + hi) / 2
        for p in parts:
            p["mesh"].apply_translation(-center)
            p["mesh"].apply_scale(scale)
            p["mesh"].apply_transform(y_up_to_z_up)
        verts = np.concatenate([p["mesh"].vertices for p in parts])
        extent = verts.max(0) - verts.min(0)
        write_usd(os.path.join(args.out, f"{name}.usdc"), name, parts, mass, tex_dir)
        entry = {"file": f"{name}.usdc", "size": [round(float(e), 4) for e in extent], "mass": mass,
                 "rest_height": round(float(min(extent)) / 2, 4), "source": src["url"], "author": src["author"],
                 "license": src["license"]}
        if kind == "bag":
            entry["kind"] = "bag"
        if note == "upright":
            entry["rest"] = "upright"
            entry["rest_height"] = round(float(extent[2]) / 2, 4)
        manifest[name] = entry
        print(name, entry["size"], sum(len(p["mesh"].faces) for p in parts), "faces")
        credits.append(f"- `{name}`: \"{src['title']}\" by {src['author']}, {src['license']}, {src['url']}"
                       + (f" ({note})" if note and note != "upright" else "") + ".")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=1)
    with open(os.path.join(args.out, "ATTRIBUTION.md"), "w") as f:
        f.write("# Litter and trash-bag models\n\n")
        f.write("The `.usda` items are generated by `scripts/assets/make_nyc_trash.py` (no third-party content). The `.usdc`\n")
        f.write("items are imported by `scripts/assets/import_litter_assets.py` from the models below, which were rescaled,\n")
        f.write("re-centered, reduced to the listed parts, converted to USD and given a convex collision hull and a mass.\n\n")
        f.write("\n".join(credits) + "\n")


if __name__ == "__main__":
    main()
