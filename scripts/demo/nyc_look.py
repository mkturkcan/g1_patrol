"""Visual dressing for the New York demo (rendering only; requires a running Kit app).

Builds textured, per-material meshes from the tile layouts of the NYC terrain (asphalt, concrete sidewalks with granite
curbs, zebra crossings, brick facades, green sidewalk sheds, orange cones, ...), adds visual-only city blocks above the
tiles' 4 m facades (brick, windows, cornices), street trees, a sky and a sun. The training meshes (``/World/ground``,
``/World/structures``) are hidden; their geometry, which is what the robot and its sensors interact with, is unchanged.
"""

from __future__ import annotations

import math

import numpy as np
import trimesh

import isaaclab.sim as sim_utils
from pxr import Gf, Sdf, UsdGeom, UsdLux, Vt

from g1_patrol.tasks.navigation.nyc.tiles import BUILDING_Y, HALF_L, HALF_W, HF_RESOLUTION, TILE_SIZE, TileLayout

NVIDIA = "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/6.1/NVIDIA"
ROOT = "/World/Demo"

TEXTURED = {  # material -> (MDL below NVIDIA/Materials, texture repeat [m])
    "asphalt": ("vMaterials_2/Ground/Asphalt_Fine.mdl", 3.0),
    "sidewalk": ("Base/Masonry/Concrete_Smooth.mdl", 1.5),
    "curb": ("Base/Stone/Granite_Light.mdl", 1.0),
    "plaza": ("vMaterials_2/Ground/Large_Granite_Paving.mdl", 2.5),
    "tree_pit": ("Base/Natural/Mulch_Brown.mdl", 1.0),
    "steel_plate": ("Base/Metals/Steel_Cast.mdl", 1.2),
    "facade": ("Base/Masonry/Brick_Wall_Red.mdl", 1.6),
    "brick_red": ("Base/Masonry/Brick_Wall_Red.mdl", 1.6),
    "brick_brown": ("Base/Masonry/Brick_Wall_Brown.mdl", 1.6),
    "limestone": ("Base/Masonry/Stucco.mdl", 2.0),
    "concrete_wall": ("vMaterials_2/Concrete/Concrete_Wall_Aged.mdl", 3.0),
    "jersey_barrier": ("vMaterials_2/Concrete/Concrete_Precast.mdl", 1.0),
    "planter": ("vMaterials_2/Concrete/Concrete_Wall_Even.mdl", 1.0),
    "construction_fence": ("Base/Wood/Plywood.mdl", 1.2),
    "partition": ("Base/Wood/Plywood.mdl", 1.2),
    "lumber": ("Base/Wood/Timber.mdl", 1.0),
}
EMISSIVE = {  # material -> (diffuse rgb, emissive rgb)
    "billboard_0": ((0.05, 0.05, 0.05), (0.9, 0.12, 0.2)),
    "billboard_1": ((0.05, 0.05, 0.05), (0.1, 0.35, 0.9)),
    "billboard_2": ((0.05, 0.05, 0.05), (0.9, 0.65, 0.1)),
    "billboard_3": ((0.05, 0.05, 0.05), (0.12, 0.8, 0.3)),
    "billboard_4": ((0.05, 0.05, 0.05), (0.8, 0.8, 0.85)),
}
COLORS = {  # material -> (diffuse rgb, roughness, metallic)
    "pothole": ((0.07, 0.07, 0.075), 0.35, 0.0),
    "leaves": ((0.13, 0.26, 0.07), 0.85, 0.0),
    "tower_glass": ((0.3, 0.38, 0.45), 0.1, 0.6),
    "stairwell": ((0.12, 0.12, 0.12), 0.8, 0.0),
    "stripe_white": ((0.86, 0.86, 0.82), 0.55, 0.0),
    "stripe_yellow": ((0.88, 0.66, 0.08), 0.55, 0.0),
    "storefront": ((0.08, 0.09, 0.1), 0.3, 0.6),
    "scaffold_steel": ((0.58, 0.59, 0.6), 0.35, 0.9),
    "shed_plywood": ((0.1, 0.22, 0.13), 0.8, 0.0),  # NYC sidewalk sheds: dark green painted plywood
    "pole": ((0.16, 0.18, 0.18), 0.45, 0.7),
    "signal_pole": ((0.22, 0.24, 0.24), 0.45, 0.7),
    "meter": ((0.32, 0.34, 0.35), 0.45, 0.8),
    "hydrant": ((0.62, 0.07, 0.04), 0.5, 0.2),
    "litter_basket": ((0.07, 0.25, 0.12), 0.6, 0.5),  # green wire litter baskets
    "newsbox": ((0.08, 0.22, 0.55), 0.4, 0.3),
    "bike_rack": ((0.4, 0.42, 0.45), 0.3, 0.9),
    "tree_guard": ((0.04, 0.04, 0.04), 0.5, 0.6),
    "bollard": ((0.18, 0.18, 0.18), 0.5, 0.6),
    "cone": ((0.95, 0.33, 0.04), 0.5, 0.0),
    "subway_rail": ((0.06, 0.24, 0.12), 0.4, 0.6),  # subway entrance railings are green
    "cafe_table": ((0.42, 0.43, 0.45), 0.35, 0.8),
    "kiosk": ((0.1, 0.22, 0.16), 0.4, 0.5),  # green newsstand
    "concrete": ((0.55, 0.55, 0.53), 0.8, 0.0),
    "window": ((0.16, 0.2, 0.24), 0.08, 0.5),
    "shop_window": ((0.025, 0.03, 0.035), 0.04, 0.85),  # storefront glass: dark and mirror-like in daylight
    "window_frame": ((0.12, 0.12, 0.12), 0.5, 0.3),
    "cornice": ((0.72, 0.7, 0.64), 0.7, 0.0),
    "bark": ((0.22, 0.16, 0.11), 0.9, 0.0),
    "awning": ((0.45, 0.06, 0.06), 0.8, 0.0),
}
BUILDING_MATERIALS = ["brick_red", "brick_brown", "limestone", "brick_red", "concrete_wall"]


def _material_path(name: str) -> str:
    return f"{ROOT}/Looks/{name}"


def create_materials():
    """Spawn every material once under ``/World/Demo/Looks``."""
    for name, (mdl, _) in TEXTURED.items():
        cfg = sim_utils.MdlFileCfg(mdl_path=f"{NVIDIA}/Materials/{mdl}")
        cfg.func(_material_path(name), cfg)
    for name, (rgb, rough, metal) in COLORS.items():
        cfg = sim_utils.PreviewSurfaceCfg(diffuse_color=rgb, roughness=rough, metallic=metal)
        cfg.func(_material_path(name), cfg)
    for name, (rgb, emissive) in EMISSIVE.items():
        cfg = sim_utils.PreviewSurfaceCfg(diffuse_color=rgb, emissive_color=emissive, roughness=0.2)
        cfg.func(_material_path(name), cfg)


def _triplanar_uv(vertices: np.ndarray, faces: np.ndarray, repeat: float) -> np.ndarray:
    """Face-varying texture coordinates (F*3, 2): each face is projected on the plane its normal faces most."""
    tri = vertices[faces]  # (F, 3, 3)
    normal = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    axis = np.abs(normal).argmax(axis=1)
    uv = np.where(
        (axis == 2)[:, None, None],
        tri[..., [0, 1]],
        np.where((axis == 1)[:, None, None], tri[..., [0, 2]], tri[..., [1, 2]]),
    )
    return (uv / repeat).reshape(-1, 2)


def add_mesh(stage, path: str, mesh: trimesh.Trimesh, material: str):
    """Define a USD mesh with triplanar texture coordinates and bind a demo material."""
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    usd_mesh = UsdGeom.Mesh.Define(stage, path)
    usd_mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(vertices.astype(np.float32)))
    usd_mesh.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(np.full(len(faces), 3, dtype=np.int32)))
    usd_mesh.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(faces.reshape(-1).astype(np.int32)))
    usd_mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    repeat = TEXTURED.get(material, (None, 1.0))[1]
    st = UsdGeom.PrimvarsAPI(usd_mesh).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.faceVarying)
    st.Set(Vt.Vec2fArray.FromNumpy(_triplanar_uv(vertices, faces, repeat).astype(np.float32)))
    sim_utils.bind_visual_material(path, _material_path(material))


def _smooth_ground_mesh(lay: TileLayout) -> trimesh.Trimesh:
    """The tile's ground as a plain grid of the height field (two triangles per cell). Unlike the collision mesh it
    has no vertical walls, so potholes and curbs render as smooth dips and edges instead of saw teeth."""
    nx, ny = lay.heights.shape
    x, y = np.meshgrid(-TILE_SIZE[0] / 2 + np.arange(nx) * HF_RESOLUTION, -TILE_SIZE[1] / 2 + np.arange(ny) * HF_RESOLUTION, indexing="ij")
    vertices = np.stack([x, y, lay.heights], axis=-1).reshape(-1, 3)
    idx = np.arange(nx * ny).reshape(nx, ny)
    a, b, c, d = idx[:-1, :-1], idx[1:, :-1], idx[1:, 1:], idx[:-1, 1:]
    faces = np.concatenate([np.stack([a, b, c], -1).reshape(-1, 3), np.stack([a, c, d], -1).reshape(-1, 3)])
    return trimesh.Trimesh(vertices=vertices, faces=faces, process=False)


def _submesh(vertices: np.ndarray, faces: np.ndarray) -> trimesh.Trimesh:
    used, inverse = np.unique(faces.reshape(-1), return_inverse=True)
    return trimesh.Trimesh(vertices=vertices[used], faces=inverse.reshape(-1, 3), process=False)


def dress_tile(stage, lay: TileLayout, center: tuple[float, float], name: str, rng: np.random.Generator,
               far_side: bool = True):
    """Textured ground and structures, city block and trees for one tile; ``center`` is the tile center (world xy)."""
    root = f"{ROOT}/Tiles/{name}"
    offset = np.array([center[0], center[1], 0.0])
    # ground: one mesh per painted material; steep faces (curb sides, pothole walls) are curb/asphalt
    ground = _smooth_ground_mesh(lay)
    vertices = np.asarray(ground.vertices) + offset
    faces = np.asarray(ground.faces)
    cells = lay.ground_materials()
    centroid = np.asarray(ground.vertices)[faces].mean(axis=1)
    i = np.clip(((centroid[:, 0] + TILE_SIZE[0] / 2) / HF_RESOLUTION).astype(int), 0, cells.shape[0] - 1)
    j = np.clip(((centroid[:, 1] + TILE_SIZE[1] / 2) / HF_RESOLUTION).astype(int), 0, cells.shape[1] - 1)
    face_mat = cells[i, j].astype(str)
    steep = np.abs(np.asarray(ground.face_normals)[:, 2]) < 0.5
    face_mat = np.where(steep & np.isin(face_mat, ["sidewalk", "curb"]), "curb", face_mat)
    for mat in np.unique(face_mat):
        add_mesh(stage, f"{root}/ground_{mat}", _submesh(vertices, faces[face_mat == mat]), mat)
    # structures, per material
    for mat, mesh in lay.structure_meshes_by_material().items():
        mesh.apply_translation(offset)
        add_mesh(stage, f"{root}/structure_{mat}", mesh, mat if mat in TEXTURED or mat in COLORS else "concrete")
    # city block above the facade (tiles with a facade) or on both sides (open tiles)
    has_facade = any(m == "facade" for m in lay.box_mats)
    edges = [BUILDING_Y] if has_facade else [HALF_W + 0.3, -HALF_W - 0.3]
    for k, y in enumerate(edges):
        _city_block(stage, f"{root}/block_{k}", lay, center, y, facing=-1.0 if y > 0 else 1.0,
                    base=4.0 if y == BUILDING_Y else 0.0, rng=rng)
    if has_facade and far_side:
        _far_side(stage, f"{root}/far_side", lay, center, rng)
    _storefronts(stage, f"{root}/storefronts", lay, center, rng)
    if lay.kind == "plaza":
        _times_square(stage, f"{root}/times_square", lay, center, rng)
    for k, (kind, x, y) in enumerate(lay.decor):
        if kind == "tree":
            _tree(stage, f"{root}/tree_{k}", center[0] + x, center[1] + y, lay.ground(x, y), rng)


def _box(center, extents, yaw: float = 0.0) -> trimesh.Trimesh:
    m = trimesh.creation.box(extents=extents)
    if yaw:
        m.apply_transform(trimesh.transformations.rotation_matrix(yaw, (0, 0, 1)))
    m.apply_translation(center)
    return m


def _city_block(stage, root: str, lay: TileLayout, center, y: float, facing: float, base: float, rng):
    """A row of buildings (brick or limestone, 3-8 storeys) along the tile edge at ``y``, facing the tile.

    ``base`` is the height up to which the tile's own facade (doors, storefronts) already exists.
    """
    cx, cy = center
    front = y if base > 0 else y  # facade plane (tile-local y)
    depth = 0.6
    by = cy + front - facing * depth / 2  # slab center: behind the facade plane
    x = -HALF_L
    walls: dict[str, list] = {}
    glass, frames, cornices, storefronts = [], [], [], []
    while x < HALF_L - 0.5:
        width = min(float(rng.uniform(4.0, 8.0)), HALF_L - x)
        storeys = int(rng.integers(3, 9))
        top = 4.0 + 3.2 * (storeys - 1) + 1.0
        mat = BUILDING_MATERIALS[int(rng.integers(len(BUILDING_MATERIALS)))]
        z0 = base
        walls.setdefault(mat, []).append(_box((cx + x + width / 2, by, (z0 + top) / 2), (width, depth, top - z0)))
        cornices.append(_box((cx + x + width / 2, by + facing * 0.1, top + 0.15), (width, depth + 0.5, 0.3)))
        if base == 0.0:  # street-level shop windows
            storefronts.append(_box((cx + x + width / 2, cy + front + facing * 0.03, 1.55), (width - 0.8, 0.06, 2.3)))
            frames += _window_frames(cx + x + width / 2, cy + front + facing * 0.07, 1.55, width - 0.8, 2.3)
        # windows: a grid on the front face, from the second storey up
        n_win = max(1, int(width / 1.9))
        for s in range(1, storeys):
            z = 4.0 + 3.2 * (s - 1) + 1.6
            for w in range(n_win):
                wx = cx + x + (w + 0.5) * width / n_win
                fy = cy + front + facing * 0.02
                glass.append(_box((wx, fy, z), (0.95, 0.04, 1.5)))
                frames.append(_box((wx, fy + facing * 0.01, z - 0.82), (1.15, 0.12, 0.12)))  # sill
        x += width
    for mat, ms in walls.items():
        add_mesh(stage, f"{root}/walls_{mat}", trimesh.util.concatenate(ms), mat)
    add_mesh(stage, f"{root}/windows", trimesh.util.concatenate(glass), "window")
    add_mesh(stage, f"{root}/sills", trimesh.util.concatenate(frames + cornices), "cornice")
    if storefronts:
        add_mesh(stage, f"{root}/storefronts", trimesh.util.concatenate(storefronts), "shop_window")


def _storefronts(stage, root: str, lay: TileLayout, center, rng):
    """Shop windows on the tile's ground-floor facade, some with an awning (above head height, visual only)."""
    cx, cy = center
    glass, awnings, frames = [], [], []
    for (bx, by, _, hx, _, _, _), mat in zip(lay.boxes, lay.box_mats):
        if mat != "facade" or 2 * hx < 1.6:
            continue
        z = lay.ground(bx, BUILDING_Y - 0.3)
        width = 2 * hx - 0.8
        glass.append(_box((cx + bx, cy + BUILDING_Y - 0.02, z + 1.65), (width, 0.04, 2.3)))
        frames += _window_frames(cx + bx, cy + BUILDING_Y - 0.05, z + 1.65, width, 2.3)
        if rng.uniform() < 0.5:
            awnings.append(_box((cx + bx, cy + BUILDING_Y - 0.45, z + 3.1), (2 * hx - 0.6, 0.9, 0.08)))
    if glass:
        add_mesh(stage, f"{root}/glass", trimesh.util.concatenate(glass), "shop_window")
        add_mesh(stage, f"{root}/frames", trimesh.util.concatenate(frames), "window_frame")
    if awnings:
        add_mesh(stage, f"{root}/awnings", trimesh.util.concatenate(awnings), "awning")


def _times_square(stage, root: str, lay: TileLayout, center, rng):
    """Lit billboards on the buildings around the plaza and shrubs in its planters."""
    cx, cy = center
    boards: dict[str, list] = {}
    for facing, y in ((-1.0, HALF_W + 0.3), (1.0, -HALF_W - 0.3)):
        x = -HALF_L + rng.uniform(0.0, 2.0)
        while x < HALF_L - 2.0:
            w, h = rng.uniform(3.0, 7.0), rng.uniform(2.5, 6.0)
            z = rng.uniform(4.5, 14.0)
            name = f"billboard_{int(rng.integers(5))}"
            boards.setdefault(name, []).append(_box((cx + x + w / 2, cy + y + facing * 0.12, z + h / 2), (w, 0.1, h)))
            x += w + rng.uniform(0.5, 3.0)
    for name, ms in boards.items():
        add_mesh(stage, f"{root}/{name}", trimesh.util.concatenate(ms), name)
    shrubs = []
    for (bx, by, _, hx, hy, _, z1), mat in zip(lay.boxes, lay.box_mats):
        if mat != "planter":
            continue
        for _ in range(4):
            blob = trimesh.creation.icosphere(subdivisions=2, radius=float(rng.uniform(0.3, 0.45)))
            blob.apply_scale((1.0, 1.0, 0.8))
            blob.apply_translation((cx + bx + rng.uniform(-0.3, 0.3), cy + by + rng.uniform(-0.3, 0.3), lay.ground(bx, by) + z1 + 0.15))
            shrubs.append(blob)
    if shrubs:
        add_mesh(stage, f"{root}/shrubs", trimesh.util.concatenate(shrubs), "leaves")


def _window_frames(x: float, y: float, z: float, width: float, height: float, pane: float = 1.4) -> list:
    """A frame around a shop window and vertical mullions every ``pane`` meters (in front of the glass at ``y``)."""
    t = 0.07
    parts = [_box((x, y, z + height / 2), (width + t, 0.08, t)), _box((x, y, z - height / 2), (width + t, 0.08, t))]
    n = max(1, round(width / pane))
    for k in range(n + 1):
        parts.append(_box((x - width / 2 + k * width / n, y, z), (t, 0.08, height)))
    return parts


def add_backdrop(stage, generator, dressed: set, margin: float = 300.0):
    """Asphalt at street level over the undressed tiles and around the terrain, so nothing shows as a void."""
    sx, sy = generator.cfg.size
    rows, cols = generator.cfg.num_rows, generator.cfg.num_cols
    x0, y0 = generator.shift[0], generator.shift[1]
    quads = [
        _box((x0 + (r + 0.5) * sx, y0 + (c + 0.5) * sy, -0.05), (sx, sy, 0.1))
        for r in range(rows) for c in range(cols) if (r, c) not in dressed
    ]
    w, h = rows * sx, cols * sy
    quads += [
        _box((x0 - margin / 2, 0.0, -0.05), (margin, h + 2 * margin, 0.1)),
        _box((x0 + w + margin / 2, 0.0, -0.05), (margin, h + 2 * margin, 0.1)),
        _box((x0 + w / 2, y0 - margin / 2, -0.05), (w, margin, 0.1)),
        _box((x0 + w / 2, y0 + h + margin / 2, -0.05), (w, margin, 0.1)),
    ]
    add_mesh(stage, f"{ROOT}/backdrop", trimesh.util.concatenate(quads), "asphalt")


def _far_side(stage, root: str, lay: TileLayout, center, rng):
    """The other side of the street, beyond the tile (for tiles with a facade): yellow center lines, a sidewalk with
    trees and another row of buildings facing the tile."""
    cx, cy = center
    # the tile's curb: first row (from the street side) that is raised
    raised = np.nonzero((lay.heights > 0.05).mean(axis=0) > 0.5)[0]
    curb_y = -HALF_W + raised[0] * HF_RESOLUTION if len(raised) else -HALF_W
    far_curb = curb_y - 10.0
    walk = 3.5
    mid = (curb_y + far_curb) / 2
    lines = [_box((cx, cy + mid + dy, 0.004), (2 * HALF_L, 0.11, 0.01)) for dy in (-0.09, 0.09)]
    add_mesh(stage, f"{root}/center_lines", trimesh.util.concatenate(lines), "stripe_yellow")
    add_mesh(stage, f"{root}/sidewalk", _box((cx, cy + far_curb - walk / 2, 0.075), (2 * HALF_L, walk, 0.15)), "sidewalk")
    add_mesh(stage, f"{root}/curb", _box((cx, cy + far_curb - 0.12, 0.08), (2 * HALF_L, 0.24, 0.16)), "curb")
    _city_block(stage, f"{root}/block", lay, center, far_curb - walk, facing=1.0, base=0.0, rng=rng)
    for k, x in enumerate(np.arange(-HALF_L + rng.uniform(1.0, 4.0), HALF_L, rng.uniform(6.0, 8.0))):
        _tree(stage, f"{root}/tree_{k}", cx + x, cy + far_curb - 0.8, 0.15, rng)


def _tree(stage, root: str, x: float, y: float, z: float, rng):
    """A street tree: trunk and a canopy of a few overlapping blobs (all above head height)."""
    height = float(rng.uniform(2.6, 3.4))
    trunk = trimesh.creation.cylinder(radius=0.09, height=height, sections=12)
    trunk.apply_translation((x, y, z + height / 2))
    add_mesh(stage, f"{root}/trunk", trunk, "bark")
    blobs = []
    for _ in range(14):
        r = float(rng.uniform(0.45, 0.8))
        blob = trimesh.creation.icosphere(subdivisions=2, radius=r)
        blob.apply_scale((1.0, 1.0, 0.8))
        blob.apply_translation((x + rng.uniform(-1.0, 1.0), y + rng.uniform(-0.8, 0.8), z + height + rng.uniform(0.2, 1.8)))
        blobs.append(blob)
    add_mesh(stage, f"{root}/canopy", trimesh.util.concatenate(blobs), "leaves")


def add_skyline(stage, center, rng, inner: float = 45.0, outer: float = 130.0, count: int = 70):
    """Distant towers all around, so streets end in a city skyline instead of the sky dome's horizon."""
    groups: dict[str, list] = {}
    for _ in range(count):
        a, d = rng.uniform(0, 2 * math.pi), rng.uniform(inner, outer)
        w, l, h = rng.uniform(12, 30), rng.uniform(12, 30), rng.uniform(25, 120) * (0.6 + 0.4 * d / outer)
        mat = ["limestone", "concrete_wall", "brick_brown", "tower_glass", "tower_glass"][int(rng.integers(5))]
        groups.setdefault(mat, []).append(
            _box((center[0] + d * math.cos(a), center[1] + d * math.sin(a), h / 2), (w, l, h), yaw=float(rng.uniform(-0.1, 0.1)))
        )
    for mat, boxes in groups.items():
        add_mesh(stage, f"{ROOT}/skyline_{mat}", trimesh.util.concatenate(boxes), mat)


def add_sky_and_sun(stage, sky: str = "Skies/Cloudy/kloofendal_48d_partly_cloudy_4k.hdr"):
    dome = UsdLux.DomeLight.Define(stage, f"{ROOT}/Sky")
    dome.CreateIntensityAttr(900.0)
    dome.CreateTextureFileAttr(f"{NVIDIA}/Assets/{sky}")
    dome.CreateTextureFormatAttr(UsdLux.Tokens.latlong)
    sun = UsdLux.DistantLight.Define(stage, f"{ROOT}/Sun")
    sun.CreateIntensityAttr(2500.0)
    sun.CreateAngleAttr(0.6)
    sun.CreateColorAttr(Gf.Vec3f(1.0, 0.95, 0.86))
    xf = UsdGeom.Xformable(sun.GetPrim())
    xf.AddRotateXYZOp().Set(Gf.Vec3f(48.0, 0.0, 35.0))


def hide_training_visuals(stage, paths=("/World/ground", "/World/structures")):
    """Hide the plain training meshes (their collision and ray-cast geometry is unaffected) and the training scene's
    lights. Every mesh is hidden individually: the renderer may not pick up visibility inherited from a parent."""
    from pxr import Usd

    hidden = 0
    for path in paths:
        root = stage.GetPrimAtPath(path)
        if not root:
            print(f"[nyc_look] {path} not found", flush=True)
            continue
        for prim in Usd.PrimRange(root):
            if prim.IsA(UsdGeom.Imageable):
                UsdGeom.Imageable(prim).MakeInvisible()
                hidden += prim.IsA(UsdGeom.Mesh)
    for prim in stage.Traverse():
        if prim.GetPath().pathString.startswith(ROOT):
            continue
        if prim.IsA(UsdLux.DomeLight) or prim.IsA(UsdLux.DistantLight):
            UsdGeom.Imageable(prim).MakeInvisible()
    print(f"[nyc_look] hid {hidden} training meshes", flush=True)


def tile_center(generator, r: int, c: int) -> tuple[float, float]:
    return (r + 0.5) * generator.cfg.size[0] + generator.shift[0], (c + 0.5) * generator.cfg.size[1] + generator.shift[1]


def dress(stage, generator, tiles: list[tuple[int, int]], seed: int = 0):
    """Dress the given (row, col) tiles of an :class:`NycTerrainGenerator` and add sky and sun."""
    create_materials()
    rng = np.random.default_rng(seed)
    dressed = set(tiles)
    for r, c in tiles:
        lay = generator.layouts[(r, c)]
        dress_tile(stage, lay, tile_center(generator, r, c), f"r{r}_c{c}", rng, far_side=(r, c - 1) not in dressed)
    add_backdrop(stage, generator, dressed)
    centers = np.array([tile_center(generator, r, c) for r, c in tiles])
    add_skyline(stage, centers.mean(axis=0), rng, inner=max(45.0, float(np.ptp(centers, axis=0).max()) / 2 + 35.0))
    add_sky_and_sun(stage)
    hide_training_visuals(stage)
