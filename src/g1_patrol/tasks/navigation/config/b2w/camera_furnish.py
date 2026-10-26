"""Furniture and clutter for the indoor tiles as the camera students see them (rendering only; requires a running Kit
app): the boxes that stand for benches, tables, desks, machines, racks, shelves, cabinets, carts, cartons and the like
are drawn as NVIDIA SimReady props (:mod:`g1_patrol.tasks.navigation.nyc.simready`) fitted to each box (tiled along its
length, scaled within limits, the front towards the room), with tools, monitors, computers and boxes on the work
surfaces, boards, panels, fire extinguishers and exit signs on the walls, and ducts, cable trays and pipes under the
halls' ceilings. Everything drawn stays within a box, on a wall or under the ceiling: the geometry the robot and its
sensors interact with is unchanged.
"""

from __future__ import annotations

import copy
import math

import numpy as np

SURFACE = {  # prop: (height (m) of its work surface, side of its back panel along its local x: +1, -1 or 0)
    "furniture_workbench_standard_d01": (0.78, -1),
    "furniture_workbench_modular_a01": (0.92, 1),
    "desk_a01": (0.76, 0),
}
FURNITURE = {  # box material -> props it may be drawn as
    "bench": ["furniture_workbench_standard_d01", "furniture_workbench_modular_a01"],
    "table": ["furniture_workbench_standard_d01", "desk_a01"],
    "desk": ["desk_a01"],
    "machine": ["equipment_3d_printer_a01", "equipment_airCompressor_a01", "equipment_tools_trolley_a01",
                "storage_locker_double_bench_b01", "equipment_liftTable_scissorMobile_a01", "cabinet_file_a01"],
    "rack": ["storage_shelf_basic_a01", "storage_shelf_basic_b01", "storage_shelf_basic_c01", "rack_bulk_storage_a03",
             "rack_composition_a04", "rack_composition_a05", "rack_composition_a06"],
    "shelf": ["storage_shelf_basic_a01", "storage_shelf_basic_b01", "storage_shelf_basic_c01"],
    "cabinet": ["cabinet_file_a01", "storage_locker_double_bench_b01"],
    "toolboxes": ["equipment_tools_trolley_a01", "storage_toolbox_standard_hardCase_d01"],
    "cart": ["cart_mobile_a01", "cart_mobile_b01", "cart_mobile_c01", "cart_mobile_d01", "cart_computer_a01"],
    "carton": ["box_cardboard_a01", "box_cardboard_b02", "box_cardboard_c03"],
    "pallet": ["pallet_a01"],
    "barrel": ["drum_plastic_blue_a01", "drum_steel_a01"],
    "bucket": ["pail_closedHead_blue_a01", "pail_plastic_green_a02", "pail_plastic_red_a02"],
    "bin": ["trashCan_office_gray_a03", "trashCan_thin_gray_a03", "bin_waste_a01"],
    "steel_bin": ["trashCan_wheeled_green_a01"],
}  # fmt: skip
ON_SURFACES = ["computer_monitor_office_a01", "computer_desktop_tower_a01", "equipment_bench_vise_a01",
               "storage_toolbox_carry_hardCase_b01", "ppe_helmet_hardHat_a01", "box_cardboard_a01",
               "box_cardboard_c03", "pail_plastic_red_a02", "storage_toolbox_standard_hardCase_d01"]  # fmt: skip
SCALE_LIMITS = (0.55, 1.8)
MATERIALS = {  # (diffuse rgb, roughness, metallic)
    "duct": ((0.62, 0.64, 0.66), 0.35, 0.9),
    "cable_tray": ((0.45, 0.46, 0.47), 0.4, 0.8),
    "pipe_red": ((0.6, 0.06, 0.05), 0.4, 0.3),
    "pipe_gray": ((0.5, 0.5, 0.52), 0.35, 0.7),
    "whiteboard": ((0.93, 0.94, 0.94), 0.15, 0.0),
    "board_frame": ((0.62, 0.63, 0.65), 0.3, 0.9),
    "cork": ((0.58, 0.42, 0.26), 0.9, 0.0),
    "panel_gray": ((0.6, 0.62, 0.62), 0.4, 0.4),
    "extinguisher": ((0.72, 0.04, 0.03), 0.3, 0.2),
    "poster_0": ((0.15, 0.3, 0.6), 0.6, 0.0),
    "poster_1": ((0.85, 0.75, 0.2), 0.6, 0.0),
    "poster_2": ((0.85, 0.85, 0.82), 0.6, 0.0),
    "poster_3": ((0.7, 0.15, 0.12), 0.6, 0.0),
}
EMISSIVE = {"exit_sign": ((0.05, 0.3, 0.1), (0.1, 0.9, 0.3))}


def register_materials(nyc_look):
    """Add the clutter's materials to the dressing's (call before ``nyc_look.create_materials``)."""
    nyc_look.COLORS.update(MATERIALS)
    nyc_look.EMISSIVE.update(EMISSIVE)


def _fit(lay_box, names, catalog, rng, surface: bool):
    """The prop, number of copies along the box's long axis and scales (along the prop's long axis, short axis,
    height) that fit the box best, chosen at random among good fits; None if none fits within :data:`SCALE_LIMITS`."""
    cx, cy, yaw, hx, hy, z0, z1 = lay_box
    length, width, height = 2 * max(hx, hy), 2 * min(hx, hy), z1 - z0
    fits = []
    for name in names:
        lo, hi = np.asarray(catalog[name]["min"]), np.asarray(catalog[name]["max"])
        ext = hi - lo
        pl, pw = max(ext[0], ext[1]), min(ext[0], ext[1])
        ph = SURFACE[name][0] if surface and name in SURFACE else ext[2]
        n = max(1, int(round(length / pl)))
        scales = np.array([length / (n * pl), width / pw, height / ph])
        if scales.min() < SCALE_LIMITS[0] or scales.max() > SCALE_LIMITS[1]:
            continue
        fits.append((float(np.abs(np.log(scales)).sum()), name, n, scales))
    if not fits:
        return None
    costs = np.array([f[0] for f in fits])
    p = np.exp(-(costs - costs.min()) / 0.25)
    return fits[int(rng.choice(len(fits), p=p / p.sum()))][1:]


def _place(stage, path: str, name: str, x: float, y: float, z: float, yaw: float, scale=(1.0, 1.0, 1.0)):
    """A reference to a SimReady prop at (x, y, z) (its base), turned by ``yaw`` and scaled along its own axes,
    without physics (as ``nyc_look._props``)."""
    from pxr import Gf, Usd, UsdGeom, UsdPhysics

    from g1_patrol.tasks.navigation.nyc.simready import root_usd

    holder = UsdGeom.Xform.Define(stage, path)
    holder.AddTranslateOp().Set(Gf.Vec3d(x, y, z))
    holder.AddRotateZOp().Set(math.degrees(yaw))
    holder.AddScaleOp().Set(Gf.Vec3f(*map(float, scale)))
    prim = stage.DefinePrim(f"{path}/{name}", "Xform")
    prim.GetReferences().AddReference(root_usd(name))
    variant_sets = prim.GetVariantSets()
    for physics in ("PhysX", "Newton", "MuJoCo"):
        if variant_sets.HasVariantSet(physics):
            variant_sets.GetVariantSet(physics).SetVariantSelection("Disabled")
    for part in Usd.PrimRange(prim):
        if part.HasAPI(UsdPhysics.RigidBodyAPI):
            part.RemoveAPI(UsdPhysics.RigidBodyAPI)
        if part.HasAPI(UsdPhysics.CollisionAPI):
            UsdPhysics.CollisionAPI(part).CreateCollisionEnabledAttr(False)


def _inside(part, box) -> bool:
    """Whether a drawn-only part (legs, tops) lies within a box's volume (it belongs to the furniture the box is)."""
    cx, cy, yaw, hx, hy, z0, z1 = box
    if part[0] == "box":
        _, x, y, _, _, _, pz0, pz1, _ = part
    elif part[0] == "circle":
        _, x, y, _, pz0, pz1, _ = part
    else:
        return False
    c, s = math.cos(yaw), math.sin(yaw)
    u, v = (x - cx) * c + (y - cy) * s, -(x - cx) * s + (y - cy) * c
    return abs(u) <= hx + 0.02 and abs(v) <= hy + 0.02 and pz0 >= z0 - 0.02 and pz1 <= z1 + 0.02


def furnish(stage, nyc_look, root: str, lay, center, rng: np.random.Generator):
    """Draw an indoor tile's furniture boxes as SimReady props with clutter on their surfaces (under ``root``); returns
    a copy of the layout without the drawn boxes' plain meshes, for ``nyc_look.dress_tile``."""
    from g1_patrol.tasks.navigation.nyc.simready import catalog as simready_catalog

    catalog = simready_catalog()
    cx0, cy0 = center
    replaced, k = set(), 0
    for i, (box, mat) in enumerate(zip(lay.boxes, lay.box_mats)):
        if mat not in FURNITURE or box[5] > 0.15:
            continue
        surface = mat in ("bench", "table", "desk")
        fit = _fit(box, FURNITURE[mat], catalog, rng, surface)
        if fit is None:
            continue
        name, n, (sl, sw, sh) = fit
        cx, cy, yaw, hx, hy, z0, z1 = box
        long_dir = yaw if hx >= hy else yaw + math.pi / 2
        lo, hi = np.asarray(catalog[name]["min"]), np.asarray(catalog[name]["max"])
        ext = hi - lo
        prop_long_is_x = ext[0] >= ext[1]
        # the prop's long axis along the box's; its front (away from its back panel) towards the room's middle
        base_yaw = long_dir if prop_long_is_x else long_dir - math.pi / 2
        back = SURFACE.get(name, (0.0, 0))[1]
        if back and not prop_long_is_x:
            front = np.array([-back * math.cos(base_yaw), -back * math.sin(base_yaw)])  # local -back*x in the world
            if front @ np.array([-cx, -cy]) < 0:  # (tile-local box center; the room's middle is the tile center)
                base_yaw += math.pi
        elif rng.uniform() < 0.5:
            base_yaw += math.pi
        scale = (sl, sw, sh) if prop_long_is_x else (sw, sl, sh)
        mid = (lo + hi) / 2
        c, s = math.cos(base_yaw), math.sin(base_yaw)
        step = 2 * max(hx, hy) / n
        ground = lay.ground(cx, cy)
        for j in range(n):
            along = -max(hx, hy) + step * (j + 0.5)
            px, py = cx + along * math.cos(long_dir), cy + along * math.sin(long_dir)
            ox, oy = mid[0] * scale[0], mid[1] * scale[1]  # the prop's footprint center in the world, relative
            _place(stage, f"{root}/f{k:03d}", name, cx0 + px - (c * ox - s * oy), cy0 + py - (s * ox + c * oy),
                   ground + z0, base_yaw, scale)
            k += 1
        replaced.add(i)
        if surface:  # tools, monitors, computers and boxes on the work surface
            length = 2 * max(hx, hy)
            count = int(rng.integers(1, 3)) * max(1, int(length // 1.2))
            spots = np.sort(rng.uniform(-length / 2 + 0.3, length / 2 - 0.3, count))
            spots = [u for t, u in enumerate(spots) if t == 0 or u - spots[t - 1] > 0.5]
            depth = 2 * min(hx, hy)
            for u in spots:
                item = ON_SURFACES[int(rng.integers(len(ON_SURFACES)))]
                v = rng.uniform(-0.25, 0.15) * depth
                px = cx + u * math.cos(long_dir) - v * math.sin(long_dir)
                py = cy + u * math.sin(long_dir) + v * math.cos(long_dir)
                _place(stage, f"{root}/f{k:03d}", item, cx0 + px, cy0 + py, ground + z1,
                       long_dir + math.pi * int(rng.integers(2)) + rng.uniform(-0.3, 0.3))
                k += 1
    drawn = copy.copy(lay)
    drawn.unmeshed = set(lay.unmeshed) | {("box", i) for i in replaced}
    boxes = [lay.boxes[i] for i in replaced]
    drawn.mesh_parts = [p for p in lay.mesh_parts if not any(_inside(p, b) for b in boxes)]
    return drawn


def wall_decor(stage, nyc_look, root: str, lay, center, rng: np.random.Generator):
    """Boards, panels, posters, fire extinguishers and exit signs on the room's long walls (on the faces towards the
    room, within 0.2 m of them)."""
    import trimesh

    cx0, cy0 = center
    parts = {}

    def add(mat, mesh):
        parts.setdefault(mat, []).append(mesh)

    for box, mat in zip(lay.boxes, lay.box_mats):
        cx, cy, yaw, hx, hy, z0, z1 = box
        if mat != "wall" or min(hx, hy) > 0.12 or max(hx, hy) < 1.0 or z1 < 2.0:
            continue
        long_dir = yaw if hx >= hy else yaw + math.pi / 2
        t = np.array([math.cos(long_dir), math.sin(long_dir)])
        nrm = np.array([-t[1], t[0]])
        if nrm @ np.array([-cx, -cy]) < 0:  # the face towards the room's middle
            nrm = -nrm
        face = np.array([cx, cy]) + nrm * min(hx, hy)
        half = max(hx, hy)
        u = -half + 0.4
        while u < half - 0.6:
            kind = rng.choice(["whiteboard", "cork", "panel", "poster", "extinguisher", "none", "none"])
            w = {"whiteboard": 1.8, "cork": 1.2, "panel": 0.6, "poster": 0.6, "extinguisher": 0.3, "none": 1.5}[kind]
            if u + w > half - 0.3:
                break
            p = face + t * (u + w / 2)
            yaw_w = math.atan2(nrm[1], nrm[0])

            def slab(width, height, depth, z, m):
                q = p + nrm * depth / 2
                add(m, nyc_look._box((cx0 + q[0], cy0 + q[1], z), (depth, width, height), yaw_w))

            if kind == "whiteboard":
                slab(1.84, 1.04, 0.03, 1.5, "board_frame")
                slab(1.8, 1.0, 0.035, 1.5, "whiteboard")
            elif kind == "cork":
                slab(1.24, 0.94, 0.03, 1.45, "board_frame")
                slab(1.2, 0.9, 0.035, 1.45, "cork")
            elif kind == "panel":
                slab(0.6, 0.8, 0.18, 1.35, "panel_gray")
            elif kind == "poster":
                slab(0.6, 0.85, 0.01, 1.55, f"poster_{int(rng.integers(4))}")
            elif kind == "extinguisher":
                q = p + nrm * 0.1
                body = trimesh.creation.cylinder(radius=0.08, height=0.5, sections=16)
                body.apply_translation((cx0 + q[0], cy0 + q[1], 1.0))
                add("extinguisher", body)
                slab(0.3, 0.3, 0.01, 1.75, "poster_3")
            u += w + rng.uniform(0.3, 1.6)
        if rng.uniform() < 0.5:  # an exit sign high on the wall
            p = face + t * rng.uniform(-half + 0.5, half - 0.5) + nrm * 0.04
            sign = nyc_look._box((cx0 + p[0], cy0 + p[1], 2.45), (0.06, 0.36, 0.16), math.atan2(nrm[1], nrm[0]))
            add("exit_sign", sign)
    for mat, meshes in parts.items():
        nyc_look.add_mesh(stage, f"{root}/{mat}", trimesh.util.concatenate(meshes), mat)


def hall_services(stage, nyc_look, root: str, center, height: float, side: float):
    """Ducts, cable trays and pipes along a hall tile under its ceiling, on the ``side`` (+-1) of the hall: the same
    for every tile of a hall, so that they continue from tile to tile."""
    import trimesh

    from g1_patrol.tasks.navigation.nyc.tiles import HALF_L

    cx, cy = center
    duct = nyc_look._box((cx, cy + side * 2.2, height - 0.55), (2 * HALF_L, 0.8, 0.5))
    nyc_look.add_mesh(stage, f"{root}/duct", duct, "duct")
    tray = nyc_look._box((cx, cy - side * 2.8, height - 0.35), (2 * HALF_L, 0.4, 0.08))
    nyc_look.add_mesh(stage, f"{root}/cable_tray", tray, "cable_tray")
    for k, (y, mat) in enumerate(((-side * 3.6, "pipe_red"), (-side * 3.8, "pipe_gray"))):
        pipe = trimesh.creation.cylinder(radius=0.05, height=2 * HALF_L, sections=12)
        pipe.apply_transform(trimesh.transformations.rotation_matrix(math.pi / 2, (0, 1, 0)))
        pipe.apply_translation((cx, cy + y, height - 0.25))
        nyc_look.add_mesh(stage, f"{root}/pipe_{k}", pipe, mat)
