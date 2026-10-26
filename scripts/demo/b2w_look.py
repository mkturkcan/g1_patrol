"""Looks for the B2-W demo (rendering only; requires a running Kit app): Unitree's B2-W on the simulated robot's pose,
and materials for the indoor and lab tiles.

The simulated B2-W is a kinematic box driven through a model of its built-in controller (``G1Patrol-B2W-*``). Here it
is drawn with the visual meshes of Unitree's description of the robot (``b2w_description`` in
github.com/unitreerobotics/unitree_ros, BSD-3-Clause; not part of this repository, its path is passed in), standing at
the episode's base height -- the legs posed by inverse kinematics so that the wheels touch the floor -- with the wheels
turning as the robot drives and turns.
"""

from __future__ import annotations

import math
import os
import xml.etree.ElementTree as ET

import numpy as np
import trimesh

import isaaclab.sim as sim_utils
from pxr import Gf, UsdGeom

import nyc_look

STAND = {"thigh": 0.67, "calf": -1.3}
"""Unitree's standing pose of the legs (rad)."""
WHEEL_RADIUS = 0.1125
PARTS = {  # mesh color in the description (rounded) -> (material, diffuse rgb, roughness, metallic)
    (0.85, 0.85, 0.85): ("b2w_shell", (0.32, 0.33, 0.35), 0.42, 0.65),
    (0.75, 0.75, 0.75): ("b2w_shell", (0.32, 0.33, 0.35), 0.42, 0.65),
    (0.65, 0.68, 0.65): ("b2w_cover", (0.16, 0.17, 0.18), 0.7, 0.1),
    (0.07, 0.07, 0.07): ("b2w_rubber", (0.035, 0.035, 0.035), 0.85, 0.0),
    (0.0, 0.0, 0.0): ("b2w_black", (0.02, 0.02, 0.022), 0.45, 0.2),
    (1.0, 1.0, 1.0): ("b2w_logo", (0.85, 0.85, 0.85), 0.4, 0.0),
}

# indoor and lab tiles (materials of g1_patrol.tasks.navigation.nyc.indoor and .makerlab)
INDOOR_TEXTURED = {
    "floor_concrete_polished": ("Base/Masonry/Concrete_Smooth.mdl", 2.0),
    "floor_concrete": ("Base/Masonry/Concrete_Smooth.mdl", 2.0),
    "floor_tiles": ("vMaterials_2/Ground/Large_Granite_Paving.mdl", 1.2),
    "wall": ("vMaterials_2/Concrete/Concrete_Wall_Even.mdl", 3.0),
    "column": ("vMaterials_2/Concrete/Concrete_Wall_Even.mdl", 3.0),
    "pallet": ("Base/Wood/Timber.mdl", 1.0),
    "pallet_stack": ("Base/Wood/Timber.mdl", 1.0),
    "bench": ("Base/Wood/Plywood.mdl", 1.2),
    "table": ("Base/Wood/Plywood.mdl", 1.2),
    "desk": ("Base/Wood/Plywood.mdl", 1.2),
    "coffee_table": ("Base/Wood/Plywood.mdl", 1.2),
    "concrete_specimen": ("Base/Masonry/Concrete_Smooth.mdl", 1.0),
}
INDOOR_COLORS = {  # material -> (diffuse rgb, roughness, metallic)
    "floor_vinyl": ((0.5, 0.5, 0.47), 0.45, 0.0),
    "floor_stone": ((0.42, 0.41, 0.39), 0.3, 0.0),
    "carpet": ((0.2, 0.22, 0.26), 0.95, 0.0),
    "floor_line": ((0.92, 0.72, 0.08), 0.5, 0.0),
    "floor_line_zone": ((0.92, 0.72, 0.08), 0.5, 0.0),
    "hazard_stripe": ((0.92, 0.72, 0.08), 0.5, 0.0),
    "cable_cover": ((0.06, 0.06, 0.06), 0.7, 0.0),
    "door": ((0.42, 0.3, 0.2), 0.6, 0.0),
    "cabinet": ((0.55, 0.57, 0.6), 0.4, 0.6),
    "shelf": ((0.35, 0.37, 0.4), 0.45, 0.7),
    "rack": ((0.12, 0.25, 0.55), 0.45, 0.6),
    "cart": ((0.45, 0.47, 0.5), 0.35, 0.8),
    "carton": ((0.62, 0.47, 0.3), 0.85, 0.0),
    "chair": ((0.08, 0.08, 0.09), 0.6, 0.2),
    "armchair": ((0.3, 0.22, 0.18), 0.85, 0.0),
    "sofa": ((0.25, 0.27, 0.3), 0.9, 0.0),
    "fume_hood": ((0.75, 0.77, 0.78), 0.3, 0.6),
    "gas_cylinder": ((0.15, 0.32, 0.18), 0.4, 0.5),
    "pedestal": ((0.8, 0.8, 0.78), 0.6, 0.0),
    "bench_seat": ((0.25, 0.27, 0.3), 0.6, 0.2),
    "bin": ((0.2, 0.22, 0.25), 0.5, 0.4),
    "fountain": ((0.7, 0.72, 0.74), 0.3, 0.8),
    "wet_floor_sign": ((0.95, 0.8, 0.1), 0.5, 0.0),
    "pallet_jack": ((0.85, 0.35, 0.05), 0.5, 0.4),
    "railing": ((0.6, 0.62, 0.64), 0.3, 0.9),
    "reception": ((0.55, 0.42, 0.3), 0.5, 0.0),
    "stanchion": ((0.7, 0.62, 0.35), 0.3, 0.9),
    "stanchion_belt": ((0.6, 0.08, 0.08), 0.6, 0.0),
    "barrel": ((0.12, 0.28, 0.55), 0.5, 0.3),
    "bucket": ((0.85, 0.85, 0.82), 0.6, 0.0),
    "machine": ((0.55, 0.6, 0.55), 0.45, 0.5),
    "scaffold": ((0.55, 0.57, 0.6), 0.35, 0.9),
    "steel_bin": ((0.5, 0.52, 0.55), 0.4, 0.8),
    "toolboxes": ((0.7, 0.1, 0.08), 0.45, 0.4),
    "tripod": ((0.08, 0.08, 0.08), 0.5, 0.5),
    # the civil engineering lab
    "floor_line_white": ((0.9, 0.9, 0.88), 0.5, 0.0),
    "floor_insert": ((0.7, 0.7, 0.68), 0.3, 0.9),
    "load_frame": ((0.62, 0.64, 0.66), 0.4, 0.5),
    "steel": ((0.45, 0.47, 0.5), 0.4, 0.8),
    "steel_polished": ((0.8, 0.8, 0.82), 0.2, 1.0),
    "specimen": ((0.6, 0.6, 0.62), 0.5, 0.6),
    "chamber": ((0.55, 0.56, 0.58), 0.45, 0.3),
    "chamber_door": ((0.48, 0.49, 0.5), 0.4, 0.4),
    "actuator": ((0.36, 0.38, 0.4), 0.5, 0.5),
    "column_wood": ((0.78, 0.68, 0.5), 0.7, 0.0),
}


def add_room_light(stage, path: str, center, intensity: float = 600.0, height: float = 7.0):
    """A ceiling's worth of light over a roofless indoor tile: an area light the size of the room, shining down from
    above the cameras (which look down into the room from up to 6 m) and invisible to them."""
    from pxr import Gf, Sdf, UsdLux

    from g1_patrol.tasks.navigation.nyc.indoor import IN_L, IN_W

    light = UsdLux.RectLight.Define(stage, path)
    light.CreateWidthAttr(2 * IN_L)
    light.CreateHeightAttr(2 * IN_W)
    light.CreateIntensityAttr(intensity)
    light.CreateColorAttr(Gf.Vec3f(1.0, 0.97, 0.92))
    light.GetPrim().CreateAttribute("visibleInPrimaryRay", Sdf.ValueTypeNames.Bool).Set(False)
    UsdGeom.Xformable(light.GetPrim()).AddTranslateOp().Set(Gf.Vec3d(center[0], center[1], height))


def register_indoor_materials():
    """Add the indoor and lab materials to the dressing's (call before ``nyc_look.dress``)."""
    nyc_look.TEXTURED.update(INDOOR_TEXTURED)
    nyc_look.COLORS.update(INDOOR_COLORS)


def _origin(el) -> np.ndarray:
    """4x4 transform of a URDF ``<origin xyz rpy>`` (fixed-axis roll, pitch, yaw)."""
    T = np.eye(4)
    if el is None:
        return T
    rpy = [float(v) for v in el.get("rpy", "0 0 0").split()]
    T = trimesh.transformations.euler_matrix(*rpy, axes="sxyz")
    T[:3, 3] = [float(v) for v in el.get("xyz", "0 0 0").split()]
    return T


def _rotation(axis, angle: float) -> np.ndarray:
    return trimesh.transformations.rotation_matrix(angle, axis)


def leg_angles(base_height: float, foot_x: float = -0.011, l1: float = 0.35, l2: float = 0.35) -> tuple[float, float]:
    """Thigh and calf angles (rad) that put the wheel under the hip (``foot_x`` behind the thigh joint, as standing)
    with the base ``base_height`` above the floor."""
    u, w = -foot_x, base_height - WHEEL_RADIUS
    c = np.clip((u * u + w * w - l1 * l1 - l2 * l2) / (2 * l1 * l2), -1.0, 1.0)
    calf = -math.acos(c)
    thigh = math.atan2(u, w) - math.atan2(l2 * math.sin(calf), l1 + l2 * math.cos(calf))
    return thigh, calf


class B2WModel:
    """Unitree's B2-W, drawn on a robot's pose: ``update(pos, quat, base_height, velocity_b, dt)`` every frame."""

    def __init__(self, stage, path: str, description: str):
        urdf = ET.parse(os.path.join(description, "urdf", "b2w_description.urdf")).getroot()
        self.joints = {}
        for j in urdf.findall("joint"):
            axis = j.find("axis")
            self.joints[j.find("child").get("link")] = {
                "name": j.get("name"), "type": j.get("type"), "parent": j.find("parent").get("link"),
                "origin": _origin(j.find("origin")),
                "axis": np.array([float(v) for v in (axis.get("xyz") if axis is not None else "0 0 0").split()]),
            }
        self.links = []
        self.path = path
        self.stage = stage
        root = UsdGeom.Xform.Define(stage, path)
        self.root_op = root.AddTransformOp()
        self.ops = {}
        loaded = []
        for link in urdf.findall("link"):
            visual = link.find("visual")
            if visual is None or visual.find("geometry/mesh") is None:
                continue
            mesh_file = visual.find("geometry/mesh").get("filename").split("/")[-1]
            scene = trimesh.load(os.path.join(description, "meshes", mesh_file), force="scene")
            groups: dict[str, list] = {}
            for m in scene.dump():
                color = getattr(getattr(m.visual, "material", None), "main_color", None)
                color = (0.85, 0.85, 0.85) if color is None else tuple(np.round(np.asarray(color[:3]) / 255.0, 2))
                key = min(PARTS, key=lambda c: sum((a - b) ** 2 for a, b in zip(c, color)))
                groups.setdefault(key, []).append(m)
            loaded.append((link.get("name"), _origin(visual.find("origin")), groups))
        for key in {k for _, _, groups in loaded for k in groups}:  # materials first, then the meshes bound to them
            material, rgb, rough, metal = PARTS[key]
            if not stage.GetPrimAtPath(nyc_look._material_path(material)).IsValid():  # (shared by the robots)
                cfg = sim_utils.PreviewSurfaceCfg(diffuse_color=rgb, roughness=rough, metallic=metal)
                cfg.func(nyc_look._material_path(material), cfg)
        for name, origin, groups in loaded:
            xf = UsdGeom.Xform.Define(stage, f"{path}/{name}")
            self.ops[name] = xf.AddTransformOp()
            for k, (key, ms) in enumerate(groups.items()):
                mesh = trimesh.util.concatenate(ms)
                mesh.apply_transform(origin)
                nyc_look.add_mesh(stage, f"{path}/{name}/part_{k}", mesh, PARTS[key][0])
            self.links.append(name)
        self.wheel = {name: 0.0 for name in self.links if name.endswith("_foot")}
        self.base_height = None

    def _link_transforms(self, base_height: float) -> dict[str, np.ndarray]:
        thigh, calf = leg_angles(base_height)
        q = {}
        for child, j in self.joints.items():
            if j["name"].endswith("_thigh_joint"):
                q[child] = thigh
            elif j["name"].endswith("_calf_joint"):
                q[child] = calf
            elif child in self.wheel:
                q[child] = self.wheel[child]
        out = {"base_link": np.eye(4)}

        def world(link):
            if link not in out:
                j = self.joints[link]
                T = world(j["parent"]) @ j["origin"]
                if j["type"] in ("revolute", "continuous") and link in q:
                    T = T @ _rotation(j["axis"], q[link])
                out[link] = T
            return out[link]

        for name in self.links:
            world(name)
        return out

    def update(self, pos, quat, base_height: float, velocity_b, dt: float):
        """Pose the model: base ``pos`` (world, m) and ``quat`` (w, x, y, z), legs for ``base_height``; the wheels
        roll with the forward speed and yaw rate in ``velocity_b`` (forward, lateral, yaw rate) over ``dt``."""
        transforms = self._link_transforms(base_height)
        for name in self.wheel:
            y = transforms[name][1, 3]
            self.wheel[name] += (float(velocity_b[0]) - float(velocity_b[2]) * y) / WHEEL_RADIUS * dt
        transforms = self._link_transforms(base_height)
        for name, T in transforms.items():
            if name in self.ops:
                self.ops[name].Set(Gf.Matrix4d(T.T.tolist()))
        w, x, y, z = (float(v) for v in quat)
        R = trimesh.transformations.quaternion_matrix([w, x, y, z])
        R[:3, 3] = [float(v) for v in pos]
        self.root_op.Set(Gf.Matrix4d(R.T.tolist()))
