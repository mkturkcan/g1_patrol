"""Pedestrians as the lidar and the depth cameras see them: an Isaac Sim character in a walking pose instead of a capsule.

Each pedestrian's capsule (a kinematic rigid body: physics, contacts) gets a child mesh ``person``: one of the posed
characters baked by ``scripts/b2w/bake_people_meshes.py`` (10 characters at 8 phases of the gait cycle, about 4,000
triangles each; ``$G1_PATROL_PEOPLE_MESHES``, default ``~/.cache/g1_patrol/people_meshes``), facing the walking
direction (the capsule's yaw) with its feet on the ground, never drawn (purpose ``guide``). :func:`use_people_meshes`
makes an environment's lidar and depth cameras cast their rays at these meshes instead of the capsules; the pedestrians
keep their slot's pose for the whole run (the characters' limbs do not move).
"""

from __future__ import annotations

import glob
import os

PEDESTRIAN_CENTER = 0.86
"""Height (m) of a pedestrian capsule's center above the ground (``NycScenarioCommandCfg.pedestrian_center_height``)."""


def people_meshes_dir() -> str:
    return os.environ.get("G1_PATROL_PEOPLE_MESHES", os.path.expanduser("~/.cache/g1_patrol/people_meshes"))


def _mesh_file(index: int) -> str:
    files = sorted(glob.glob(os.path.join(people_meshes_dir(), "*.npz")))
    if not files:
        raise FileNotFoundError(f"no people meshes in {people_meshes_dir()}: run scripts/b2w/bake_people_meshes.py")
    return files[(index * 11) % len(files)]  # 11 is coprime with 80: neighbouring slots differ in character and pose


def _add_person(stage, prim_path: str):
    import numpy as np
    from pxr import Gf, UsdGeom, Vt

    data = np.load(_mesh_file(int(str(prim_path).rsplit("_", 1)[-1])))
    person = UsdGeom.Mesh.Define(stage, f"{prim_path}/person")
    person.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(data["vertices"].astype(np.float32)))
    person.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(np.full(len(data["faces"]), 3, dtype=np.int32)))
    person.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(data["faces"].reshape(-1).astype(np.int32)))
    person.CreatePurposeAttr().Set(UsdGeom.Tokens.guide)
    person.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, -PEDESTRIAN_CENTER))


def _capsule_with_person(prim_path, cfg, translation=None, orientation=None, hide_capsule: bool = False, **kwargs):
    from pxr import Usd, UsdGeom

    from isaaclab.sim.spawners.shapes.shapes import spawn_capsule

    prim = spawn_capsule.__wrapped__(prim_path, cfg, translation, orientation, **kwargs)
    if hide_capsule:
        for part in Usd.PrimRange(prim):
            if part.IsA(UsdGeom.Gprim):
                UsdGeom.Imageable(part).CreatePurposeAttr().Set(UsdGeom.Tokens.guide)
    _add_person(prim.GetStage(), str(prim.GetPath()))
    return prim


def spawn_capsule_with_person(prim_path, cfg, translation=None, orientation=None, **kwargs):
    """A pedestrian's capsule with a ``person`` mesh (USD modules are imported here, once the app runs)."""
    from isaaclab.sim.utils.prims import clone

    return clone(_capsule_with_person)(prim_path, cfg, translation, orientation, **kwargs)


def spawn_hidden_capsule_with_person(prim_path, cfg, translation=None, orientation=None, **kwargs):
    """As :func:`spawn_capsule_with_person`, the capsule not drawn (the camera environment draws characters)."""
    from isaaclab.sim.utils.prims import clone

    return clone(_capsule_with_person)(prim_path, cfg, translation, orientation, hide_capsule=True, **kwargs)


def use_people_meshes(env_cfg) -> None:
    """Give the pedestrians ``person`` meshes and make the lidar and the depth cameras see these instead of the capsules."""
    from .camera_env_cfg import spawn_hidden_pedestrian

    for ped in env_cfg.scene.pedestrians.rigid_objects.values():
        hidden = ped.spawn.func is spawn_hidden_pedestrian
        ped.spawn = ped.spawn.replace(func=spawn_hidden_capsule_with_person if hidden else spawn_capsule_with_person)
    for name in ("lidar", "front_depth", "rear_depth"):
        sensor = getattr(env_cfg.scene, name, None)
        if sensor is None:
            continue
        targets = []
        for target in sensor.mesh_prim_paths:
            expr = getattr(target, "prim_expr", None)
            if expr is not None and expr.rsplit("/", 1)[-1].startswith("Ped_"):
                target = target.replace(prim_expr=f"{expr}/person")
            targets.append(target)
        sensor.mesh_prim_paths = targets


def _people_env_cfgs():
    from isaaclab.utils import configclass

    from .nav_env_cfg import B2WLabFusedStudentEnvCfg, B2WLabLidarStudentEnvCfg

    @configclass
    class B2WLabFusedStudentPeopleEnvCfg(B2WLabFusedStudentEnvCfg):
        """The lab fused student, its lidar and depth cameras seeing the pedestrians as posed characters."""

        def __post_init__(self):
            super().__post_init__()
            use_people_meshes(self)

    @configclass
    class B2WLabLidarStudentPeopleEnvCfg(B2WLabLidarStudentEnvCfg):
        """The lab lidar-only student, its lidar seeing the pedestrians as posed characters."""

        def __post_init__(self):
            super().__post_init__()
            use_people_meshes(self)

    return B2WLabFusedStudentPeopleEnvCfg, B2WLabLidarStudentPeopleEnvCfg


B2WLabFusedStudentPeopleEnvCfg, B2WLabLidarStudentPeopleEnvCfg = _people_env_cfgs()
