"""Pedestrians as the lidar and the depth cameras see them: Isaac Sim characters in walking poses instead of capsules.

Each pedestrian's capsule (a kinematic rigid body: physics, contacts) gets a child mesh ``person``: one of the posed
characters baked by ``scripts/b2w/bake_people_meshes.py`` (10 characters at 8 phases of the gait cycle, about 4,000
triangles each; ``$G1_PATROL_PEOPLE_MESHES``, default ``~/.cache/g1_patrol/people_meshes``), facing the walking
direction (the capsule's yaw) with its feet on the ground, never drawn (purpose ``guide``). :func:`use_people_meshes`
makes an environment's lidar and depth cameras cast their rays at these meshes instead of the capsules; each pedestrian
keeps its slot's pose, unless :class:`walking_people` poses it along its gait. Slot ``i`` takes mesh ``11 (i + offset)``
(mod 80) of the sorted files, the same in every environment: with the default offset 0 the 20 slots of a scene take 20
of the 80 meshes (all 10 characters); ``$G1_PATROL_PEOPLE_OFFSET=40`` gives them 20 others (the same characters in
other poses).
"""

from __future__ import annotations

import glob
import os

from isaaclab.managers import ManagerTermBase

PEDESTRIAN_CENTER = 0.86
"""Height (m) of a pedestrian capsule's center above the ground (``NycScenarioCommandCfg.pedestrian_center_height``)."""


def people_meshes_dir() -> str:
    return os.environ.get("G1_PATROL_PEOPLE_MESHES", os.path.expanduser("~/.cache/g1_patrol/people_meshes"))


def _mesh_file(index: int) -> str:
    files = sorted(glob.glob(os.path.join(people_meshes_dir(), "*.npz")))
    if not files:
        raise FileNotFoundError(f"no people meshes in {people_meshes_dir()}: run scripts/b2w/bake_people_meshes.py")
    offset = int(os.environ.get("G1_PATROL_PEOPLE_OFFSET", "0"))
    return files[11 * (index + offset) % len(files)]  # 11 is coprime with 80: neighbours differ in character and pose


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


GAIT_CYCLE = 1.45
"""Distance (m) walked per gait cycle by the characters' walk clip (2.9 m per clip of two cycles, ``people.py``)."""


def _poses_by_character() -> dict[str, list[str]]:
    """The baked meshes by character (name order), each character's in phase order."""
    by_character: dict[str, list[str]] = {}
    for f in sorted(glob.glob(os.path.join(people_meshes_dir(), "*.npz"))):
        by_character.setdefault(os.path.basename(f).rsplit("_w", 1)[0], []).append(f)
    if not by_character:
        raise FileNotFoundError(f"no people meshes in {people_meshes_dir()}: run scripts/b2w/bake_people_meshes.py")
    return {c: sorted(fs, key=lambda f: float(f.rsplit("_w", 1)[1][: -len(".npz")])) for c, fs in by_character.items()}


def walking_characters(subset: str) -> list[str]:
    """The characters of ``subset``: ``all``, ``train`` (the first 8 in name order) or ``held_out`` (the last 2)."""
    names = list(_poses_by_character())
    return {"all": names, "train": names[:8], "held_out": names[8:]}[subset]


class walking_people(ManagerTermBase):
    """Event term (every step): every pedestrian of every environment, as the lidar and the depth cameras see it, is
    one of the baked characters in the pose of its own gait.

    The ray casters cast each environment's rays at the meshes of a table with one column per target. This term builds
    a mesh per baked pose (characters x gait phases, in the pedestrian's body frame like the spawned ``person`` mesh)
    and points each pedestrian's column at the pose of its character at its gait phase, which advances with its
    walking speed (:data:`GAIT_CYCLE`). Characters are drawn per environment and pedestrian from
    :func:`walking_characters` of ``characters``; ``seed`` fixes the draw and the initial phases.
    """

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._targets = None

    def _setup(self, env, characters: str, seed: int):
        import numpy as np
        import torch
        import warp as wp

        from isaaclab.utils.warp import convert_to_warp_mesh

        poses = _poses_by_character()
        names = walking_characters(characters)
        self.num_phases = len(poses[names[0]])
        self._meshes = []  # the warp meshes must outlive their ids in the sensors' tables
        for name in names:
            for f in poses[name]:
                data = np.load(f)
                vertices = data["vertices"].astype(np.float32) - np.array([0.0, 0.0, PEDESTRIAN_CENTER], np.float32)
                self._meshes.append(convert_to_warp_mesh(vertices, data["faces"].astype(np.int32), device=env.device))
        self.table = torch.tensor([m.id for m in self._meshes], dtype=torch.uint64).view(torch.int64).to(env.device)
        self._targets = []
        for sensor in env.scene.sensors.values():
            if not hasattr(sensor, "_mesh_ids_wp"):
                continue
            columns, offset = {}, 0
            for target in sensor._raycast_targets_cfg:
                name = target.prim_expr.rstrip("/").rsplit("/", 2)
                if name[-1] == "person" and name[-2].startswith("Ped_"):
                    columns[int(name[-2][len("Ped_") :])] = offset
                offset += sensor._num_meshes_per_env[target.prim_expr]
            if columns:
                ids = wp.to_torch(sensor._mesh_ids_wp).view(torch.int64)  # aliases the sensor's table
                self._targets.append((ids, torch.tensor([columns[k] for k in sorted(columns)], device=env.device)))
        generator = torch.Generator(device=env.device).manual_seed(seed)
        n, p = env.num_envs, len(self._targets[0][1]) if self._targets else 0
        self.character = torch.randint(len(names), (n, p), generator=generator, device=env.device)
        self.phase = torch.rand((n, p), generator=generator, device=env.device)

    def __call__(self, env, env_ids, characters: str = "all", seed: int = 0, command_name: str = "nyc"):
        import torch

        if self._targets is None:
            self._setup(env, characters, seed)
        if not self._targets:
            return
        scenario = env.command_manager.get_term(command_name)
        speed = torch.linalg.norm(scenario.ped_vel[:, : self.phase.shape[1]], dim=-1)
        self.phase = (self.phase + speed * env.step_dt / GAIT_CYCLE) % 1.0
        pose = self.character * self.num_phases + (self.phase * self.num_phases).long().clamp(max=self.num_phases - 1)
        mesh_ids = self.table[pose]
        for ids, columns in self._targets:
            ids[:, columns] = mesh_ids


def use_people_meshes(env_cfg, walking: str | None = None, seed: int = 0) -> None:
    """Give the pedestrians ``person`` meshes; the lidar and the depth cameras see these instead of the capsules.

    With ``walking`` (``all``, ``train`` or ``held_out``), they see characters of that subset whose poses follow their
    gait (:class:`walking_people`) instead of each slot's fixed pose.
    """
    from isaaclab.managers import EventTermCfg

    from .camera_env_cfg import spawn_hidden_pedestrian

    if walking is not None:
        env_cfg.events.walking_people = EventTermCfg(
            func=walking_people, mode="interval", interval_range_s=(0.0, 0.0), is_global_time=True,
            params={"characters": walking, "seed": seed})
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
