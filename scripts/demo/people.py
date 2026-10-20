"""Animated people for the New York demo (rendering only; requires a running Kit app).

Each pedestrian of the filmed environments gets an Isaac Sim character. Its skeleton is posed every rendered frame
from clips baked by ``bake_people_animations.py``: an in-place walk cycle whose phase advances with the pedestrian's
speed along its facing direction (so feet do not slide), blended into an idle loop as the pedestrian slows down and
stops. A character faces along its path when it enters the scene and turns towards its direction of motion; when that
direction is behind it, it turns on the spot instead of stepping backwards. Characters stand on the tile's ground
(and on the stairs of subway entrances). The kinematic capsules that the robot and its sensors interact with are
hidden; their motion is what the characters follow.
"""

from __future__ import annotations

import math
import os

import numpy as np
import torch
from pxr import Gf, Usd, UsdGeom, UsdSkel, Vt

PEOPLE = "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/6.1/Isaac/People"
CHARACTERS = {
    "F_Business_02": "Characters/F_Business_02/F_Business_02.usd",
    "F_Medical_01": "Characters/F_Medical_01/F_Medical_01.usd",
    "M_Medical_01": "Characters/M_Medical_01/M_Medical_01.usd",
    "female_police_01": "Characters/female_adult_police_01_new/female_adult_police_01_new.usd",
    "female_police_02": "Characters/female_adult_police_02/female_adult_police_02.usd",
    "female_police_03": "Characters/female_adult_police_03_new/female_adult_police_03_new.usd",
    "male_construction_01": "Characters/male_adult_construction_01_new/male_adult_construction_01_new.usd",
    "male_construction_03": "Characters/male_adult_construction_03/male_adult_construction_03.usd",
    "male_construction_05": "Characters/male_adult_construction_05_new/male_adult_construction_05_new.usd",
    "male_police_04": "Characters/male_adult_police_04/male_adult_police_04.usd",
}
# how often each character appears (a street crowd, not a police parade)
WEIGHTS = {"F_Business_02": 4, "F_Medical_01": 2, "M_Medical_01": 2, "male_construction_01": 1,
           "male_construction_03": 1, "male_construction_05": 1, "female_police_01": 0.3, "female_police_02": 0.3,
           "female_police_03": 0.3, "male_police_04": 0.3}  # fmt: skip
# clothing tints (multiplied with the clothes' textures): city colors, mostly dark and muted
TOPS = [(0.05, 0.05, 0.06), (0.12, 0.13, 0.2), (0.25, 0.25, 0.27), (0.55, 0.42, 0.28), (0.35, 0.08, 0.1),
        (0.2, 0.25, 0.15), (0.85, 0.85, 0.85), (0.45, 0.55, 0.7), (0.7, 0.2, 0.15), (0.9, 0.75, 0.3),
        (0.15, 0.3, 0.45), (0.6, 0.6, 0.62)]  # fmt: skip
BOTTOMS = [(0.05, 0.1, 0.22), (0.1, 0.17, 0.32), (0.04, 0.04, 0.05), (0.3, 0.3, 0.32), (0.45, 0.38, 0.27),
           (0.2, 0.2, 0.25)]  # fmt: skip
SHOES = [(0.9, 0.9, 0.9), (0.08, 0.08, 0.08), (0.35, 0.22, 0.12), (0.6, 0.6, 0.6)]
WALK_SPEED = 1.08
"""Walking speed (m/s) of the walk clip: its root-motion version covers 2.9 m per 80-frame cycle."""
FPS = 30.0
YAW_OFFSET = math.pi / 2
"""The characters face -y in their own frame."""
MIRROR_YAW = math.pi
"""The ``_mirror`` walk and idle clips are mirrored front to back: posed like the others, a character faces the opposite
way (measured with ``check_character_facing.py``), so it is turned by this angle to face its direction of motion."""


def _load_clip(path: str) -> dict:
    stage = Usd.Stage.Open(path)
    anim = UsdSkel.Animation(stage.GetDefaultPrim())
    times = anim.GetRotationsAttr().GetTimeSamples()
    rot = np.stack([np.array(anim.GetRotationsAttr().Get(t), dtype=np.float32) for t in times])  # (F, J, 4) xyzw
    trans = np.stack([np.array(anim.GetTranslationsAttr().Get(t), dtype=np.float32) for t in times])
    return {
        "rot": rot[:-1],  # the last frame repeats the first: loop over F - 1 frames
        "trans": trans[:-1],
        "scales": anim.GetScalesAttr().Get(times[0]),
        "joints": anim.GetJointsAttr().Get(),
    }


def _sample(clip: dict, phase: float) -> tuple[np.ndarray, np.ndarray]:
    """Pose at a loop phase in [0, 1): linear interpolation of translations, normalized lerp of rotations."""
    n = len(clip["rot"])
    f = (phase % 1.0) * n
    a, w = int(f) % n, f - int(f)
    b = (a + 1) % n
    return _nlerp(clip["rot"][a], clip["rot"][b], w), (1 - w) * clip["trans"][a] + w * clip["trans"][b]


def _nlerp(qa: np.ndarray, qb: np.ndarray, w: float) -> np.ndarray:
    sign = np.where((qa * qb).sum(-1, keepdims=True) < 0, -1.0, 1.0)
    q = (1 - w) * qa + w * sign * qb
    return q / np.linalg.norm(q, axis=-1, keepdims=True)


class AnimatedCrowd:
    """Characters standing in for the pedestrians of ``env_ids`` in the scenario command ``scenario``."""

    def __init__(self, scenario, env_ids: list[int], clips_dir: str, root: str = "/World/Demo/People", seed: int = 0,
                 stairs: list | None = None, slots: list[int] | None = None, offsets: dict | None = None):
        import omni.usd

        self.stage = stage = omni.usd.get_context().get_stage()
        self.scenario = scenario
        self.env_ids = torch.as_tensor(env_ids, device=scenario.device)
        # subway stairs per filmed environment: (x0, x1, y0, y1) tile-local, rising from x0 to street level at x1
        self.stairs = stairs or [[] for _ in env_ids]
        rng = np.random.default_rng(seed)
        names = list(CHARACTERS)
        weights = np.array([WEIGHTS[n] for n in names], dtype=np.float64)
        self.clips = {
            name: {
                kind: [_load_clip(os.path.join(clips_dir, f"{name}.stand_{kind}{suffix}.skelanim.usd")) for suffix in ("", "_mirror")]
                for kind in ("walk_loop_in_place", "idle_loop")
            }
            for name in names
        }
        p = scenario.num_peds
        self.slots = list(range(p)) if slots is None else list(slots)  # pedestrian slots shown as people
        self.offsets = offsets or {}  # slot -> distance (m) of the character ahead of the slot's position (negative: behind)
        # per filmed environment and person, after each update: shown, chest position (world), moving backwards
        self.visible = np.zeros((len(env_ids), p), dtype=bool)
        self.points = np.zeros((len(env_ids), p, 3))
        self.backward = np.zeros((len(env_ids), p), dtype=bool)
        self.people = []
        for i, e in enumerate(env_ids):
            for k in self.slots:
                name = names[rng.choice(len(names), p=weights / weights.sum())]
                path = f"{root}/env_{e}/person_{k}"
                prim = stage.DefinePrim(path, "Xform")
                prim.GetReferences().AddReference(f"{PEOPLE}/{CHARACTERS[name]}")
                xf = UsdGeom.Xformable(prim)
                xf.ClearXformOpOrder()
                translate, rotate = xf.AddTranslateOp(), xf.AddRotateZOp()
                anim = UsdSkel.Animation.Define(stage, f"{root}/env_{e}/anim_{k}")
                mirror = int(rng.integers(2))  # the same version of both clips, so that walk and idle poses blend
                walk = self.clips[name]["walk_loop_in_place"][mirror]
                idle = self.clips[name]["idle_loop"][mirror]
                anim.CreateJointsAttr().Set(walk["joints"])
                anim.CreateScalesAttr().Set(walk["scales"])
                rot_attr, trans_attr = anim.CreateRotationsAttr(), anim.CreateTranslationsAttr()
                skel = next(q for q in Usd.PrimRange(prim) if q.IsA(UsdSkel.Skeleton))
                UsdSkel.BindingAPI.Apply(skel).CreateAnimationSourceRel().SetTargets([anim.GetPath()])
                self._dress_up(prim, rng)
                self.people.append({
                    "env": i, "slot": k,
                    "prim": UsdGeom.Imageable(prim), "translate": translate, "rotate": rotate,
                    "rot": rot_attr, "trans": trans_attr, "walk": walk, "idle": idle,
                    "walk_phase": float(rng.uniform()), "idle_phase": float(rng.uniform()),
                    "stride": float(rng.uniform(0.92, 1.08)),  # people's step length varies a little
                    "yaw": None, "blend": 0.0, "visible": None, "facing": MIRROR_YAW if mirror else 0.0,
                })  # fmt: skip
        # hide the capsules these characters replace
        for e in env_ids:
            for k in self.slots:
                capsule = stage.GetPrimAtPath(f"/World/envs/env_{e}/Ped_{k}")
                if capsule:
                    UsdGeom.Imageable(capsule).MakeInvisible()

    @staticmethod
    def _dress_up(prim, rng: np.random.Generator):
        """Give this person their own clothing colors (per-instance overrides of the clothes' tints)."""
        from pxr import Sdf, UsdShade

        for q in Usd.PrimRange(prim):
            if not q.IsA(UsdShade.Shader):
                continue
            name = q.GetPath().pathString.lower()
            if "fabric" in name and "safety" not in name:
                palette = BOTTOMS if any(k in name for k in ("jeans", "pants", "trousers", "slacks")) else TOPS
            elif "leather" in name and "boots" not in name:
                palette = SHOES
            else:
                continue
            color = np.array(palette[int(rng.integers(len(palette)))]) * rng.uniform(0.85, 1.15)
            UsdShade.Shader(q).CreateInput("diffuse_tint", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*np.clip(color, 0, 1)))

    def _ground(self, ids) -> np.ndarray:
        """Ground height under each pedestrian: the tile's (repeated beyond its ends), and the stairs of subway entrances
        instead of the floor of their stairwell."""
        s = self.scenario
        local = s.tile_local(s.ped_pos[ids])
        ground = s.layout.ground_height(s.tile_row[ids], s.tile_col[ids], local).cpu().numpy()
        local = local.cpu().numpy()
        for e, stairs in enumerate(self.stairs):
            for x0, x1, y0, y1 in stairs:
                lay_top = s.layout.ground_height(s.tile_row[ids][e : e + 1], s.tile_col[ids][e : e + 1],
                                                 torch.tensor([[[x1 + 0.3, (y0 + y1) / 2]]], device=s.device))
                top = float(lay_top.flatten()[0])
                x, y = local[e, :, 0], local[e, :, 1]
                inside = (x >= x0) & (x <= x1) & (y >= y0) & (y <= y1)
                ramp = top - 1.0 * (1.0 - np.clip((x - x0) / (x1 - x0), 0.0, 1.0))
                ground[e] = np.where(inside, ramp, ground[e])
        return ground

    def yaw(self, env_index: int, slot: int) -> float | None:
        """Facing (rad) of the character of ``slot`` in the ``env_index``-th filmed environment (None: not shown)."""
        for person in self.people:
            if person["env"] == env_index and person["slot"] == slot:
                return person["yaw"]
        return None

    def update(self, dt: float):
        """Move and pose the characters to the pedestrians' current state (call before each rendered frame)."""
        s, ids = self.scenario, self.env_ids
        pos = (s.ped_pos[ids] + s.tile_center[ids][:, None]).cpu().numpy()
        ground = self._ground(ids)
        vel = s.ped_vel[ids].cpu().numpy()
        path_dir = s.walking_direction()[ids].cpu().numpy()
        visible = (s.ped_mode[ids] != 0).cpu().numpy()  # PARKED = 0
        shown = np.zeros_like(visible)
        shown[:, self.slots] = True
        visible &= shown  # the other slots (wheelchairs, dogs) are drawn elsewhere
        self.visible[:] = visible
        self.points[..., :2], self.points[..., 2] = pos[..., :2], ground + 1.2
        self.backward[:] = False
        for person in self.people:
            e, k = person["env"], person["slot"]
            if person["visible"] != bool(visible[e, k]):
                person["visible"] = bool(visible[e, k])
                person["prim"].MakeVisible() if visible[e, k] else person["prim"].MakeInvisible()
            if not visible[e, k]:
                person["yaw"] = None
                continue
            v = vel[e, k]
            speed = float(np.hypot(*v))
            if person["yaw"] is None:  # entering the scene (or standing at a spot): face along the path
                person["yaw"] = math.atan2(path_dir[e, k, 1], path_dir[e, k, 0])
            # turn towards the direction of motion while walking; when the motion is behind the person, pivot on the
            # spot (also at low speed) instead of stepping backwards
            along = float(v[0] * math.cos(person["yaw"]) + v[1] * math.sin(person["yaw"]))
            if speed > 0.2 or (speed > 0.03 and along < 0.0):
                diff = (math.atan2(v[1], v[0]) - person["yaw"] + math.pi) % (2 * math.pi) - math.pi
                rate = (4.0 + 8.0 * min(1.0, speed)) * dt * (2.5 if abs(diff) > math.pi / 2 else 1.0)
                person["yaw"] += max(-rate, min(rate, diff))
            # the walk cycle follows the speed along the facing direction and never runs backwards
            along = float(v[0] * math.cos(person["yaw"]) + v[1] * math.sin(person["yaw"]))
            self.backward[e, k] = along < -0.05
            along = max(0.0, along)
            person["walk_phase"] += along / (WALK_SPEED * person["stride"]) * dt * FPS / len(person["walk"]["rot"])
            person["idle_phase"] += dt * FPS / len(person["idle"]["rot"])
            target_blend = min(1.0, along / 0.5)
            person["blend"] += (target_blend - person["blend"]) * min(1.0, 6.0 * dt)
            rot_w, trans_w = _sample(person["walk"], person["walk_phase"])
            rot_i, trans_i = _sample(person["idle"], person["idle_phase"])
            w = person["blend"]
            rot = _nlerp(rot_i, rot_w, w) if w < 0.999 else rot_w
            trans = (1 - w) * trans_i + w * trans_w
            person["rot"].Set(Vt.QuatfArray.FromNumpy(rot.astype(np.float32)))
            person["trans"].Set(Vt.Vec3fArray.FromNumpy(trans.astype(np.float32)))
            shift = self.offsets.get(k, 0.0)
            x = float(pos[e, k, 0]) + shift * math.cos(person["yaw"])
            y = float(pos[e, k, 1]) + shift * math.sin(person["yaw"])
            person["translate"].Set(Gf.Vec3d(x, y, float(ground[e, k])))
            person["rotate"].Set(math.degrees(person["yaw"] + YAW_OFFSET + person["facing"]))
