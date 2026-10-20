"""Looks of the civic tasks for the New York demo (rendering only; requires a running Kit app): pedestrian signal heads
that show the simulated phase, a person lying on the ground, dogs on leashes, wheelchair users, strollers, and captions
saying what the robot is doing."""

from __future__ import annotations

import math
import os

import numpy as np
from pxr import Gf, Usd, UsdGeom, UsdShade, UsdSkel, Vt

from people import CHARACTERS, PEOPLE, _load_clip, _sample

LOOKS = "/World/Demo/Looks"


def _bind(prim, material: str):
    UsdShade.MaterialBindingAPI.Apply(prim).Bind(UsdShade.Material.Get(prim.GetStage(), f"{LOOKS}/{material}"))


class SignalHeads:
    """A pedestrian signal head on each signal pole beside a filmed crosswalk, facing the far curb; its lens shows the
    scenario's phase: WALK (white), flashing DON'T WALK, DON'T WALK (orange)."""

    def __init__(self, stage, scenario, tiles: list, generator, centers: list, root: str = "/World/Demo/Signals"):
        self.scenario, self.heads = scenario, []  # (env, lens prim)
        for e, (tile, center) in enumerate(zip(tiles, centers)):
            lay = generator.layouts[tile]
            if not lay.crossings:
                continue
            for k, ((x, y, r, z0, z1), mat) in enumerate(zip(lay.circles, lay.circle_mats)):
                if mat != "signal_pole":
                    continue
                facing = -1.0 if y > 0 else 1.0  # towards the far curb
                base = lay.ground(x, y)
                box = UsdGeom.Cube.Define(stage, f"{root}/env_{e}/head_{k}")
                box.CreateSizeAttr(1.0)
                xf = UsdGeom.Xformable(box)
                xf.AddTranslateOp().Set(Gf.Vec3d(center[0] + x, center[1] + y + facing * (r + 0.15), base + 2.7))
                xf.AddScaleOp().Set(Gf.Vec3f(0.34, 0.24, 0.42))
                _bind(box.GetPrim(), "signal_pole")
                lens = UsdGeom.Mesh.Define(stage, f"{root}/env_{e}/lens_{k}")
                ly = center[1] + y + facing * (r + 0.27 + 0.002)
                pts = [(center[0] + x - 0.13, ly, base + 2.53), (center[0] + x + 0.13, ly, base + 2.53),
                       (center[0] + x + 0.13, ly, base + 2.87), (center[0] + x - 0.13, ly, base + 2.87)]
                lens.CreatePointsAttr(Vt.Vec3fArray([Gf.Vec3f(*p) for p in pts]))
                lens.CreateFaceVertexCountsAttr([4])
                lens.CreateFaceVertexIndicesAttr([0, 1, 2, 3] if facing > 0 else [0, 3, 2, 1])
                lens.CreateDoubleSidedAttr(True)
                _bind(lens.GetPrim(), "signal_off")
                self.heads.append((e, lens.GetPrim()))
        self.shown = {}

    def update(self, t: float):
        if not self.heads:
            return
        phase = self.scenario.signal_phase.cpu().numpy()
        valid = self.scenario.signal_valid.cpu().numpy()
        for e, lens in self.heads:
            if not valid[e]:
                state = "signal_off"
            elif phase[e] == 0:
                state = "signal_walk"
            elif phase[e] == 1:
                state = "signal_dont" if (t % 1.0) < 0.5 else "signal_off"  # flashing
            else:
                state = "signal_dont"
            if self.shown.get(lens) != state:
                _bind(lens, state)
                self.shown[lens] = state


class FallenPerson:
    """A character lying on its back where the scenario's fallen person is (the capsule it replaces is hidden)."""

    def __init__(self, stage, scenario, env_ids: list[int], clips_dir: str, root: str = "/World/Demo/Fallen"):
        self.scenario, self.env_ids, self.people = scenario, env_ids, []
        clip = _load_clip(os.path.join(clips_dir, "F_Business_02.stand_idle_loop.skelanim.usd"))
        rot, trans = _sample(clip, 0.0)
        for e in env_ids:
            prim = stage.DefinePrim(f"{root}/env_{e}", "Xform")
            prim.GetReferences().AddReference(f"{PEOPLE}/{CHARACTERS['F_Business_02']}")
            xf = UsdGeom.Xformable(prim)
            xf.ClearXformOpOrder()
            ops = (xf.AddTranslateOp(), xf.AddRotateZOp(), xf.AddRotateXOp())
            anim = UsdSkel.Animation.Define(stage, f"{root}/anim_{e}")
            anim.CreateJointsAttr().Set(clip["joints"])
            anim.CreateScalesAttr().Set(clip["scales"])
            anim.CreateRotationsAttr().Set(Vt.QuatfArray.FromNumpy(rot.astype(np.float32)))
            anim.CreateTranslationsAttr().Set(Vt.Vec3fArray.FromNumpy(trans.astype(np.float32)))
            skel = next(q for q in Usd.PrimRange(prim) if q.IsA(UsdSkel.Skeleton))
            UsdSkel.BindingAPI.Apply(skel).CreateAnimationSourceRel().SetTargets([anim.GetPath()])
            capsule = stage.GetPrimAtPath(f"/World/envs/env_{e}/Fallen")
            if capsule:
                UsdGeom.Imageable(capsule).MakeInvisible()
            self.people.append((UsdGeom.Imageable(prim), ops))

    def update(self):
        s = self.scenario
        on = s.has_fallen.cpu().numpy()
        pos = (s.fallen_pos + s.tile_center).cpu().numpy()
        yaw = s.fallen_yaw.cpu().numpy()
        ground = s.layout.ground_height(s.tile_row, s.tile_col, s.fallen_pos[:, None])[:, 0].cpu().numpy()
        for e, (img, (translate, rotate_z, rotate_x)) in zip(self.env_ids, self.people):
            if not on[e]:
                img.MakeInvisible()
                continue
            img.MakeVisible()
            # the character stands on its origin facing -y; lying on its back, its head points along the body's yaw
            head = np.array([math.cos(yaw[e]), math.sin(yaw[e])])
            feet = pos[e] - 0.85 * head
            translate.Set(Gf.Vec3d(float(feet[0]), float(feet[1]), float(ground[e]) + 0.14))
            rotate_z.Set(math.degrees(float(yaw[e])) - 90.0)
            rotate_x.Set(-90.0)


def caption(scenario, hands, e: int) -> str | None:
    """What the robot ``e`` is doing, for the overlay (None: nothing particular)."""
    from g1_patrol.tasks.navigation.mdp.nyc_manipulation import (
        ALERT, APPROACH, CROUCH, GRASP, LIFT, LOWER, RAISE, REACH, RELEASE, STAND,
    )  # fmt: skip

    s = scenario
    phase = int(hands.phase[e])
    holding = int(s.held_item[e])
    item = int(hands.item[e])
    bag = bool(s.is_bag[max(item, 0)]) if s.num_trash else False
    held_bag = holding >= 0 and bool(s.is_bag[holding])
    if phase == ALERT:
        return "Calling for help"
    if bool(s.under_attack[e]):
        return "Attacked: braced, getting away" if bool(hands.bracing[e]) else "Attacked: getting away"
    if phase in (APPROACH, CROUCH, REACH, GRASP, LIFT, STAND) and holding < 0 and item >= 0:
        return "Picking up a trash bag" if bag else "Picking up litter"
    if phase in (LIFT, STAND) and holding >= 0:
        return "Picking up a trash bag" if held_bag else "Picking up litter"
    if phase == APPROACH and holding >= 0:
        return "Setting the bag down at the pile" if held_bag else "Litter into the basket"
    if phase in (RAISE, RELEASE, LOWER):
        return "Setting the bag down at the pile" if bool(hands.drop_bag[e]) else "Litter into the basket"
    if holding >= 0:
        return "Carrying a bag to the curbside pile" if held_bag else "Carrying litter to a basket"
    if bool(s.has_fallen[e]) and not bool(s.assisted[e]):
        return "Someone has fallen: going to help"
    if int(s.incident[e]) == 2 and int(s.incident_phase[e]) == 1:
        return "Someone is blocking the way"
    if bool(s.signal_valid[e]) and int(s.signal_phase[e]) != 0 and bool(s.forbidden_valid[e, -1]):
        speed = float(np.hypot(*s.robot.data.root_lin_vel_w.torch[e, :2].tolist()))
        if speed < 0.15 and float(s.street_clearance()[e]) < 1.0:
            return "Waiting for WALK"
    return None


ACTORS = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "src", "g1_patrol",
                      "assets", "data", "actors")
DOG_CLIPS = {"walk": ("Wolf_Walk_cycle_", 1.5), "run": ("Wolf_Run_Cycle_", 1.9), "idle": ("Wolf_Idle_", None)}
"""Dog clips: animation prim and the distance (m) covered per cycle (the clips are in place)."""


def _skel_clip(stage, path) -> dict:
    """Joint rotations and translations per frame of a converted ``SkelAnimation`` (the last frame repeats the first)."""
    anim = UsdSkel.Animation(stage.GetPrimAtPath(path))
    times = anim.GetRotationsAttr().GetTimeSamples()
    rot = np.stack([np.array(anim.GetRotationsAttr().Get(t), dtype=np.float32) for t in times])
    trans = np.stack([np.array(anim.GetTranslationsAttr().Get(t), dtype=np.float32) for t in times])
    return {"rot": rot[:-1], "trans": trans[:-1], "scales": anim.GetScalesAttr().Get(times[0]), "joints": anim.GetJointsAttr().Get()}


class Dogs:
    """The scenario's dogs (pedestrian slots of the dog role) as an animated dog (walk, run and idle cycles timed to the
    dog's speed) on a leash held by its owner; the capsules they replace are hidden."""

    def __init__(self, stage, scenario, env_ids: list[int], root: str = "/World/Demo/Dogs"):
        self.scenario, self.env_ids = scenario, env_ids
        path = os.path.join(ACTORS, "dog", "dog.usd")
        src = Usd.Stage.Open(path)
        anims = {q.GetName(): q.GetPath() for q in src.Traverse() if q.IsA(UsdSkel.Animation)}
        self.clips = {k: (_skel_clip(src, anims[name]), stride) for k, (name, stride) in DOG_CLIPS.items()}
        self.slots = [k for k in range(scenario.num_peds) if int(scenario.slot_kind[k]) == 2]
        self.dogs = []
        for i, e in enumerate(env_ids):
            for k in self.slots:
                prim = stage.DefinePrim(f"{root}/env_{e}/dog_{k}", "Xform")  # our pose; the asset (with its own) below
                xf = UsdGeom.Xformable(prim)
                translate, rotate = xf.AddTranslateOp(), xf.AddRotateZOp()
                stage.DefinePrim(f"{root}/env_{e}/dog_{k}/asset", "Xform").GetReferences().AddReference(path)
                for q in Usd.PrimRange(prim):
                    if q.IsA(UsdGeom.Mesh) and q.GetName() == "Wolf_obj_body":  # fur cards (blended alpha renders opaque)
                        UsdGeom.Imageable(q).MakeInvisible()
                anim = UsdSkel.Animation.Define(stage, f"{root}/env_{e}/anim_{k}")
                walk = self.clips["walk"][0]
                anim.CreateJointsAttr().Set(walk["joints"])
                anim.CreateScalesAttr().Set(walk["scales"])
                skel = next(q for q in Usd.PrimRange(prim) if q.IsA(UsdSkel.Skeleton))
                UsdSkel.BindingAPI.Apply(skel).CreateAnimationSourceRel().SetTargets([anim.GetPath()])
                leash = UsdGeom.BasisCurves.Define(stage, f"{root}/env_{e}/leash_{k}")
                leash.CreateTypeAttr(UsdGeom.Tokens.linear)
                leash.CreateCurveVertexCountsAttr([8])
                leash.CreateWidthsAttr(Vt.FloatArray([0.012] * 8))
                leash.SetWidthsInterpolation(UsdGeom.Tokens.vertex)
                _bind(leash.GetPrim(), "tree_guard")  # dark, slightly shiny
                capsule = stage.GetPrimAtPath(f"/World/envs/env_{e}/Ped_{k}")
                if capsule:
                    UsdGeom.Imageable(capsule).MakeInvisible()
                self.dogs.append({"env": i, "slot": k, "img": UsdGeom.Imageable(prim), "leash": leash, "translate": translate,
                                  "rotate": rotate, "rot": anim.CreateRotationsAttr(), "trans": anim.CreateTranslationsAttr(),
                                  "yaw": None, "phase": {"walk": 0.0, "run": 0.0, "idle": float(np.random.uniform())},
                                  "visible": None})

    def update(self, dt: float):
        import torch

        s = self.scenario
        ids = torch.as_tensor(self.env_ids, device=s.device)
        pos = (s.ped_pos[ids] + s.tile_center[ids][:, None]).cpu().numpy()
        vel = s.ped_vel[ids].cpu().numpy()
        shown = ((s.ped_mode[ids] != 0) & (s.ped_role[ids] == 4)).cpu().numpy()
        ground = s.layout.ground_height(s.tile_row[ids], s.tile_col[ids], s.tile_local(s.ped_pos[ids])).cpu().numpy()
        owner = s.ped_link[ids].clamp(min=0).cpu().numpy()
        for dog in self.dogs:
            e, k = dog["env"], dog["slot"]
            visible = bool(shown[e, k])
            if dog["visible"] != visible:
                dog["visible"] = visible
                for img in (dog["img"], UsdGeom.Imageable(dog["leash"].GetPrim())):
                    img.MakeVisible() if visible else img.MakeInvisible()
            if not visible:
                dog["yaw"] = None
                continue
            v = vel[e, k]
            speed = float(np.hypot(*v))
            if dog["yaw"] is None:
                dog["yaw"] = math.atan2(v[1], v[0]) if speed > 0.05 else 0.0
            if speed > 0.15:
                diff = (math.atan2(v[1], v[0]) - dog["yaw"] + math.pi) % (2 * math.pi) - math.pi
                rate = 6.0 * dt * (2.0 if abs(diff) > math.pi / 2 else 1.0)
                dog["yaw"] += max(-rate, min(rate, diff))
            # the clip and its phase follow the speed along the dog's heading
            along = max(0.0, float(v[0] * math.cos(dog["yaw"]) + v[1] * math.sin(dog["yaw"])))
            kind = "idle" if along < 0.12 else ("run" if along > 1.8 else "walk")
            clip, stride = self.clips[kind]
            n = len(clip["rot"])
            if stride is None:
                dog["phase"][kind] += dt * 24.0 / n
            else:
                dog["phase"][kind] += along / stride * dt
            rot, trans = _sample(clip, dog["phase"][kind])
            dog["rot"].Set(Vt.QuatfArray.FromNumpy(rot.astype(np.float32)))
            dog["trans"].Set(Vt.Vec3fArray.FromNumpy(trans.astype(np.float32)))
            x, y, z = float(pos[e, k, 0]), float(pos[e, k, 1]), float(ground[e, k])
            dog["translate"].Set(Gf.Vec3d(x, y, z))
            dog["rotate"].Set(math.degrees(dog["yaw"]) + 90.0)  # the model faces -y
            # leash: from the owner's right hand to the dog's collar, sagging when slack
            o = owner[e, k]
            ox, oy = float(pos[e, o, 0]), float(pos[e, o, 1])
            heading = math.atan2(vel[e, o, 1], vel[e, o, 0]) if float(np.hypot(*vel[e, o])) > 0.1 else dog["yaw"]
            hand = np.array([ox + 0.22 * math.sin(heading), oy - 0.22 * math.cos(heading), float(ground[e, o]) + 0.9])
            collar = np.array([x + 0.32 * math.cos(dog["yaw"]), y + 0.32 * math.sin(dog["yaw"]), z + 0.48])
            span = float(np.linalg.norm(hand[:2] - collar[:2]))
            sag = max(0.02, 0.25 * (1.0 - span / 2.2))
            t = np.linspace(0.0, 1.0, 8)
            pts = hand[None] + t[:, None] * (collar - hand)[None]
            pts[:, 2] -= sag * 4.0 * t * (1.0 - t)
            dog["leash"].GetPointsAttr().Set(Vt.Vec3fArray([Gf.Vec3f(*map(float, p)) for p in pts]))


def wide_slots(scenario) -> tuple[list[int], list[int]]:
    """The wide-body pedestrian slots shown as wheelchair users and as parents pushing a stroller (alternately)."""
    wide = [k for k in range(scenario.num_peds) if int(scenario.slot_kind[k]) == 1]
    return wide[0::2], wide[1::2]


SEATED = (("Thigh", 90.0), ("Calf", -70.0), ("Upperarm", 20.0), ("Forearm", 75.0))
"""Seated pose: flexion (deg) of both legs and arms about each joint's local x axis, added to the idle clip's first
frame (thighs level, feet on the footrests, hands in the lap)."""
SEAT_HIP = (0.10, 0.58)
"""Hip joints of a seated person in the wheelchair's frame: behind its center and above the ground (m)."""
WHEELCHAIR_PEOPLE = {"F_Business_02": 3, "F_Medical_01": 1, "M_Medical_01": 1, "male_construction_03": 1}
"""Characters in the wheelchairs and how often each appears."""


def _seated(clips_dir: str, name: str):
    """A character's seated pose (joint rotations and translations) and its hip joints' position in the character's
    frame (forward kinematics; the flexion leaves them in place)."""
    clip = _load_clip(os.path.join(clips_dir, f"{name}.stand_idle_loop.skelanim.usd"))
    rot, trans = _sample(clip, 0.0)
    paths = [str(j) for j in clip["joints"]]
    names = [j.split("/")[-1] for j in paths]
    for joint, angle in SEATED:
        flex = Gf.Quatf(Gf.Rotation(Gf.Vec3d(1, 0, 0), angle).GetQuat())
        for side in "LR":
            i = names.index(f"{side}_{joint}")
            q = Gf.Quatf(float(rot[i, 3]), Gf.Vec3f(*map(float, rot[i, :3]))) * flex  # the clips store xyzw
            rot[i] = (*q.GetImaginary(), q.GetReal())
    world = {}
    for j, path in enumerate(paths):
        local = (Gf.Matrix4d().SetScale(Gf.Vec3d(*map(float, clip["scales"][j])))
                 * Gf.Matrix4d().SetRotate(Gf.Quatd(float(rot[j, 3]), Gf.Vec3d(*map(float, rot[j, :3]))))
                 * Gf.Matrix4d().SetTranslateOnly(Gf.Vec3d(*map(float, trans[j]))))
        parent = path.rsplit("/", 1)[0]
        world[path] = local * world[parent] if parent in world else local
    hips = [np.array(world[p].ExtractTranslation()) for p, n in zip(paths, names) if n in ("L_Thigh", "R_Thigh")]
    return clip, rot, trans, np.mean(hips, axis=0)


class Wheelchairs:
    """A wheelchair user for each wheelchair slot (``wide_slots``): a seated character in a wheelchair that turns
    towards its direction of motion (on the spot when that is behind it); the capsules they replace are hidden."""

    def __init__(self, stage, scenario, env_ids: list[int], clips_dir: str, root: str = "/World/Demo/Wheelchairs",
                 seed: int = 1):
        from people import AnimatedCrowd

        self.scenario, self.env_ids = scenario, env_ids
        self.slots = wide_slots(scenario)[0]
        self.visible = np.zeros((len(env_ids), scenario.num_peds), dtype=bool)  # after each update
        rng = np.random.default_rng(seed)
        names = list(WHEELCHAIR_PEOPLE)
        weights = np.array([WHEELCHAIR_PEOPLE[n] for n in names], dtype=np.float64)
        chair = os.path.join(ACTORS, "wheelchair", "wheelchair.usd")
        poses, self.items = {}, []
        for i, e in enumerate(env_ids):
            for k in self.slots:
                name = names[rng.choice(len(names), p=weights / weights.sum())]
                if name not in poses:
                    poses[name] = _seated(clips_dir, name)
                clip, rot, trans, hip = poses[name]
                path = f"{root}/env_{e}/wheelchair_{k}"
                prim = stage.DefinePrim(path, "Xform")  # our pose; the wheelchair (with its own) and its user below
                xf = UsdGeom.Xformable(prim)
                translate, rotate = xf.AddTranslateOp(), xf.AddRotateZOp()
                UsdGeom.Xformable(stage.DefinePrim(f"{path}/chair", "Xform")).AddRotateXOp().Set(90.0)  # a Y-up asset
                stage.DefinePrim(f"{path}/chair/asset", "Xform").GetReferences().AddReference(chair)
                person = stage.DefinePrim(f"{path}/person", "Xform")
                UsdGeom.Xformable(person).AddTranslateOp().Set(Gf.Vec3d(-hip[0], SEAT_HIP[0] - hip[1], SEAT_HIP[1] - hip[2]))
                character = stage.DefinePrim(f"{path}/person/character", "Xform")
                character.GetReferences().AddReference(f"{PEOPLE}/{CHARACTERS[name]}")
                anim = UsdSkel.Animation.Define(stage, f"{root}/env_{e}/anim_{k}")
                anim.CreateJointsAttr().Set(clip["joints"])
                anim.CreateScalesAttr().Set(clip["scales"])
                anim.CreateRotationsAttr().Set(Vt.QuatfArray.FromNumpy(rot.astype(np.float32)))
                anim.CreateTranslationsAttr().Set(Vt.Vec3fArray.FromNumpy(trans.astype(np.float32)))
                skel = next(q for q in Usd.PrimRange(character) if q.IsA(UsdSkel.Skeleton))
                UsdSkel.BindingAPI.Apply(skel).CreateAnimationSourceRel().SetTargets([anim.GetPath()])
                AnimatedCrowd._dress_up(character, rng)
                capsule = stage.GetPrimAtPath(f"/World/envs/env_{e}/Ped_{k}")
                if capsule:
                    UsdGeom.Imageable(capsule).MakeInvisible()
                self.items.append({"env": i, "slot": k, "img": UsdGeom.Imageable(prim), "translate": translate,
                                   "rotate": rotate, "yaw": None, "visible": None})

    def update(self, dt: float):
        import torch

        from g1_patrol.tasks.navigation.mdp.nyc_social import WHEELCHAIR

        s = self.scenario
        ids = torch.as_tensor(self.env_ids, device=s.device)
        pos = (s.ped_pos[ids] + s.tile_center[ids][:, None]).cpu().numpy()
        vel = s.ped_vel[ids].cpu().numpy()
        path_dir = s.walking_direction()[ids].cpu().numpy()
        ground = s.layout.ground_height(s.tile_row[ids], s.tile_col[ids], s.tile_local(s.ped_pos[ids])).cpu().numpy()
        shown = ((s.ped_mode[ids] != 0) & (s.ped_role[ids] == WHEELCHAIR)).cpu().numpy()
        self.visible[:] = False
        for item in self.items:
            e, k = item["env"], item["slot"]
            visible = bool(shown[e, k])
            self.visible[e, k] = visible
            if item["visible"] != visible:
                item["visible"] = visible
                item["img"].MakeVisible() if visible else item["img"].MakeInvisible()
            if not visible:
                item["yaw"] = None
                continue
            v = vel[e, k]
            speed = float(np.hypot(*v))
            if item["yaw"] is None:  # entering the scene: face along the path
                item["yaw"] = math.atan2(path_dir[e, k, 1], path_dir[e, k, 0])
            if speed > 0.05:
                diff = (math.atan2(v[1], v[0]) - item["yaw"] + math.pi) % (2 * math.pi) - math.pi
                rate = (3.0 + 5.0 * min(1.0, speed)) * dt * (2.0 if abs(diff) > math.pi / 2 else 1.0)
                item["yaw"] += max(-rate, min(rate, diff))
            item["translate"].Set(Gf.Vec3d(float(pos[e, k, 0]), float(pos[e, k, 1]), float(ground[e, k])))
            item["rotate"].Set(math.degrees(item["yaw"]) + 90.0)  # the wheelchair and its user face -y


class Strollers:
    """A stroller pushed by the character of each stroller slot (``wide_slots``; the character walks behind it)."""

    def __init__(self, stage, scenario, env_ids: list[int], root: str = "/World/Demo/Strollers"):
        self.scenario, self.env_ids = scenario, env_ids
        self.slots = wide_slots(scenario)[1]
        path = os.path.join(ACTORS, "stroller", "stroller.usd")
        self.items = []
        for i, e in enumerate(env_ids):
            for k in self.slots:
                prim = stage.DefinePrim(f"{root}/env_{e}/stroller_{k}", "Xform")  # our pose; the asset (with its own) below
                xf = UsdGeom.Xformable(prim)
                UsdGeom.Xformable(stage.DefinePrim(f"{root}/env_{e}/stroller_{k}/up", "Xform")).AddRotateXOp().Set(90.0)  # a Y-up asset
                stage.DefinePrim(f"{root}/env_{e}/stroller_{k}/up/asset", "Xform").GetReferences().AddReference(path)
                self.items.append({"env": i, "slot": k, "img": UsdGeom.Imageable(prim), "translate": xf.AddTranslateOp(),
                                   "rotate": xf.AddRotateZOp(), "visible": None})

    def update(self, crowd):
        import torch

        s = self.scenario
        ids = torch.as_tensor(self.env_ids, device=s.device)
        pos = (s.ped_pos[ids] + s.tile_center[ids][:, None]).cpu().numpy()
        ground = s.layout.ground_height(s.tile_row[ids], s.tile_col[ids], s.tile_local(s.ped_pos[ids])).cpu().numpy()
        for item in self.items:
            e, k = item["env"], item["slot"]
            yaw = crowd.yaw(e, k)
            visible = yaw is not None and bool(crowd.visible[e, k])
            if item["visible"] != visible:
                item["visible"] = visible
                item["img"].MakeVisible() if visible else item["img"].MakeInvisible()
            if not visible:
                continue
            x = float(pos[e, k, 0]) + STROLLER_AHEAD * math.cos(yaw)
            y = float(pos[e, k, 1]) + STROLLER_AHEAD * math.sin(yaw)
            item["translate"].Set(Gf.Vec3d(x, y, float(ground[e, k])))
            item["rotate"].Set(math.degrees(yaw) + STROLLER_YAW)


STROLLER_AHEAD, PUSHER_BEHIND = 0.38, -0.42
"""Stroller center and pushing parent relative to a wide-body slot's position (m, along its heading)."""
STROLLER_YAW = 90.0
"""Rotation (deg) that turns the upright stroller model to roll along +x, its handle at -x (checked with a render)."""
