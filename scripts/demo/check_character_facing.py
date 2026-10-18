"""Check that every demo character faces its direction of motion with every baked clip.

Each character is posed as ``people.py`` poses a pedestrian walking along +x (yaw 0, including the turn applied to the
mirrored clips); its facing is measured from the skinned skeleton, as the horizontal direction from the head to the
eyes and from the ankles to the toes, averaged over the clip. Every direction must be within ``--tolerance`` degrees
of +x. Requires the clips baked by ``bake_people_animations.py`` and access to NVIDIA's asset server.

Usage:
    python scripts/demo/check_character_facing.py --clips src/g1_patrol/assets/data/people
"""

import argparse
import math
import os
import sys

from isaacsim import SimulationApp

parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.add_argument("--clips", required=True, help="Directory of the baked clips.")
parser.add_argument("--tolerance", type=float, default=20.0, help="Largest accepted deviation from +x (degrees).")
args = parser.parse_args()
app = SimulationApp({"headless": True})

import numpy as np  # noqa: E402
import omni.usd  # noqa: E402
from pxr import Gf, Usd, UsdGeom, UsdSkel  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from people import CHARACTERS, MIRROR_YAW, PEOPLE, YAW_OFFSET  # noqa: E402

stage = omni.usd.get_context().get_stage()
stage.SetTimeCodesPerSecond(30.0)
UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
UsdGeom.SetStageMetersPerUnit(stage, 1.0)
setups = []
for i, name in enumerate(CHARACTERS):
    for kind in ("walk_loop_in_place", "idle_loop"):
        for suffix, turn in (("", 0.0), ("_mirror", MIRROR_YAW)):
            tag = f"{i}_{kind}{suffix}"
            char = stage.DefinePrim(f"/World/Character_{tag}", "Xform")
            char.GetReferences().AddReference(f"{PEOPLE}/{CHARACTERS[name]}")
            xf = UsdGeom.Xformable(char)
            xf.ClearXformOpOrder()
            xf.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, 0.0))
            xf.AddRotateZOp().Set(math.degrees(YAW_OFFSET + turn))
            anim = stage.DefinePrim(f"/World/Clip_{tag}")
            anim.GetReferences().AddReference(os.path.join(args.clips, f"{name}.stand_{kind}{suffix}.skelanim.usd"))
            skel = next(p for p in Usd.PrimRange(char) if p.IsA(UsdSkel.Skeleton))
            UsdSkel.BindingAPI.Apply(skel).CreateAnimationSourceRel().SetTargets([anim.GetPath()])
            setups.append((name, kind + suffix, skel))
for _ in range(5):
    app.update()

cache = UsdSkel.Cache()
for prim in stage.Traverse(Usd.TraverseInstanceProxies()):
    if prim.IsA(UsdSkel.Root):
        cache.Populate(UsdSkel.Root(prim), Usd.TraverseInstanceProxies())


def heading(v: np.ndarray) -> float:
    return math.degrees(math.atan2(v[1], v[0]))


failures = 0
for name, clip, skel in setups:
    query = cache.GetSkelQuery(UsdSkel.Skeleton(skel))
    index = {j.split("/")[-1]: k for k, j in enumerate(query.GetJointOrder())}
    eyes, toes = [], []
    for t in range(0, 80, 5):
        world = query.ComputeJointWorldTransforms(UsdGeom.XformCache(Usd.TimeCode(t)))
        pos = lambda j: np.array(world[index[j]].ExtractTranslation())  # noqa: E731
        eyes.append((pos("L_Eye") + pos("R_Eye")) / 2 - pos("Head"))
        toes.append(pos("L_ToeBase") - pos("L_Foot") + pos("R_ToeBase") - pos("R_Foot"))
    eye, toe = heading(np.mean(eyes, axis=0)), heading(np.mean(toes, axis=0))
    ok = abs(eye) < args.tolerance and abs(toe) < args.tolerance
    failures += not ok
    print(f"{'ok  ' if ok else 'FAIL'} {name:22s} {clip:34s} eyes {eye:7.1f} deg, toes {toe:7.1f} deg", flush=True)
print(f"{failures} of {len(setups)} character clips do not face their direction of motion", flush=True)
app.close()
sys.exit(1 if failures else 0)
