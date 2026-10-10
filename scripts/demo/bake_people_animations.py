"""Bake Isaac Sim's biped walk/idle clips onto the People characters' own skeletons (offline retargeting).

The clips in ``Isaac/People/Animations`` are authored for an 81-joint biped skeleton; the characters in
``Isaac/People/Characters`` use a 101-joint Reallusion skeleton. Isaac Sim normally retargets at runtime through an
animation graph; for the demo we want plain ``SkelAnimation`` prims we can bind and time-offset per pedestrian, so
this script retargets the clips once with ``omni.anim.retarget.core`` and saves one ``.skelanim.usd`` per character
and clip.

Usage (headless):
    python scripts/demo/bake_people_animations.py --out src/g1_patrol/assets/data/people
"""

import argparse
import os

from isaacsim import SimulationApp

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
CLIPS = ["stand_walk_loop_in_place", "stand_walk_loop_in_place_mirror", "stand_idle_loop", "stand_idle_loop_mirror"]
SOURCE_SKELETON = "Characters/biped_demo/biped_demo_meters.usd"

# Human retarget rig tag -> joint of the biped skeleton the clips are authored for
BIPED_TAGS = {
    "Hips": "Pelvis", "Spine1": "Spine1", "Spine2": "Spine2", "Chest": "Chest", "Neck1": "Neck1", "Neck2": "Neck2",
    "Head": "Head",
}  # fmt: skip
for side, s in (("Left", "L"), ("Right", "R")):
    BIPED_TAGS.update({
        f"{side}Leg": f"{s}_UpLeg", f"{side}Shin": f"{s}_LoLeg", f"{side}Foot": f"{s}_Ankle",
        f"{side}ToeBase": f"{s}_Ball", f"{side}Shoulder": f"{s}_Clavicle", f"{side}Arm": f"{s}_UpArm",
        f"{side}ForeArm": f"{s}_LoArm", f"{side}Hand": f"{s}_Wrist",
        f"{side}HandThumb1": f"{s}_ThumbFinger1", f"{side}HandThumb2": f"{s}_ThumbFinger2",
        f"{side}HandThumb3": f"{s}_ThumbFinger3",
    })  # fmt: skip
    for finger in ("Index", "Middle", "Ring", "Pinky"):
        for k in range(1, 5):
            BIPED_TAGS[f"{side}Hand{finger}{k}"] = f"{s}_{finger}Finger{k}"


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", required=True, help="Output directory for the baked clips.")
    parser.add_argument("--characters", nargs="*", default=list(CHARACTERS), help="Subset of characters to bake.")
    args = parser.parse_args()
    app = SimulationApp({"headless": True})

    import omni.kit.app

    omni.kit.app.get_app().get_extension_manager().set_extension_enabled_immediate("omni.anim.retarget.core", True)
    import omni.kit.commands
    import omni.usd
    from omni.anim.retarget.core.scripts.extension import auto_setup_by_prim, has_retarget_setup
    from omni.anim.retarget.core.scripts.rig import RetargetRig
    from pxr import Sdf, Usd, UsdGeom, UsdSkel

    def first_skeleton(root):
        return next(p for p in Usd.PrimRange(root) if p.IsA(UsdSkel.Skeleton))

    os.makedirs(args.out, exist_ok=True)
    for name in args.characters:
        ctx = omni.usd.get_context()
        ctx.new_stage()
        stage = ctx.get_stage()
        stage.SetTimeCodesPerSecond(30.0)
        UsdGeom.SetStageMetersPerUnit(stage, 1.0)
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        source = stage.DefinePrim("/Source", "Xform")
        source.GetReferences().AddReference(f"{PEOPLE}/{SOURCE_SKELETON}")
        target = stage.DefinePrim("/Target", "Xform")
        target.GetReferences().AddReference(f"{PEOPLE}/{CHARACTERS[name]}")
        anim_paths = []
        for clip in CLIPS:
            prim = stage.DefinePrim(f"/Clips/{clip}")
            prim.GetReferences().AddReference(f"{PEOPLE}/Animations/{clip}.skelanim.usd")
            anim_paths.append(prim.GetPath().pathString)
        src_skel, tgt_skel = first_skeleton(source), first_skeleton(target)

        # the biped's own tags predate the Human rig: re-tag it and recompute its retarget pose
        rig = RetargetRig("Human")
        rig.set_skeleton(UsdSkel.Skeleton(src_skel))
        rig.set_joint_mappings(BIPED_TAGS)
        ok = auto_setup_by_prim("Human", src_skel, use_mapping=False, auto_facing=True, auto_tagging=False)
        print(f"[{name}] biped setup: {ok}, target has retarget setup: {has_retarget_setup(tgt_skel, 20)}", flush=True)

        stage.DefinePrim("/Baked", "Scope")
        omni.kit.commands.execute(
            "CreateRetargetAnimations",
            source_skeleton_path=src_skel.GetPath().pathString,
            target_skeleton_path=tgt_skel.GetPath().pathString,
            source_animation_paths=anim_paths,
            target_animation_parent_path="/Baked",
            set_root_identity=False,
        )
        for clip in CLIPS:
            baked = stage.GetPrimAtPath(f"/Baked/{clip}")
            if not baked:
                print(f"[{name}] {clip}: retargeting failed", flush=True)
                continue
            layer = Sdf.Layer.CreateAnonymous()
            Sdf.CopySpec(stage.GetRootLayer(), baked.GetPath(), layer, Sdf.Path("/Root"))
            layer.defaultPrim = "Root"
            samples = UsdSkel.Animation(baked).GetRotationsAttr().GetTimeSamples()
            layer.timeCodesPerSecond = 30.0
            layer.startTimeCode, layer.endTimeCode = samples[0], samples[-1]
            out = os.path.join(args.out, f"{name}.{clip}.skelanim.usd")
            layer.Export(out)
            print(f"[{name}] {clip}: {len(samples)} frames -> {out}", flush=True)
    app.close()


if __name__ == "__main__":
    main()
