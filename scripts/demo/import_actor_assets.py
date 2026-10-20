"""Import the demo's street actors (a dog with walk, run and idle cycles, a wheelchair, a stroller; CC0 / CC-BY 4.0
models) into USD with Isaac Sim's asset converter (rendering only; the simulation's bodies stay
the kinematic capsules).

Each model of ``ACTORS`` is converted from glTF (skeleton, skinning, animations and materials kept) into
``<out>/<name>/<name>.usd``; ``<out>/ATTRIBUTION.md`` credits them. The downloads are data only (nothing in them is
executed); ``--downloads`` is the directory with their ``manifest.json``.

Usage:
    python scripts/demo/import_actor_assets.py --downloads <street_actors> --out src/g1_patrol/assets/data/actors
"""

import argparse
import asyncio
import json
import os
import shutil

from isaaclab.app import add_launcher_args, launch_simulation

ACTORS = {  # name: download id
    "dog": "dog__wolf_scene_realistic__5d555064",
    "wheelchair": "wheelchair__wheelchair_01",
    "stroller": "stroller__stroller_scan__b8bc996d",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--downloads", required=True)
    parser.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "..", "..", "src", "g1_patrol", "assets", "data", "actors"))
    add_launcher_args(parser)
    args = parser.parse_args()
    import isaaclab.sim as sim_utils

    with launch_simulation(sim_utils.SimulationCfg(device=args.device), args):
        import omni.kit.app

        omni.kit.app.get_app().get_extension_manager().set_extension_enabled_immediate("omni.kit.asset_converter", True)
        import omni.kit.asset_converter as converter

        with open(os.path.join(args.downloads, "manifest.json")) as f:
            sources = {a["id"]: a for a in json.load(f)}

        async def convert(src: str, dst: str):
            ctx = converter.AssetConverterContext()
            ctx.ignore_animations = False
            ctx.ignore_materials = False
            ctx.embed_textures = False
            ctx.use_meter_as_world_unit = True
            task = converter.get_instance().create_converter_task(src, dst, None, ctx)
            ok = await task.wait_until_finished()
            return ok, task.get_error_message()

        credits = []
        for name, source_id in ACTORS.items():
            src = sources[source_id]
            out_dir = os.path.join(args.out, name)
            if os.path.isdir(out_dir):
                shutil.rmtree(out_dir)
            os.makedirs(out_dir)
            future = asyncio.ensure_future(convert(os.path.join(args.downloads, src["local_path"]), os.path.join(out_dir, f"{name}.usd")))
            while not future.done():
                omni.kit.app.get_app().update()
            ok, error = future.result()
            print(f"[IMPORT] {name}: {'ok' if ok else 'FAILED ' + error}", flush=True)
            credits.append(f"- `{name}`: \"{src['title']}\" by {src['author']}, {src['license']}, {src['url']}.")
        with open(os.path.join(args.out, "ATTRIBUTION.md"), "w") as f:
            f.write("# Demo street actors\n\n")
            f.write("Imported by `scripts/demo/import_actor_assets.py` (converted to USD; rendering only).\n\n")
            f.write("\n".join(credits) + "\n")


if __name__ == "__main__":
    main()
