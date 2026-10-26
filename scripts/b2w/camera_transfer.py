"""Image-to-image transfer of the camera student's replays (``camera_replays.py``) to realistic video with NVIDIA
Cosmos 3 Nano (diffusers modular pipeline), for the post-training alignment of its encoder (``camera_align.py``).

The Canny edges of a replay and its depth video are the controls, so that the transfer keeps the scene's layout; the
text prompt of its scene (``camera_prompts/<kind>.json``) decides the look, with ``--looks`` one appearance variant per
replay (wall paint, floor, ceiling, light, equipment colors, clothing; :data:`LOOKS`), drawn from a generator seeded by
the replay's name. Writes ``<out>/<replay>.mp4`` and a line per replay to ``<out>/manifest.jsonl`` (with the edge F1
of the transfer against the render, a check that the layout was kept). Replays of kinds without a prompt (the street
tiles) are skipped. Replays are split over workers (``--worker k --workers n``); finished ones are skipped.

Runs in an environment with diffusers' Cosmos 3 pipelines (``pip install git+https://github.com/huggingface/diffusers``
at a revision that has ``Cosmos3OmniModularPipeline``, and opencv-python), not the Isaac Lab one; ``--model`` is a local
download of ``nvidia/Cosmos3-Nano`` (OpenMDW-1.1).

Usage:
    python scripts/b2w/camera_transfer.py --replays <dir> [<dir> ...] --out <dir> --model <Cosmos3-Nano dir> --looks \\
        --device cuda:0 --worker 0 --workers 2
"""

import argparse
import copy
import datetime
import glob
import hashlib
import json
import os
import time

import cv2
import imageio.v3 as iio
import numpy as np
import torch
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
LOOKS = {
    "walls": ["white painted concrete block walls", "light grey painted drywall", "pale blue-grey painted walls",
              "warm beige painted walls", "exposed red brick walls painted white up to two meters",
              "corrugated sheet-metal wall panels", "unpainted grey concrete walls with formwork marks"],
    "floor": ["a polished grey concrete floor", "a grey epoxy-coated floor with yellow safety lines",
              "a dark green epoxy floor", "speckled beige vinyl floor tiles",
              "a sealed concrete floor with worn, scuffed patches", "light grey rubber floor tiles"],
    "ceiling_hall": ["an exposed steel roof deck with silver ducts and cable trays",
                     "a high bay roof with skylights and steel trusses",
                     "a white acoustic drop ceiling with recessed LED panels"],
    "ceiling_room": ["a white acoustic drop ceiling with recessed LED panels",
                     "white ceiling tiles with fluorescent troffers", "an exposed painted ceiling with ducts"],
    "light": [("Bright, cool white LED lighting (5000 K)", "Crisp reflections of the lights on the floor"),
              ("Warm white fluorescent lighting (3500 K)", "Soft warm reflections on the floor"),
              ("Mixed daylight from high windows and LED lights", "Patches of daylight on the floor"),
              ("Dim lighting with some fixtures switched off", "Low-key, darker corners"),
              ("Even, slightly overcast-looking light", "Matte, low-contrast reflections")],
    "equipment": ["blue steel cabinets, red tool chests and grey machines",
                  "grey cabinets, green machinery and orange accents",
                  "yellow safety railings, black equipment and stainless steel",
                  "white lab furniture with blue accents", "worn industrial equipment in mixed colors"],
    "clothing": ["casual clothes", "white lab coats over casual clothes", "high-visibility safety vests and hard hats",
                 "dark work clothes", "a mix of casual clothes and safety vests"],
}  # fmt: skip
SCENES = {  # the scene, without its finishes (the look decides them)
    "makerlab": "A fabrication and testing laboratory: a long hall with workbenches, machines and shelving along both "
                "sides of a central aisle, red exit signs and fire extinguishers on the walls",
    "civil_lab": "A civil engineering testing laboratory: a high bay with load frames, test specimens, carts and "
                 "equipment",
    "lab": "A research laboratory with long lab benches, fume hoods, cabinets and shelving",
    "office": "An open-plan office with desks, office chairs, monitors and low partitions",
    "warehouse": "A warehouse with tall pallet racks loaded with boxes along the aisles",
    "corridor": "A corridor in an institutional building with doors, notice boards and a few pieces of furniture",
    "doorway": "Rooms connected by doorways in an institutional building, with shelves, cabinets and tables",
    "lobby": "A building lobby with a reception desk, sofas, armchairs, planters and columns",
}  # fmt: skip
HALLS = ("makerlab", "civil_lab", "warehouse")


def varied_prompt(prompt: dict, scene: str, rng: np.random.Generator) -> tuple[dict, dict]:
    """The scene's prompt with one appearance variant, and the variant."""
    look = {k: v[int(rng.integers(len(v)))] for k, v in LOOKS.items() if not k.startswith("ceiling")}
    ceilings = LOOKS["ceiling_hall" if scene in HALLS else "ceiling_room"]
    look["ceiling"] = ceilings[int(rng.integers(len(ceilings)))]
    p = copy.deepcopy(prompt)
    p["background_setting"] = (f"{SCENES.get(scene, p.get('background_setting', ''))}. The room has {look['walls']}, "
                               f"{look['floor']} and {look['ceiling']}.")
    p.setdefault("lighting", {})
    p["lighting"]["conditions"], p["lighting"]["illumination_effect"] = look["light"]
    p.setdefault("aesthetics", {})["color_scheme"] = f"Realistic colors: {look['equipment']}"
    for subject in p.get("subjects", []):
        subject["appearance_details"] = (f"{subject.get('appearance_details', '')} Furniture and equipment: "
                                         f"{look['equipment']}. People wear {look['clothing']}.").strip()
    look["light"] = look["light"][0]
    return p, look


def edges(frames: np.ndarray) -> list:
    out = []
    for f in frames:
        e = cv2.Canny(cv2.GaussianBlur(cv2.cvtColor(f, cv2.COLOR_RGB2GRAY), (5, 5), 0), 100, 200)
        out.append(Image.fromarray(np.repeat(e[..., None], 3, axis=-1)))
    return out


def edge_f1(a: np.ndarray, b: np.ndarray, tol: int = 2) -> float:
    """F1 of the Canny edges of two frames, within ``tol`` pixels."""
    ea, eb = (cv2.Canny(cv2.GaussianBlur(cv2.cvtColor(f, cv2.COLOR_RGB2GRAY), (5, 5), 0), 100, 200) > 0 for f in (a, b))
    kernel = np.ones((2 * tol + 1, 2 * tol + 1), np.uint8)
    da, db = (cv2.dilate(e.astype(np.uint8), kernel) > 0 for e in (ea, eb))
    precision, recall = (eb & da).sum() / max(eb.sum(), 1), (ea & db).sum() / max(ea.sum(), 1)
    return float(2 * precision * recall / max(precision + recall, 1e-9))


def to_frames(x) -> list:
    """uint8 (H, W, 3) frames from a pipeline output (PIL images, arrays or tensors, possibly nested)."""
    if isinstance(x, Image.Image):
        return [np.asarray(x.convert("RGB"))]
    if isinstance(x, torch.Tensor):
        x = x.float().cpu().numpy()
    if isinstance(x, np.ndarray):
        if x.ndim == 5:
            x = x[0]
        if x.ndim == 4 and x.shape[1] in (1, 3) and x.shape[-1] not in (1, 3):
            x = x.transpose(0, 2, 3, 1)
        if x.dtype != np.uint8:
            x = (np.clip(x, 0, 1) * 255).astype(np.uint8)
        return list(x) if x.ndim == 4 else [x]
    if isinstance(x, (list, tuple)):
        return [f for item in x for f in to_frames(item)]
    raise TypeError(type(x))


def local_index(model: str, out: str) -> str:
    """A copy of the model's modular_model_index.json whose components point at the local download."""
    spec = json.load(open(os.path.join(model, "modular_model_index.json")))
    for value in spec.values():
        if isinstance(value, list) and len(value) == 3 and isinstance(value[2], dict):
            value[2]["pretrained_model_name_or_path"] = model
            value[2]["revision"] = None
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "modular_model_index.json"), "w") as fh:
        json.dump(spec, fh, indent=1)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--replays", nargs="+", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--model", default=os.environ.get("COSMOS3_NANO"), help="Local nvidia/Cosmos3-Nano download.")
    parser.add_argument("--prompts", default=os.path.join(HERE, "camera_prompts"))
    parser.add_argument("--looks", action="store_true", help="An appearance variant per replay.")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--worker", type=int, default=0)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--height", type=int, default=368)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--steps", type=int, default=18)
    parser.add_argument("--fps", type=float, default=10.0, help="Frame rate of the replays.")
    parser.add_argument("--frames", type=int, default=121)
    parser.add_argument("--until", default=None, help="Start no replay after this time (UTC, HH:MM).")
    args = parser.parse_args()
    assert args.model, "--model (or COSMOS3_NANO): a local download of nvidia/Cosmos3-Nano"
    os.makedirs(args.out, exist_ok=True)
    stop = None
    if args.until:
        stop = datetime.datetime.combine(datetime.datetime.now(datetime.timezone.utc).date(),
                                         datetime.time.fromisoformat(args.until), datetime.timezone.utc)
    jobs = []
    for d in args.replays:
        for clip in sorted(glob.glob(os.path.join(d, "*.mp4"))):
            if clip.endswith(("_depth.mp4", "_policy.mp4")) or not os.path.exists(clip.replace(".mp4", "_depth.mp4")):
                continue
            scene = os.path.basename(clip).split("_L")[0]
            if os.path.exists(os.path.join(args.prompts, f"{scene}.json")):
                jobs.append((clip, scene))
    jobs = jobs[args.worker :: args.workers]
    print(f"[camera_transfer] worker {args.worker}: {len(jobs)} replays", flush=True)

    from diffusers import ModularPipeline

    pipe = ModularPipeline.from_pretrained(local_index(args.model, os.path.join(args.out, ".cosmos_index")))
    pipe.load_components(torch_dtype=torch.bfloat16)
    pipe.to(args.device)
    pipe.disable_safety_checker()  # simulation renders only; the guardrail would also blur faces
    negative = json.load(open(os.path.join(args.model, "assets", "negative_prompt.json")))
    manifest = os.path.join(args.out, "manifest.jsonl")
    done = {json.loads(line)["output"] for line in open(manifest)} if os.path.exists(manifest) else set()
    for clip, scene in jobs:
        if stop is not None and datetime.datetime.now(datetime.timezone.utc) > stop:
            print("[camera_transfer] deadline reached", flush=True)
            break
        name = f"{os.path.basename(os.path.dirname(clip))}_{os.path.basename(clip)[:-4]}"
        out = os.path.join(args.out, f"{name}.mp4")
        if out in done:
            continue
        rgb = iio.imread(clip, plugin="pyav")[: args.frames]
        depth = iio.imread(clip.replace(".mp4", "_depth.mp4"), plugin="pyav")[: args.frames, ..., 0].astype(np.float32)
        lo, hi = np.percentile(depth, [1, 99])
        depth = np.clip((depth - lo) / max(hi - lo, 1e-3), 0, 1)
        gray = [np.repeat((255 * d).astype(np.uint8)[..., None], 3, axis=-1) for d in depth]
        controls = {"edge": edges(rgb), "depth": [Image.fromarray(g) for g in gray]}
        prompt = json.load(open(os.path.join(args.prompts, f"{scene}.json")))
        seed = int(hashlib.sha256(name.encode()).hexdigest()[:8], 16)  # per replay
        look = None
        if args.looks:
            prompt, look = varied_prompt(prompt, scene, np.random.default_rng(seed))
        t0 = time.time()
        result = pipe(prompt=json.dumps(prompt), negative_prompt=json.dumps(negative), control_videos=controls,
                      num_frames=len(rgb), height=args.height, width=args.width, fps=args.fps,
                      num_inference_steps=args.steps, guidance_scale=3.0, control_guidance=1.5,
                      generator=torch.Generator(device=args.device).manual_seed(seed), output="videos")
        frames = np.stack(to_frames(result))
        iio.imwrite(out, frames, fps=args.fps, codec="libx264", quality=9, macro_block_size=8)
        n = min(len(rgb), len(frames))
        sim = [cv2.resize(f, (frames.shape[2], frames.shape[1]), interpolation=cv2.INTER_AREA) for f in rgb[:n]]
        f1 = float(np.mean([edge_f1(a, b) for a, b in zip(sim, frames[:n])]))
        row = {"output": out, "source": clip, "frames": n, "scene": scene, "seed": seed, "look": look,
               "size": [args.width, args.height], "steps": args.steps, "seconds": round(time.time() - t0, 1),
               "edge_f1": round(f1, 3)}
        with open(manifest, "a") as fh:
            fh.write(json.dumps(row) + "\n")
        print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
