"""Post-training alignment of the camera student's encoder: the last blocks of its frozen DINOv2-small are fine-tuned so
that a realistic image (the transfer of a replay frame, ``camera_transfer.py``) gives the patch tokens the original
encoder gives the render it was made from (``camera_replays.py``); the policy is unchanged.

Fits on the transfers of all but ``--val`` of the replays (with exposure, white-balance and contrast jitter) and reports
on the held-out ones: the cosine similarity of the policy's camera features (class token and pooled patch tokens) of
the transfers to those of the renders, and the mean change of the policy's actions (tanh of the mean action) from the
renders' features, with the original and the aligned encoder, and with the camera features replaced by their mean (what
the camera adds). Writes ``<out>/encoder`` (the aligned encoder, Hugging Face format: run the policy with
``G1_PATROL_ENCODER=<out>/encoder``) and ``<out>/report.json``.

Usage:
    python scripts/b2w/camera_align.py --transfers <dir> [<dir> ...] --policy <camera student model.pt> --out <dir> \\
        --device cuda:0
"""

import argparse
import copy
import json
import os

import imageio.v2 as imageio
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--transfers", nargs="+", required=True, help="Directories with a manifest.jsonl.")
    parser.add_argument("--policy", required=True, help="The camera student's checkpoint.")
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--val", type=float, default=0.2, help="Share of the replays held out.")
    parser.add_argument("--blocks", type=int, default=8, help="Trained blocks at the end of the encoder (of 12).")
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--batch", type=int, default=48)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    from rsl_rl.models import CNNModel
    from tensordict import TensorDict

    from g1_patrol.tasks.navigation.mdp.camera import PATCH, FrozenImageEncoder

    frozen = FrozenImageEncoder(args.device)
    rows = [json.loads(line) for d in args.transfers for line in open(os.path.join(d, "manifest.jsonl"))]
    clips = []
    for row in rows:
        rec = np.load(row["source"][:-4] + ".npz")
        h, w = rec["front"].shape[1:3]
        with imageio.get_reader(row["output"]) as reader:
            real = np.stack([np.asarray(Image.fromarray(f).resize((w, h), Image.BICUBIC)) for f in reader])
        n = min(len(rec["front"]), len(real))
        clips.append({"sim": rec["front"][:n], "real": real[:n], "camera": rec["camera"][:n].astype(np.float32),
                      "student": rec["student"][:n].astype(np.float32),
                      "height_map": rec["height_map"][:n].astype(np.float32)})
    order = rng.permutation(len(clips))
    n_val = max(1, int(round(args.val * len(clips))))
    val, train = [clips[i] for i in order[:n_val]], [clips[i] for i in order[n_val:]]
    frames = sum(len(c["sim"]) for c in train)
    print(f"[camera_align] {len(train)} replays ({frames} frames) to fit, {len(val)} held out", flush=True)

    teacher = frozen.model
    student = copy.deepcopy(teacher).float()
    student.requires_grad_(False)
    for block in student.encoder.layer[len(student.encoder.layer) - args.blocks :]:
        block.requires_grad_(True)
    student.layernorm.requires_grad_(True)
    opt = torch.optim.AdamW([p for p in student.parameters() if p.requires_grad], lr=args.lr, weight_decay=0.01)
    mean, std = frozen.mean.float(), frozen.std.float()
    h, w = train[0]["sim"].shape[1:3]

    def tokens(net, rgb, dtype):
        x = torch.from_numpy(rgb).to(args.device).permute(0, 3, 1, 2).float() / 255.0
        return net(pixel_values=((x - mean) / std).to(dtype)).last_hidden_state.float()

    def pooled(tok, grid=(2, 3)):  # the policy's camera features (mdp.camera.FrozenImageEncoder)
        rows_, cols = h // PATCH, w // PATCH
        patches = tok[:, 1 : 1 + rows_ * cols].reshape(len(tok), rows_, cols, -1).permute(0, 3, 1, 2)
        return torch.cat([tok[:, 0], F.adaptive_avg_pool2d(patches, grid).flatten(1)], dim=1)

    def jitter(rgb):  # exposure, white balance and contrast as a camera's vary
        x = rgb.astype(np.float32)
        gain = rng.uniform(0.75, 1.25, size=(len(x), 1, 1, 1)) * rng.uniform(0.92, 1.08, size=(len(x), 1, 1, 3))
        mid = x.mean(axis=(1, 2, 3), keepdims=True)
        x = (x - mid) * rng.uniform(0.85, 1.15, size=(len(x), 1, 1, 1)) + mid
        return np.clip(x * gain, 0, 255).astype(np.uint8)

    sim_all = np.concatenate([c["sim"] for c in train])
    real_all = np.concatenate([c["real"] for c in train])
    for step in range(args.steps):
        idx = rng.integers(0, len(sim_all), args.batch)
        with torch.no_grad():
            target = tokens(teacher, sim_all[idx], torch.bfloat16)
        pred = tokens(student, jitter(real_all[idx]), torch.float32)
        loss = F.mse_loss(pred, target) + (1 - F.cosine_similarity(pooled(pred), pooled(target), dim=1)).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 500 == 0 or step == args.steps - 1:
            print(f"[camera_align] step {step}: loss {loss.item():.4f}", flush=True)
    student.eval()
    student.save_pretrained(os.path.join(args.out, "encoder"))

    @torch.no_grad()
    def features(net, frames, dtype):
        out = [pooled(tokens(net, frames[i : i + 64], dtype)) for i in range(0, len(frames), 64)]
        return torch.cat(out).cpu().numpy()

    def cosine(a, b):
        return float(np.mean(np.sum(a * b, 1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-9)))

    target = np.concatenate([c["camera"] for c in val])
    real = np.concatenate([c["real"] for c in val])
    before, after = features(teacher, real, torch.bfloat16), features(student, real, torch.float32)
    report = {"replays": {"fit": len(train), "held_out": len(val)}, "blocks": args.blocks, "steps": args.steps,
              "cosine_to_render_features": {"original": cosine(before, target), "aligned": cosine(after, target)}}

    c0 = val[0]
    obs0 = TensorDict({"student": torch.zeros(1, c0["student"].shape[1]), "camera": torch.zeros(1, target.shape[1]),
                       "height_map": torch.zeros(1, *c0["height_map"].shape[1:])}, batch_size=[1])
    policy = CNNModel(obs0, {"student": ["student", "camera", "height_map"]}, "student", 3, hidden_dims=[256, 256, 128],
                      activation="elu", obs_normalization=True,
                      distribution_cfg={"class_name": "GaussianDistribution", "init_std": 0.1},
                      cnn_cfg=dict(output_channels=[16, 32, 32, 32], kernel_size=[5, 3, 3, 3], stride=[2, 2, 2, 2],
                                   activation="elu", max_pool=False))
    policy.load_state_dict(torch.load(args.policy, map_location="cpu", weights_only=False)["student_state_dict"])
    policy.to(args.device).eval()
    st = torch.from_numpy(np.concatenate([c["student"] for c in val])).to(args.device)
    hm = torch.from_numpy(np.concatenate([c["height_map"] for c in val])).to(args.device)

    @torch.no_grad()
    def act(camera):
        obs = TensorDict({"student": st, "camera": torch.from_numpy(camera.astype(np.float32)).to(args.device),
                          "height_map": hm}, batch_size=[len(st)])
        return torch.tanh(policy.mlp(policy.get_latent(obs))).cpu().numpy()

    a_sim = act(target)
    at_mean = np.tile(target.mean(0), (len(target), 1))
    report["action_change"] = {"original_encoder": float(np.abs(act(before) - a_sim).mean()),
                               "aligned_encoder": float(np.abs(act(after) - a_sim).mean()),
                               "camera_features_at_their_mean": float(np.abs(act(at_mean) - a_sim).mean())}
    json.dump(report, open(os.path.join(args.out, "report.json"), "w"), indent=1)
    print(json.dumps(report, indent=1), flush=True)


if __name__ == "__main__":
    main()
