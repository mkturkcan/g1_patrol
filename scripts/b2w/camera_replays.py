"""Record replays of the camera student's environment (G1Patrol-B2W-Navigate-Lab-Camera: the training environment drawn
as in the demo) for the post-training alignment of its encoder (``camera_transfer.py``, ``camera_align.py``).

Per robot, clips of ``--frames`` policy steps (10 Hz) without a reset, each with
  <kind>_L<level>_env<e>_clip<k>.mp4        the camera at --scale times the policy's resolution: the transfer's input
  <kind>_L<level>_env<e>_clip<k>_depth.mp4  its inverse depth (0.5 m white to 20 m black): the transfer's depth control
  <kind>_L<level>_env<e>_clip<k>_policy.mp4 the policy camera's frames
  <kind>_L<level>_env<e>_clip<k>.npz        the policy's observations at every step (``student``, ``camera``: the
                                            encoder features, ``height_map``) and its camera's frames (``front``)
The robots are driven by ``--checkpoint``, a student of ``--policy_task`` (default: the fused student, which does not
use the camera; the replays only have to show the environment as the camera student sees it).

Usage:
    python scripts/b2w/camera_replays.py --checkpoint <fused student model.pt> --num_envs 32 --clips 2 --out <dir> \\
        --device cuda:0 physics=isaacsim_physx
"""

import argparse
import os
import sys

import numpy as np
import torch

from isaaclab.app import add_launcher_args, launch_simulation

from isaaclab_tasks.utils import resolve_task_config, setup_preset_cli

import g1_patrol.tasks  # noqa: F401

NEAR, FAR = 0.5, 20.0
"""Depth (m) drawn white and black in the depth videos (inverse depth in between), as in
``scripts/demo/b2w_demo.py``."""


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--task", default="G1Patrol-B2W-Navigate-Lab-Camera")
    parser.add_argument("--policy_task", default="G1Patrol-B2W-Navigate-Lab-Fused",
                        help="Task of --checkpoint's agent.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--num_envs", type=int, default=32)
    parser.add_argument("--frames", type=int, default=121)
    parser.add_argument("--clips", type=int, default=2, help="Clips per robot.")
    parser.add_argument("--scale", type=float, default=2.5,
                        help="Resolution of the transfer's input, times the policy's.")
    parser.add_argument("--seed", type=int, default=None, help="Environment seed (tiles, scenarios).")
    parser.add_argument("--out", required=True)
    add_launcher_args(parser)
    args, remaining = setup_preset_cli(parser, sys.argv[1:])
    sys.argv = [sys.argv[0], *remaining]
    args.enable_cameras = True

    env_cfg, _ = resolve_task_config(args.task, "rsl_rl_cfg_entry_point")
    _, agent_cfg = resolve_task_config(args.policy_task, "rsl_rl_cfg_entry_point")
    env_cfg.scene.num_envs = args.num_envs
    if args.seed is not None:
        env_cfg.seed = args.seed
    front = env_cfg.scene.front_rgb
    env_cfg.scene.rec_rgb = front.replace(prim_path="{ENV_REGEX_NS}/Robot/rec_rgb",
                                          data_types=["rgb", "distance_to_image_plane"],
                                          width=int(round(front.width * args.scale)),
                                          height=int(round(front.height * args.scale)))
    with launch_simulation(env_cfg, args):
        import gymnasium as gym
        import imageio.v2 as imageio

        from isaaclab_rl.rsl_rl import (
            RslRlVecEnvWrapper,
            check_rsl_rl_version,
            create_rsl_rl_runner,
            handle_deprecated_rsl_rl_cfg,
        )

        from g1_patrol.tasks.navigation.nyc.kinds import ALL_KINDS

        agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, check_rsl_rl_version())
        env = RslRlVecEnvWrapper(gym.make(args.task, cfg=env_cfg), clip_actions=agent_cfg.clip_actions)
        base = env.unwrapped
        runner = create_rsl_rl_runner(env, agent_cfg)
        runner.load(args.checkpoint)
        policy = runner.get_inference_policy(device=base.device)
        scenario = base.command_manager.get_term("nyc")
        camera, recorder = base.scene["front_rgb"], base.scene["rec_rgb"]
        os.makedirs(args.out, exist_ok=True)
        n = args.num_envs

        def fresh():
            return {"rec": [], "depth": [], "front": [], "student": [], "camera": [], "height_map": []}

        buffers = [fresh() for _ in range(n)]
        clip_index = [0] * n
        written = 0

        def flush(e):
            b = buffers[e]
            kind = ALL_KINDS[int(scenario.layout.kind[scenario.tile_row[e], scenario.tile_col[e]])]
            stem = os.path.join(args.out, f"{kind}_L{int(scenario.tile_row[e])}_env{e:03d}_clip{clip_index[e]:02d}")
            for key, suffix in (("rec", ""), ("depth", "_depth"), ("front", "_policy")):
                imageio.mimwrite(f"{stem}{suffix}.mp4", b[key], fps=10, codec="libx264", quality=9, macro_block_size=2)
            np.savez_compressed(f"{stem}.npz", **{k: np.stack(b[k]) for k in ("student", "camera", "height_map")},
                                front=np.stack(b["front"]))
            clip_index[e] += 1

        obs = env.get_observations()
        step = 0
        with torch.inference_mode():
            while min(clip_index) < args.clips and step < args.clips * args.frames * 3:
                obs, _, dones, _ = env.step(policy(obs))
                policy.reset(dones)
                step += 1
                rgb = recorder.data.output["rgb"][..., :3].cpu().numpy()
                depth = recorder.data.output["distance_to_image_plane"][..., 0].cpu().numpy()
                frames = camera.data.output["rgb"][..., :3].cpu().numpy()
                student = obs["student"].cpu().numpy().astype(np.float32)
                features = obs["camera"].cpu().numpy().astype(np.float16)
                height_map = obs["height_map"].cpu().numpy().astype(np.float16)
                for e in range(n):
                    if clip_index[e] >= args.clips:
                        continue
                    if bool(dones[e]):  # a clip never spans a reset
                        buffers[e] = fresh()
                        continue
                    d = np.nan_to_num(depth[e], nan=FAR, posinf=FAR)
                    inv = (1.0 / np.clip(d, NEAR, FAR) - 1.0 / FAR) / (1.0 / NEAR - 1.0 / FAR)
                    b = buffers[e]
                    b["rec"].append(rgb[e])
                    b["depth"].append(np.repeat((255 * inv).astype(np.uint8)[..., None], 3, axis=-1))
                    b["front"].append(frames[e])
                    b["student"].append(student[e])
                    b["camera"].append(features[e])
                    b["height_map"].append(height_map[e])
                    if len(b["rec"]) == args.frames:
                        flush(e)
                        written += 1
                        buffers[e] = fresh()
                if step % 100 == 0:
                    print(f"[camera_replays] step {step}: {written} clips", flush=True)
        print(f"[camera_replays] {written} clips of {args.frames} frames in {args.out} after {step} steps", flush=True)
        env.close()


if __name__ == "__main__":
    main()
