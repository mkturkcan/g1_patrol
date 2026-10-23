"""Export a B2-W student for the robot: ``policy.onnx``, ``deploy.json`` and ``samples.npz``.

Loads a student checkpoint (``--task G1Patrol-B2W-Navigate-Fused``: lidar and depth cameras, the default;
``G1Patrol-B2W-Navigate-Lidar``: lidar only), exports the deterministic policy to ONNX (inputs ``obs`` (1, 13) and the
height map (1, 3, 80, 80), output ``actions`` (1, 3)), writes the settings the robot-side code needs
(:func:`g1_patrol.b2w.deploy_config`) and rolls the policy out in simulation for ``--samples`` steps, saving its inputs
and outputs. The
ONNX model is checked against the PyTorch policy on these samples here; ``scripts/b2w/benchmark.py --check`` repeats
the check on the robot's computer (same inputs, same actions: the inference stack is fine).

Usage:
    python scripts/b2w/export.py --checkpoint logs/rsl_rl/b2w_nav_fused_student/<run>/model_1499.pt --out deploy/b2w/release
"""

import argparse
import json
import os
import sys

import numpy as np

from isaaclab.app import add_launcher_args, launch_simulation

from isaaclab_rl.entrypoints.backends import cli_args_rsl_rl as cli_args
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, check_rsl_rl_version, create_rsl_rl_runner, handle_deprecated_rsl_rl_cfg

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import resolve_task_config, setup_preset_cli

import g1_patrol.tasks  # noqa: F401
from g1_patrol.b2w import LIDAR, deploy_config

MAPS = {"G1Patrol-B2W-Navigate-Lab-Fused": "height_map", "G1Patrol-B2W-Navigate-Lab-Lidar": "lidar_map",
        "G1Patrol-B2W-Navigate-Fused": "height_map", "G1Patrol-B2W-Navigate-Lidar": "lidar_map"}


def main(argv):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", required=True, help="Output directory.")
    parser.add_argument("--samples", type=int, default=200)
    parser.add_argument("--lidar", default=LIDAR)
    parser.add_argument("--task", default="G1Patrol-B2W-Navigate-Lab-Fused", choices=list(MAPS))
    cli_args.add_rsl_rl_args(parser)
    add_launcher_args(parser)
    args, remaining = setup_preset_cli(parser, argv)
    sys.argv = [sys.argv[0], *remaining]  # Hydra overrides only
    env_cfg, agent_cfg = resolve_task_config(args.task, "rsl_rl_cfg_entry_point", play_mode=True)
    group = MAPS[args.task]
    env_cfg.scene.num_envs = 4
    with launch_simulation(env_cfg, args):
        import gymnasium as gym
        import onnxruntime as ort
        import torch

        agent_cfg = handle_deprecated_rsl_rl_cfg(cli_args.update_rsl_rl_cfg(agent_cfg, args), check_rsl_rl_version())
        env = RslRlVecEnvWrapper(gym.make(args.task, cfg=env_cfg), clip_actions=agent_cfg.clip_actions)
        runner = create_rsl_rl_runner(env, agent_cfg)
        runner.load(os.path.abspath(args.checkpoint))
        out = os.path.abspath(args.out)
        os.makedirs(out, exist_ok=True)
        runner.export_policy_to_onnx(path=out, filename="policy.onnx")
        runner.export_policy_to_jit(path=out, filename="policy.pt")
        with open(os.path.join(out, "deploy.json"), "w") as f:
            json.dump(deploy_config(args.lidar, depth=group == "height_map") | {"task": args.task,
                      "checkpoint": os.path.abspath(args.checkpoint)}, f, indent=2)
        # roll out, keep the student's inputs and actions
        policy = runner.get_inference_policy(device=env.unwrapped.device)
        session = ort.InferenceSession(os.path.join(out, "policy.onnx"), providers=["CPUExecutionProvider"])
        names = [i.name for i in session.get_inputs()]
        obs = env.get_observations()
        samples = {"obs": [], "lidar_map": [], "actions": []}  # the map under its lidar-only name, for benchmark.py
        worst = 0.0
        with torch.inference_mode():
            for _ in range(args.samples):
                actions = policy(obs)
                o1, o2 = obs["student"][:1].cpu().numpy(), obs[group][:1].cpu().numpy()
                onnx_actions = session.run(None, {names[0]: o1, names[1]: o2})[0]
                worst = max(worst, float(np.abs(onnx_actions - actions[:1].cpu().numpy()).max()))
                samples["obs"].append(o1[0]), samples["lidar_map"].append(o2[0]), samples["actions"].append(onnx_actions[0])
                obs, _, _, _ = env.step(actions)
        np.savez_compressed(os.path.join(out, "samples.npz"), **{k: np.stack(v) for k, v in samples.items()})
        print(f"[INFO] exported {args.checkpoint} -> {out}; ONNX vs PyTorch max |action difference| {worst:.2e}")
        env.close()


if __name__ == "__main__":
    main(sys.argv[1:])
