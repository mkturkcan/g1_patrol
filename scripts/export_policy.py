"""Export a trained RSL-RL policy to TorchScript (for example the frozen walker used by the navigation tasks).

Usage:
    python scripts/export_policy.py --task G1Patrol-Walk-City --checkpoint <model.pt> \
        --out src/g1_patrol/assets/data/policies/g1_walk_city.pt
"""

import argparse
import os
import sys

from isaaclab.app import add_launcher_args, launch_simulation

from isaaclab_rl.entrypoints.backends import cli_args_rsl_rl as cli_args
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, check_rsl_rl_version, create_rsl_rl_runner, handle_deprecated_rsl_rl_cfg

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import resolve_task_config, setup_preset_cli

import g1_patrol.tasks  # noqa: F401


def main(argv):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--task", required=True)
    parser.add_argument("--out", required=True, help="Output TorchScript file.")
    cli_args.add_rsl_rl_args(parser)
    add_launcher_args(parser)
    args, remaining = setup_preset_cli(parser, argv)
    sys.argv = [sys.argv[0], *remaining]  # Hydra overrides only
    env_cfg, agent_cfg = resolve_task_config(args.task, "rsl_rl_cfg_entry_point", play_mode=True)
    env_cfg.scene.num_envs = 4
    with launch_simulation(env_cfg, args):
        import gymnasium as gym

        agent_cfg = handle_deprecated_rsl_rl_cfg(cli_args.update_rsl_rl_cfg(agent_cfg, args), check_rsl_rl_version())
        env = RslRlVecEnvWrapper(gym.make(args.task, cfg=env_cfg), clip_actions=agent_cfg.clip_actions)
        runner = create_rsl_rl_runner(env, agent_cfg)
        runner.load(os.path.abspath(args.checkpoint))
        out = os.path.abspath(args.out)
        runner.export_policy_to_jit(path=os.path.dirname(out), filename=os.path.basename(out))
        print(f"[INFO] exported {args.checkpoint} -> {out}")
        env.close()


if __name__ == "__main__":
    main(sys.argv[1:])
