"""Print the input/output interface of a trained policy (for deployment): observation terms in order with sizes,
history, noise and latency, the joint order, default joint positions, action scale and PD gains.

Usage:
    python scripts/describe_policy_io.py --task G1Patrol-Walk-City > interface.md
"""

import argparse
import sys

from isaaclab.app import add_launcher_args, launch_simulation

from isaaclab_tasks.utils import resolve_task_config, setup_preset_cli

import isaaclab_tasks  # noqa: F401

import g1_patrol.tasks  # noqa: F401


def describe_noise(noise) -> str:
    """Uniform noise per step, plus a bias drawn once per episode for noise models with an additive bias."""
    if noise is None:
        return "-"
    per_step = getattr(noise, "noise_cfg", noise)
    text = f"uniform ±{per_step.n_max:g}" if hasattr(per_step, "n_max") else "-"
    bias = getattr(noise, "bias_noise_cfg", None)
    if bias is not None and hasattr(bias, "n_max"):
        text += f", bias ±{bias.n_max:g} per episode"
    return text


def main(argv):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--task", required=True)
    add_launcher_args(parser)
    args, remaining = setup_preset_cli(parser, argv)
    sys.argv = [sys.argv[0], *remaining]
    env_cfg, _ = resolve_task_config(args.task, "rsl_rl_cfg_entry_point", play_mode=True)
    env_cfg.scene.num_envs = 2
    with launch_simulation(env_cfg, args):
        import gymnasium as gym

        env = gym.make(args.task, cfg=env_cfg).unwrapped
        om = env.observation_manager
        print(f"# Interface of `{args.task}`\n")
        print(f"Policy rate: {1.0 / env.step_dt:.0f} Hz; physics {1.0 / env.physics_dt:.0f} Hz.\n")
        for group, names in om.active_terms.items():
            cfgs = om._group_obs_term_cfgs[group]
            dims = om.group_obs_term_dim[group]
            total = om.group_obs_dim[group]
            print(f"## Observation group `{group}` (total {total})\n")
            print("| # | Term | Size per step | History | Noise | Latency (steps) | Scale / clip |")
            print("|---|---|---|---|---|---|---|")
            for k, (name, cfg, dim) in enumerate(zip(names, cfgs, dims)):
                noise_s = describe_noise(cfg.noise)
                hist = cfg.history_length or 1
                per_step = tuple(dim)[1:] if hist > 1 and cfg.flatten_history_dim is False else tuple(dim)
                if hist > 1 and cfg.flatten_history_dim:
                    per_step = (int(dim[0]) // hist,)
                lag = f"{cfg.delay_min_lag}-{cfg.delay_max_lag}" if cfg.delay_max_lag else "0"
                extra = ", ".join(x for x in (f"scale {cfg.scale}" if cfg.scale is not None else "", f"clip {cfg.clip}" if cfg.clip else "") if x) or "-"
                print(f"| {k} | `{name}` | {per_step} | {hist} | {noise_s} | {lag} | {extra} |")
            if any((c.history_length or 0) > 1 for c in cfgs):
                print("\nWith history, each term contributes its frames oldest first, term after term (flattened).")
            print()
        robot = env.scene["robot"]
        print("## Joints (policy order)\n")
        print("| # | Joint | Default position (rad) | Stiffness | Damping | Effort limit |")
        print("|---|---|---|---|---|---|")
        import re

        default = robot.data.default_joint_pos.torch[0].tolist()

        def resolve(value, joint):
            if isinstance(value, dict):
                return next((v for k, v in value.items() if re.fullmatch(k, joint)), float("nan"))
            return float("nan") if value is None else value

        for j, name in enumerate(robot.joint_names):
            act = next(a for a in env_cfg.scene.robot.actuators.values() if any(re.fullmatch(e, name) for e in a.joint_names_expr))
            limit = act.actuator_effort_limit if act.actuator_effort_limit is not None else act.joint_effort_limit
            print(f"| {j} | `{name}` | {default[j]:.3f} | {resolve(act.stiffness, name):g} | {resolve(act.damping, name):g} | "
                  f"{resolve(limit, name):g} |")
        am = env.action_manager
        print("\n## Actions\n")
        for name, term in am._terms.items():
            cfg = term.cfg
            scale = getattr(cfg, "scale", None)
            print(f"- `{name}`: {term.action_dim} values; {type(term).__name__}" + (f", scale {scale}" if scale is not None else ""))
        env.close()


if __name__ == "__main__":
    main(sys.argv[1:])
