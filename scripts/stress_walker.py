"""Stress tests of the locomotion policy: :mod:`evaluate` on ``G1Patrol-Walk-City`` with every robot at, or beyond, the
edge of the training randomization (heavier, weaker, slower, more slippery, noisier) instead of sampled within it.

Usage:
    python scripts/stress_walker.py --condition payload --checkpoint <model.pt> --num_envs 1024
    conditions: nominal (the training randomization), payload, weak_motors, latency, slippery, sensor_noise, combined
"""

import sys

import gymnasium as gym

import evaluate

CONDITIONS = ("nominal", "payload", "weak_motors", "latency", "slippery", "sensor_noise", "combined")


def stressed_cfg(condition: str):
    from isaaclab.utils import configclass

    from g1_patrol.tasks.locomotion.config.g1.walk_env_cfg import G1WalkCityEnvCfg

    apply = CONDITIONS[1:-1] if condition == "combined" else (condition,)

    @configclass
    class Stressed(G1WalkCityEnvCfg):
        def __post_init__(self):
            super().__post_init__()
            if "payload" in apply:  # a 5 kg backpack or sensor box on the torso (training: -1 to +5 kg)
                self.events.add_base_mass.params.update(
                    {"mass_distribution_params": (5.0, 5.0), "operation": "add", "distribution": "uniform"}
                )
            if "weak_motors" in apply:  # hot motors / low battery: 20% less torque on every joint (training: 0-20%)
                self.events.motor_strength.params["strength_range"] = (0.8, 0.8)
            if "latency" in apply:  # 15-25 ms actuation latency (training: 0-20 ms)
                for name in ("legs", "feet"):
                    self.scene.robot.actuators[name].min_delay = 3
                    self.scene.robot.actuators[name].max_delay = 5
            if "slippery" in apply:  # wet paint, steel plates: the training's lowest friction on every robot
                self.events.physics_material.params["static_friction_range"] = (0.3, 0.3)
                self.events.physics_material.params["dynamic_friction_range"] = (0.25, 0.25)
            if "sensor_noise" in apply:  # twice the training noise and bias on IMU and encoders
                for term in ("base_ang_vel", "projected_gravity", "joint_pos", "joint_vel"):
                    noise = getattr(self.observations.policy, term).noise
                    for cfg in (getattr(noise, "noise_cfg", None), getattr(noise, "bias_noise_cfg", None), noise):
                        if hasattr(cfg, "n_min"):
                            cfg.n_min, cfg.n_max = 2 * cfg.n_min, 2 * cfg.n_max

    return Stressed


def main(argv):
    condition = argv[argv.index("--condition") + 1]
    argv = [a for i, a in enumerate(argv) if a != "--condition" and (i == 0 or argv[i - 1] != "--condition")]
    assert condition in CONDITIONS, condition
    task = f"G1Patrol-Walk-City-Stress-{condition}"
    gym.register(id=task, entry_point="isaaclab.envs:ManagerBasedRLEnv", disable_env_checker=True,
                 kwargs={"env_cfg_entry_point": stressed_cfg(condition),
                         "rsl_rl_cfg_entry_point": "g1_patrol.tasks.locomotion.config.g1.agents.rsl_rl_ppo_cfg:G1WalkCityPPORunnerCfg"})
    evaluate.main(["--task", task, *argv])


if __name__ == "__main__":
    main(sys.argv[1:])
