"""Stress tests of the locomotion policy: :mod:`evaluate` on ``G1Patrol-Walk-City`` (or, with ``--walker reach``,
``G1Patrol-Walk-Reach``) with every robot at, or beyond, the edge of the training randomization (heavier, weaker,
slower, more slippery, noisier) instead of sampled within it.

Usage:
    python scripts/stress_walker.py --condition payload --checkpoint <model.pt> --num_envs 1024 [--walker reach]
    conditions: nominal (the training randomization), payload, weak_motors, latency, slippery, sensor_noise, combined;
    reaching walker only: attacks (shoves and kicks half of the time), deep_squat (only deep postures, held while
    standing), brace_retreat (braced stance, walking at 0.8-1.3 m/s and turning while attacked), hand_load (3 kg in
    one hand all the time)
"""

import sys

import gymnasium as gym

import evaluate

CONDITIONS = ("nominal", "payload", "weak_motors", "latency", "slippery", "sensor_noise", "combined")
REACH_CONDITIONS = ("attacks", "deep_squat", "brace_retreat", "hand_load")


def stressed_cfg(condition: str, walker: str = "city"):
    from isaaclab.utils import configclass

    from g1_patrol.tasks.locomotion.config.g1.walk_env_cfg import G1WalkCityEnvCfg, G1WalkReachEnvCfg

    apply = CONDITIONS[1:-1] if condition == "combined" else (condition,)
    base = G1WalkReachEnvCfg if walker == "reach" else G1WalkCityEnvCfg

    @configclass
    class Stressed(base):
        def play_mode(self):
            super().play_mode()
            if "attacks" in apply or "brace_retreat" in apply:  # play mode turns the attacks off
                self.actions.disturbance.attack_ratio = 0.5

        def __post_init__(self):
            super().__post_init__()
            command = self.commands.base_velocity
            if "deep_squat" in apply:  # picking things up from the ground, again and again
                command.posture_ratio, command.deep_ratio, command.guard_ratio = 1.0, 1.0, 0.0
            if "brace_retreat" in apply:  # braced, getting away from an attacker
                command.posture_ratio, command.guard_ratio = 0.0, 1.0
                command.rel_standing_envs = 0.0
                command.ranges.lin_vel_x = (0.8, 1.3)
                command.ranges.ang_vel_z = (-1.5, 1.5)
            if "hand_load" in apply:
                self.actions.disturbance.hand_load_mass = (3.0, 3.0)
                self.actions.disturbance.hand_load_probability = 1.0
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


def _pop(argv: list[str], flag: str, default=None):
    if flag not in argv:
        return argv, default
    i = argv.index(flag)
    return argv[:i] + argv[i + 2 :], argv[i + 1]


def main(argv):
    argv, condition = _pop(argv, "--condition")
    argv, walker = _pop(argv, "--walker", "city")
    assert condition in CONDITIONS + (REACH_CONDITIONS if walker == "reach" else ()), condition
    name, runner = ("Reach", "G1WalkReachPPORunnerCfg") if walker == "reach" else ("City", "G1WalkCityPPORunnerCfg")
    task = f"G1Patrol-Walk-{name}-Stress-{condition}"
    gym.register(id=task, entry_point="isaaclab.envs:ManagerBasedRLEnv", disable_env_checker=True,
                 kwargs={"env_cfg_entry_point": stressed_cfg(condition, walker),
                         "rsl_rl_cfg_entry_point": f"g1_patrol.tasks.locomotion.config.g1.agents.rsl_rl_ppo_cfg:{runner}"})
    evaluate.main(["--task", task, *argv])


if __name__ == "__main__":
    main(sys.argv[1:])
