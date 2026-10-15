"""Stress tests of a navigation policy: :mod:`evaluate_nav` with conditions outside the training distribution.

Usage:
    python scripts/stress_nav.py --condition sensors --task G1Patrol-Navigate-NYC-Lidar --checkpoint <model.pt>
    conditions: nominal, sensors (2x sensor noise and odometry drift, 0-200 ms latency), pedestrians (20% faster,
    30% fewer pedestrians notice the robot), robot (5 kg payload, 20% weaker motors, 15-25 ms actuation latency,
    slippery ground 0.3), combined
"""

import sys

import gymnasium as gym

import evaluate_nav

CONDITIONS = ("nominal", "sensors", "pedestrians", "robot", "combined")


def stressed_cfg(base_cls, condition: str):
    from isaaclab.utils import configclass

    apply = CONDITIONS[1:-1] if condition == "combined" else (condition,)

    @configclass
    class Stressed(base_cls):
        def __post_init__(self):
            super().__post_init__()
            obs = self.observations
            if "sensors" in apply:
                if hasattr(obs, "student"):
                    for term in ("goal", "base_lin_vel", "base_ang_vel", "projected_gravity"):
                        noise = getattr(obs.student, term).noise
                        for cfg in (getattr(noise, "noise_cfg", None), getattr(noise, "bias_noise_cfg", None), noise):
                            if hasattr(cfg, "n_min"):
                                cfg.n_min, cfg.n_max = 2 * cfg.n_min, 2 * cfg.n_max
                for group in ("lidar_map", "depth_map", "depth"):
                    if hasattr(obs, group):
                        term = getattr(getattr(obs, group), group)
                        term.delay_max_lag = 2
                        for key in ("noise_std", "drift_pos", "drift_yaw", "dropout"):
                            if key in term.params:
                                term.params[key] = 2 * term.params[key]
            if "pedestrians" in apply:
                nyc = self.commands.nyc
                nyc.speed_easy = tuple(1.2 * v for v in nyc.speed_easy)
                nyc.speed_hard = tuple(1.2 * v for v in nyc.speed_hard)
                nyc.aware_easy, nyc.aware_hard = nyc.aware_easy - 0.3, nyc.aware_hard - 0.3
            if "robot" in apply:
                self.events.add_base_mass.params.update(
                    {"mass_distribution_params": (5.0, 5.0), "operation": "add", "distribution": "uniform"}
                )
                self.events.motor_strength.params["strength_range"] = (0.8, 0.8)
                for name in ("legs", "feet"):
                    self.scene.robot.actuators[name].min_delay = 3
                    self.scene.robot.actuators[name].max_delay = 5
                self.events.physics_material.params["static_friction_range"] = (0.3, 0.3)
                self.events.physics_material.params["dynamic_friction_range"] = (0.25, 0.25)

    return Stressed


def main(argv):
    condition = argv[argv.index("--condition") + 1]
    task = argv[argv.index("--task") + 1]
    argv = [a for i, a in enumerate(argv) if a not in ("--condition", "--task") and argv[i - 1] not in ("--condition", "--task")]
    assert condition in CONDITIONS, condition
    import g1_patrol.tasks  # noqa: F401

    spec = gym.spec(task)
    from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

    base_cls = type(load_cfg_from_registry(task, "env_cfg_entry_point"))
    stressed = f"{task}-Stress-{condition}"
    gym.register(id=stressed, entry_point=spec.entry_point, disable_env_checker=True,
                 kwargs={**spec.kwargs, "env_cfg_entry_point": stressed_cfg(base_cls, condition)})
    evaluate_nav.main(["--task", stressed, *argv])


if __name__ == "__main__":
    main(sys.argv[1:])
