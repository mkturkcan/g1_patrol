"""Unitree B2-W navigation tasks."""

import gymnasium as gym

from . import agents

gym.register(
    id="G1Patrol-B2W-Navigate",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.nav_env_cfg:B2WNavEnvCfg",
        "default_agent": "rsl_rl",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:B2WTeacherPPORunnerCfg",
    },
)

gym.register(
    id="G1Patrol-B2W-Navigate-Lidar",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.nav_env_cfg:B2WNavLidarStudentEnvCfg",
        "default_agent": "rsl_rl",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:B2WLidarDistillationRunnerCfg",
    },
)
