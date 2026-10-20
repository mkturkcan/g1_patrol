"""Unitree G1 navigation tasks."""

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##

gym.register(
    id="G1Patrol-Navigate-Crowd",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.crowd_env_cfg:G1CrowdNavEnvCfg",
        "default_agent": "rsl_rl",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1CrowdNavPPORunnerCfg",
    },
)

gym.register(
    id="G1Patrol-Navigate-NYC",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.nyc_env_cfg:G1NycNavEnvCfg",
        "default_agent": "rsl_rl",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1NycTeacherPPORunnerCfg",
    },
)

gym.register(
    id="G1Patrol-Navigate-NYC-Lidar",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.nyc_env_cfg:G1NycNavLidarStudentEnvCfg",
        "default_agent": "rsl_rl",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1NycLidarDistillationRunnerCfg",
    },
)

gym.register(
    id="G1Patrol-Navigate-NYC-Camera",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.nyc_env_cfg:G1NycNavCameraStudentEnvCfg",
        "default_agent": "rsl_rl",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1NycCameraDistillationRunnerCfg",
    },
)

gym.register(
    id="G1Patrol-Navigate-Civic",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.civic_env_cfg:G1CivicNavEnvCfg",
        "default_agent": "rsl_rl",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1CivicTeacherPPORunnerCfg",
    },
)

gym.register(
    id="G1Patrol-Navigate-Civic-Lidar",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.civic_env_cfg:G1CivicNavLidarStudentEnvCfg",
        "default_agent": "rsl_rl",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1CivicLidarDistillationRunnerCfg",
    },
)

gym.register(
    id="G1Patrol-Navigate-Civic-Camera",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.civic_env_cfg:G1CivicNavCameraStudentEnvCfg",
        "default_agent": "rsl_rl",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1CivicCameraDistillationRunnerCfg",
    },
)
