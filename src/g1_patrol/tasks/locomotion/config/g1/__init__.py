"""Unitree G1 velocity-tracking ("walk") tasks."""

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##

gym.register(
    id="G1Patrol-Walk-Flat",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.walk_env_cfg:G1WalkFlatEnvCfg",
        "default_agent": "rsl_rl",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1WalkFlatPPORunnerCfg",
    },
)

gym.register(
    id="G1Patrol-Walk-Rough",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.walk_env_cfg:G1WalkRoughEnvCfg",
        "default_agent": "rsl_rl",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1WalkRoughPPORunnerCfg",
    },
)

gym.register(
    id="G1Patrol-Walk-Street",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.walk_env_cfg:G1WalkStreetEnvCfg",
        "default_agent": "rsl_rl",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1WalkStreetPPORunnerCfg",
    },
)

gym.register(
    id="G1Patrol-Walk-City",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.walk_env_cfg:G1WalkCityEnvCfg",
        "default_agent": "rsl_rl",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1WalkCityPPORunnerCfg",
    },
)
