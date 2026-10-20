"""Unitree G1 velocity-tracking ("walk") tasks."""

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##

gym.register(
    id="G1Patrol-Walk-Flat",
    entry_point="g1_patrol.tasks.locomotion.env:WalkEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.walk_env_cfg:G1WalkFlatEnvCfg",
        "default_agent": "rsl_rl",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1WalkFlatPPORunnerCfg",
    },
)

gym.register(
    id="G1Patrol-Walk-Street",
    entry_point="g1_patrol.tasks.locomotion.env:WalkEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.walk_env_cfg:G1WalkStreetEnvCfg",
        "default_agent": "rsl_rl",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1WalkStreetPPORunnerCfg",
    },
)

gym.register(
    id="G1Patrol-Walk-City",
    entry_point="g1_patrol.tasks.locomotion.env:WalkEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.walk_env_cfg:G1WalkCityEnvCfg",
        "default_agent": "rsl_rl",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1WalkCityPPORunnerCfg",
    },
)

gym.register(
    id="G1Patrol-Walk-Reach",
    entry_point="g1_patrol.tasks.locomotion.env:WalkEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.walk_env_cfg:G1WalkReachEnvCfg",
        "default_agent": "rsl_rl",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1WalkReachPPORunnerCfg",
    },
)
