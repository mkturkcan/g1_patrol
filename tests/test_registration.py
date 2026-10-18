# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the project's task registrations."""

import gymnasium as gym

import g1_patrol.tasks  # noqa: F401


def test_task_registrations():
    """The project tasks must expose valid environment and agent entry points."""
    expected = {
        "G1Patrol-Walk-Flat": {
            "entry_point": "g1_patrol.tasks.locomotion.env:WalkEnv",
            "env_cfg_entry_point": "g1_patrol.tasks.locomotion.config.g1.walk_env_cfg:G1WalkFlatEnvCfg",
            "default_agent": "rsl_rl",
        },
        "G1Patrol-Walk-Street": {
            "entry_point": "g1_patrol.tasks.locomotion.env:WalkEnv",
            "env_cfg_entry_point": "g1_patrol.tasks.locomotion.config.g1.walk_env_cfg:G1WalkStreetEnvCfg",
            "default_agent": "rsl_rl",
        },
        "G1Patrol-Walk-City": {
            "entry_point": "g1_patrol.tasks.locomotion.env:WalkEnv",
            "env_cfg_entry_point": "g1_patrol.tasks.locomotion.config.g1.walk_env_cfg:G1WalkCityEnvCfg",
            "default_agent": "rsl_rl",
        },
        "G1Patrol-Navigate-NYC": {
            "entry_point": "isaaclab.envs:ManagerBasedRLEnv",
            "env_cfg_entry_point": "g1_patrol.tasks.navigation.config.g1.nyc_env_cfg:G1NycNavEnvCfg",
            "default_agent": "rsl_rl",
        },
        "G1Patrol-Navigate-NYC-Lidar": {
            "entry_point": "isaaclab.envs:ManagerBasedRLEnv",
            "env_cfg_entry_point": "g1_patrol.tasks.navigation.config.g1.nyc_env_cfg:G1NycNavLidarStudentEnvCfg",
            "default_agent": "rsl_rl",
        },
        "G1Patrol-Navigate-NYC-Camera": {
            "entry_point": "isaaclab.envs:ManagerBasedRLEnv",
            "env_cfg_entry_point": "g1_patrol.tasks.navigation.config.g1.nyc_env_cfg:G1NycNavCameraStudentEnvCfg",
            "default_agent": "rsl_rl",
        },
        "G1Patrol-Navigate-Crowd": {
            "entry_point": "isaaclab.envs:ManagerBasedRLEnv",
            "env_cfg_entry_point": "g1_patrol.tasks.navigation.config.g1.crowd_env_cfg:G1CrowdNavEnvCfg",
            "default_agent": "rsl_rl",
        },
    }

    for task_id, expected_values in expected.items():
        spec = gym.spec(task_id)
        assert spec.entry_point == expected_values["entry_point"]
        assert spec.kwargs["env_cfg_entry_point"] == expected_values["env_cfg_entry_point"]
        if "default_agent" in expected_values:
            assert spec.kwargs["default_agent"] == expected_values["default_agent"]
