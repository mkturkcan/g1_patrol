# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Isaac Lab tasks for training a Unitree G1 humanoid, building towards outdoor patrol."""


def register_tasks() -> None:
    """Register this project's tasks.

    ``isaaclab train`` and ``isaaclab play`` find them through the ``isaaclab.tasks`` entry point; launchers that do
    not, such as ``isaaclab train_multigpu``, need ``--external_callback g1_patrol.register_tasks``.
    """
    import g1_patrol.tasks  # noqa: F401
