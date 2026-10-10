"""Curriculum terms for the locomotion tasks."""

from __future__ import annotations

from typing import TYPE_CHECKING

from isaaclab.envs.mdp import modify_env_param

if TYPE_CHECKING:
    from collections.abc import Sequence

    from isaaclab.envs import ManagerBasedRLEnv


def ramp_push_velocity(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    data: dict,
    initial: float,
    final: float,
    start_step: int,
    end_step: int,
):
    """Grow the planar velocity kick of the ``push_by_setting_velocity`` event from ``initial`` to ``final`` m/s.

    Intended as the ``modify_fn`` of :class:`isaaclab.envs.mdp.modify_term_cfg` with the address of the push event's
    ``params``. The kick ramps linearly between ``start_step`` and ``end_step`` environment steps (one PPO iteration is
    ``num_steps_per_env`` steps). The params dict is updated in place.
    """
    frac = min(max((env.common_step_counter - start_step) / (end_step - start_step), 0.0), 1.0)
    v = initial + frac * (final - initial)
    data["velocity_range"]["x"] = (-v, v)
    data["velocity_range"]["y"] = (-v, v)
    return modify_env_param.NO_CHANGE
