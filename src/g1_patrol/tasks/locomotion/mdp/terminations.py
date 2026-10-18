"""Termination terms for the locomotion tasks."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def non_finite_state(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), max_joint_vel: float = 1000.0
) -> torch.Tensor:
    """Robots whose simulated state is not finite, or whose joint speeds are far beyond any motor's limit.

    This is a failure of the physics solver rather than a consequence of the policy's actions: on Newton it happened
    once in about 10^10 robot steps, and a single robot with a NaN state stops training on all GPUs (rsl_rl rejects
    non-finite observations). Use as a time-out termination with :class:`g1_patrol.tasks.locomotion.env.WalkEnv`, which
    zeroes the rewards these robots receive in the step that ends their episode.
    """
    data = env.scene[asset_cfg.name].data
    state = torch.cat(
        (data.root_pos_w.torch, data.root_quat_w.torch, data.root_lin_vel_w.torch, data.root_ang_vel_w.torch,
         data.joint_pos.torch, data.joint_vel.torch),
        dim=-1,
    )  # fmt: skip
    return ~torch.isfinite(state).all(dim=-1) | (data.joint_vel.torch.abs() > max_joint_vel).any(dim=-1)
