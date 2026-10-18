"""Tests for the locomotion termination that catches failures of the physics solver."""

from types import SimpleNamespace

import torch

from g1_patrol.tasks.locomotion.mdp.terminations import non_finite_state


def _env(joint_vel: torch.Tensor) -> SimpleNamespace:
    n = joint_vel.shape[0]
    wrap = lambda x: SimpleNamespace(torch=x)  # noqa: E731
    data = SimpleNamespace(root_pos_w=wrap(torch.zeros(n, 3)), root_quat_w=wrap(torch.tensor([[0.0, 0.0, 0.0, 1.0]] * n)),
                           root_lin_vel_w=wrap(torch.zeros(n, 3)), root_ang_vel_w=wrap(torch.zeros(n, 3)),
                           joint_pos=wrap(torch.zeros_like(joint_vel)), joint_vel=wrap(joint_vel))  # fmt: skip
    return SimpleNamespace(scene={"robot": SimpleNamespace(data=data)})


def test_non_finite_state_flags_only_failed_robots():
    joint_vel = torch.zeros(4, 29)
    joint_vel[1, 3] = float("nan")
    joint_vel[2, 0] = float("inf")
    joint_vel[3, 5] = 5000.0  # far beyond any motor's speed limit
    assert non_finite_state(_env(joint_vel)).tolist() == [False, True, True, True]
    assert not non_finite_state(_env(torch.full((2, 29), 20.0))).any()
