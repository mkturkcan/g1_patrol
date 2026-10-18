"""Environment class of the locomotion tasks."""

from __future__ import annotations

import torch

from isaaclab.envs import ManagerBasedRLEnv


class _ZeroOnSolverFailure:
    """A reward term that is zero for robots flagged by the ``non_finite_state`` termination in the current step.

    Other attributes (e.g. ``reset`` of class-based terms) are those of the wrapped term.
    """

    def __init__(self, func, env: ManagerBasedRLEnv):
        self._func, self._env = func, env

    def __call__(self, env, **params):
        value = self._func(env, **params)
        failed = self._env.termination_manager.get_term("non_finite_state")
        if isinstance(value, torch.Tensor) and value.shape == failed.shape and failed.any():
            value = torch.where(failed, torch.zeros_like(value), value)
        return value

    def __getattr__(self, name):
        return getattr(self._func, name)


class WalkEnv(ManagerBasedRLEnv):
    """:class:`ManagerBasedRLEnv` that recovers from a failure of the physics solver in single robots.

    The ``non_finite_state`` termination (computed before the rewards) resets a robot whose simulated state is not
    finite. The rewards of that step are still computed from the failed state; they are set to zero for the robot, so
    that one failure does not make the rewards, and with them training, NaN. Elsewhere a NaN is still reported.
    """

    def load_managers(self):
        super().load_managers()
        if "non_finite_state" in self.termination_manager.active_terms:
            for cfg in self.reward_manager._term_cfgs:
                cfg.func = _ZeroOnSolverFailure(cfg.func, self)
