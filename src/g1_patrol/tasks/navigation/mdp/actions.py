"""Action term that drives a frozen walking policy with velocity commands."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.utils import configclass

from isaaclab_tasks.contrib.navigation.mdp.pre_trained_policy_action import PreTrainedPolicyAction
from isaaclab_tasks.contrib.navigation.mdp.pre_trained_policy_action_cfg import PreTrainedPolicyActionCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class WalkerPolicyAction(PreTrainedPolicyAction):
    """:class:`PreTrainedPolicyAction` that keeps commands inside the range the walker was trained on and feeds it
    noise-free observations."""

    cfg: WalkerPolicyActionCfg

    def __init__(self, cfg: WalkerPolicyActionCfg, env: ManagerBasedRLEnv):
        cfg.low_level_observations.enable_corruption = False
        super().__init__(cfg, env)
        # the walker is frozen: never track gradients through it, even outside torch.inference_mode()
        self.policy = torch.no_grad()(self.policy)
        ranges = torch.tensor(cfg.command_ranges, device=self.device)
        self._low, self._high = ranges[:, 0], ranges[:, 1]

    def process_actions(self, actions: torch.Tensor):
        self._raw_actions[:] = torch.clamp(actions, self._low, self._high)


@configclass
class WalkerPolicyActionCfg(PreTrainedPolicyActionCfg):
    class_type: type = WalkerPolicyAction
    command_ranges: tuple[tuple[float, float], ...] = ((-0.5, 1.0), (-0.5, 0.5), (-1.0, 1.0))
    """(min, max) of the forward, lateral and yaw-rate commands (m/s, m/s, rad/s)."""
