"""Sensor bias drawn anew in every episode.

Isaac Lab 3.0's :class:`~isaaclab.utils.noise.NoiseModelWithAdditiveBias` resets the bias by applying the bias
distribution to the current bias, ``bias = func(bias)``; with an additive distribution the bias therefore performs a
random walk over the episodes of an environment instead of being redrawn. After a few hundred resets the IMU and encoder
offsets of the walker's observations exceed their configured ranges many times over, the walker falls, the fall resets
the episode and the bias grows further: in navigation training, where episodes are short, robots ended up falling within
a second of every reset. :class:`PerEpisodeAdditiveBias` draws the bias from the distribution at every reset.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

from isaaclab.utils import configclass
from isaaclab.utils.noise import NoiseModelWithAdditiveBias, NoiseModelWithAdditiveBiasCfg


class PerEpisodeAdditiveBias(NoiseModelWithAdditiveBias):
    """:class:`NoiseModelWithAdditiveBias` whose bias is drawn from ``bias_noise_cfg`` at every reset."""

    def reset(self, env_ids: Sequence[int] | None = None):
        if env_ids is None:
            env_ids = slice(None)
        self._bias[env_ids] = self._bias_noise_cfg.func(torch.zeros_like(self._bias[env_ids]), self._bias_noise_cfg)


@configclass
class PerEpisodeAdditiveBiasCfg(NoiseModelWithAdditiveBiasCfg):
    """Noise with an additive bias that is drawn anew in every episode (see :class:`PerEpisodeAdditiveBias`)."""

    class_type: type = PerEpisodeAdditiveBias
