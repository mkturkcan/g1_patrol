"""Tests for the per-episode sensor bias."""

import torch

from isaaclab.utils.noise import UniformNoiseCfg

from g1_patrol.noise import PerEpisodeAdditiveBias, PerEpisodeAdditiveBiasCfg


def test_bias_is_redrawn_at_every_reset():
    cfg = PerEpisodeAdditiveBiasCfg(noise_cfg=UniformNoiseCfg(n_min=0.0, n_max=0.0),
                                    bias_noise_cfg=UniformNoiseCfg(n_min=-0.02, n_max=0.02))
    model = cfg.class_type(cfg, num_envs=512, device="cpu")
    assert isinstance(model, PerEpisodeAdditiveBias)
    data = torch.zeros(512, 3)
    model(data)  # the first call sizes the bias per component
    first = model(data).clone()
    for _ in range(200):  # Isaac Lab 3.0's model would add 200 draws: a random walk with a spread of about 0.16
        model.reset()
    biased = model(data)
    assert biased.abs().max() <= 0.02
    assert not torch.equal(biased, first)
    model.reset(env_ids=[0])  # only the reset environments change
    assert torch.equal(model(data)[1:], biased[1:])
