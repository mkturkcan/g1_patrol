"""Tests for the warm start of a student with new inputs (the camera student from the fused student)."""

import types

import torch
from rsl_rl.models import CNNModel
from tensordict import TensorDict

from g1_patrol.tasks.navigation.config.g1.agents.distillation import WarmStartDistillation

CNN = dict(output_channels=[8, 8], kernel_size=[3, 3], stride=[2, 2], activation="elu", max_pool=False)


def _model(obs, groups):
    return CNNModel(obs, {"student": groups}, "student", 3, hidden_dims=[32, 16], activation="elu", obs_normalization=True,
                    distribution_cfg={"class_name": "GaussianDistribution", "init_std": 0.1}, cnn_cfg=CNN)


def _act(model, obs):
    with torch.no_grad():
        return model.mlp(model.get_latent(obs))


def test_warm_started_student_acts_as_the_trained_one(tmp_path):
    torch.manual_seed(0)
    n = 64
    obs = TensorDict({"student": torch.randn(n, 5), "camera": 3 * torch.randn(n, 12) + 1,
                      "height_map": torch.randn(n, 2, 16, 16)}, batch_size=[n])
    trained = _model(obs, ["student", "height_map"])
    trained.obs_normalizer.update(obs["student"])  # trained statistics
    for p in trained.parameters():
        p.data.normal_()
    path = tmp_path / "student.pt"
    torch.save({"student_state_dict": trained.state_dict()}, path)

    student = _model(obs, ["student", "camera", "height_map"])
    alg = types.SimpleNamespace(device="cpu", _raw_student=student)
    WarmStartDistillation._warm_start(alg, str(path), new_input_samples=2 * n)
    trained.eval(), student.eval()
    assert torch.allclose(_act(student, obs), _act(trained, obs), atol=1e-6)  # the camera's weights start at zero

    # the trained inputs keep their statistics; the new inputs' are estimated from the first samples, then fixed
    mean = student.obs_normalizer._mean.clone()
    student.obs_normalizer.update(torch.cat([obs["student"], obs["camera"]], dim=-1))
    student.obs_normalizer.update(torch.cat([obs["student"], obs["camera"]], dim=-1) + 100.0)
    norm = student.obs_normalizer
    assert torch.equal(norm._mean[0, :5], mean[0, :5])
    both = torch.cat([obs["camera"], obs["camera"] + 100.0])
    assert torch.allclose(norm._mean[0, 5:], both.mean(0), atol=1e-4)
    assert torch.allclose(norm._var[0, 5:], both.var(0, unbiased=False), rtol=1e-4)
    student.obs_normalizer.update(torch.zeros(n, 17))
    assert torch.allclose(norm._mean[0, 5:], both.mean(0), atol=1e-4)
