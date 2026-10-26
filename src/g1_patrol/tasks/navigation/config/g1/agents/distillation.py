"""Distillation in command space."""

import torch

from rsl_rl.algorithms import Distillation


class CommandSpaceDistillation(Distillation):
    """:class:`rsl_rl.algorithms.Distillation` with the behavior-cloning loss on tanh-squashed actions.

    The navigation actions are mapped to velocity commands by a tanh (:class:`WalkerPolicyAction`), so errors between
    large outputs of the same sign barely change the command. Comparing the squashed actions weights the errors by
    their effect on the command.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        loss_fn = self.loss_fn
        self.loss_fn = lambda student, teacher: loss_fn(torch.tanh(student), torch.tanh(teacher))


class WarmStartDistillation(CommandSpaceDistillation):
    """:class:`CommandSpaceDistillation` with the student starting from a trained student whose inputs are this
    student's without one or more observation groups appended to its one-dimensional inputs (``student_init``: the
    trained student's distillation checkpoint). Its weights are copied and the weights of the new inputs start at zero,
    so that the student starts out acting as the trained one. The input normalization keeps the trained student's
    statistics; the new inputs' are estimated from the first ``new_input_samples`` samples, and all are then fixed."""

    def __init__(self, *args, student_init: str | None = None, new_input_samples: int = 8192, **kwargs):
        super().__init__(*args, **kwargs)
        if student_init:
            self._warm_start(student_init, new_input_samples)

    @torch.no_grad()
    def _warm_start(self, path: str, new_input_samples: int):
        old = torch.load(path, map_location=self.device, weights_only=False)["student_state_dict"]
        new = self._raw_student.state_dict()
        n_old = old["obs_normalizer._mean"].shape[-1]  # one-dimensional inputs of the trained student
        added = new["obs_normalizer._mean"].shape[-1] - n_old
        first = next(k for k in new if k.startswith("mlp.") and k.endswith(".weight"))
        for key, value in old.items():
            if key == first:  # inputs: [trained 1-D inputs, new inputs, CNN features]
                new[key].zero_()
                new[key][:, :n_old] = value[:, :n_old]
                new[key][:, n_old + added :] = value[:, n_old:]
            elif key.startswith("obs_normalizer.") and key != "obs_normalizer.count":
                new[key][..., :n_old] = value
            else:
                new[key].copy_(value)
        self._raw_student.load_state_dict(new)
        norm = self._raw_student.obs_normalizer
        new_inputs = slice(n_old, n_old + added)
        stats = {"n": 0, "mean": None, "m2": None}

        @torch.no_grad()
        def update(x: torch.Tensor):  # replaces the normalizer's running update
            if stats["n"] >= new_input_samples:
                return
            x = x[:, new_inputs].float()
            n_x, mean_x = x.shape[0], x.mean(0, keepdim=True)
            m2_x = ((x - mean_x) ** 2).sum(0, keepdim=True)
            if stats["mean"] is None:
                stats["mean"], stats["m2"] = mean_x, m2_x
            else:  # Chan et al.'s parallel update of mean and sum of squares
                n = stats["n"]
                delta = mean_x - stats["mean"]
                stats["mean"] = stats["mean"] + delta * n_x / (n + n_x)
                stats["m2"] = stats["m2"] + m2_x + delta**2 * n * n_x / (n + n_x)
            stats["n"] += n_x
            norm._mean[:, new_inputs] = stats["mean"]
            norm._var[:, new_inputs] = stats["m2"] / stats["n"]
            norm._std[:, new_inputs] = torch.sqrt(norm._var[:, new_inputs])

        norm.update = update
        print(f"[WarmStartDistillation] student from {path}: {added} new inputs (weights zero)", flush=True)
