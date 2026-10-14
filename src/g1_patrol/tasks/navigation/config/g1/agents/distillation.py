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
