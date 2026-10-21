"""PPO with the action noise held near a target."""

import math

import torch

from rsl_rl.algorithms import PPO


class TargetNoisePPO(PPO):
    """:class:`rsl_rl.algorithms.PPO` whose entropy coefficient follows the action noise.

    After every update the coefficient is multiplied by ``(target_std / mean std) ** entropy_rate`` and kept within
    ``entropy_range``, so that the mean standard deviation of the Gaussian action distribution settles near
    ``target_std``. A fixed coefficient either let the noise collapse (the policy stopped exploring before it had
    learned to deliver litter) or inflate it (noisy commands, more collisions), depending on the phase of training.
    """

    def __init__(
        self,
        *args,
        target_std: float = 0.35,
        entropy_rate: float = 0.02,
        entropy_range: tuple[float, float] = (1.0e-4, 1.0e-2),
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.target_std = target_std
        self.entropy_rate = entropy_rate
        self.entropy_range = tuple(entropy_range)

    def mean_std(self) -> float:
        """Mean standard deviation of the action distribution."""
        distribution = self._raw_actor.distribution
        std = torch.exp(distribution.log_std_param) if distribution.std_type == "log" else distribution.std_param.abs()
        return float(std.detach().mean())

    def update(self) -> dict[str, float]:
        losses = super().update()
        scale = math.exp(self.entropy_rate * math.log(self.target_std / max(self.mean_std(), 1.0e-6)))
        self.entropy_coef = min(max(self.entropy_coef * scale, self.entropy_range[0]), self.entropy_range[1])
        losses["entropy_coef"] = self.entropy_coef
        return losses
