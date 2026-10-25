"""The floor's height below the robot, measured from its own sensor returns (numpy only; runs on the robot).

The B2-W's controller stands the body at a height that is adjustable and not reported reliably, and the navigation
policy's map is built from heights above the floor. Rather than assume a standing height, the robot measures the
floor in every scan: among the returns 0.7-3 m around it (gravity-aligned, relative to the base), the floor is the
lowest height that many returns share -- a histogram peak -- within the heights the body can stand at. A running
estimate follows it; it moves by at most ``max_step`` per scan, so that a large low platform or a step down does not
yank it.
"""

from __future__ import annotations

import numpy as np


class FloorEstimator:
    """Height (m, negative) of the floor below the base in a gravity-aligned frame centred on the base."""

    def __init__(self, prior: float = -0.65, band: tuple[float, float] = (-0.9, -0.4), ring: tuple[float, float] = (0.7, 3.0),
                 bin_size: float = 0.02, share: float = 0.3, min_points: int = 50, gain: float = 0.3, max_step: float = 0.05):
        self.prior, self.band, self.ring = prior, band, ring
        self.bin_size, self.share, self.min_points = bin_size, share, min_points
        self.gain, self.max_step = gain, max_step
        self.value: float | None = None
        self.measured: float | None = None
        """The latest scan's own measurement (None if the scan showed too little floor)."""

    def reset(self):
        self.value, self.measured = None, None

    def measure(self, rel: np.ndarray) -> float | None:
        """The floor in one scan: ``rel`` (k, 3) are returns relative to the base in a gravity-aligned frame (the
        odometry frame's axes, origin at the base)."""
        rel = np.asarray(rel, dtype=np.float64).reshape(-1, 3)
        rel = rel[np.isfinite(rel).all(axis=1)]
        r = np.hypot(rel[:, 0], rel[:, 1])
        z = rel[(r > self.ring[0]) & (r < self.ring[1]) & (rel[:, 2] > self.band[0]) & (rel[:, 2] < self.band[1]), 2]
        if len(z) < self.min_points:
            return None
        edges = np.arange(self.band[0], self.band[1] + self.bin_size, self.bin_size)
        counts, _ = np.histogram(z, edges)
        # the lowest well-populated height: the floor lies below box tops and platforms, and a drop is rarely as large
        k = int(np.nonzero(counts >= self.share * counts.max())[0][0])
        centre = (edges[k] + edges[k + 1]) / 2
        near = z[np.abs(z - centre) < 1.5 * self.bin_size]
        return float(np.median(near))

    def update(self, rel: np.ndarray) -> float:
        """Add a scan; the running estimate (the prior until the floor has been seen)."""
        self.measured = self.measure(rel)
        if self.measured is not None:
            if self.value is None:
                self.value = self.measured
            else:
                step = np.clip(self.measured - self.value, -self.max_step, self.max_step)
                self.value += self.gain * float(step)
        return self.prior if self.value is None else self.value
