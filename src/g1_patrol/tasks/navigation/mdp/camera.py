"""Camera observations: features of a robot camera's image from a frozen pretrained encoder (DINOv2-small).

Students that see RGB are trained on the plain render of the training environment; after training, their encoder is
aligned to realistic images (Cosmos transfers of the student's own replays) so that the policy, unchanged, sees the
same features on the robot. The features are the encoder's class token and its patch tokens average-pooled over a
coarse grid, so that where things are in the image is kept.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import torch

from isaaclab.utils.string import string_to_callable

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

ENCODER_REPO = "facebook/dinov2-small"
ENCODER_REVISION = "ed25f3a31f01632728cabb09d1542f84ab7b0056"
PATCH = 14


def encoder_path() -> str:
    """Where the encoder's weights are (``$G1_PATROL_ENCODER``, default ``~/.cache/g1_patrol/dinov2_small``);
    downloaded at the pinned revision if missing."""
    path = os.environ.get("G1_PATROL_ENCODER", os.path.expanduser("~/.cache/g1_patrol/dinov2_small"))
    if not os.path.exists(os.path.join(path, "model.safetensors")):
        from huggingface_hub import snapshot_download

        snapshot_download(ENCODER_REPO, revision=ENCODER_REVISION, local_dir=path,
                          allow_patterns=["*.json", "*.safetensors"])
    return path


class FrozenImageEncoder(torch.nn.Module):
    """DINOv2-small, frozen, in bfloat16: RGB (N, H, W, 3) uint8 with H and W multiples of 14 -> (N, 384 * (1 + rows *
    cols)) features (class token, then patch tokens pooled to a ``grid`` of rows x cols)."""

    def __init__(self, device: str, grid: tuple[int, int] = (2, 3), chunk: int = 1024):
        super().__init__()
        from transformers import AutoModel

        self.model = AutoModel.from_pretrained(encoder_path(), torch_dtype=torch.bfloat16).to(device).eval()
        self.model.requires_grad_(False)
        self.grid, self.chunk = grid, chunk
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1))

    @torch.no_grad()
    def forward(self, rgb: torch.Tensor) -> torch.Tensor:
        n, h, w = rgb.shape[:3]
        rows, cols = h // PATCH, w // PATCH
        out = []
        for i in range(0, n, self.chunk):
            x = rgb[i : i + self.chunk, ..., :3].permute(0, 3, 1, 2).float() / 255.0
            x = ((x - self.mean) / self.std).to(torch.bfloat16)
            tokens = self.model(pixel_values=x).last_hidden_state  # (B, 1 + rows * cols, 384)
            cls = tokens[:, 0]
            patches = tokens[:, 1 : 1 + rows * cols].reshape(len(x), rows, cols, -1).permute(0, 3, 1, 2)
            pooled = torch.nn.functional.adaptive_avg_pool2d(patches.float(), self.grid).flatten(1)
            out.append(torch.cat([cls.float(), pooled], dim=1))
        return torch.cat(out)


_ENCODERS: dict[str, FrozenImageEncoder] = {}


def camera_features(
    env: ManagerBasedRLEnv, sensor_name: str = "front_rgb", grid: tuple[int, int] = (2, 3), before_render: str | None = None
) -> torch.Tensor:
    """Frozen-encoder features of the camera ``sensor_name``'s RGB image (see :class:`FrozenImageEncoder`).
    ``before_render`` ("module:function", called with the environment) updates what is drawn before the camera renders
    (the camera renders when its image is read)."""
    if before_render is not None:
        string_to_callable(before_render)(env)
    if env.device not in _ENCODERS:
        _ENCODERS[env.device] = FrozenImageEncoder(env.device, grid)
    return _ENCODERS[env.device](env.scene[sensor_name].data.output["rgb"])
