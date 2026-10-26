"""Export the B2-W camera student for the robot, with its aligned encoder (``scripts/b2w/camera_align.py``):

* ``encoder.onnx``: the front color camera's image, uint8 (1, 196, 336, 3) RGB, to the policy's camera features
  (1, 2688): the class token and the 2 x 3 average-pooled patch tokens of the aligned DINOv2-small;
* ``policy.onnx``: inputs ``obs`` (1, 14 + 2688), the fused student's observations followed by the camera features,
  and the height map (1, 3, 80, 80); output ``actions`` (1, 3), as the fused student's;
* ``deploy.json``: the fused student's settings (``--fused``) and the camera's (mount, image size, field of view);
* ``samples.npz``: frames, observations and actions of replays (``scripts/b2w/camera_replays.py``), on which both ONNX
  models are checked against PyTorch here (and can be checked on the robot's computer).

The robot stack does not run the camera student yet (``deploy/b2w`` runs the fused student).

Usage:
    python scripts/b2w/export_camera.py --checkpoint <camera student model.pt> --encoder <alignment>/encoder \\
        --replays <alignment>/replays_42 --out deploy/b2w/release_camera
"""

import argparse
import glob
import json
import os

import numpy as np
import torch


class CameraEncoder(torch.nn.Module):
    """uint8 RGB images (N, H, W, 3) -> the policy's camera features (as ``mdp.camera.FrozenImageEncoder``)."""

    def __init__(self, model, grid=(2, 3), patch: int = 14):
        super().__init__()
        self.model, self.grid, self.patch = model, grid, patch
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        n, h, w = image.shape[:3]
        x = (image[..., :3].permute(0, 3, 1, 2).float() / 255.0 - self.mean) / self.std
        tokens = self.model(pixel_values=x).last_hidden_state
        rows, cols = h // self.patch, w // self.patch
        gr, gc = self.grid
        assert rows % gr == 0 and cols % gc == 0, "the patch grid must divide into the pooling grid"
        # adaptive average pooling to the grid, written out (ONNX has no adaptive pooling with a dynamic batch)
        patches = tokens[:, 1 : 1 + rows * cols].reshape(n, gr, rows // gr, gc, cols // gc, -1).mean(dim=(2, 4))
        return torch.cat([tokens[:, 0], patches.permute(0, 3, 1, 2).flatten(1)], dim=1)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", required=True, help="The camera student's checkpoint.")
    parser.add_argument("--encoder", required=True, help="The aligned encoder (camera_align.py's <out>/encoder).")
    parser.add_argument("--replays", required=True, help="Replays for the samples (camera_replays.py's output).")
    parser.add_argument("--fused", default="deploy/b2w/release", help="The fused student's export (its deploy.json).")
    parser.add_argument("--out", required=True)
    parser.add_argument("--samples", type=int, default=1, help="Replays in samples.npz.")
    parser.add_argument("--hfov", type=float, default=90.0,
                        help="Horizontal field of view (deg) of the training camera.")
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)

    import onnxruntime as ort
    from rsl_rl.models import CNNModel
    from tensordict import TensorDict
    from transformers import AutoModel

    from g1_patrol.b2w import OPTICAL_CAMERAS

    clips = sorted(glob.glob(os.path.join(args.replays, "*.npz")))[: args.samples]
    data = [np.load(c) for c in clips]
    frames = np.concatenate([d["front"] for d in data])
    student = np.concatenate([d["student"] for d in data]).astype(np.float32)
    height_map = np.concatenate([d["height_map"] for d in data]).astype(np.float32)

    # encoder
    model = AutoModel.from_pretrained(args.encoder, torch_dtype=torch.float32, attn_implementation="eager")
    encoder = CameraEncoder(model.eval()).eval()
    height, width = frames.shape[1:3]
    dummy = torch.zeros(1, height, width, 3, dtype=torch.uint8)
    torch.onnx.export(encoder, (dummy,), os.path.join(args.out, "encoder.onnx"), input_names=["image"],
                      output_names=["features"], dynamic_axes={"image": {0: "n"}, "features": {0: "n"}},
                      opset_version=17,
                      dynamo=False)
    with torch.no_grad():
        features = torch.cat([encoder(torch.from_numpy(frames[i : i + 32])) for i in range(0, len(frames), 32)]).numpy()

    # policy
    sd = torch.load(args.checkpoint, map_location="cpu", weights_only=False)["student_state_dict"]
    obs0 = TensorDict({"student": torch.zeros(1, student.shape[1]), "camera": torch.zeros(1, features.shape[1]),
                       "height_map": torch.zeros(1, *height_map.shape[1:])}, batch_size=[1])
    model = CNNModel(obs0, {"student": ["student", "camera", "height_map"]}, "student", 3, hidden_dims=[256, 256, 128],
                     activation="elu", obs_normalization=True,
                     distribution_cfg={"class_name": "GaussianDistribution", "init_std": 0.1},
                     cnn_cfg=dict(output_channels=[16, 32, 32, 32], kernel_size=[5, 3, 3, 3], stride=[2, 2, 2, 2],
                                  activation="elu", max_pool=False))
    model.load_state_dict(sd)
    model.eval()
    onnx_policy = model.as_onnx(verbose=False).eval()
    torch.onnx.export(onnx_policy, onnx_policy.get_dummy_inputs(), os.path.join(args.out, "policy.onnx"),
                      input_names=onnx_policy.input_names, output_names=onnx_policy.output_names, opset_version=17,
                      dynamo=False)
    obs = np.concatenate([student, features], axis=1).astype(np.float32)
    with torch.no_grad():
        actions = onnx_policy(torch.from_numpy(obs), torch.from_numpy(height_map)).numpy()

    # checks against PyTorch
    enc_session = ort.InferenceSession(os.path.join(args.out, "encoder.onnx"), providers=["CPUExecutionProvider"])
    pol_session = ort.InferenceSession(os.path.join(args.out, "policy.onnx"), providers=["CPUExecutionProvider"])
    onnx_features = np.concatenate([enc_session.run(None, {"image": frames[i : i + 32]})[0]
                                    for i in range(0, len(frames), 32)])
    names = [i.name for i in pol_session.get_inputs()]
    onnx_actions = np.concatenate([pol_session.run(None, {names[0]: obs[i : i + 1], names[1]: height_map[i : i + 1]})[0]
                                   for i in range(len(obs))])
    check = {"encoder_max_abs_error": float(np.abs(onnx_features - features).max()),
             "policy_max_abs_error": float(np.abs(onnx_actions - actions).max()), "samples": int(len(frames))}
    np.savez_compressed(os.path.join(args.out, "samples.npz"), image=frames, obs=obs, height_map=height_map,
                        actions=actions)

    config = json.load(open(os.path.join(args.fused, "deploy.json")))
    pos, rpy = OPTICAL_CAMERAS["front"]
    config["camera"] = {"link": "f_oc_link", "position": list(pos), "rpy": list(rpy), "width": width, "height": height,
                        "hfov_deg": args.hfov, "features": int(features.shape[1]), "encoder": "encoder.onnx",
                        "image": "RGB uint8 (1, height, width, 3), the camera's view resized (not cropped) to this "
                                 "field of view"}
    config["policy_inputs"] = {"obs": f"the fused student's {student.shape[1]} observations, then the camera features",
                               "height_map": list(height_map.shape[1:])}
    json.dump(config, open(os.path.join(args.out, "deploy.json"), "w"), indent=1)
    print(json.dumps(check), flush=True)
    assert check["encoder_max_abs_error"] < 1e-2 and check["policy_max_abs_error"] < 1e-4, check


if __name__ == "__main__":
    main()
