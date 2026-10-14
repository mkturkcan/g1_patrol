"""Prepare a PPO checkpoint as the initialization of a new training run.

The output layer of the actor is scaled by ``--scale`` (so that the policy starts away from the saturated region of
the tanh command mapping), the action noise is reset to ``--std`` and the iteration counter to zero.

Usage:
    python scripts/warm_start.py --checkpoint <model.pt> --out <warm.pt> --scale 0.3 --std 0.5
"""

import argparse

import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--scale", type=float, default=0.3, help="Factor applied to the actor's output layer.")
    parser.add_argument("--std", type=float, default=0.5, help="Initial action noise standard deviation.")
    args = parser.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    actor = ckpt["actor_state_dict"]
    layers = sorted({k.rsplit(".", 1)[0] for k in actor if k.startswith("mlp.") and k.endswith(".weight")},
                    key=lambda k: int(k.split(".")[1]))
    last = layers[-1]
    actor[f"{last}.weight"] *= args.scale
    actor[f"{last}.bias"] *= args.scale
    for key in [k for k in actor if k.endswith("std_param")]:
        actor[key] = torch.full_like(actor[key], args.std)
    ckpt["iter"] = 0
    ckpt.setdefault("infos", None)
    torch.save(ckpt, args.out)
    print(f"[INFO] scaled {last} by {args.scale}, std {args.std}: {args.checkpoint} -> {args.out}")


if __name__ == "__main__":
    main()
