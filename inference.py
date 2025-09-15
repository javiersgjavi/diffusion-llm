import argparse

import torch

from engine import LLADAEngine


def main(args):
    torch.set_float32_matmul_precision("medium")

    engine = LLADAEngine.load_from_checkpoint(
        "weights/model.ckpt",
    )

    engine.generate(
        t_steps=args.t_steps, n_tokens=args.n_tokens, sampling=args.sampling
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run inference with LLADAEngine")
    parser.add_argument("--t_steps", type=int, default=512, help="Number of timesteps")
    parser.add_argument(
        "--n_tokens", type=int, default=30, help="Number of tokens to generate"
    )
    parser.add_argument(
        "--sampling", type=str, default="greedy", help="Sampling method"
    )
    args = parser.parse_args()

    main(args)
