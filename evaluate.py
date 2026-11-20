# MTech IIT Madras || System Engineering for Deep Learning || Assignment 2
# Created by Tanuj Gupta (CS25M540)
"""Evaluate a saved model on the CIFAR-10 test set and print its in-memory size.

    python evaluate.py --checkpoint checkpoints/baseline/best.pt          # fp32 baseline
    python evaluate.py --compressed checkpoints/best_compressed/compressed_model.pt   # packed model
"""
import argparse

import torch

from compression.size import MB, resident_bytes
from mbv2.data import get_loaders
from mbv2.engine import evaluate
from mbv2.utils import get_device, load_float_model


def main():
    p = argparse.ArgumentParser()
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--checkpoint", help="fp32 checkpoint written by train.py")
    g.add_argument("--compressed", help="compressed_model.pt written by test.py")
    p.add_argument("--data_dir", default="./data")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num_workers", type=int, default=2)
    args = p.parse_args()

    device = get_device()
    if args.checkpoint:
        model, _ = load_float_model(args.checkpoint, device)
    else:
        model = torch.load(args.compressed, map_location=device, weights_only=False)
    _, val_loader, test_loader = get_loaders(args.data_dir, 128, seed=args.seed, num_workers=args.num_workers)
    _, val_acc = evaluate(model, val_loader, device)
    _, test_acc = evaluate(model, test_loader, device)
    print(f"val acc {100 * val_acc:.2f}% | test acc {100 * test_acc:.2f}% | "
          f"in-memory size {resident_bytes(model) / MB:.3f} MB")


if __name__ == "__main__":
    main()
