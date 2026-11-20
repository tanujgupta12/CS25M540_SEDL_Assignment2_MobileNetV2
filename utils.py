# MTech IIT Madras || System Engineering for Deep Learning || Assignment 2
# Created by Tanuj Gupta (CS25M540)
"""Seeding, device selection, checkpoint loading and local plotting helpers."""
import json
import os
import random

import numpy as np
import torch

from .model import MobileNetV2


def set_seed(seed, deterministic=True):
    """Seeds Python, NumPy and PyTorch; optionally makes cuDNN deterministic."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_float_model(checkpoint, device="cpu"):
    """Rebuilds the trained fp32 MobileNetV2 from a checkpoint written by train.py."""
    ckpt = torch.load(checkpoint, map_location=device)
    cfg = ckpt["config"]
    model = MobileNetV2(num_classes=10, width_mult=cfg["width_mult"], dropout=cfg["dropout"])
    model.load_state_dict(ckpt["model"])
    return model.to(device), cfg


def save_history(history, out_dir):
    """Writes per-epoch metrics to history.json and draws loss/accuracy curves to curves.png."""
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=2)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed: skipping curves.png (history.json was written)")
        return

    epochs = range(1, len(history["train_loss"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    ax1.plot(epochs, history["train_loss"], label="train")
    ax1.plot(epochs, history["val_loss"], label="validation")
    ax1.set(title="Loss", xlabel="epoch", ylabel="cross-entropy")
    ax2.plot(epochs, history["train_acc"], label="train")
    ax2.plot(epochs, history["val_acc"], label="validation")
    ax2.set(title="Top-1 accuracy", xlabel="epoch", ylabel="accuracy")
    for ax in (ax1, ax2):
        ax.grid(alpha=0.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "curves.png"), dpi=150)
    plt.close(fig)
