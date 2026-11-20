# MTech IIT Madras || System Engineering for Deep Learning || Assignment 2
# Created by Tanuj Gupta (CS25M540)
"""Q1: train the uncompressed (fp32) MobileNet-v2 baseline on CIFAR-10 and log it to W&B.

Example:
    python train.py --epochs 200 --save_dir checkpoints/baseline
"""
import argparse
import math
import os
import time

import torch
import wandb

from mbv2.data import get_loaders
from mbv2.engine import evaluate, train_one_epoch
from mbv2.model import MobileNetV2
from mbv2.utils import get_device, save_history, set_seed


def parse_args():
    p = argparse.ArgumentParser(description="Train MobileNet-v2 on CIFAR-10")
    # Model
    p.add_argument("--width_mult", type=float, default=1.0)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--bn_momentum", type=float, default=0.1)
    # Optimisation
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--weight_decay", type=float, default=5e-4, help="applied to conv/linear weights only")
    p.add_argument("--warmup_epochs", type=int, default=2)
    p.add_argument("--label_smoothing", type=float, default=0.1)
    p.add_argument("--no_cutout", action="store_true", help="disable Cutout (RandomErasing) augmentation")
    p.add_argument("--no_amp", action="store_true", help="disable mixed precision on CUDA")
    # Reproducibility / IO
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no_deterministic", action="store_true", help="allow non-deterministic (faster) cuDNN")
    p.add_argument("--data_dir", default="./data")
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--save_dir", default="checkpoints/baseline")
    p.add_argument("--project", default="CS6886W-SEDL-Assignment2-MobileNetV2")
    p.add_argument("--no_wandb", action="store_true")
    return p.parse_args()


def param_groups(model, weight_decay):
    """Weight decay on conv/linear weights only; none on BatchNorm parameters or biases."""
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        (no_decay if p.ndim <= 1 else decay).append(p)
    return [{"params": decay, "weight_decay": weight_decay}, {"params": no_decay, "weight_decay": 0.0}]


def warmup_cosine(total_steps, warmup_steps):
    """LR multiplier: linear warm-up, then cosine decay to 0 (stepped every batch)."""
    def fn(step):
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * progress))
    return fn


def main():
    args = parse_args()
    run = wandb.init(project=args.project, job_type="train", config=vars(args),
                     mode="disabled" if args.no_wandb else None)
    run.name = f"baseline-w{args.width_mult:g}-ep{args.epochs}-lr{args.lr:g}"

    set_seed(args.seed, deterministic=not args.no_deterministic)
    device = get_device()
    print(f"Device: {device} | config: {vars(args)}")

    train_loader, val_loader, test_loader = get_loaders(args.data_dir, args.batch_size, seed=args.seed,
                                                        num_workers=args.num_workers, cutout=not args.no_cutout)
    model = MobileNetV2(10, args.width_mult, args.dropout, args.bn_momentum).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"MobileNetV2: {n_params / 1e6:.3f}M parameters")
    wandb.run.summary["num_params"] = n_params

    optimizer = torch.optim.SGD(param_groups(model, args.weight_decay), lr=args.lr,
                                momentum=args.momentum, nesterov=True)
    steps = args.epochs * len(train_loader)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, warmup_cosine(steps, args.warmup_epochs * len(train_loader)))
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" and not args.no_amp else None

    os.makedirs(args.save_dir, exist_ok=True)
    ckpt_path = os.path.join(args.save_dir, "best.pt")
    history = {k: [] for k in ("train_loss", "train_acc", "val_loss", "val_acc")}
    best_val_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, device, scheduler,
                                                args.label_smoothing, scaler)
        val_loss, val_acc = evaluate(model, val_loader, device)
        for k, v in zip(history, (train_loss, train_acc, val_loss, val_acc)):
            history[k].append(v)
        wandb.log({"epoch": epoch, "train_loss": train_loss, "train_acc": train_acc, "val_loss": val_loss,
                   "val_acc": val_acc, "lr": optimizer.param_groups[0]["lr"]})

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({"model": model.state_dict(), "config": vars(args), "epoch": epoch, "val_acc": val_acc},
                       ckpt_path)
        print(f"epoch {epoch:3d} | train loss {train_loss:.4f} acc {train_acc:.4f} | "
              f"val loss {val_loss:.4f} acc {val_acc:.4f} | {time.time() - t0:.1f}s", flush=True)

    # Test accuracy of the best-validation checkpoint (the test set is used only here)
    model.load_state_dict(torch.load(ckpt_path, map_location=device)["model"])
    test_loss, test_acc = evaluate(model, test_loader, device)
    wandb.run.summary.update({"best_val_acc": best_val_acc, "test_acc": test_acc, "test_loss": test_loss})
    save_history(history, args.save_dir)
    if not args.no_wandb and os.path.exists(os.path.join(args.save_dir, "curves.png")):
        wandb.log({"curves": wandb.Image(os.path.join(args.save_dir, "curves.png"))})
    print(f"Best val acc {best_val_acc:.4f} | test acc {test_acc:.4f} | saved to {ckpt_path}")
    wandb.finish()


if __name__ == "__main__":
    main()
