# MTech IIT Madras || System Engineering for Deep Learning || Assignment 2
# Created by Tanuj Gupta (CS25M540)
"""One epoch of training and a full pass of evaluation (shared by training and QAT fine-tuning)."""
import torch
import torch.nn.functional as F


def train_one_epoch(model, loader, optimizer, device, scheduler=None, label_smoothing=0.0,
                    scaler=None, on_step=None):
    """Trains for one epoch; `scheduler` (if given) is stepped every batch.

    scaler: a torch.amp.GradScaler to train in mixed precision on CUDA, or None for fp32.
    on_step(batch_loss, batch_acc) is called after every batch.
    """
    model.train()
    total_loss, correct, seen = 0.0, 0, 0
    use_amp = scaler is not None
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=use_amp):
            out = model(x)
            loss = F.cross_entropy(out, y, label_smoothing=label_smoothing)
        if use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        if scheduler is not None:
            scheduler.step()

        batch_correct = (out.argmax(1) == y).sum().item()
        total_loss += loss.item() * y.size(0)
        correct += batch_correct
        seen += y.size(0)
        if on_step:
            on_step(loss.item(), batch_correct / y.size(0))
    return total_loss / seen, correct / seen


@torch.no_grad()
def evaluate(model, loader, device):
    """Returns (mean cross-entropy, top-1 accuracy) over the loader."""
    model.eval()
    total_loss, correct, seen = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        out = model(x).float()
        total_loss += F.cross_entropy(out, y, reduction="sum").item()
        correct += (out.argmax(1) == y).sum().item()
        seen += y.size(0)
    return total_loss / seen, correct / seen
