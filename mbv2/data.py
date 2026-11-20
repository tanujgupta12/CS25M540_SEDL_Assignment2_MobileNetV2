# MTech IIT Madras || System Engineering for Deep Learning || Assignment 2
# Created by Tanuj Gupta (CS25M540)
"""CIFAR-10 loading: normalisation, augmentation and a fixed 45k/5k train/val split."""
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

# Per-channel statistics of the CIFAR-10 training set
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


def build_transforms(augment=True, cutout=True):
    """Train: random crop (pad 4) + horizontal flip (+ Cutout via RandomErasing). Eval: normalise only."""
    norm = [transforms.ToTensor(), transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)]
    if not augment:
        return transforms.Compose(norm)
    train = [transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip()] + norm
    if cutout:
        # One 8x8-ish patch (6.25% of the image) erased per image, like Cutout(16) at p=0.5
        train.append(transforms.RandomErasing(p=0.5, scale=(0.0625, 0.0625), ratio=(1.0, 1.0), value=0))
    return transforms.Compose(train)


def get_loaders(data_dir, batch_size, seed=42, val_size=5000, num_workers=2, cutout=True):
    """Returns train, validation and test loaders.

    The 50k training images are split once (seeded) into 45k train / 5k validation.
    Validation images use the eval transform. The 10k test set is only used for reporting.
    """
    train_aug = datasets.CIFAR10(data_dir, train=True, download=True, transform=build_transforms(True, cutout))
    train_plain = datasets.CIFAR10(data_dir, train=True, download=True, transform=build_transforms(False))
    test_set = datasets.CIFAR10(data_dir, train=False, download=True, transform=build_transforms(False))

    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(train_aug), generator=g).tolist()
    val_idx, train_idx = perm[:val_size], perm[val_size:]

    pin = torch.cuda.is_available()
    kw = dict(num_workers=num_workers, pin_memory=pin, persistent_workers=num_workers > 0)
    train_loader = DataLoader(Subset(train_aug, train_idx), batch_size=batch_size, shuffle=True,
                              drop_last=True, generator=g, **kw)
    val_loader = DataLoader(Subset(train_plain, val_idx), batch_size=500, shuffle=False, **kw)
    test_loader = DataLoader(test_set, batch_size=500, shuffle=False, **kw)
    return train_loader, val_loader, test_loader


def get_calibration_loader(data_dir, num_images=2048, batch_size=256, seed=42, val_size=5000, num_workers=2):
    """Un-augmented images from the 45k training split, used to calibrate activation ranges.

    Uses training images (never test images) so calibration does not leak test data.
    """
    train_plain = datasets.CIFAR10(data_dir, train=True, download=True, transform=build_transforms(False))
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(train_plain), generator=g).tolist()
    idx = perm[val_size:val_size + num_images]
    return DataLoader(Subset(train_plain, idx), batch_size=batch_size, shuffle=False, num_workers=num_workers)
