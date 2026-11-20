# MTech IIT Madras || System Engineering for Deep Learning || Assignment 2
# Created by Tanuj Gupta (CS25M540)
"""Q2-Q4: compress the trained MobileNetV2, evaluate it, measure its in-memory size, log to W&B.

Same interface as the template repository:
    python test.py --weight_quant_bits 8 --activation_quant_bits 8

Steps: load fp32 checkpoint -> fold BN + quantize -> calibrate activation ranges on training
images -> (optional) QAT fine-tuning -> bit-pack weights -> test accuracy + size report.
"""
import argparse
import json
import os
import time

import torch
import wandb

from compression.qmodel import calibrate, compress_model, describe_layers, finalize, quant_layers
from compression.size import (MB, activation_breakdown, compressed_breakdown, float_breakdown,
                              measure_cuda_bytes, resident_bytes)
from mbv2.data import get_calibration_loader, get_loaders
from mbv2.engine import evaluate, train_one_epoch
from mbv2.utils import get_device, load_float_model, set_seed


def parse_args():
    p = argparse.ArgumentParser(description="Compress MobileNetV2 and evaluate it")
    p.add_argument("--checkpoint", default="checkpoints/baseline/best.pt")
    p.add_argument("--weight_quant_bits", type=int, default=8)
    p.add_argument("--activation_quant_bits", type=int, default=8)
    p.add_argument("--first_last_bits", type=int, default=8,
                   help="weight bits of the stem conv and classifier; 0 = same as --weight_quant_bits")
    p.add_argument("--dw_bits", type=int, default=0, help="weight bits of depthwise convs; 0 = same as weights")
    p.add_argument("--input_bits", type=int, default=8, help="bits for the input image")
    p.add_argument("--granularity", default="per_channel", choices=["per_channel", "per_tensor"])
    p.add_argument("--scheme", default="symmetric", choices=["symmetric", "asymmetric"],
                   help="weight quantization scheme (activations are always asymmetric)")
    p.add_argument("--weight_clip", default="mse", choices=["mse", "max"])
    p.add_argument("--act_clip", default="mse", choices=["mse", "percentile", "max"])
    p.add_argument("--calib_images", type=int, default=2048)
    p.add_argument("--qat_epochs", type=int, default=0, help="quantization-aware fine-tuning epochs (0 = PTQ only)")
    p.add_argument("--qat_lr", type=float, default=0.005)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--data_dir", default="./data")
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--save_dir", default=None, help="save the packed model + report here (default: runs/<run name>)")
    p.add_argument("--project", default="CS6886W-SEDL-Assignment2-MobileNetV2")
    p.add_argument("--no_wandb", action="store_true")
    return p.parse_args()


def qat(qmodel, train_loader, val_loader, device, epochs, lr):
    """Fine-tunes the BN-folded weights/biases through the fake quantizers (STE), activation ranges fixed."""
    params = [p for p in qmodel.parameters() if p.requires_grad]
    opt = torch.optim.SGD(params, lr=lr, momentum=0.9, nesterov=True, weight_decay=0)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs * len(train_loader))
    for ep in range(1, epochs + 1):
        for layer in quant_layers(qmodel):  # follow the weights as they move
            layer.recompute_qparams()
        t0 = time.time()
        tl, ta = train_one_epoch(qmodel, train_loader, opt, device, sched, label_smoothing=0.1)
        vl, va = evaluate(qmodel, val_loader, device)
        wandb.log({"qat_epoch": ep, "qat_train_loss": tl, "qat_val_acc": va})
        print(f"QAT epoch {ep} | train loss {tl:.4f} acc {ta:.4f} | val acc {va:.4f} | {time.time() - t0:.1f}s",
              flush=True)


def main():
    args = parse_args()
    wb, ab = args.weight_quant_bits, args.activation_quant_bits
    fl_bits = args.first_last_bits or wb
    dw_bits = args.dw_bits or wb
    run = wandb.init(project=args.project, job_type="compress", config=vars(args),
                     mode="disabled" if args.no_wandb else None)
    run.name = f"w{wb}-a{ab}-fl{fl_bits}-dw{dw_bits}" + (f"-qat{args.qat_epochs}" if args.qat_epochs else "")
    set_seed(args.seed)
    device = get_device()

    # ---- fp32 baseline
    fmodel, train_cfg = load_float_model(args.checkpoint, device)
    train_loader, val_loader, test_loader = get_loaders(args.data_dir, args.batch_size, seed=args.seed,
                                                        num_workers=args.num_workers)
    _, base_acc = evaluate(fmodel, test_loader, device)
    fb = float_breakdown(fmodel)
    print(f"fp32 model: test acc {base_acc:.4f} | {fb['total'] / MB:.3f} MB in memory")

    # ---- compress
    qmodel = compress_model(fmodel, wb, ab, first_last_bits=fl_bits, dw_bits=dw_bits, input_bits=args.input_bits,
                            per_channel=args.granularity == "per_channel",
                            symmetric=args.scheme == "symmetric",
                            weight_clip=args.weight_clip, act_clip=args.act_clip).to(device)
    calib = get_calibration_loader(args.data_dir, args.calib_images, seed=args.seed, num_workers=args.num_workers)
    calibrate(qmodel, calib, device)
    _, ptq_acc = evaluate(qmodel, test_loader, device)
    print(f"PTQ (fake-quant) test acc {ptq_acc:.4f}")
    if args.qat_epochs:
        qat(qmodel, train_loader, val_loader, device, args.qat_epochs, args.qat_lr)

    # ---- pack weights; evaluate the packed model (this is the model whose memory we report)
    finalize(qmodel)
    _, q_acc = evaluate(qmodel, test_loader, device)
    qmodel.cpu()
    fmodel.cpu()

    cb = compressed_breakdown(qmodel)
    act = activation_breakdown(qmodel, "cpu")
    fp32_cuda, q_cuda = measure_cuda_bytes(fmodel), measure_cuda_bytes(qmodel)
    weight_fp32 = fb["weights"]
    weight_q = cb["weights_packed"] + cb["weight_meta"]
    results = {
        "weight_quant_bits": wb, "activation_quant_bits": ab, "first_last_bits": fl_bits, "dw_bits": dw_bits,
        "qat_epochs": args.qat_epochs,
        "baseline_acc": 100 * base_acc, "ptq_acc": 100 * ptq_acc, "quantized_acc": 100 * q_acc,
        "acc_drop": 100 * (base_acc - q_acc),
        "baseline_size_mb": fb["total"] / MB, "model_size_mb": cb["total"] / MB,
        "compression_ratio": fb["total"] / cb["total"],
        "weight_size_fp32_mb": weight_fp32 / MB, "weight_size_mb": weight_q / MB,
        "weight_compression_ratio": weight_fp32 / weight_q,
        "act_total_fp32_mb": act["fp32_total"] / MB, "act_total_mb": act["quant_total"] / MB,
        "act_peak_fp32_kb": act["fp32_peak"] / 1024, "act_peak_kb": act["quant_peak"] / 1024,
        "activation_compression_ratio": act["fp32_total"] / act["quant_total"],
        "overhead_weight_meta_kb": cb["weight_meta"] / 1024, "overhead_act_meta_kb": cb["act_meta"] / 1024,
        "bias_kb": cb["bias"] / 1024, "overhead_pct": 100 * (cb["weight_meta"] + cb["act_meta"]) / cb["total"],
        "cuda_fp32_mb": fp32_cuda / MB if fp32_cuda else None, "cuda_quant_mb": q_cuda / MB if q_cuda else None,
    }
    wandb.log(results)
    wandb.run.summary.update(results)

    print(json.dumps(results, indent=2))
    save_dir = args.save_dir or os.path.join("runs", run.name)
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, "results.json"), "w") as f:
        json.dump({"results": results, "fp32_breakdown_bytes": fb, "compressed_breakdown_bytes": cb,
                   "activation_breakdown_bytes": act, "args": vars(args),
                   "layers": [dict(name=n, kind=k, shape=list(s), bits=b, act_quant_after=a)
                              for n, k, s, b, a in describe_layers(qmodel)]}, f, indent=2)
    torch.save(qmodel, os.path.join(save_dir, "compressed_model.pt"))  # whole packed module
    print(f"Saved packed model and results to {save_dir} "
          f"(resident size {resident_bytes(qmodel) / MB:.3f} MB)")
    wandb.finish()


if __name__ == "__main__":
    main()
