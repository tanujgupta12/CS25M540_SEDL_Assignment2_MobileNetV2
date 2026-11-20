# MTech IIT Madras || System Engineering for Deep Learning || Assignment 2
# Created by Tanuj Gupta (CS25M540)
"""Export the compression runs from W&B, pick the best one and draw a parallel-coordinates figure.

    python wandb_results.py export --out results.csv           # all compression runs -> CSV
    python wandb_results.py best --max_drop 1.0                # flags of the best run (for test.py)
    python wandb_results.py plot --out parallel_coords.png     # offline copy of the W&B chart

"Best" = highest whole-model compression ratio among runs that lose at most --max_drop
accuracy points versus the fp32 baseline (ties broken by accuracy).
"""
import argparse
import csv

import wandb

COLUMNS = ["name", "weight_quant_bits", "activation_quant_bits", "first_last_bits", "dw_bits", "qat_epochs",
           "quantized_acc", "acc_drop", "compression_ratio", "model_size_mb", "weight_compression_ratio",
           "activation_compression_ratio", "weight_size_mb", "overhead_weight_meta_kb", "overhead_act_meta_kb"]
FLAGS = ["weight_quant_bits", "activation_quant_bits", "first_last_bits", "dw_bits", "qat_epochs"]


def fetch(project):
    rows = []
    for r in wandb.Api().runs(project, filters={"jobType": "compress", "state": "finished"}):
        s = r.summary
        if "quantized_acc" not in s:
            continue
        rows.append({"name": r.name, **{c: s.get(c) for c in COLUMNS[1:]}})
    return rows


def best_run(rows, max_drop):
    ok = [r for r in rows if r["acc_drop"] <= max_drop] or rows
    return max(ok, key=lambda r: (round(r["compression_ratio"], 2), r["quantized_acc"]))


def plot(rows, out):
    """Parallel coordinates with matplotlib (each axis min-max normalised, real values as tick labels)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    axes = ["activation_quant_bits", "weight_quant_bits", "compression_ratio", "model_size_mb", "quantized_acc"]
    lo = {a: min(r[a] for r in rows) for a in axes}
    hi = {a: max(r[a] for r in rows) for a in axes}
    norm = lambda a, v: 0.5 if hi[a] == lo[a] else (v - lo[a]) / (hi[a] - lo[a])
    fig, ax = plt.subplots(figsize=(11, 5))
    cmap = plt.get_cmap("viridis")
    acc_lo, acc_hi = lo["quantized_acc"], hi["quantized_acc"]
    for r in sorted(rows, key=lambda r: r["quantized_acc"]):
        c = cmap(norm("quantized_acc", r["quantized_acc"]))
        ax.plot(range(len(axes)), [norm(a, r[a]) for a in axes], color=c, alpha=0.8, lw=1.3)
    for i, a in enumerate(axes):
        ax.axvline(i, color="grey", lw=0.8)
        for t in (0, 0.5, 1):
            ax.text(i + 0.03, t, f"{lo[a] + t * (hi[a] - lo[a]):.3g}", fontsize=8, va="center")
    ax.set_xticks(range(len(axes)), axes)
    ax.set_yticks([])
    ax.set_title(f"MobileNetV2 quantization sweep ({len(rows)} runs; colour = accuracy {acc_lo:.1f}-{acc_hi:.1f}%)")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["export", "best", "plot"])
    p.add_argument("--project", default="CS6886W-SEDL-Assignment2-MobileNetV2")
    p.add_argument("--max_drop", type=float, default=1.0)
    p.add_argument("--out", default=None)
    args = p.parse_args()
    rows = fetch(args.project)
    if args.cmd == "export":
        out = args.out or "results.csv"
        with open(out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=COLUMNS)
            w.writeheader()
            w.writerows(sorted(rows, key=lambda r: -r["compression_ratio"]))
        print(f"wrote {len(rows)} runs to {out}")
    elif args.cmd == "best":
        b = best_run(rows, args.max_drop)
        # Last line = flags for test.py, so a notebook can reuse it
        print(f"# best: {b['name']} acc {b['quantized_acc']:.2f}% ratio {b['compression_ratio']:.2f}x")
        print(" ".join(f"--{k} {b[k]}" for k in FLAGS))
    else:
        plot(rows, args.out or "parallel_coords.png")


if __name__ == "__main__":
    main()
