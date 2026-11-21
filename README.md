<!-- MTech IIT Madras || System Engineering for Deep Learning || Assignment 2 -->
<!-- Created by Tanuj Gupta (CS25M540) -->
# MobileNet-v2 on CIFAR-10: training and compression

CS6886W Systems Engineering for Deep Learning, Assignment 2. Name: Tanuj. Roll no: CS25M540.

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tanujgupta12/CS25M540_SEDL_Assignment2_MobileNetV2/blob/main/colab_run.ipynb)

This repo trains MobileNet-v2 on CIFAR-10 and then compresses its weights and activations with hand-written quantization. It uses no `torch.quantization`, `torch.ao` or other compression library. Every size reported here is the model's **in-memory** size: the bytes of the tensors that stay resident in RAM/VRAM. It is never the file size on disk.

## Repository layout

```
mbv2/                 training side
  model.py            MobileNetV2 for 32x32 inputs (CIFAR strides), width multiplier, dropout
  data.py             CIFAR-10 normalisation, augmentation, fixed 45k/5k split, calibration loader
  engine.py           train_one_epoch() (AMP optional) and evaluate()
  utils.py            seeding, device, checkpoint loading, loss/accuracy curves
compression/          compression side (all hand-written)
  quant.py            weight/activation quantizers, MSE clipping, STE, bit packing
  qmodel.py           BN folding, layer replacement, calibration, QAT hooks, packing (finalize)
  size.py             in-memory size accounting: weights, metadata, activations, CUDA check
train.py              Q1: train the fp32 baseline
test.py               Q2-Q4: compress + evaluate + size report + W&B logging (template interface)
evaluate.py           evaluate an fp32 checkpoint or a packed compressed model
wandb_results.py      export sweep runs, pick the best one, offline parallel-coordinates plot
sweeps/ptq.yaml       30-run grid: weight bits x activation bits, post-training quantization
sweeps/qat.yaml       12-run grid: low-bit settings with 10 epochs of QAT
run_sweeps.sh         creates and runs both sweeps
colab_run.ipynb       one-click Colab launcher (all logic is in the .py files)
results/              results.csv (all 42 compression runs), parallel_coords.png, environment.txt
```

## Setup

| | Version |
|---|---|
| Python | 3.13.16 |
| PyTorch / torchvision | 2.14.1+cu130 / 0.29.1+cu130 (CUDA 13.0) |
| wandb | 0.28.1 |
| matplotlib | 3.10.0 |
| numpy | 2.4.6 |
| Hardware | Google Colab, 1x NVIDIA Tesla T4. The code also runs on CPU and Apple MPS. |

These are the versions recorded by W&B for the reported runs (`results/environment.txt` has the `pip freeze` / Python / GPU record from the Colab session). `requirements.txt` gives the minimum versions.

```bash
pip install -r requirements.txt
wandb login
```

CIFAR-10 downloads itself into `./data` on the first run.

## Seed

Every script takes `--seed` (default 42). The seed is applied to Python, NumPy and PyTorch. It also fixes the 45k/5k train/validation split and which images are used for calibration. cuDNN runs in deterministic mode (`train.py --no_deterministic` turns that off for speed).

## How to run

Run these from the repo root.

### Q1: train the fp32 baseline

```bash
python train.py --epochs 200 --save_dir checkpoints/baseline
python evaluate.py --checkpoint checkpoints/baseline/best.pt
```

### Q2/Q3: compress one configuration

```bash
python test.py --weight_quant_bits 8 --activation_quant_bits 8
python test.py --weight_quant_bits 4 --activation_quant_bits 4 --qat_epochs 10
```

Useful flags (defaults in brackets): `--first_last_bits` [8] sets the weight bits of the stem conv and the classifier. `--dw_bits` [same as weights] sets the bits of the depthwise convs. `--scheme` [symmetric] can also be asymmetric. `--granularity` [per_channel] can also be per_tensor. `--weight_clip` [mse] can also be max. `--act_clip` [mse] can also be percentile or max. `--calib_images` [2048]. `--qat_epochs` [0]. `--no_wandb`.

Each run writes `runs/<name>/results.json` with the full size breakdown and the per-layer bit widths. It also writes `runs/<name>/compressed_model.pt`, the packed model, which can be checked with:

```bash
python evaluate.py --compressed runs/<name>/compressed_model.pt
```

### Q3: run all the sweeps (42 runs)

```bash
bash run_sweeps.sh CS6886W-SEDL-Assignment2-MobileNetV2
python wandb_results.py export --out results.csv
python wandb_results.py plot --out parallel_coords.png
python wandb_results.py best --max_drop 1.0
```

To get the W&B parallel coordinates chart, open the project, click **Add panel → Parallel coordinates**, and choose the columns `activation_quant_bits`, `weight_quant_bits`, `compression_ratio`, `model_size_mb`, `quantized_acc`.

## Method

### Baseline (Q1)

- **Data:** the training images get `RandomCrop(32, padding=4)`, `RandomHorizontalFlip`, `Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2470, 0.2435, 0.2616))` and Cutout (`RandomErasing`, p=0.5, one 8x8 patch). Validation and test images are only normalised.
- **Model:** this is torchvision's MobileNetV2 with width multiplier 1.0, 2.24M parameters, dropout 0.2 before the classifier, and BatchNorm with momentum 0.1 and eps 1e-5. For 32x32 inputs, the stem conv and the first t=6 stage use stride 1 instead of 2, so the last feature map is 4x4 instead of 1x1.
- **Training:** SGD with Nesterov momentum 0.9, batch size 128, and lr 0.1 with a 2-epoch linear warm-up followed by per-step cosine decay to 0 over 200 epochs. Weight decay is 5e-4 on conv and linear weights only (BN parameters and biases get none). Label smoothing is 0.1, and mixed precision is used on CUDA. The checkpoint with the best validation accuracy is kept.

### Compression (Q2)

1. **BN folding.** Each conv→BN→ReLU6 becomes a single conv with a bias: `W' = W·γ/√(σ²+ε)` and `b' = β − μγ/√(σ²+ε)`. This removes all BN parameters and running statistics from memory.
2. **Weight quantization.** Weights use uniform b-bit quantization, symmetric by default, with one fp16 scale **per output channel**. The clipping range is chosen per channel by searching 40 shrink factors (α ∈ [0.3, 1]) and keeping the one that minimises the squared quantization error. At 2-4 bits this works much better than plain max scaling, because one outlier weight would otherwise use up most of the levels.
3. **Mixed precision.** The stem conv (3 input channels) and the classifier stay at 8 bits by default. They hold under 1% of the weights but are the most sensitive layers. Depthwise convs can be given their own bit width with `--dw_bits`.
4. **Activation quantization.** Activations use per-tensor asymmetric uint-b quantization with an fp16 scale and a uint8 zero point. Quantizers sit on the input image (fixed at 8 bits, since pixels are 8-bit to begin with), after every ReLU6, after every inverted-residual block (after the skip add), and after global pooling. That is 54 tensors in total. The linear-bottleneck projection output is not quantized separately, because it is added to the skip connection first and then quantized. Ranges are calibrated on 2048 un-augmented *training* images (never test images) with an MSE-optimal clip search.
5. **QAT (optional).** The folded weights and biases are fine-tuned through the fake quantizers with a straight-through estimator: SGD, lr 0.005 with cosine decay, label smoothing 0.1. The activation ranges stay fixed during this step.
6. **Bit packing.** The integer codes are packed densely into `uint8` buffers. That takes `ceil(n·b/8)` bytes per layer for any b from 1 to 8, so 3-bit weights are not padded to 4. The float weights are then deleted. During the forward pass each layer unpacks and dequantizes its weight on the fly, so the resident model really is the compressed size. `test.py` evaluates this packed model, and it gives bit-identical outputs to the fake-quant model.

### Size accounting (Q2c, Q4)

- **Model size** is the sum of the bytes of every parameter and buffer the model holds, with each storage counted once. For the fp32 model that covers the conv/linear weights, BN γ/β, the running mean/var and `num_batches_tracked`. For the compressed model it covers the packed codes, the fp16 weight scales (plus uint8 zero points if asymmetric), the fp32 folded biases, and the activation scales and zero points. `test.py` also reports `torch.cuda.memory_allocated()` before and after moving each model to the GPU, as an empirical check of the VRAM it uses.
- **Overheads** are logged separately: `overhead_weight_meta_kb` for the per-channel scales and zero points, `overhead_act_meta_kb` for the 54 per-tensor activation scales and zero points, and `bias_kb`.
- **Weight compression ratio** is the fp32 conv/linear weight bytes divided by (packed codes + weight scales/zero points).
- **Activation compression ratio:** one 3×32×32 image is passed through the model, and every activation quantizer records the number of elements in the tensor it quantizes. The fp32 cost is 4 bytes per element. The quantized cost is `ceil(n·b/8)` bytes plus that tensor's scale and zero point. The ratio is taken over the sum of all these tensors. The size of the largest single tensor (a lower bound on peak activation memory) is also logged. In this PyTorch simulation, activations are fake-quantized and still held in fp32 in memory, so this ratio is the memory an integer kernel would use to store them at b bits.

## Results

All numbers come from the W&B project `CS6886W-SEDL-Assignment2-MobileNetV2` (seed 42, Colab T4). `results/results.csv` has every compression run and `results/parallel_coords.png` is an offline copy of the parallel coordinates chart.

### Baseline (Q1)

| | Value |
|---|---|
| Epochs / training time | 200 / 2 h 12 min on a T4 |
| Best validation accuracy | 95.32% |
| **Test top-1 accuracy (fp32, uncompressed)** | **94.47%** |
| In-memory size | 8.66 MB (9,083,592 bytes; CUDA allocator: 8.74 MB) |

### Sweep summary (Q3)

42 runs: 30 post-training quantization (PTQ) runs over weight bits {8, 6, 5, 4, 3, 2} x activation bits {8, 6, 4, 3, 2}, and 12 runs with 10 epochs of QAT. The stem conv and classifier stay at 8-bit weights in every run.

| Configuration | Test acc | Drop | Model size | Model ratio | Weight ratio | Activation ratio |
|---|---|---|---|---|---|---|
| W8 A8, PTQ | 94.39% | 0.08 | 2.20 MB | 3.94x | 3.94x | 4.00x |
| W6 A8, PTQ | 94.36% | 0.11 | 1.68 MB | 5.17x | 5.21x | 4.00x |
| W4 A8, PTQ | 93.46% | 1.01 | 1.15 MB | 7.50x | 7.71x | 4.00x |
| W4 A4, QAT | 93.86% | 0.61 | 1.15 MB | 7.50x | 7.71x | 7.98x |
| **W3 A8, QAT (best)** | **94.00%** | **0.47** | **0.89 MB** | **9.69x** | **10.14x** | 4.00x |
| W3 (depthwise 8) A4, QAT | 93.52% | 0.95 | 0.93 MB | 9.30x | 9.69x | 7.98x |
| W2 (depthwise 8) A8, QAT | 92.26% | 2.21 | 0.68 MB | 12.76x | 13.70x | 4.00x |
| W2 A8, QAT | 89.68% | 4.79 | 0.63 MB | 13.69x | 14.81x | 4.00x |

Observations:
- Down to 4-bit weights, PTQ alone keeps accuracy within about 1 point. At 3 bits PTQ falls to 81.1% and at 2 bits to 10% (chance level). 10 epochs of QAT bring 3-bit weights back to 94.0% and 2-bit weights to 89.7-92.3%.
- Activations are the more sensitive side: with PTQ and weights of 4 bits or more, 3-bit activations cost 8.6-12.9 points and 2-bit activations 65.8-72.7 points. With QAT, 4-bit activations cost 0.61-0.95 points (except W3 A4 with 3-bit depthwise convs: 1.32).
- Keeping the depthwise convs at 8 bits costs little memory (0.89 → 0.93 MB at W3) and helps most at 2 bits (89.7% → 92.3%).

### Best configuration (Q4)

"Best" = highest whole-model compression ratio with at most 1 point of accuracy loss: 3-bit weights (stem conv and classifier 8-bit), 8-bit activations, 10 epochs QAT.

```bash
python test.py --weight_quant_bits 3 --activation_quant_bits 8 --first_last_bits 8 --dw_bits 3 --qat_epochs 10
```

| | Value |
|---|---|
| Test accuracy | 94.00% (fp32: 94.47%; drop 0.47 points; PTQ-only at these bits: 81.14%) |
| **Model compression ratio (in memory)** | **9.69x** (8.66 MB → 0.894 MB) |
| **Weight compression ratio** | **10.14x** (8.40 MB → 0.828 MB incl. scales) |
| **Activation compression ratio** | **4.00x** (6.27 MB → 1.57 MB per image, summed over 54 activation tensors) |
| Storage overheads | per-channel fp16 weight scales 33.3 KB, activation scales + zero points 0.16 KB (together 3.7% of the model), plus fp32 folded biases 66.7 KB |
| CUDA allocator check | 8.74 MB → 1.06 MB (the allocator rounds every tensor up to a 512-byte block, and the packed model has many small per-layer tensors, so it reads slightly above the exact tensor bytes) |
| **Final approximate model size** | **0.89 MB** |

**Reproducibility check.** This configuration was re-run from scratch in a separate Colab T4 session (fresh runtime, same `best.pt` baseline, seed 42; W&B run `ewdfy4es`). Every QAT epoch and every number above came out identical: 81.14% → 94.00% test accuracy (94.18% validation), 9.69x / 10.14x / 4.00x, 0.894 MB, 1.06 MB on the CUDA allocator. The saved packed model (`checkpoints/best_compressed/compressed_model.pt`, 1.06 MB on disk) re-evaluates to the same 94.00% with `evaluate.py --compressed`. That rerun is a duplicate of the sweep run `w3-a8-fl8-dw3-qat10`, so it is not included in `results/results.csv`.

If activation compression matters as much as model size, W3 (depthwise 8) A4 with QAT is the best balance: 9.30x model, 9.69x weights, 7.98x activations at 93.52% (drop 0.95).
