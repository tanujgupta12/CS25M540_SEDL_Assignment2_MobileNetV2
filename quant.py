# MTech IIT Madras || System Engineering for Deep Learning || Assignment 2
# Created by Tanuj Gupta (CS25M540)
"""Hand-written quantization primitives (no torch.quantization / compression library is used).

* Weight quantization: uniform, per-output-channel (or per-tensor), symmetric or asymmetric,
  with an MSE-optimal clipping range searched per channel.
* Activation quantization: uniform, per-tensor, asymmetric, with ranges calibrated on
  training images (min/max, percentile or MSE-optimal clipping).
* Straight-through estimator (STE) so the same fake-quant ops can be fine-tuned (QAT).
* Bit packing: b-bit integer codes are packed densely into a uint8 buffer, so a 4-bit
  weight really occupies half a byte in RAM/VRAM.
"""
import math

import torch
import torch.nn as nn

SCALE_DTYPE = torch.float16  # scales are stored in fp16 (2 bytes each)
MIN_SCALE = 1e-6             # smallest scale; representable in fp16 (as a subnormal)


# ----------------------------------------------------------------------------- helpers
def ste_round(x):
    """round() in the forward pass, identity gradient in the backward pass."""
    return x + (torch.round(x) - x).detach()


def qrange(bits, symmetric):
    """Integer range of the codes: symmetric -> [-(2^(b-1)-1), 2^(b-1)-1], asymmetric -> [0, 2^b-1]."""
    if symmetric:
        q = 2 ** (bits - 1) - 1
        return -q, q
    return 0, 2 ** bits - 1


# ----------------------------------------------------------------------------- weights
def _channel_view(w, per_channel):
    """Reshapes a weight to (channels, -1) for per-channel stats, or (1, -1) for per-tensor."""
    return w.reshape(w.shape[0], -1) if per_channel else w.reshape(1, -1)


def _qparams_from_range(lo, hi, bits, symmetric):
    """Scale / zero point for a clipping range [lo, hi] (tensors of shape (C,))."""
    qmin, qmax = qrange(bits, symmetric)
    if symmetric:
        scale = torch.maximum(hi.abs(), lo.abs()) / qmax
        zp = torch.zeros_like(scale)
    else:
        lo, hi = torch.clamp(lo, max=0), torch.clamp(hi, min=0)  # always represent 0 exactly
        scale = torch.clamp((hi - lo) / (qmax - qmin), min=MIN_SCALE)
        zp = torch.clamp(torch.round(-lo / scale), qmin, qmax)
    # Round to the stored fp16 precision. The floor must be representable in fp16 (1e-8 would
    # underflow to 0 and give 0/0 = NaN for all-zero channels/tensors, e.g. dead ReLU6 outputs).
    scale = torch.clamp(scale, min=MIN_SCALE).to(SCALE_DTYPE).float()
    return scale, zp


def _fake_quant(w2d, scale, zp, bits, symmetric):
    qmin, qmax = qrange(bits, symmetric)
    q = torch.clamp(torch.round(w2d / scale[:, None]) + zp[:, None], qmin, qmax)
    return (q - zp[:, None]) * scale[:, None]


@torch.no_grad()
def weight_qparams(w, bits, per_channel=True, symmetric=True, clip="mse", grid=40):
    """Chooses scale and zero point for a weight tensor.

    clip="max": use the full [min, max] range.
    clip="mse": try shrinking the range by alpha in [0.3, 1] and keep, per channel, the alpha
                with the smallest squared quantization error. At 2-4 bits this matters a lot,
                because a single outlier weight otherwise wastes most of the levels.
    """
    w2d = _channel_view(w.detach().float(), per_channel)
    lo, hi = w2d.min(1).values, w2d.max(1).values
    if clip == "max":
        return _qparams_from_range(lo, hi, bits, symmetric)

    best_err = torch.full_like(lo, float("inf"))
    best_scale, best_zp = None, None
    for alpha in torch.linspace(0.3, 1.0, grid).tolist():
        scale, zp = _qparams_from_range(lo * alpha, hi * alpha, bits, symmetric)
        err = (_fake_quant(w2d, scale, zp, bits, symmetric) - w2d).pow(2).sum(1)
        better = err < best_err
        if best_scale is None:
            best_scale, best_zp = scale.clone(), zp.clone()
        best_err = torch.where(better, err, best_err)
        best_scale = torch.where(better, scale, best_scale)
        best_zp = torch.where(better, zp, best_zp)
    return best_scale, best_zp


def fake_quant_weight(w, scale, zp, bits, symmetric):
    """Quantize-dequantize a weight with STE (gradients flow to w inside the clipping range)."""
    shape = w.shape
    w2d = w.reshape(scale.numel(), -1)
    qmin, qmax = qrange(bits, symmetric)
    q = torch.clamp(ste_round(w2d / scale[:, None]) + zp[:, None], qmin, qmax)
    return ((q - zp[:, None]) * scale[:, None]).reshape(shape)


@torch.no_grad()
def weight_to_codes(w, scale, zp, bits, symmetric):
    """Integer codes in [0, 2^bits - 1] (symmetric codes are offset by 2^(bits-1))."""
    qmin, qmax = qrange(bits, symmetric)
    q = torch.clamp(torch.round(w.reshape(scale.numel(), -1) / scale[:, None]) + zp[:, None], qmin, qmax)
    return (q - qmin).to(torch.uint8).reshape(-1)


def codes_to_weight(codes, scale, zp, bits, symmetric, shape):
    """Inverse of weight_to_codes: dequantize packed-then-unpacked codes back to float."""
    qmin, _ = qrange(bits, symmetric)
    q = codes.float().reshape(scale.numel(), -1) + qmin
    return ((q - zp[:, None]) * scale[:, None]).reshape(shape)


# ----------------------------------------------------------------------------- bit packing
_BIT_CACHE = {}


def _bit_weights(n, device):
    key = (n, device)
    if key not in _BIT_CACHE:
        _BIT_CACHE[key] = (2 ** torch.arange(n, device=device)).to(torch.int32)
    return _BIT_CACHE[key]


@torch.no_grad()
def pack_bits(codes, bits):
    """Packs a flat uint8 tensor of `bits`-bit codes into ceil(n*bits/8) bytes (LSB first).

    Works for any bit width 1..8, e.g. 3-bit codes are not padded to 4 bits.
    """
    codes = codes.reshape(-1).to(torch.int32)
    n = codes.numel()
    bitstream = ((codes[:, None] >> torch.arange(bits, device=codes.device)) & 1).reshape(-1)
    pad = (-bitstream.numel()) % 8
    if pad:
        bitstream = torch.cat([bitstream, bitstream.new_zeros(pad)])
    packed = (bitstream.reshape(-1, 8) * _bit_weights(8, codes.device)).sum(1).to(torch.uint8)
    assert packed.numel() == math.ceil(n * bits / 8)
    return packed


def unpack_bits(packed, bits, n):
    """Inverse of pack_bits: returns n codes as a flat int32 tensor."""
    bitstream = ((packed.to(torch.int32)[:, None] >> torch.arange(8, device=packed.device)) & 1).reshape(-1)
    bitstream = bitstream[: n * bits].reshape(n, bits)
    return (bitstream * _bit_weights(bits, packed.device)).sum(1)


# ----------------------------------------------------------------------------- activations
class ActQuantizer(nn.Module):
    """Per-tensor asymmetric fake quantizer for activations.

    Modes:
      "off"     - pass-through (float model behaviour)
      "observe" - pass-through, but record a sample of values to calibrate the range
      "quant"   - quantize-dequantize to `bits` bits with a fixed scale / zero point (STE in training)

    In memory the quantizer only keeps its scale (fp16) and zero point (uint8).
    """

    def __init__(self, bits, clip="mse", percentile=99.99, max_samples=200_000):
        super().__init__()
        self.bits, self.clip, self.percentile, self.max_samples = bits, clip, percentile, max_samples
        self.mode = "off"
        self.register_buffer("scale", torch.ones((), dtype=SCALE_DTYPE))
        self.register_buffer("zero_point", torch.zeros((), dtype=torch.uint8))
        self._samples, self._min, self._max = [], None, None
        self.last_numel = 0  # elements per image of the last activation seen (for size accounting)

    def extra_repr(self):
        return f"bits={self.bits}, clip={self.clip}, mode={self.mode}"

    def forward(self, x):
        self.last_numel = x[0].numel()
        if self.mode == "observe":
            self._observe(x.detach().float())
            return x
        if self.mode == "quant":
            return self._fake_quant(x)
        return x

    def _observe(self, x):
        flat = x.reshape(-1)
        bmin, bmax = flat.min(), flat.max()
        self._min = bmin if self._min is None else torch.minimum(self._min, bmin)
        self._max = bmax if self._max is None else torch.maximum(self._max, bmax)
        per_batch = max(1, self.max_samples // 8)
        idx = torch.randint(0, flat.numel(), (min(per_batch, flat.numel()),), device=flat.device)
        self._samples.append(flat[idx].cpu())

    @torch.no_grad()
    def finish_calibration(self, grid=80):
        """Turns the observed values into a scale and zero point, then switches to "quant"."""
        sample = torch.cat(self._samples)[: self.max_samples]
        lo, hi = self._min.cpu(), self._max.cpu()
        if self.clip == "percentile":
            lo = torch.quantile(sample, 1 - self.percentile / 100)
            hi = torch.quantile(sample, self.percentile / 100)
        elif self.clip == "mse":
            best = (float("inf"), lo, hi)
            for alpha in torch.linspace(0.2, 1.0, grid).tolist():
                s, z = _qparams_from_range(lo[None] * alpha, hi[None] * alpha, self.bits, symmetric=False)
                err = (_fake_quant(sample[None], s, z, self.bits, False) - sample).pow(2).mean().item()
                if err < best[0]:
                    best = (err, lo * alpha, hi * alpha)
            lo, hi = best[1], best[2]
        scale, zp = _qparams_from_range(lo.reshape(1), hi.reshape(1), self.bits, symmetric=False)
        self.scale.copy_(scale[0].to(SCALE_DTYPE))
        self.zero_point.copy_(zp[0].to(torch.uint8))
        self._samples, self._min, self._max = [], None, None
        self.mode = "quant"

    def _fake_quant(self, x):
        qmin, qmax = qrange(self.bits, symmetric=False)
        s, z = self.scale.float(), self.zero_point.float()
        # Clamp in the float domain first so the STE passes no gradient for clipped values
        lo, hi = (qmin - z) * s, (qmax - z) * s
        xc = torch.clamp(x, lo.item(), hi.item())
        q = torch.round(xc / s) + z
        return xc + ((q - z) * s - xc).detach()
