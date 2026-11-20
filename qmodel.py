# MTech IIT Madras || System Engineering for Deep Learning || Assignment 2
# Created by Tanuj Gupta (CS25M540)
"""Turns a trained fp32 MobileNetV2 into a quantized, bit-packed MobileNetV2.

Pipeline (see compress_model / calibrate / finalize):
  1. BN folding     - every ConvBNAct (conv -> BN -> act) becomes one conv with a bias,
                      so the BatchNorm parameters and running stats disappear from memory.
  2. Weight quant   - every conv and the classifier get b-bit weights (per-channel scales).
                      Mixed precision: the stem conv and classifier can keep more bits
                      (--first_last_bits) and depthwise convs can use their own bit width (--dw_bits).
  3. Activation quant - ActQuantizers on the network input, after every ReLU6, after every
                      inverted-residual block (after the skip add) and after global pooling.
  4. (optional) QAT - fine-tune the folded weights through the fake-quant ops with the STE.
  5. Packing        - weights are converted to integer codes and bit-packed into uint8
                      buffers; the float weights are deleted. Forward unpacks + dequantizes
                      on the fly, so the resident model really is the compressed size.
"""
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from mbv2.model import ConvBNAct, InvertedResidual, MobileNetV2
from .quant import (SCALE_DTYPE, ActQuantizer, codes_to_weight, fake_quant_weight, pack_bits,
                    unpack_bits, weight_qparams, weight_to_codes)


class QuantWeightMixin:
    """Shared weight handling for QConv2d / QLinear.

    Before finalize(): `weight` is a float Parameter, fake-quantized every forward (trainable via STE).
    After  finalize(): only `packed` (uint8), `w_scale` (fp16) and, for asymmetric, `w_zp` (uint8) remain.
    """

    def _init_quant(self, weight, bias, bits, per_channel, symmetric, clip):
        self.bits, self.per_channel, self.symmetric, self.clip = bits, per_channel, symmetric, clip
        self.w_shape = tuple(weight.shape)
        self.weight = nn.Parameter(weight.detach().clone())
        self.bias = nn.Parameter(bias.detach().clone()) if bias is not None else None
        self.finalized = False
        self.recompute_qparams()

    @torch.no_grad()
    def recompute_qparams(self):
        scale, zp = weight_qparams(self.weight, self.bits, self.per_channel, self.symmetric, self.clip)
        self._scale, self._zp = scale, zp  # float copies used for fake quant before finalize

    def get_weight(self):
        if not self.finalized:
            return fake_quant_weight(self.weight, self._scale.to(self.weight.device),
                                     self._zp.to(self.weight.device), self.bits, self.symmetric)
        n = 1
        for d in self.w_shape:
            n *= d
        codes = unpack_bits(self.packed, self.bits, n)
        zp = torch.zeros_like(self.w_scale, dtype=torch.float32) if self.symmetric else self.w_zp.float()
        return codes_to_weight(codes, self.w_scale.float(), zp, self.bits, self.symmetric, self.w_shape)

    @torch.no_grad()
    def finalize(self):
        """Replace the float weight with bit-packed integer codes + fp16 scales (+ uint8 zero points)."""
        if self.finalized:
            return
        # Reuse the exact scales used by fake quant, so the packed model gives identical outputs
        scale, zp = self._scale.to(self.weight.device), self._zp.to(self.weight.device)
        codes = weight_to_codes(self.weight, scale, zp, self.bits, self.symmetric)
        dev = self.weight.device
        del self.weight
        self.register_buffer("packed", pack_bits(codes, self.bits))
        self.register_buffer("w_scale", scale.to(SCALE_DTYPE))
        if not self.symmetric:
            self.register_buffer("w_zp", zp.to(torch.uint8))
        del self._scale, self._zp
        self.finalized = True
        self.to(dev)


class QConv2d(nn.Module, QuantWeightMixin):
    """A ConvBNAct with BN folded in, quantized weights, its activation and an output quantizer."""

    def __init__(self, cba: ConvBNAct, bits, act_quant, per_channel=True, symmetric=True, clip="mse"):
        super().__init__()
        conv, bn = cba.conv, cba.bn
        # BN folding: y = gamma * (Wx - mean) / sqrt(var + eps) + beta = W'x + b'
        std = torch.sqrt(bn.running_var + bn.eps)
        w = conv.weight * (bn.weight / std).reshape(-1, 1, 1, 1)
        b = bn.bias - bn.running_mean * bn.weight / std
        self.stride, self.padding, self.groups = conv.stride, conv.padding, conv.groups
        self._init_quant(w, b, bits, per_channel, symmetric, clip)
        self.act = cba.act
        # Linear-bottleneck convs (no activation) are not quantized here: their output is added to
        # the skip connection first and quantized by the block's out_quant.
        self.out_quant = act_quant if not isinstance(cba.act, nn.Identity) else nn.Identity()

    def extra_repr(self):
        return f"w_shape={self.w_shape}, bits={self.bits}, groups={self.groups}, finalized={self.finalized}"

    def forward(self, x):
        y = F.conv2d(x, self.get_weight().to(x.dtype), self.bias.to(x.dtype), self.stride, self.padding,
                     groups=self.groups)
        return self.out_quant(self.act(y))


class QLinear(nn.Module, QuantWeightMixin):
    def __init__(self, linear: nn.Linear, bits, per_channel=True, symmetric=True, clip="mse"):
        super().__init__()
        self._init_quant(linear.weight, linear.bias, bits, per_channel, symmetric, clip)

    def extra_repr(self):
        return f"w_shape={self.w_shape}, bits={self.bits}, finalized={self.finalized}"

    def forward(self, x):
        return F.linear(x, self.get_weight().to(x.dtype), self.bias.to(x.dtype))


# ----------------------------------------------------------------------------- conversion
def compress_model(model: MobileNetV2, weight_bits, act_bits, first_last_bits=8, dw_bits=None,
                   input_bits=8, per_channel=True, symmetric=True, weight_clip="mse", act_clip="mse"):
    """Returns a quantized copy of `model` (the original is untouched).

    first_last_bits: weight bits for the stem conv and the classifier (None = weight_bits).
    dw_bits:         weight bits for depthwise convs (None = weight_bits).
    input_bits:      bits for the network input (CIFAR pixels are 8-bit to begin with).
    """
    model = copy.deepcopy(model).eval()
    first_last_bits = first_last_bits or weight_bits
    dw_bits = dw_bits or weight_bits
    wkw = dict(per_channel=per_channel, symmetric=symmetric, clip=weight_clip)

    def aq(bits=act_bits):
        return ActQuantizer(bits, clip=act_clip)

    def convert_cba(cba, bits):
        if cba.conv.groups > 1 and cba.conv.groups == cba.conv.in_channels:  # depthwise
            bits = dw_bits
        return QConv2d(cba, bits, aq(), **wkw)

    model.in_quant = aq(input_bits)
    feats = model.features
    for i, m in enumerate(feats):
        if isinstance(m, ConvBNAct):
            feats[i] = convert_cba(m, first_last_bits if i == 0 else weight_bits)
        elif isinstance(m, InvertedResidual):
            for j, cba in enumerate(m.conv):
                m.conv[j] = convert_cba(cba, weight_bits)
            m.out_quant = aq()
    model.pool_quant = aq()
    model.classifier[1] = QLinear(model.classifier[1], first_last_bits, **wkw)
    return model


def act_quantizers(model):
    return [m for m in model.modules() if isinstance(m, ActQuantizer)]


def quant_layers(model):
    return [m for m in model.modules() if isinstance(m, (QConv2d, QLinear))]


@torch.no_grad()
def calibrate(model, loader, device):
    """Runs calibration images through the model (weights already fake-quantized, activations
    in float) to choose every ActQuantizer's range, then switches them all to "quant"."""
    model.eval()
    qs = act_quantizers(model)
    for q in qs:
        q.mode = "observe"
    for x, _ in loader:
        model(x.to(device))
    for q in qs:
        q.finish_calibration()


def finalize(model):
    """Packs every quantized layer's weights (irreversible: float weights are deleted)."""
    for layer in quant_layers(model):
        layer.finalize()
    return model


def describe_layers(model):
    """One row per quantized layer: name, type, weight shape and bit width (for the report, Q2b)."""
    rows = []
    for name, m in model.named_modules():
        if isinstance(m, QConv2d):
            kind = "depthwise" if m.groups > 1 else ("pointwise" if m.w_shape[-1] == 1 else "conv3x3")
            rows.append((name, kind, m.w_shape, m.bits, not isinstance(m.out_quant, nn.Identity)))
        elif isinstance(m, QLinear):
            rows.append((name, "linear", m.w_shape, m.bits, False))
    return rows
