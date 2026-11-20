# MTech IIT Madras || System Engineering for Deep Learning || Assignment 2
# Created by Tanuj Gupta (CS25M540)
"""MobileNet-v2 adapted to 32x32 CIFAR-10 images.

Follows torchvision's mobilenetv2.py (inverted residual blocks, linear bottlenecks,
ReLU6, width multiplier, channel rounding to multiples of 8). Two changes for CIFAR:
  * the stem conv uses stride 1 instead of 2, and
  * the second stage (t=6, c=24) uses stride 1 instead of 2,
so the final feature map is 4x4 instead of 1x1 (ImageNet strides would shrink a
32x32 image to 1x1 and lose most spatial information).

Every conv is wrapped in ConvBNAct, and every block output passes through an
`out_quant` placeholder (nn.Identity in the float model). The compression code
(compression/qmodel.py) relies on this structure: it folds each ConvBNAct into a
single quantized conv and swaps each placeholder for an activation quantizer.
"""
import torch
import torch.nn as nn


def make_divisible(v, divisor=8, min_value=None):
    """Rounds a channel count to the nearest multiple of `divisor` (as torchvision does)."""
    min_value = min_value or divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:  # never round down by more than 10%
        new_v += divisor
    return new_v


class ConvBNAct(nn.Module):
    """Conv2d (no bias) -> BatchNorm2d -> ReLU6, or no activation for linear bottlenecks."""

    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, groups=1, act=True, bn_momentum=0.1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, stride, (kernel_size - 1) // 2,
                              groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_ch, momentum=bn_momentum)
        self.act = nn.ReLU6(inplace=True) if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class InvertedResidual(nn.Module):
    """Expand (1x1) -> depthwise (3x3) -> project (1x1, linear), with a skip when shapes match."""

    def __init__(self, in_ch, out_ch, stride, expand_ratio, bn_momentum=0.1):
        super().__init__()
        hidden = int(round(in_ch * expand_ratio))
        self.use_res = stride == 1 and in_ch == out_ch
        layers = []
        if expand_ratio != 1:
            layers.append(ConvBNAct(in_ch, hidden, 1, bn_momentum=bn_momentum))
        layers += [
            ConvBNAct(hidden, hidden, 3, stride, groups=hidden, bn_momentum=bn_momentum),  # depthwise
            ConvBNAct(hidden, out_ch, 1, act=False, bn_momentum=bn_momentum),              # linear projection
        ]
        self.conv = nn.Sequential(*layers)
        # Placeholder for the activation quantizer on the block output (after the residual add)
        self.out_quant = nn.Identity()

    def forward(self, x):
        out = self.conv(x)
        if self.use_res:
            out = x + out
        return self.out_quant(out)


# (expansion t, output channels c, repeats n, first stride s); CIFAR strides marked *
CIFAR_SETTINGS = [
    [1, 16, 1, 1],
    [6, 24, 2, 1],  # * ImageNet uses 2
    [6, 32, 3, 2],
    [6, 64, 4, 2],
    [6, 96, 3, 1],
    [6, 160, 3, 2],
    [6, 320, 1, 1],
]


class MobileNetV2(nn.Module):
    def __init__(self, num_classes=10, width_mult=1.0, dropout=0.2, bn_momentum=0.1):
        super().__init__()
        in_ch = make_divisible(32 * width_mult)
        self.last_ch = make_divisible(1280 * max(1.0, width_mult))

        self.in_quant = nn.Identity()  # placeholder: quantizes the network input
        features = [ConvBNAct(3, in_ch, 3, stride=1, bn_momentum=bn_momentum)]  # * ImageNet uses stride 2
        for t, c, n, s in CIFAR_SETTINGS:
            out_ch = make_divisible(c * width_mult)
            for i in range(n):
                features.append(InvertedResidual(in_ch, out_ch, s if i == 0 else 1, t, bn_momentum))
                in_ch = out_ch
        features.append(ConvBNAct(in_ch, self.last_ch, 1, bn_momentum=bn_momentum))
        self.features = nn.Sequential(*features)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.pool_quant = nn.Identity()  # placeholder: quantizes the pooled feature vector
        self.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(self.last_ch, num_classes))
        self._init_weights()

    def _init_weights(self):
        # Same initialisation as torchvision's MobileNetV2
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.features(self.in_quant(x))
        x = self.pool_quant(torch.flatten(self.pool(x), 1))
        return self.classifier(x)
