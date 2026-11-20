# MTech IIT Madras || System Engineering for Deep Learning || Assignment 2
# Created by Tanuj Gupta (CS25M540)
"""In-memory size accounting for the float and the compressed model (Q2c, Q4).

Everything is measured from the tensors the model actually keeps resident (parameters and
buffers, deduplicated by storage), i.e. what occupies RAM/VRAM once the model is loaded,
not from a file on disk. measure_cuda_bytes() cross-checks this with the CUDA allocator.
"""
import math

import torch
import torch.nn as nn

from .qmodel import QConv2d, QLinear, act_quantizers

MB = 1024 ** 2


def resident_bytes(module):
    """Bytes of all parameters + buffers held by `module` (each storage counted once)."""
    seen, total = set(), 0
    for t in list(module.parameters()) + list(module.buffers()):
        key = (t.untyped_storage().data_ptr(), t.device)
        if key not in seen:
            seen.add(key)
            total += t.untyped_storage().nbytes()
    return total


def float_breakdown(model):
    """Bytes of the fp32 model, split into conv/linear weights, BN parameters + stats, and biases."""
    out = {"weights": 0, "bn": 0, "bias": 0, "other": 0}
    for m in model.modules():
        for name, t in list(m.named_parameters(recurse=False)) + list(m.named_buffers(recurse=False)):
            if isinstance(m, nn.BatchNorm2d):
                out["bn"] += t.nbytes
            elif name == "weight":
                out["weights"] += t.nbytes
            elif name == "bias":
                out["bias"] += t.nbytes
            else:
                out["other"] += t.nbytes
    out["total"] = sum(out.values())
    return out


def compressed_breakdown(qmodel):
    """Bytes of a finalized compressed model.

    weights_packed  - bit-packed integer weight codes (ceil(n*bits/8) bytes per layer)
    weight_meta     - per-channel fp16 scales (+ uint8 zero points for asymmetric)  [overhead]
    bias            - fp32 biases produced by BN folding (+ classifier bias)
    act_meta        - per-tensor fp16 scale + uint8 zero point of every ActQuantizer  [overhead]
    """
    out = {"weights_packed": 0, "weight_meta": 0, "bias": 0, "act_meta": 0, "other": 0}
    for m in qmodel.modules():
        if isinstance(m, (QConv2d, QLinear)):
            assert m.finalized, "call compression.qmodel.finalize() before measuring"
            out["weights_packed"] += m.packed.nbytes
            out["weight_meta"] += m.w_scale.nbytes + (m.w_zp.nbytes if hasattr(m, "w_zp") else 0)
            out["bias"] += m.bias.nbytes if m.bias is not None else 0
    for q in act_quantizers(qmodel):
        out["act_meta"] += q.scale.nbytes + q.zero_point.nbytes
    out["total"] = sum(out.values())
    resident = resident_bytes(qmodel)
    out["other"] = resident - out["total"]  # anything not covered above (should be 0)
    out["total"] = resident
    return out


@torch.no_grad()
def activation_breakdown(qmodel, device, act_bits_override=None):
    """Activation memory for ONE 3x32x32 image, fp32 vs quantized.

    Measured by running one image through the model and recording, at every ActQuantizer,
    the number of elements of the activation tensor it quantizes. These are exactly the
    tensors passed between layers (input image, every ReLU6 output, every block output, the
    pooled vector). fp32 cost = 4 bytes/element; quantized cost = ceil(n*bits/8) bytes plus the
    tensor's scale + zero point. Reported both as the sum over all tensors (total traffic) and
    as the largest single tensor (a lower bound on the peak working set).
    """
    qmodel.eval()
    qmodel(torch.zeros(1, 3, 32, 32, device=device))
    fp32_total = q_total = fp32_peak = q_peak = 0
    for q in act_quantizers(qmodel):
        n = q.last_numel
        bits = act_bits_override or q.bits
        fq = 4 * n
        qq = math.ceil(n * bits / 8) + q.scale.nbytes + q.zero_point.nbytes
        fp32_total, q_total = fp32_total + fq, q_total + qq
        fp32_peak, q_peak = max(fp32_peak, fq), max(q_peak, qq)
    return {"fp32_total": fp32_total, "quant_total": q_total, "fp32_peak": fp32_peak, "quant_peak": q_peak,
            "num_tensors": len(act_quantizers(qmodel))}


def measure_cuda_bytes(model_cpu):
    """Bytes the CUDA caching allocator reserves for the model's tensors (None without CUDA).

    The allocator rounds each tensor up to a 512-byte block, so this is slightly above the
    exact tensor bytes; it confirms the packed model really is small in VRAM.
    """
    if not torch.cuda.is_available():
        return None
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    before = torch.cuda.memory_allocated()
    m = model_cpu.to("cuda")
    torch.cuda.synchronize()
    used = torch.cuda.memory_allocated() - before
    m.to("cpu")
    torch.cuda.empty_cache()
    return used
