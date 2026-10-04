"""Stage 2 Task 2: manual INT8 PTQ pipeline for MobileNetV3-Small (CPU/fbgemm).

torchvision ships no `models.quantization.mobilenet_v3_small` convenience wrapper
(unlike ResNet-18), so this builds the pipeline directly on FX graph-mode
quantization (torch.ao.quantization.quantize_fx -- already-installed, handles
conv/bn/relu fusion and observer insertion generically, no hand-written
QuantizableMobileNetV3 needed) and layers the Nagel et al. (2021) procedure on
top of its default round-to-nearest conversion:

  Nagel, M., Fournarakis, M., Amjad, R. A., Bondarenko, Y., van Baalen, M., &
  Blankevoort, T. (2021). A White Paper on Neural Network Quantization.
  arXiv:2106.08295.

Cross-layer equalization is in that procedure but is SKIPPED here, disclosed,
not silently dropped: CLE's purpose is fixing per-channel weight-range
imbalance for backends that quantize weights per-tensor. fbgemm's default
qconfig already uses a per-channel weight observer, so the problem CLE exists
to fix is already handled by the backend -- same "tooling changes what the
proposal's six states actually need" pattern as the INT4 Conv2d-kernel finding
in Stage 1. What IS implemented: bias correction + AdaRound (the two
accuracy-recovery steps with no backend-level substitute).

AdaRound here is layer-wise, not the paper's full sequential/network-wise
calibration (each layer's reconstruction loss uses the FP32 model's own
activations as input, not the already-quantized upstream layers' outputs).
Simpler, weaker, but the core idea -- learned per-weight round-up/down instead
of round-to-nearest -- is faithfully implemented. Disclosed as a simplification,
not hidden.

Usage: python3 quantize_mobilenet_ptq.py --checkpoint checkpoints/mobilenet_v3_small_fp32.pt --out checkpoints/mobilenet_v3_small_int8.pt
"""
import argparse, json, copy
from pathlib import Path

import torch, torchvision
from torch import nn
from torchvision import transforms
from torch.ao.quantization import get_default_qconfig_mapping
from torch.ao.quantization.quantize_fx import prepare_fx, convert_fx
from torch.ao.quantization.observer import PerChannelMinMaxObserver

CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2470, 0.2435, 0.2616)
ZETA, GAMMA = 1.1, -0.1  # AdaRound soft-rounding stretch constants, per the paper


def build_model(arch: str):
    m = (torchvision.models.mobilenet_v3_small if arch == 'mobilenet_v3_small'
         else torchvision.models.efficientnet_b0)(weights=None, num_classes=10)
    m.features[0][0].stride = (1, 1)
    return m


def evaluate(model, dl, device='cpu'):
    model.eval()
    correct, n = 0, 0
    with torch.no_grad():
        for x, y in dl:
            x, y = x.to(device), y.to(device)
            out = model(x)
            correct += (out.argmax(1) == y).sum().item()
            n += x.size(0)
    return correct / n


def weighted_modules(model):
    return [(name, m) for name, m in model.named_modules() if isinstance(m, (nn.Conv2d, nn.Linear))]


def per_channel_scale(weight: torch.Tensor) -> torch.Tensor:
    """Symmetric per-output-channel int8 scale, matching fbgemm's default weight observer."""
    flat = weight.reshape(weight.shape[0], -1)
    amax = flat.abs().amax(dim=1).clamp(min=1e-8)
    return amax / 127.0


def adaround_weight(weight: torch.Tensor, scale: torch.Tensor, x_calib: torch.Tensor, module: nn.Module,
                     iters: int = 200) -> torch.Tensor:
    """Learn per-weight round-up/down (soft relaxation) to minimize this layer's
    output reconstruction error on calibration data, instead of round-to-nearest."""
    s = scale.view(-1, *([1] * (weight.dim() - 1)))
    w_floor = torch.floor(weight / s)
    frac = (weight / s) - w_floor
    # invert sigmoid to initialize alpha so h(alpha) ~= frac at step 0
    frac_c = frac.clamp(1e-4, 1 - 1e-4)
    v = -torch.log((ZETA - GAMMA) / (frac_c - GAMMA) - 1)
    v.requires_grad_(True)
    opt = torch.optim.Adam([v], lr=0.01)

    with torch.no_grad():
        target = (nn.functional.linear(x_calib, weight) if isinstance(module, nn.Linear) else
                  nn.functional.conv2d(x_calib, weight, bias=None, stride=module.stride, padding=module.padding,
                                        dilation=module.dilation, groups=module.groups))

    for it in range(iters):
        h = torch.clamp(torch.sigmoid(v) * (ZETA - GAMMA) + GAMMA, 0, 1)
        w_soft = s * torch.clamp(w_floor + h, -128, 127)
        out = (nn.functional.linear(x_calib, w_soft) if isinstance(module, nn.Linear) else
               nn.functional.conv2d(x_calib, w_soft, bias=None, stride=module.stride, padding=module.padding,
                                     dilation=module.dilation, groups=module.groups))
        recon = (out - target).pow(2).mean()
        beta = 20 - (20 - 2) * (it / iters)  # anneal, per the paper
        reg = (1 - (2 * h - 1).abs().pow(beta)).mean()
        loss = recon + 0.01 * reg
        opt.zero_grad(); loss.backward(); opt.step()

    with torch.no_grad():
        h = (torch.sigmoid(v) > 0.5).float()  # hard round at the end
        w_final = s * torch.clamp(w_floor + h, -128, 127)
    return w_final.detach()


def apply_adaround_and_bias_correction(model: nn.Module, calib_batches: list[torch.Tensor]) -> dict:
    """Walk the FP32 model, replacing each conv/linear weight with its AdaRound
    grid-aligned value and correcting bias for the resulting mean output shift.
    Captured via forward hooks so each layer sees the real (not re-derived) input
    it gets during a normal forward pass on calibration data."""
    captured = {}
    hooks = []

    def make_hook(name):
        def hook(mod, inp, out):
            captured.setdefault(name, []).append((inp[0].detach(), out.detach()))
        return hook

    targets = weighted_modules(model)
    for name, m in targets:
        hooks.append(m.register_forward_hook(make_hook(name)))
    with torch.no_grad():
        for x in calib_batches:
            model(x)
    for h in hooks:
        h.remove()

    report = {}
    for name, m in targets:
        xs = torch.cat([c[0] for c in captured[name]], dim=0)
        fp32_outs = torch.cat([c[1] for c in captured[name]], dim=0)
        scale = per_channel_scale(m.weight.data)
        w_new = adaround_weight(m.weight.data, scale, xs, m)
        m.weight.data.copy_(w_new)
        if m.bias is not None:
            with torch.no_grad():
                q_out = m(xs)
                err = (fp32_outs - q_out)
                reduce_dims = [0] + list(range(2, err.dim())) if err.dim() > 2 else [0]
                correction = err.mean(dim=reduce_dims)
                m.bias.data.add_(correction)
        report[name] = 'adaround+bias-corrected'
    return report


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--arch', required=True, choices=['mobilenet_v3_small', 'efficientnet_b0'])
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--data', default='/home/shohan/green-ai-research/data')
    p.add_argument('--out', required=True)
    p.add_argument('--calib-batches', type=int, default=8)
    a = p.parse_args()

    torch.backends.quantized.engine = 'fbgemm'
    torch.manual_seed(2026)
    test_tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(CIFAR_MEAN, CIFAR_STD)])
    train_ds = torchvision.datasets.CIFAR10(a.data, train=True, download=False, transform=test_tf)
    test_ds = torchvision.datasets.CIFAR10(a.data, train=False, download=False, transform=test_tf)
    calib_dl = torch.utils.data.DataLoader(train_ds, batch_size=64, shuffle=True)
    test_dl = torch.utils.data.DataLoader(test_ds, batch_size=256, shuffle=False)

    model = build_model(a.arch)
    model.load_state_dict(torch.load(a.checkpoint, map_location='cpu', weights_only=True))
    model.eval()
    fp32_acc = evaluate(model, test_dl)

    calib_batches = [x for i, (x, _) in enumerate(calib_dl) if i < a.calib_batches]

    adaround_model = copy.deepcopy(model)
    layer_report = apply_adaround_and_bias_correction(adaround_model, calib_batches)
    adaround_fp32_acc = evaluate(adaround_model, test_dl)  # sanity: still float here, should be close to fp32_acc

    qconfig_mapping = get_default_qconfig_mapping('fbgemm')
    example_inputs = (calib_batches[0],)
    prepared = prepare_fx(adaround_model, qconfig_mapping, example_inputs)
    with torch.no_grad():
        for x in calib_batches:
            prepared(x)
    quantized = convert_fx(prepared)

    is_int8 = any('quantized' in type(m).__module__ for m in quantized.modules())
    int8_acc = evaluate(quantized, test_dl) if is_int8 else None

    out_path = Path(a.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.jit.save(torch.jit.script(quantized), str(out_path))
    meta = dict(arch=a.arch, verdict='pass' if is_int8 else 'fail-silent-fallback',
                fp32_acc=fp32_acc, adaround_weights_still_float_acc=adaround_fp32_acc, int8_acc=int8_acc,
                kernel_dtype_check='torch.ao.nn.quantized modules found' if is_int8 else 'NO quantized modules found',
                backend='fbgemm (CPU)', pipeline='FX graph-mode fuse+observe+convert, '
                'AdaRound (layer-wise, disclosed simplification) + bias correction applied pre-convert, '
                'cross-layer equalization skipped (fbgemm already does per-channel weight quant -- CLE redundant, see module docstring)',
                num_adaround_layers=len(layer_report), calib_batches=a.calib_batches,
                citation='Nagel et al. (2021), A White Paper on Neural Network Quantization, arXiv:2106.08295')
    out_path.with_suffix('.json').write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))


if __name__ == '__main__':
    main()
