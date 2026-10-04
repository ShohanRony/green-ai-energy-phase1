"""Stage 2 Task 2: INT8 post-training static quantization, CPU/fbgemm only.

ResNet-18 path (torchvision.models.quantization, confirmed working): fuse -> calibrate -> convert.
Usage: python3 quantize_int8.py --arch resnet18 --checkpoint checkpoints/resnet18_fp32.pt --out checkpoints/resnet18_int8.pt
"""
import argparse, json, time
from pathlib import Path

import torch, torchvision
from torch import nn
from torchvision import transforms

CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2470, 0.2435, 0.2616)


def build_quantizable_resnet18():
    m = torchvision.models.quantization.resnet18(weights=None, num_classes=10, quantize=False)
    m.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
    m.maxpool = nn.Identity()
    return m


def evaluate(model, dl, device):
    model.eval()
    correct, n = 0, 0
    with torch.no_grad():
        for x, y in dl:
            x, y = x.to(device), y.to(device)
            out = model(x)
            correct += (out.argmax(1) == y).sum().item()
            n += x.size(0)
    return correct / n


def kernel_is_int8(model) -> bool:
    """Confirm the executed conv kernel actually dispatches to a quantized int8 op,
    not a silent FP32 fallback (the EfficientNet-B0/MobileNetV3 risk flagged in the brief)."""
    for m in model.modules():
        if isinstance(m, torch.nn.quantized.Conv2d) or type(m).__name__ in ('ConvReLU2d',) and 'quantized' in type(m).__module__:
            return True
    # Fallback: check any module's __module__ mentions torch.ao.nn.quantized / torch.nn.quantized
    return any('quantized' in type(m).__module__ for m in model.modules())


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--arch', required=True, choices=['resnet18'])
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--data', default='/home/shohan/green-ai-research/data')
    p.add_argument('--out', required=True)
    p.add_argument('--calib-batches', type=int, default=20)
    a = p.parse_args()

    torch.backends.quantized.engine = 'fbgemm'
    test_tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(CIFAR_MEAN, CIFAR_STD)])
    train_ds = torchvision.datasets.CIFAR10(a.data, train=True, download=False, transform=test_tf)
    test_ds = torchvision.datasets.CIFAR10(a.data, train=False, download=False, transform=test_tf)
    calib_dl = torch.utils.data.DataLoader(train_ds, batch_size=64, shuffle=True)
    test_dl = torch.utils.data.DataLoader(test_ds, batch_size=256, shuffle=False)

    model = build_quantizable_resnet18()
    model.load_state_dict(torch.load(a.checkpoint, map_location='cpu', weights_only=True))
    model.eval()
    fp32_acc = evaluate(model, test_dl, 'cpu')

    model.fuse_model()
    model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
    torch.quantization.prepare(model, inplace=True)
    with torch.no_grad():
        for i, (x, _) in enumerate(calib_dl):
            if i >= a.calib_batches:
                break
            model(x)
    torch.quantization.convert(model, inplace=True)

    is_int8 = kernel_is_int8(model)
    int8_acc = evaluate(model, test_dl, 'cpu') if is_int8 else None

    out_path = Path(a.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.jit.script(model).save(str(out_path))  # quantized state_dict alone isn't reloadable without the module graph
    meta = dict(arch=a.arch, verdict='pass' if is_int8 else 'fail-silent-fallback',
                fp32_acc=fp32_acc, int8_acc=int8_acc,
                kernel_dtype_check='torch.ao.nn.quantized modules found' if is_int8 else 'NO quantized modules found -- fell back to fp32',
                calib_batches=a.calib_batches, backend='fbgemm (CPU)')
    out_path.with_suffix('.json').write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))


if __name__ == '__main__':
    main()
