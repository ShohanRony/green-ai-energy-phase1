"""Train an FP32 CIFAR-10 baseline checkpoint for Stage 2 (fresh training, not pretrained).

Usage: python3 train_baseline.py --arch resnet18 --epochs 30 --out checkpoints/resnet18_fp32.pt
"""
import argparse, json, time, random
from pathlib import Path

import torch, torchvision
from torch import nn
from torchvision import transforms

CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2470, 0.2435, 0.2616)

ARCHES = {
    'resnet18': torchvision.models.resnet18,
    'mobilenet_v3_small': torchvision.models.mobilenet_v3_small,
    'efficientnet_b0': torchvision.models.efficientnet_b0,
}


def build_model(arch: str) -> nn.Module:
    model = ARCHES[arch](weights=None, num_classes=10)
    # CIFAR-10 is 32x32; the ImageNet-default stem downsamples 4x before the
    # first residual/block stage, collapsing spatial resolution too early.
    # Standard "CIFAR-ResNet" fix: drop the initial stride-2 conv + maxpool
    # to stride-1, applied consistently across all three architectures here.
    if arch == 'resnet18':
        model.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        model.maxpool = nn.Identity()
    elif arch == 'mobilenet_v3_small':
        model.features[0][0].stride = (1, 1)
    elif arch == 'efficientnet_b0':
        model.features[0][0].stride = (1, 1)
    return model


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--arch', required=True, choices=sorted(ARCHES))
    p.add_argument('--data', default='data')
    p.add_argument('--epochs', type=int, default=30)
    p.add_argument('--batch-size', type=int, default=128)
    p.add_argument('--lr', type=float, default=0.1)
    p.add_argument('--out', required=True)
    a = p.parse_args()

    torch.manual_seed(2026)
    random.seed(2026)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    train_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
    test_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
    train_ds = torchvision.datasets.CIFAR10(a.data, train=True, download=False, transform=train_tf)
    test_ds = torchvision.datasets.CIFAR10(a.data, train=False, download=False, transform=test_tf)
    train_dl = torch.utils.data.DataLoader(train_ds, batch_size=a.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    test_dl = torch.utils.data.DataLoader(test_ds, batch_size=256, shuffle=False, num_workers=4, pin_memory=True)

    model = build_model(a.arch).to(device)
    opt = torch.optim.SGD(model.parameters(), lr=a.lr, momentum=0.9, weight_decay=5e-4, nesterov=True)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs)
    crit = nn.CrossEntropyLoss()

    log = []
    t0 = time.time()
    for epoch in range(a.epochs):
        model.train()
        running_loss, correct, n = 0.0, 0, 0
        for x, y in train_dl:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            opt.zero_grad()
            out = model(x)
            loss = crit(out, y)
            loss.backward()
            opt.step()
            running_loss += loss.item() * x.size(0)
            correct += (out.argmax(1) == y).sum().item()
            n += x.size(0)
        sched.step()
        train_acc = correct / n
        epoch_time = time.time() - t0
        print(f'epoch {epoch+1}/{a.epochs} loss={running_loss/n:.4f} train_acc={train_acc:.4f} elapsed={epoch_time:.0f}s', flush=True)
        log.append(dict(epoch=epoch + 1, loss=running_loss / n, train_acc=train_acc))

    model.eval()
    correct, n = 0, 0
    with torch.no_grad():
        for x, y in test_dl:
            x, y = x.to(device), y.to(device)
            out = model(x)
            correct += (out.argmax(1) == y).sum().item()
            n += x.size(0)
    test_acc = correct / n
    print(f'FINAL test_acc={test_acc:.4f}', flush=True)

    out_path = Path(a.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_path)
    meta = dict(arch=a.arch, epochs=a.epochs, lr=a.lr, batch_size=a.batch_size, seed=2026,
                test_acc=test_acc, train_log=log, torch=torch.__version__,
                torchvision=torchvision.__version__, device=device,
                wall_clock_s=time.time() - t0, provenance='trained-here-fresh')
    out_path.with_suffix('.json').write_text(json.dumps(meta, indent=2))
    print(f'Saved {out_path} and {out_path.with_suffix(".json")}')


if __name__ == '__main__':
    main()
