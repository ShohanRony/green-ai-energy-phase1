"""Stage 2 Task 5: EfficientNet-B0 per-epoch timing, same conditions (GPU, batch
size=64) as the prior validation sprint's VGG-19-BN/ResNet-18/MobileNetV3-Small
timing table (research/mitra-reproduction/run_experiment.py, batch_size=64).
"""
import json, time

import torch, torchvision
from torch import nn
from torchvision import transforms
from train_baseline import build_model

CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2470, 0.2435, 0.2616)
BATCH_SIZE = 64


def main():
    device = 'cuda'
    torch.manual_seed(2026)
    train_tf = transforms.Compose([transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(), transforms.Normalize(CIFAR_MEAN, CIFAR_STD)])
    train_ds = torchvision.datasets.CIFAR10('/home/shohan/green-ai-research/data', train=True, download=False,
                                             transform=train_tf)
    train_dl = torch.utils.data.DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

    model = build_model('efficientnet_b0').to(device)
    opt = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    crit = nn.CrossEntropyLoss()

    epoch_times = []
    for epoch in range(2):  # epoch 1 absorbs cuDNN autotune/warmup; epoch 2 is steady-state
        model.train()
        torch.cuda.synchronize()
        t0 = time.time()
        for x, y in train_dl:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            opt.zero_grad()
            loss = crit(model(x), y)
            loss.backward()
            opt.step()
        torch.cuda.synchronize()
        epoch_times.append(time.time() - t0)
        print(f'epoch {epoch+1}: {epoch_times[-1]:.1f}s')

    result = dict(arch='efficientnet_b0', batch_size=BATCH_SIZE, warmup_epoch_s=epoch_times[0],
                  steady_state_epoch_s=epoch_times[1], gpu=torch.cuda.get_device_name(0))
    print(json.dumps(result, indent=2))
    with open('results_stage2/efficientnet_b0_timing.json', 'w') as f:
        json.dump(result, f, indent=2)


if __name__ == '__main__':
    main()
