"""Stage 2 Task 4 decision checkpoint: zero-finetune vs brief-recovery structured
pruning at 70% on ONE model (ResNet-18), via torch-pruning. Per the brief, this
stops here for a decision -- it does NOT run the full 3-model x 3-ratio grid.

Usage: python3 prune_decision_checkpoint.py
"""
import copy, json, time

import torch, torch_pruning as tp
from train_baseline import build_model

CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2470, 0.2435, 0.2616)
RECOVERY_EPOCHS = 3


def evaluate(model, dl, device):
    model.eval()
    correct, n = 0, 0
    with torch.no_grad():
        for x, y in dl:
            x, y = x.to(device), y.to(device)
            correct += (model(x).argmax(1) == y).sum().item()
            n += x.size(0)
    return correct / n


def count_params_flops(model, example_inputs):
    macs, params = tp.utils.count_ops_and_params(model, example_inputs)
    return macs, params


def measure_latency(model, x, reps=50, warmup=10):
    model.eval()
    with torch.no_grad():
        for _ in range(warmup):
            model(x)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(reps):
            model(x)
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / reps * 1000


def prune_resnet18(model, ratio, example_inputs):
    imp = tp.importance.MagnitudeImportance(p=2)
    ignored_layers = [model.fc]
    pruner = tp.pruner.MagnitudePruner(
        model, example_inputs, importance=imp, pruning_ratio=ratio, ignored_layers=ignored_layers)
    pruner.step()
    return model


def main():
    from torchvision import transforms
    import torchvision

    device = 'cuda'
    torch.manual_seed(2026)
    data_dir = '/home/shohan/green-ai-research/data'
    train_tf = transforms.Compose([transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(), transforms.Normalize(CIFAR_MEAN, CIFAR_STD)])
    test_tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(CIFAR_MEAN, CIFAR_STD)])
    train_ds = torchvision.datasets.CIFAR10(data_dir, train=True, download=False, transform=train_tf)
    test_ds = torchvision.datasets.CIFAR10(data_dir, train=False, download=False, transform=test_tf)
    train_dl = torch.utils.data.DataLoader(train_ds, batch_size=128, shuffle=True, num_workers=4)
    test_dl = torch.utils.data.DataLoader(test_ds, batch_size=256, shuffle=False, num_workers=4)

    base = build_model('resnet18')
    base.load_state_dict(torch.load('checkpoints/resnet18_fp32.pt', map_location='cpu', weights_only=True))
    base = base.to(device).eval()
    example_inputs = torch.randn(1, 3, 32, 32, device=device)

    base_macs, base_params = count_params_flops(copy.deepcopy(base), example_inputs)
    base_acc = evaluate(base, test_dl, device)
    base_latency = measure_latency(base, torch.randn(128, 3, 32, 32, device=device))

    results = dict(baseline=dict(acc=base_acc, macs=base_macs, params=base_params, latency_ms=base_latency))

    # --- Variant A: zero-finetune ---
    model_a = copy.deepcopy(base)
    prune_resnet18(model_a, 0.7, example_inputs)
    macs_a, params_a = count_params_flops(model_a, example_inputs)
    acc_a = evaluate(model_a, test_dl, device)
    latency_a = measure_latency(model_a, torch.randn(128, 3, 32, 32, device=device))
    results['zero_finetune'] = dict(
        acc=acc_a, macs=macs_a, params=params_a, latency_ms=latency_a,
        nominal_sparsity=0.7, realized_macs_reduction=1 - macs_a / base_macs,
        realized_params_reduction=1 - params_a / base_params, realized_speedup=base_latency / latency_a)

    # --- Variant B: brief recovery (short fine-tune after pruning) ---
    model_b = copy.deepcopy(base)
    prune_resnet18(model_b, 0.7, example_inputs)
    opt = torch.optim.SGD(model_b.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
    crit = torch.nn.CrossEntropyLoss()
    model_b.train()
    for epoch in range(RECOVERY_EPOCHS):
        for x, y in train_dl:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = crit(model_b(x), y)
            loss.backward()
            opt.step()
    macs_b, params_b = count_params_flops(model_b, example_inputs)
    acc_b = evaluate(model_b, test_dl, device)
    latency_b = measure_latency(model_b, torch.randn(128, 3, 32, 32, device=device))
    results['brief_recovery'] = dict(
        acc=acc_b, macs=macs_b, params=params_b, latency_ms=latency_b, recovery_epochs=RECOVERY_EPOCHS,
        nominal_sparsity=0.7, realized_macs_reduction=1 - macs_b / base_macs,
        realized_params_reduction=1 - params_b / base_params, realized_speedup=base_latency / latency_b)

    print(json.dumps(results, indent=2))
    with open('results_stage2/task4_decision_checkpoint.json', 'w') as f:
        json.dump(results, f, indent=2)


if __name__ == '__main__':
    main()
