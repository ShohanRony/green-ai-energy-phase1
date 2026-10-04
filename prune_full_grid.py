"""Stage 2 Task 4 full run: zero-finetune structured pruning at 30/50/70% for all
3 models, via torch-pruning. Policy confirmed after the single-model decision
checkpoint (results_stage2/task4_decision_checkpoint.json): zero-finetune
uniformly, report collapse where it happens, do not silently switch to recovery.
"""
import argparse, copy, json, time
from pathlib import Path

import torch, torch_pruning as tp
import torchvision
from torchvision import transforms
from train_baseline import build_model

CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2470, 0.2435, 0.2616)
RATIOS = [0.3, 0.5, 0.7]
FINAL_LAYER = {
    'resnet18': lambda m: m.fc,
    'mobilenet_v3_small': lambda m: m.classifier[3],
    'efficientnet_b0': lambda m: m.classifier[1],
}


def evaluate(model, dl, device):
    model.eval()
    correct, n = 0, 0
    with torch.no_grad():
        for x, y in dl:
            x, y = x.to(device), y.to(device)
            correct += (model(x).argmax(1) == y).sum().item()
            n += x.size(0)
    return correct / n


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


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--archs', nargs='+', default=['resnet18', 'mobilenet_v3_small', 'efficientnet_b0'],
                    choices=['resnet18', 'mobilenet_v3_small', 'efficientnet_b0'])
    a = p.parse_args()

    device = 'cuda'
    torch.manual_seed(2026)
    data_dir = '/home/shohan/green-ai-research/data'
    test_tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(CIFAR_MEAN, CIFAR_STD)])
    test_ds = torchvision.datasets.CIFAR10(data_dir, train=False, download=False, transform=test_tf)
    test_dl = torch.utils.data.DataLoader(test_ds, batch_size=256, shuffle=False, num_workers=4)
    example_inputs = torch.randn(1, 3, 32, 32, device=device)
    bench_input = torch.randn(128, 3, 32, 32, device=device)

    out_file = Path('results_stage2/task4_full_grid.json')
    results = json.loads(out_file.read_text()) if out_file.exists() else {}
    for arch in a.archs:
        base = build_model(arch)
        base.load_state_dict(torch.load(f'checkpoints/{arch}_fp32.pt', map_location='cpu', weights_only=True))
        base = base.to(device).eval()
        base_macs, base_params = tp.utils.count_ops_and_params(copy.deepcopy(base), example_inputs)
        base_acc = evaluate(base, test_dl, device)
        base_latency = measure_latency(base, bench_input)
        results[arch] = {'baseline': dict(acc=base_acc, macs=base_macs, params=base_params, latency_ms=base_latency)}

        for ratio in RATIOS:
            model = copy.deepcopy(base)
            imp = tp.importance.MagnitudeImportance(p=2)
            pruner = tp.pruner.MagnitudePruner(
                model, example_inputs, importance=imp, pruning_ratio=ratio,
                ignored_layers=[FINAL_LAYER[arch](model)])
            pruner.step()
            macs, params = tp.utils.count_ops_and_params(model, example_inputs)
            acc = evaluate(model, test_dl, device)
            latency = measure_latency(model, bench_input)
            collapsed = acc <= 0.15  # near chance-level for CIFAR-10 (10 classes)
            key = f'prune_{int(ratio*100)}'
            results[arch][key] = dict(
                acc=acc, macs=macs, params=params, latency_ms=latency,
                nominal_sparsity=ratio, realized_macs_reduction=1 - macs / base_macs,
                realized_params_reduction=1 - params / base_params,
                realized_speedup=base_latency / latency,
                flagged_collapse=collapsed, policy='zero-finetune')
            print(f'{arch} @ {int(ratio*100)}%: acc={acc:.4f} macs_reduction={1-macs/base_macs:.3f} '
                  f'speedup={base_latency/latency:.2f}x{" -- COLLAPSED" if collapsed else ""}')

    out_file.write_text(json.dumps(results, indent=2))
    print(f'\nSaved {out_file}')


if __name__ == '__main__':
    main()
