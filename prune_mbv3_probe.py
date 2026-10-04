"""Task A probe: zero-finetune pruning on MobileNetV3-Small at 10% sparsity,
to tell genuine architectural fragility apart from a torch-pruning dependency-
graph bug on depthwise-separable/grouped convs. If 10% also collapses, that
points to a structural mismatch at the depthwise/pointwise boundary, not
'no redundancy to prune'.
"""
import copy, json

import torch, torch_pruning as tp
import torchvision
from torchvision import transforms
from train_baseline import build_model

CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2470, 0.2435, 0.2616)


def evaluate(model, dl, device):
    model.eval()
    correct, n = 0, 0
    with torch.no_grad():
        for x, y in dl:
            x, y = x.to(device), y.to(device)
            correct += (model(x).argmax(1) == y).sum().item()
            n += x.size(0)
    return correct / n


def main():
    device = 'cuda'
    torch.manual_seed(2026)
    data_dir = '/home/shohan/green-ai-research/data'
    test_tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(CIFAR_MEAN, CIFAR_STD)])
    test_ds = torchvision.datasets.CIFAR10(data_dir, train=False, download=False, transform=test_tf)
    test_dl = torch.utils.data.DataLoader(test_ds, batch_size=256, shuffle=False, num_workers=4)
    example_inputs = torch.randn(1, 3, 32, 32, device=device)

    base = build_model('mobilenet_v3_small')
    base.load_state_dict(torch.load('checkpoints/mobilenet_v3_small_fp32.pt', map_location='cpu', weights_only=True))
    base = base.to(device).eval()
    base_macs, base_params = tp.utils.count_ops_and_params(copy.deepcopy(base), example_inputs)
    base_acc = evaluate(base, test_dl, device)

    results = {'baseline': dict(acc=base_acc, macs=base_macs, params=base_params)}
    for ratio in [0.05, 0.10, 0.15, 0.20]:
        model = copy.deepcopy(base)
        imp = tp.importance.MagnitudeImportance(p=2)
        pruner = tp.pruner.MagnitudePruner(
            model, example_inputs, importance=imp, pruning_ratio=ratio,
            ignored_layers=[model.classifier[3]])
        pruner.step()
        macs, params = tp.utils.count_ops_and_params(model, example_inputs)
        acc = evaluate(model, test_dl, device)
        key = f'prune_{int(ratio*100)}'
        results[key] = dict(acc=acc, macs=macs, params=params,
                             nominal_sparsity=ratio, realized_macs_reduction=1 - macs / base_macs,
                             realized_params_reduction=1 - params / base_params)
        print(f'{int(ratio*100)}%: acc={acc:.4f} macs_reduction={1-macs/base_macs:.3f} '
              f'params_reduction={1-params/base_params:.3f}')

    with open('results_stage2/mbv3_sparsity_probe.json', 'w') as f:
        json.dump(results, f, indent=2)


if __name__ == '__main__':
    main()
