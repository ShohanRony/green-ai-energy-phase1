"""Task D: materialize FP16 (x3) and pruned (x9) states as standalone checkpoint
files, same self-contained TorchScript-trace format the INT8 checkpoints already
use (no architecture reconstruction needed to reload -- torch.jit.load alone
is enough). Deterministic from the existing FP32 checkpoints (seed 2026), same
pruning calls as prune_full_grid.py.
"""
import argparse, copy

import torch, torch_pruning as tp
from train_baseline import build_model

RATIOS = [0.3, 0.5, 0.7]
FINAL_LAYER = {
    'resnet18': lambda m: m.fc,
    'mobilenet_v3_small': lambda m: m.classifier[3],
    'efficientnet_b0': lambda m: m.classifier[1],
}


def trace_and_save(model, example_input, path):
    model.eval()
    traced = torch.jit.trace(model, example_input)
    traced.save(path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--archs', nargs='+', default=['resnet18', 'mobilenet_v3_small', 'efficientnet_b0'],
                    choices=['resnet18', 'mobilenet_v3_small', 'efficientnet_b0'])
    a = p.parse_args()

    device = 'cuda'
    torch.manual_seed(2026)
    example_inputs = torch.randn(1, 3, 32, 32, device=device)

    for arch in a.archs:
        base = build_model(arch)
        base.load_state_dict(torch.load(f'checkpoints/{arch}_fp32.pt', map_location='cpu', weights_only=True))
        base = base.to(device).eval()

        fp16_model = copy.deepcopy(base).half()
        trace_and_save(fp16_model, example_inputs.half(), f'checkpoints/{arch}_fp16.pt')
        print(f'{arch}_fp16.pt saved')

        for ratio in RATIOS:
            pruned = copy.deepcopy(base)
            imp = tp.importance.MagnitudeImportance(p=2)
            pruner = tp.pruner.MagnitudePruner(
                pruned, example_inputs, importance=imp, pruning_ratio=ratio,
                ignored_layers=[FINAL_LAYER[arch](pruned)])
            pruner.step()
            trace_and_save(pruned, example_inputs, f'checkpoints/{arch}_pruned{int(ratio*100)}.pt')
            print(f'{arch}_pruned{int(ratio*100)}.pt saved')


if __name__ == '__main__':
    main()
