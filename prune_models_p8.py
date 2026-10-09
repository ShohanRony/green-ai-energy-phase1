import argparse
import torch
import torch_pruning as tp
from pathlib import Path
from train_baseline import build_model

FINAL_LAYER = {
    'resnet18': lambda m: m.fc,
    'mobilenet_v3_small': lambda m: m.classifier[3],
    'efficientnet_b0': lambda m: m.classifier[1],
}

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--arch', required=True)
    p.add_argument('--ratio', type=float, required=True)
    p.add_argument('--seed', type=int, required=True)
    a = p.parse_args()

    in_path = Path(f"checkpoints_p8/{a.arch}_fp32_{a.seed}.pt")
    if not in_path.exists():
        raise FileNotFoundError(f"Missing base model {in_path}")
    
    ratio_int = int(a.ratio * 100)
    out_path = Path(f"checkpoints_p8/{a.arch}_pruned{ratio_int}_{a.seed}.pt")
    
    if out_path.exists():
        print(f"Already exists: {out_path}")
        return

    model = build_model(a.arch)
    model.load_state_dict(torch.load(in_path, map_location='cpu', weights_only=False))
    model.eval()
    
    example_inputs = torch.randn(1, 3, 32, 32)
    imp = tp.importance.MagnitudeImportance(p=2)
    ignored_layers = [FINAL_LAYER[a.arch](model)]

    pruner = tp.pruner.MagnitudePruner(
        model,
        example_inputs,
        importance=imp,
        ch_sparsity=a.ratio,
        ignored_layers=ignored_layers,
    )

    pruner.step()
    
    # Save the full pruned module
    torch.save(model, out_path)
    print(f"Pruned {a.arch} at {a.ratio} (seed {a.seed}) -> {out_path}")

if __name__ == '__main__':
    main()
