#!/usr/bin/env python3
"""Export FP32 TorchScript checkpoints for all architectures.

Matches the method and example input used in materialize_checkpoints.py:
1x3x32x32 trace, eval() mode, exported to checkpoints/{arch}_fp32_ts.pt.
"""
from pathlib import Path
import sys
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from train_baseline import build_model

ARCHITECTURES = ['resnet18', 'mobilenet_v3_small', 'efficientnet_b0']


def export_and_verify(ckpt_dir=None):
    if ckpt_dir is None:
        ckpt_dir = ROOT / 'checkpoints'
    else:
        ckpt_dir = Path(ckpt_dir)

    torch.manual_seed(2026)
    example_input = torch.randn(1, 3, 32, 32)

    results = {}
    for arch in ARCHITECTURES:
        src_path = ckpt_dir / f'{arch}_fp32.pt'
        dst_path = ckpt_dir / f'{arch}_fp32_ts.pt'

        model = build_model(arch)
        state_dict = torch.load(src_path, map_location='cpu', weights_only=True)
        model.load_state_dict(state_dict)
        model.eval()

        traced = torch.jit.trace(model, example_input)
        traced.save(str(dst_path))

        # Verification: load via torch.jit.load
        loaded = torch.jit.load(str(dst_path))
        loaded.eval()

        torch.manual_seed(9999)
        test_input = torch.randn(1, 3, 32, 32)
        with torch.no_grad():
            eager_out = model(test_input)
            ts_out = loaded(test_input)
            max_abs_diff = (eager_out - ts_out).abs().max().item()

        results[arch] = {
            'dst_path': str(dst_path),
            'max_abs_diff': max_abs_diff,
        }
        print(f"Exported {arch}: {dst_path} (max abs diff vs eager: {max_abs_diff:.6e})")

    return results


def main():
    export_and_verify()


if __name__ == '__main__':
    main()
