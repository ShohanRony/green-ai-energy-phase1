#!/usr/bin/env python3
"""Throwaway M1 load check (outside the repo, not committed).
For each of the 23 registered session-1 conditions: verify checkpoint
SHA-256, load the TorchScript checkpoint (qnnpack engine set first for INT8),
and run one forward pass on a single CIFAR-10 test image. No powermetrics,
no guards, no energy measurement -- this only confirms every checkpoint
loads and runs under the pinned environment before a real session.
"""
import sys
import traceback
from pathlib import Path

REPO = Path.home() / 'green-ai-research' / 'green-ai-energy-phase1'
sys.path.insert(0, str(REPO / 'scripts' / 'm1'))

from m1_session_runner import M1_STRUCTURAL_EXCLUSIONS, checkpoint_for, conditions_for  # noqa: E402
from m1_condition_runner import DEFAULT_HASHES_FILE, verify_checkpoint_sha256  # noqa: E402
sys.path.insert(0, str(REPO))
from train_baseline import build_model  # noqa: E402

CHECKPOINTS_DIR = REPO / 'checkpoints'
ARCHITECTURES = ['resnet18', 'mobilenet_v3_small', 'efficientnet_b0']


def build_pairs():
    pairs = []
    for arch in ARCHITECTURES:
        for cond in conditions_for(arch, include_collapsed_bnrecal=False):
            pairs.append((arch, cond))
    return pairs


def load_one_image(target_dtype):
    import torch
    import torchvision
    from torchvision import transforms

    data_dir = CHECKPOINTS_DIR.parent / 'data'
    if not data_dir.exists():
        data_dir = Path('/tmp/cifar10')
    try:
        ds = torchvision.datasets.CIFAR10(str(data_dir), train=False, download=False,
                                           transform=transforms.ToTensor())
        raw = ds[0][0].unsqueeze(0)
    except Exception:
        torch.manual_seed(2026)
        raw = torch.randn(1, 3, 32, 32)
    norm_mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1)
    norm_std = torch.tensor([0.247, 0.243, 0.261]).view(1, 3, 1, 1)
    x = (raw - norm_mean) / norm_std
    return x.to(target_dtype)


def check_one(arch, cond):
    import torch
    engine = ''
    load_status = 'FAIL'
    forward_status = 'FAIL'
    error = ''

    try:
        if (arch, cond) == (arch, 'int8') and (arch, 'int8') in M1_STRUCTURAL_EXCLUSIONS:
            raise ValueError(f"{arch} INT8 structurally excluded under qnnpack")

        if 'int8' in cond:
            torch.backends.quantized.engine = 'qnnpack'
            engine = 'qnnpack'
        else:
            engine = str(torch.backends.quantized.engine)

        ckpt_name = checkpoint_for(arch, cond)
        if ckpt_name is None:
            # fp32_eager: no checkpoint file, mirrors measure_condition's else branch exactly.
            model = build_model(arch)
            load_status = 'OK'
        else:
            ckpt_path = CHECKPOINTS_DIR / ckpt_name
            verify_checkpoint_sha256(ckpt_path, DEFAULT_HASHES_FILE)
            try:
                model = torch.jit.load(str(ckpt_path), map_location='cpu')
            except RuntimeError:
                model = build_model(arch)
                model.load_state_dict(torch.load(str(ckpt_path), map_location='cpu', weights_only=True))
            load_status = 'OK'
        model.eval()

        try:
            target_dtype = next(model.parameters()).dtype
        except StopIteration:
            target_dtype = torch.float32
        x = load_one_image(target_dtype)
        with torch.no_grad():
            _ = model(x)
        forward_status = 'OK'

    except Exception as e:
        error = f"{type(e).__name__}: {e}"
        if load_status != 'OK':
            # truncate traceback for one-line reporting; full traceback to stderr
            traceback.print_exc(file=sys.stderr)

    return (arch, cond, engine, load_status, forward_status, error)


def main():
    pairs = build_pairs()
    print(f"{len(pairs)} conditions to check\n")
    header = f"{'arch':<20} {'condition':<22} {'engine':<10} {'load':<6} {'forward':<8} error"
    print(header)
    print('-' * len(header))
    n_fail = 0
    for arch, cond in pairs:
        arch_r, cond_r, engine, load_s, fwd_s, err = check_one(arch, cond)
        line = f"{arch_r:<20} {cond_r:<22} {engine:<10} {load_s:<6} {fwd_s:<8} {err}"
        print(line)
        if load_s == 'FAIL' or fwd_s == 'FAIL':
            n_fail += 1
    print(f"\n{len(pairs) - n_fail}/{len(pairs)} conditions OK")


if __name__ == '__main__':
    main()
