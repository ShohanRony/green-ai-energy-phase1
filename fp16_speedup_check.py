"""Stage 2 Task 3: FP16 conversion check -- dtype confirmation + realized GPU speedup.

Usage: python3 fp16_speedup_check.py --arch efficientnet_b0 --checkpoint checkpoints/efficientnet_b0_fp32.pt
"""
import argparse, json, time

import torch
from train_baseline import build_model


def timed_forward(model, x, reps, warmup=10):
    with torch.no_grad():
        for _ in range(warmup):
            model(x)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(reps):
            out = model(x)
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / reps, out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--arch', required=True, choices=['resnet18', 'mobilenet_v3_small', 'efficientnet_b0'])
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--batch-size', type=int, default=128)
    p.add_argument('--reps', type=int, default=50)
    a = p.parse_args()

    assert torch.cuda.is_available(), 'FP16 speedup is a CUDA-only comparison on this hardware'
    device = 'cuda'
    torch.manual_seed(2026)

    model = build_model(a.arch)
    model.load_state_dict(torch.load(a.checkpoint, map_location='cpu', weights_only=True))
    model = model.eval().to(device)
    x = torch.randn(a.batch_size, 3, 32, 32, device=device)

    fp32_time, fp32_out = timed_forward(model, x, a.reps)

    model_fp16 = model.half()
    x_fp16 = x.half()
    fp16_time, fp16_out = timed_forward(model_fp16, x_fp16, a.reps)

    dtype_ok = fp16_out.dtype == torch.float16
    speedup = fp32_time / fp16_time

    result = dict(arch=a.arch, fp32_ms=fp32_time * 1000, fp16_ms=fp16_time * 1000,
                  speedup=speedup, dtype_confirmed=dtype_ok, batch_size=a.batch_size, reps=a.reps)
    print(json.dumps(result, indent=2))
    assert dtype_ok, 'FP16 forward pass did not produce float16 output -- silent fallback'


if __name__ == '__main__':
    main()
