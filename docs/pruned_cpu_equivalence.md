# Zero-Finetune Pruned CPU Functional-Equivalence Check (Read-Only, No Energy Measurement)

## Background and Purpose

All 9 zero-finetune structured pruning checkpoints (`checkpoints/{model}_{state}.pt` for `model` $\in$ `{resnet18, mobilenet_v3_small, efficientnet_b0}` and `state` $\in$ `{pruned30, pruned50, pruned70}`) had accuracy previously evaluated on **CUDA** in `results_accuracy/summary.csv` (`*_preds.npy`), whereas their energy consumption was measured on **x86-CPU** in the confirmatory CPU block.

Prior to this check, CPU-vs-CUDA functional equivalence had been empirically documented only for the 9 `_bnrecal` checkpoints (`docs/bnrecal_cpu_equivalence.md`, deviation D37). On 2026-10-16, the researcher requested this check to verify whether zero-finetune pruned checkpoints behave equivalently on CPU versus CUDA.

## Methodology and Script Pipeline

The evaluation is performed by `scripts/eval_pruned_cpu.py` on the full CIFAR-10 test set ($n = 10{,}000$, `shuffle=False`) using `measurement_env/bin/python3` on CPU only. Preprocessing, loading, and batching match `scripts/eval_artifacts.py` exactly:
- **Model loading:** `torch.jit.load(map_location='cpu')` with fallback to `build_model(arch)` + `load_state_dict(...)` (copied from `scripts/eval_artifacts.py`, lines 49–55).
- **Normalization constants:** `MEAN = [0.4914, 0.4822, 0.4465]`, `STD = [0.247, 0.243, 0.261]` (copied from `scripts/eval_artifacts.py`, lines 38–39).
- **Test transform:** `transforms.ToTensor()` on the full CIFAR-10 test split (copied from `scripts/eval_artifacts.py`, lines 69–72).
- **Batching & dtype casting:** Batch size 200, matching `eval_artifacts.py` dtype detection rule (copied from `scripts/eval_artifacts.py`, lines 83–95).

Predictions were saved to new files `results_accuracy/{model}_{state}_cpu_preds.npy` without modifying existing prediction files.

## Equivalence Rule (Plan v1 Section 9)

Per `docs/stage5_plan_v1.md` section 9 (and `docs/bnrecal_cpu_equivalence.md`):
- A checkpoint is functionally equivalent between CPU and CUDA if:
  $$\text{McNemar exact two-sided } p \ge 0.05 \quad \text{AND} \quad |\text{accuracy difference}| \le 0.05\text{ percentage points}$$
- Here, $b$ is the count of images correct on CPU and incorrect on CUDA; $c$ is the count of images incorrect on CPU and correct on CUDA. The exact two-sided test uses $\text{Binomial}(n=b+c, p=0.5)$ (or $p=1.0$ when $b+c=0$).

## Results Table

| Model | State | CPU Correct / 10000 | CUDA Correct / 10000 | Diff (pp) | $b$ | $c$ | McNemar $p$ | Class Diffs | Equivalent |
|---|---|---|---|---|---|---|---|---|---|
| resnet18 | pruned30 | 6482 | 6482 | +0.00 | 0 | 0 | 1.0000 | 1 | **yes** |
| resnet18 | pruned50 | 1761 | 1761 | +0.00 | 0 | 0 | 1.0000 | 3 | **yes** |
| resnet18 | pruned70 | 1035 | 1035 | +0.00 | 0 | 0 | 1.0000 | 0 | **yes** |
| mobilenet_v3_small | pruned30 | 1716 | 1716 | +0.00 | 1 | 1 | 1.0000 | 4 | **yes** |
| mobilenet_v3_small | pruned50 | 1000 | 1000 | +0.00 | 0 | 0 | 1.0000 | 0 | **yes** |
| mobilenet_v3_small | pruned70 | 1000 | 1000 | +0.00 | 0 | 0 | 1.0000 | 0 | **yes** |
| efficientnet_b0 | pruned30 | 6030 | 6029 | +0.01 | 1 | 0 | 1.0000 | 1 | **yes** |
| efficientnet_b0 | pruned50 | 1333 | 1335 | -0.02 | 0 | 2 | 0.5000 | 3 | **yes** |
| efficientnet_b0 | pruned70 | 1000 | 1000 | +0.00 | 0 | 0 | 1.0000 | 0 | **yes** |

## Summary of Findings

- **9 of 9 checkpoints pass** the registered functional equivalence rule.
- Net accuracy differences range from $-0.02$ to $+0.01$ percentage points (at most 2 images out of 10,000 net change), well within the $\le 0.05$ pp threshold.
- McNemar exact $p$-values are all $\ge 0.5000$, well above the $\alpha = 0.05$ significance threshold.
- Individual prediction disagreements (differing predicted class) range from 0 to 4 images out of 10,000, consistent with expected floating-point kernel differences across CPU and CUDA backends.
- **Deployability tiers:** All 9 zero-finetune pruned checkpoints remain classified as **not deployable** under Plan v1 section 9 (accuracies are 10.0%–64.8%, far below 99% of the respective FP32 reference model). No state changes tier.
