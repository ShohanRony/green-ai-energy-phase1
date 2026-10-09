# Feasibility checks for an x86-CPU primary block — smoke tests only, no measurement

Run 2026-10-09 as part of the pre-analysis review, ahead of the researcher's decision to make an
x86-CPU block (plus an M1-CPU block, same protocol) primary, with the x86-GPU block supplementary.
Everything here is a **smoke test**: load/run checks and static capability checks, not energy
measurement. No `pilot.py` run collected or retained any result.

---

## (a) CPU ISA flags relevant to FP16/BF16/INT8, and whether FP16 inference actually works

**Flags present** (`lscpu`, this machine — 13th Gen Intel Core i5-13450HX):
```
avx, avx2, avx_vnni, f16c
```
**Absent:** `avx512*` (any variant), `amx*`. Consistent with Intel's known Raptor Lake/Alder Lake
hybrid-core behavior — AVX-512 is fused off at the factory on these dies specifically because the
E-cores don't implement it, even where the P-cores physically could; AMX is server/Xeon-only.
`avx_vnni` (consumer AVX-VNNI, distinct from server AVX512-VNNI) accelerates INT8 dot-products —
relevant to this project's existing INT8/fbgemm CPU path. `f16c` provides hardware FP16↔FP32
**conversion** instructions only (`vcvtph2ps`/`vcvtps2ph`), not native FP16 arithmetic.

**Does FP16 inference work on this CPU, in this PyTorch build (2.7.1+cu118)? Yes — but it is
software-emulated, not hardware-accelerated.** Measured directly:

| Test | FP32 | FP16 | Ratio |
|---|---|---|---|
| Single `Conv2d(3,64,3)`, 50 iters | 1.78ms/50 | 95.78ms/50 | **54.0x slower** |
| Full ResNet-18 forward, 30 iters | 5.51ms/image | 356.60ms/image | **64.7x slower** |

No error, no crash — but FP16 on this CPU is ~54-65x slower than FP32, consistent with
`f16c`-only (conversion, not compute) support and no AVX512-FP16. Not representative of any
realistic deployment regime; an x86-CPU FP16 condition, if ever added, would need this disclosed
prominently, not treated as a normal data point.

## (b) Do pruned30/50/70 (+ `_bnrecal`) load and run on CPU, eager and TorchScript, identically?

All 18 checkpoints tested (9 zero-finetune pruned + 9 `_bnrecal`, all 3 architectures × 3 ratios),
CPU device, batch=4:

**Zero-finetune pruned (9/9): TorchScript load/run OK, eager rebuild OK, identity check PASS,
max-abs-diff = 0.0 (exact) for all nine** — eager rebuild used the same deterministic procedure
(`torch_pruning.MagnitudePruner`, seed 2026, from the FP32 base) already verified bit-identical to
the traced checkpoints on CUDA in the earlier D16-era audit; re-verified here on CPU.

**`_bnrecal` (9/9): TorchScript load/run on CPU confirmed OK** (correct output shape, no errors).
**No independent eager-rebuild identity check was possible** — these checkpoints were materialized
by a different process than this review's own scripts (confirmed as real, separate `.pt` files
under `checkpoints/`, used in `results_stage4b/*/{...}_bnrecal/`), and the exact BN-recalibration
procedure (image sample, seed) that produced them is not independently reconstructable from this
review alone. Marked **unknown**, not assumed equivalent.

## (c) Is INT8-cpu and FP32-cpu still the only x86 CPU states measured?

**Yes, confirmed directly** from every condition's own `environment.json` `arguments.device`
field across the full 30-condition Stage 4b matrix (checked Session 1): exactly 6 of 30
conditions run on `cpu` — `{arch}_int8` and `{arch}_fp32_cpu` for each of the 3 architectures.
Every FP16, pruned, and `_bnrecal` state runs on `cuda`. **No pruned or FP32-TorchScript state has
ever been measured on the x86 CPU in this project** — matches the phase-1 audit's C5 finding
exactly. An x86-CPU *block* with pruned states in it, as the new Paper-1 framing wants, does not
exist yet and would require new measurement (explicitly out of scope for this review).

## (d) CodeCarbon 3.3.1: is there a supported way to force CPU-load or TDP estimation?

**Yes — a real, documented, supported override, confirmed with file/line evidence from the
installed package** (`measurement_env/lib/python3.12/site-packages/codecarbon/`):

- `emissions_tracker.py:419` / `:424` — `OfflineEmissionsTracker`/`EmissionsTracker` accept
  `force_cpu_power: Optional[int]` and `force_mode_cpu_load: Optional[bool]` constructor
  parameters.
- `emissions_tracker.py:498-499` — docstring: *"force_cpu_power: Force the CPU max power
  consumption in watts. Use this if you know the TDP of your machine."*
- `emissions_tracker.py:512` — docstring: *"force_mode_cpu_load: Force the addition of a CPU in
  MODE_CPU_LOAD"*.
- `core/resource_tracker.py:249-278` (`set_CPU_tracking`) — confirmed from the actual control
  flow, not just the docstring: `force_mode_cpu_load` is checked **before** the RAPL/platform
  auto-detection path (`_try_platform_cpu_backend()`, where `is_rapl_available()` lives), so
  setting it **does** override RAPL even when RAPL is available on the machine — it is not merely
  a fallback for when RAPL is absent. `force_cpu_power` similarly short-circuits ahead of RAPL
  detection when set.

This directly unblocks the phase-1 audit's RQ1b (CodeCarbon with counters available vs. forced
into its TDP/load estimation mode, against a reference) — the forcing mechanism exists and is
exactly where the docstrings say it is. Not exercised here (no CodeCarbon run performed).
