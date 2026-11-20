# Deviation: M1 v3 eager-weights and input-source fix (recorded 2026-11-20, before any M1 session data)

(a) The first complete session-1 attempt (v2, commit 2e645f8) voided itself
after 17 of 23 conditions: `from train_baseline import build_model` raised
`ModuleNotFoundError: No module named 'train_baseline'` because neither
entry script (`m1_session_runner.py`, `m1_condition_runner.py`) added the
repo root to `sys.path` -- only their own `scripts/m1/` directory, where
`train_baseline.py` does not live. This branch runs only for `fp32_eager`
(the one condition with no checkpoint file), so all three `fp32_eager`
conditions voided identically; the >2-void rule then aborted the session.
14 conditions ran cleanly; 6 (one per remaining `fp32_eager` x86-vs-M1
condition-list position) never ran.

(b) Before this fix, the `fp32_eager` branch called `build_model(arch)`
with no seeding at all, so it would have produced PyTorch's default
unseeded random weights -- not matching x86 `pilot.py`'s deterministic
`torch.manual_seed(2026); random.seed(2026)` (called unconditionally,
before `build_model`, for every condition). This was never exercised in
the v2 run because the import failed first.

(c) The successfully-completed conditions in the v2 run did not use the
synthetic-random-tensor fallback that session-1 attempt 2 (v1,
2026-11-20) would have hit -- by the time v2 ran, no diagnostic check had
confirmed this either way. Read-only diagnosis found: neither
`<repo>/data` nor `/tmp/cifar10` (the two paths the pre-v3 code checked)
exist on the Mac, so every M1 condition that reached the data-loading step
before v3 would have silently fallen through to
`torch.manual_seed(2026); raw = torch.randn(1, 3, 32, 32)` -- a synthetic
noise image, not real CIFAR-10 -- with no error or warning surfaced
anywhere. The real CIFAR-10 test split exists on the Mac, but one
directory level higher than the code checked: `~/green-ai-m1/data`
(sibling of the `repo/` checkout), not `~/green-ai-m1/repo/data`.

Fix (v3, commit e7160a8, tag m1-measurement-v3):
1. Both entry scripts add the repo root (`Path(__file__).resolve().parents[2]`)
   to `sys.path`, alongside the existing `scripts/m1/` insert.
2. `fp32_eager` now calls `torch.manual_seed(2026); random.seed(2026)`
   immediately before `build_model(arch)`, mirroring `pilot.py` exactly.
   `environment.json` records `weights` ("seeded random weights (seed
   2026), timing only, as x86 pilot.py" for eager, "checkpoint" otherwise)
   and a SHA-256 of the eager state_dict, for traceability.
3. Default CIFAR-10 resolution now also tries
   `checkpoints_dir.resolve().parent.parent / 'data'` (matches the real Mac
   layout) in addition to the original `checkpoints_dir.parent / 'data'`.
   If no candidate loads, `measure_condition` raises `RuntimeError` --
   the synthetic-tensor fallback was removed entirely. `environment.json`
   records the actual `input_source` used (dataset, path, split, index,
   transform) for every condition.

Verified through the real CLI (`m1_session_runner.py --single-condition`,
session 0, protocol-check path) for `resnet18`, `mobilenet_v3_small`, and
`efficientnet_b0` `fp32_eager`, plus `resnet18 fp32_ts` as a trained-weight
reference: all 4 ran to completion (exit 0, 24 windows.csv rows,
`condition_voided: false`, `thermal_flagged: false`), with `input_source`
confirming real CIFAR-10 from `/Users/shohan/green-ai-m1/data` in every
case -- no synthetic fallback triggered.

A one-off top-1 accuracy check (1000 CIFAR-10 test images, not a repo
file) confirmed the eager models are genuinely untrained: `resnet18
fp32_ts` (trained checkpoint) scored 0.9290; `resnet18`, `mobilenet_v3_small`,
and `efficientnet_b0` `fp32_eager` scored 0.1000, 0.1010, and 0.0840
respectively -- chance level for 10 classes, as expected under the
confirmed design (fp32_eager mirrors x86's seeded-random, timing-only
baseline; it is not meant to match fp32_ts's accuracy).

The v2 session-1 output (17 conditions, 3 of them voided on
ModuleNotFoundError) is void and is kept, not used as data.
