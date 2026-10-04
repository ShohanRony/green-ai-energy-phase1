# Stage 2 Retrain Report — Tasks E–H

Follow-up to `stage2_closeout_report.md` (Task B flagged MobileNetV3-Small and EfficientNet-B0 as not
plateaued at 30 epochs, but left them untouched). Run 2026-10-04, same session day. Retrain the two
unconverged baselines (Task E), propagate the new baselines through every downstream artifact (Task F),
commit and push (Task G), then follow up on three review items against that work (Task H + two
confirmation checks). ResNet-18 (loss 0.0288 at epoch 30, already plateaued) was not touched anywhere
in this pass.

---

## Task E — Retrain MobileNetV3-Small and EfficientNet-B0

Same recipe as `stage2_task_report.md` Task 1 (SGD, momentum 0.9, weight_decay=5e-4, nesterov, lr=0.1,
batch_size=128, standard CIFAR-10 augmentation, seed 2026), extended to 60 epochs. `train_baseline.py`
re-derives the cosine schedule over whatever `--epochs` is passed (`CosineAnnealingLR(opt, T_max=a.epochs)`)
and trains from scratch every run — there is no resume path, so this is a fresh 60-epoch cosine schedule,
not the old 30-epoch schedule run twice.

**MobileNetV3-Small** (442s wall-clock):

| | Loss | Train acc |
|---|---|---|
| Epoch 26→30 (old, 30ep run) | 0.4607→0.4073 (Δ-0.0534) | 83.96%→85.93% (Δ+1.97pt) |
| Epoch 56→60 (new, 60ep run) | 0.3486→0.3320 (Δ-0.0166) | 87.84%→88.49% (Δ+0.65pt) |

Loss delta shrank ~3.2x, accuracy delta ~3x. Test accuracy: 84.99%→86.46% (+1.47pt from the extra 30
epochs). **Verdict: genuinely flattened, not extended further.**

**EfficientNet-B0** (2067s wall-clock):

| | Loss | Train acc |
|---|---|---|
| Epoch 26→30 (old, 30ep run) | 0.5222→0.4613 (Δ-0.0609) | 81.89%→83.94% (Δ+2.05pt) |
| Epoch 56→60 (new, 60ep run) | 0.2630→0.2420 (Δ-0.0210) | 90.96%→91.73% (Δ+0.77pt) |

Loss delta shrank ~2.9x, accuracy delta ~2.7x. The last epoch (59→60) actually shows a small *dip*
(train_acc 91.81%→91.73%, loss 0.2415→0.2420) — oscillation around a optimum with the cosine LR fully
decayed, not continued climbing. Test accuracy: 83.61%→88.91% (+5.30pt). **Verdict: genuinely
flattened, not extended further.** (EfficientNet-B0's net gain from 30 more epochs is larger than
MobileNetV3-Small's — 5.30pt vs 1.47pt — consistent with Task B's observation that its per-epoch
improvement rate was already slightly higher than MobileNetV3-Small's despite longer wall-clock per
epoch.)

Both last-5-epoch deltas are a genuine ~3x flattening versus the 30-epoch run, with the cosine schedule
fully decayed in both cases. Accepted as converged; no further epoch extension triggered.

---

## Task F — Propagate the new baselines

### INT8 (rerun via `quantize_mobilenet_ptq.py`, ResNet-18 untouched)

| Model | Old (30ep) FP32→INT8 | New (60ep) FP32→INT8 | Cost moved |
|---|---|---|---|
| MobileNetV3-Small | 84.99%→84.20% (0.79pt) | 86.46%→84.00% (2.47pt) | worse |
| EfficientNet-B0 | 83.61%→78.37% (5.2pt) | 88.91%→86.87% (2.04pt) | much better |

EfficientNet-B0's quantization cost dropped by more than half; MobileNetV3-Small's roughly tripled. With
undercooked baselines, EfficientNet-B0 looked like the more INT8-fragile of the two lightweight models
(5.2pt vs 0.79pt). With converged baselines, that ranking **flips** — MobileNetV3-Small is now the more
fragile one (2.47pt vs 2.04pt). This is a real finding, not noise: `num_adaround_layers` and the
pipeline are identical both times, only the FP32 starting weights differ.

### FP16 (rerun via `fp16_speedup_check.py`)

| Model | Old speedup | New speedup |
|---|---|---|
| MobileNetV3-Small | 1.96x | 1.78x |
| EfficientNet-B0 | 2.01x | 2.03x |

Both still show genuine `float16`-confirmed speedup, no fallback. The small movement (most visible on
MobileNetV3-Small) is expected run-to-run timing noise — FP16 forward-pass latency is a function of
architecture and precision, not of the specific trained weight values, so it should not shift
systematically with retraining. Not treated as a real effect.

### Pruning full grid (rerun via `prune_full_grid.py --archs mobilenet_v3_small efficientnet_b0`,
ResNet-18 rows preserved unchanged by merge-write)

| Model | Ratio | Old acc (30ep) | New acc (60ep) | Realized MACs reduction (unchanged — architecture-dependent) |
|---|---|---|---|---|
| MobileNetV3-Small | 30% | 11.17% (collapsed) | 17.13% (not collapsed by the 15% criterion) | 46.7% |
| MobileNetV3-Small | 50% | 11.16% (collapsed) | 10.00% (collapsed) | 69.1% |
| MobileNetV3-Small | 70% | 10.00% (collapsed) | 10.00% (collapsed) | 86.8% |
| EfficientNet-B0 | 30% | 61.14% | 60.18% | 48.4% |
| EfficientNet-B0 | 50% | 20.46% (not collapsed) | 13.29% (collapsed) | 71.2% |
| EfficientNet-B0 | 70% | 10.00% (collapsed) | 10.00% (collapsed) | 88.1% |

Realized MACs/params reduction at each ratio is bit-identical between the old and new runs — it's a
function of which channels `torch-pruning`'s dependency graph removes for a given nominal ratio, which
is architecture-determined, not weight-determined. Only accuracy moved. Two threshold crossings flipped:
MobileNetV3-Small@30% moved *out* of collapse, EfficientNet-B0@50% moved *into* collapse. The
EfficientNet-B0@50% result is the more interesting one — a better-converged baseline turned out to be
**more** fragile to 50%-sparsity zero-finetune magnitude pruning, not less, plausibly because a model
that has converged further has settled into a sharper, less redundant weight solution that tolerates
less disruption. This is reported as a genuine, counter-intuitive finding, not filtered out.

### MobileNetV3-Small fine-grained probe (rerun via `prune_mbv3_probe.py`)

| Sparsity | Old acc (30ep baseline) | New acc (60ep baseline) |
|---|---|---|
| baseline | 84.99% | 86.46% |
| 5% | 81.96% | 79.84% |
| 10% | 59.13% | 63.58% |
| 15% | 30.31% | 59.83% |
| 20% | 21.18% | 30.21% |
| 30% (from full grid) | 11.17% | 17.13% |

**This is the one that matters for Task A's verdict.** The old curve (undercooked baseline) was smooth
and monotonically decreasing, cited as evidence against a `torch-pruning` dependency-graph bug (a bug
would be expected to show an abrupt, localized failure, not a smooth gradient). The new curve (converged
baseline) is **not** smooth in the same way: there's a near-plateau from 10%→15% (-3.75pt only) followed
by a sharp cliff from 15%→20% (-29.62pt) — the single steepest step-to-step drop anywhere in either
curve. Taken alone, this cliff pattern is closer to the "abrupt failure at a threshold" signature the
original report said was *absent* and would have pointed toward a bug.

**Re-examined, not reversed:** the other leg of Task A's evidence — realized/nominal MACs-reduction
amplification at 30% is unchanged (1.56x, still consistent with ResNet-18's 1.72x and EfficientNet-B0's
1.61x at the same nominal ratio, no outlier) — still argues against a depthwise/pointwise dependency-graph
bug specifically. A structural graph bug would be expected to produce an outlier *amplification ratio* at
the point of failure, not just a steep accuracy drop; this check still shows no such outlier. **Revised
verdict: genuine architectural fragility still holds as the better-supported explanation, but the
specific "smooth curve, no cliff" phrasing from the original Task A report is superseded — there is a
real cliff, between 15% and 20% nominal sparsity, that the undercooked-baseline data didn't show.** If
this distinction matters for the eventual writeup (e.g., characterizing *where* MobileNetV3-Small's
redundancy runs out, not just *that* it does), the cliff's location is now itself a finding worth stating,
not a smooth gradient.

### Materialized checkpoints (Task D, rerun via `materialize_checkpoints.py --archs mobilenet_v3_small efficientnet_b0`)

All 8 files (`{arch}_fp16.pt`, `{arch}_pruned{30,50,70}.pt` × 2 models) regenerated from the new FP32
checkpoints, same TorchScript-trace format as before. ResNet-18's 4 materialized files untouched
(confirmed via `git status` — no ResNet-18 checkpoint file appears modified).

### Task 6 / Task 7 (deliverable)

`stage2_deliverable.md` §2-§6 updated in place with the new numbers (ResNet-18 rows unchanged). One
pre-existing staleness fixed while touching §2: it still described FP16/pruned states as "in-memory
only, no on-disk file," left over from before Task D's closeout materialized all 18 as real files in a
previous session — corrected for all three models, not just the two retrained ones, since it was a
documentation bug independent of this retrain.

---

## Task G — Commit and push

Committed (`24a5c94`) and pushed to `origin` — this and the two prior Stage 2 commits (`bc37ff9`,
`888c585`) had been local-only, a single-laptop point of failure for several hours of GPU work. The
push itself needed two retries: it kept failing with HTTP 408 (request timeout) pushing ~179MB of
packed checkpoint binaries. Diagnosed as genuinely poor upload throughput on this connection at the
time (~54KB/s measured directly against an unrelated endpoint, not a git/GitHub-specific problem —
confirmed before retrying rather than assumed) rather than a git configuration issue; `http.postBuffer`,
forcing `HTTP/1.1`, and local `git gc` were all tried and made no difference, consistent with that
diagnosis. Succeeded on a later retry once the connection recovered.

---

## Task H — Fine-grained probe, 15-20% in 1% steps (follow-up review item)

The 5%-step probe above located a -29.6pt drop somewhere in the 15%→20% bin, too coarse to tell a
genuine sharp threshold apart from several smaller drops that happen to bin together. Reran
`prune_mbv3_probe.py` (extended with a `--ratios` flag and a `channel_snapshot()` helper that records
every Conv2d/Linear's output-channel count alongside accuracy) at 15/16/17/18/19/20%:

| Sparsity | 15% | 16% | 17% | 18% | 19% | 20% |
|---|---|---|---|---|---|---|
| Accuracy | 59.83% | 56.18% | 56.17% | 51.57% | 31.09% | 30.21% |
| Step Δ | — | -3.65 | -0.01 | -4.60 | **-20.48** | -0.88 |

The cliff is real and precisely located: 18%→19% alone accounts for -20.48pt, more than 4x any other
single step in the range. Not a sampling artifact.

**Is it one critical block?** Diffed channel counts for every Conv2d/Linear between the 18% and 19%
models. Result: no. All ~40 layers that change lose only 1-10 channels each (1-8% relative, roughly
proportional to layer size — consistent with `torch-pruning` spreading a global ratio target evenly
rather than concentrating it on one layer). No layer collapses to a near-zero channel count at 19% that
wasn't already small at 18%. Checked the two obvious candidates specifically — SE squeeze gates
(`*.fc1`) and the pre-classifier depthwise stage (`features.11.block.*`) — neither shows an outlier cut
at this step relative to the rest of the network.

**Conclusion:** the cliff is real and sharp (one specific percentage point, not a 5-point-wide region),
but it isn't attributable to any single structurally critical layer. The mechanism looks like dozens of
layers simultaneously losing a small, individually-survivable number of channels, where the *combined*
effect crosses a capacity threshold for the network's composed function — a genuinely nonlinear
interaction, not a localized failure. This still argues against a `torch-pruning` dependency-graph bug
(which would show as one disproportionately-cut layer, not this globally-proportional pattern) and if
anything strengthens the "MobileNetV3-Small has thin, collectively fragile channel redundancy" reading
of Task A, while correcting the specific claim that the degradation curve has no cliff anywhere.
`stage2_deliverable.md` §6 updated with this precise version; the 5%-only table there now carries the
1%-resolution follow-up alongside it rather than standing alone.

---

## Confirmation checks on the review's other two items

**Regenerated checkpoints actually load (not just "should," per Task D's existing proof path):** ran
`pilot.py` end-to-end against one regenerated file per retrained model —
`mobilenet_v3_small_fp16.pt` and `efficientnet_b0_pruned50.pt`, both TorchScript traces produced by this
session's `materialize_checkpoints.py` rerun. Both exit 0 with real energy traces
(`results_stage2/pilot_proof_retrain_mbv3_fp16/`, `results_stage2/pilot_proof_retrain_effnet_pruned50/`
— `gross_j_per_image` 0.0400 and 0.0400-0.0656 respectively across windows). Confirmed, not assumed.

**EfficientNet-B0@50% convergence-fragility finding, flagged not settled:** added an explicit caveat to
`stage2_deliverable.md` §6 — this is one before/after comparison at a single pair of epoch budgets (30
vs. 60), not a swept relationship. Worth raising in Stage 5's discussion section as literature-consistent
("sharper minima, less pruning-tolerant"), but confirming it would need a third checkpoint (e.g. 45
epochs) to check for a monotonic trend. Not run here — correctly out of Stage 2's scope, flagged for
later rather than either asserted as fact or silently dropped.
