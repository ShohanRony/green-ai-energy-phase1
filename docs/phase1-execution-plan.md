# Phase 1 Execution Plan — Energy-Aware Efficiency of Lightweight Vision Models Under Post-Training Compression

Built from: `Proposal_GreenAI_taught.NEW.docx` (Masaryk University application, Spring 2027) + `Energy Measurement of Compressed Deep Learning Models: A Literature Review` (AITA submission). This plan operationalizes **Proposal §3 (Phase 1 methodology)** only — nothing from Phase 2 (§4) is in scope, by design, because Phase 2 needs institutional resources Shohan doesn't have yet.

Status: **not starting from zero.** A 2026-09-14→09-21 validation sprint on this exact laptop (done for a *different, earlier* proposal variant — see §1.7) already built most of the Stage 1 harness and surfaced hardware facts that materially change this plan. See §1.5.

**Decision (confirmed 2026-10-03): full push.** Shohan is executing Stage 1 through Stage 6, not stopping at the harness/artifacts boundary the proposal's §5 timeline originally scoped for the pre-admission phase. Implication carried forward: §5 of the proposal will need rewriting once Phase 1 is actually complete pre-admission — that's a writing task for later, flagged here so it isn't forgotten.

---

## 0. Scope lock — what runs on owned hardware, what doesn't

| Item | In scope for this plan? | Why |
|---|---|---|
| Full RQ1–RQ3 factorial: 3 models × 6 compression states × 2 platforms × ≥30 runs (1,080 batches) | **Yes, with the state substitution in §1.6** | Proposal §6 states explicitly this needs no purchase — both platforms are already owned |
| Raspberry Pi + inline USB power meter (3rd platform, RAPL ground-truthing) | **No** | Not owned; Proposal §9 lists it as optional, no stated result depends on it |
| Supplementary 224×224 experiment (FP32/INT8 subset) | **Stretch, attempt last** | Needs a resized dataset subset; feasible but not core — do only after the core matrix is banked |
| Phase 2 WP1 (ImageNet, ViT) | **No** | Needs GPU cluster + ~150GB storage (Proposal §4.1) |
| Phase 2 WP2 (datacenter accelerators) | **No** | Needs exclusive-node scheduling + counter access (Proposal §4.2) |

**Conclusion: the entire core contribution (RQ1, RQ2, RQ3) is executable on the two laptops Shohan already owns** — with one compression state substituted for tooling reasons, decided in §1.6.

---

## 1. Hardware feasibility audit

**x86 platform — Lenovo LOQ 15IRX9**
Intel Core i5-13450HX, RTX 3050 6GB (GA107), 24GB DDR5-4800, Linux Mint 22.3 XFCE (dual-boot with Windows).

- CPU energy: RAPL via `perf-events`, per Raffin & Trystram (2024) — their preferred access path (fastest, lowest idle-inflation). The psys-vs-package question and the root/capability question are addressed empirically in §1.5 — a udev rule already exists making RAPL sysfs world-readable, no sudo needed. **Still unconfirmed:** whether that rule exposes `psys` specifically or only `package` — needs a one-line check (`ls /sys/class/powercap/intel-rapl/intel-rapl:*/name`) before Stage 1 is marked complete.
- GPU energy: NVML via `nvidia-smi` / `pynvml`. §1.5 already found the critical fact here: **don't trust a sampling rate faster than ~0.3–0.5s on this specific GPU** — it's not a tuning choice, it's a measured counter artifact.
- **Critical pre-check, now answered (and it's bad news) — see §1.6:** PyTorch INT8 on this GPU does not run; it segfaults. The GA107 kernel-dispatch question from the previous version of this plan is resolved, just not the way the proposal assumed.

**ARM platform — MacBook Air M1**
8GB RAM, fanless/passive cooling.

- No RAPL equivalent. Use `powermetrics` (needs sudo per invocation — set up a narrowly-scoped passwordless sudo rule for just that binary, not global, for repeatable scripting). **Not yet set up** — §1.5's validation sprint was entirely on the LOQ; nothing has been run on the M1 yet.
- 8GB RAM is tight for this workload. Swapping during a measurement run contaminates the energy reading (disk I/O power ≠ inference power) and must be avoided — close everything else, monitor memory pressure, keep batch sizes modest.
- Passive cooling means this is also the platform where the proposal's "thermal observation" (§3, loosely held 4th hypothesis) will actually show up. Don't fight the throttling — record it.

**Raspberry Pi:** not owned. Excluded, per Scope Lock above.

---

## 1.5 Prior validation work on this exact hardware — what's already done, reusable now

A 2026-09-14→09-21 sprint (documented in `STATUS_REPORT.md`, uploaded 2026-10-03) already built and stress-tested most of what Stage 1 of this plan asks for, on this exact LOQ. None of it needs redoing; it needs **porting from its current repo layout into this proposal's harness** and **extending from pilot scale to the full factorial**.

**Already built and working:**
- `measurement_env` — Python 3.12 venv, torch 2.7.1+cu118, torchvision/torchaudio, onnxruntime-gpu 1.29.0, numpy/pandas/sklearn/matplotlib/jupyter/psutil, `perf`, `powerstat`, and a udev rule making RAPL sysfs world-readable (no sudo needed). **This is Stage 1's environment-setup bullet, done.**
- CIFAR-10 (163MB) and CIFAR-10-C (2.8GB), both downloaded and MD5-verified. **This is Stage 2's dataset-acquisition step, done.**
- `research/energy-pilot` — a working RAPL+NVML paired measurement harness, 4/4 unit tests passing, already run for real: idle 17–29W, active 71–78W, 0.0447 J/image, physically plausible. **This is most of Stage 1's "build the harness" + a head start on Stage 3's pilot** — currently at 3 A/A pairs; needs scaling to ≥30 and extending across all states.
- `torch-pruning` confirmed as the library to use for structured pruning. Native `torch.nn.utils.prune` produces **zero real speedup** (masks weights, still computes/stores densely — the nominal-vs-realized trap, reproduced empirically). `torch-pruning` does genuine physical channel removal, verified up to ~3.5x real speedup at 70%. **Use `torch-pruning` for all three pruning states in Stage 2.**
- Training-time feasibility for a 10-seed × 3-architecture grid (VGG-19-BN, ResNet-18, MobileNetV3-Small) at ≈68.6h / 2.9 days sequential, plus a GPU-contention finding: two concurrent GPU jobs slow both by ~2x. **New checklist item:** never run a second GPU job during a measurement run. EfficientNet-B0 wasn't benchmarked in this sprint — get its own timing check in Stage 2.

---

## 1.6 Decided: INT4 substitution

Validated finding: **INT4 has no `Conv2d` kernel in any installed library** (`bitsandbytes`-style libraries only cover `Linear4bit`; ResNet-18, MobileNetV3-Small, and EfficientNet-B0 are convolution-dominated). This is a missing-tooling problem, not a measurement problem, and it conflicts directly with Proposal §3's six named states.

INT8 is also narrower than the proposal assumed: works only via `torchvision.models.quantization.resnet18` (CPU-only, fbgemm) — GPU dispatch **segfaults**. MobileNetV3 has no official quantized variant. ONNX Runtime's CUDA/TensorRT providers claim GPU support but silently fall back to CPU.

**Decision (confirmed 2026-10-03):**
1. **INT4 → FP16, as the sixth compression state**, for all three models. FP16 is already verified working (1.7x genuine speedup on ResNet-18 and MobileNetV3-Small, dtype-confirmed).
2. **The substitution is disclosed, not hidden.** Stage 2's methodology notes, and eventually the paper's methodology section, state plainly: *"INT4 post-training quantization was attempted and found infeasible for convolutional layers on available consumer tooling (no Conv2d INT4 kernel in any evaluated library as of 2026); FP16 was substituted as the sixth compression state. This finding is itself consistent with the nominal-vs-realized compression gap documented in the literature review."* That sentence goes into Stage 2's deliverable notes now, so it isn't reconstructed from memory later.
3. **MobileNetV3-Small INT8:** build a manual PTQ pipeline with `torch.quantization` (stays faithful to Nagel et al.'s procedure) rather than substituting a different model.
4. **INT8, all models:** runs on CPU/fbgemm. The GPU energy column for INT8 is structurally empty — stated as a limitation, not pursued further.

**Revised six states, all three models: FP32, INT8 (CPU/fbgemm), FP16, pruning 30%, pruning 50%, pruning 70%.**

---

## 2. Pre-flight checklist (apply to every run, not just the first)

1. Instrument + accuracy class + measurement boundary stated for every number (RAPL package vs psys vs NVML vs powermetrics — never let a number float without its boundary).
2. Idle baseline measured before **and** after each condition block, reported, not silently subtracted.
3. First run of every configuration discarded (cold-cache/cold-thermal).
4. **Two separate sampling-rate rules:**
   - RAPL/CPU (perf-events): cap **≤100 Hz**.
   - NVML/GPU on this specific RTX 3050: sample **no faster than every 0.3–0.5s** (counter-telescoping artifact below that; swept empirically: 20ms→291W, 50ms→127W, 100ms→85W, 200ms→51W, 300ms→26W plausible, 500ms→16W).
5. Batch size recorded for every figure; vary 2–3 values as a declared control variable.
6. CPU governor fixed to performance mode for every run, both platforms.
7. Driver/library/runtime versions pinned and logged.
8. Nominal compression reported **together with** realized-compression evidence (kernel trace, or `torch-pruning`'s own FLOPs/parameter report).
9. Thermal state and session length logged on the M1 specifically.
10. ≥30 runs per condition; mean, SD, 95% CI reported.
11. Phase separation: inference energy isolated from data-loading/post-processing.
12. Raw traces retained, not just summary statistics.
13. **No concurrent GPU jobs during any measurement run.**

---

## 3. Stage-by-stage execution plan

### Stage 1 — Harness & instrumentation build
- ~~RAPL access~~ — **done**; confirm psys-vs-package naming only.
- ~~NVML reads~~ — **done**, safe sampling interval known.
- Set up `powermetrics` on the M1 with a scoped sudo rule — **not started.**
- Port `research/energy-pilot` into this proposal's own harness repo (see §1.7 — don't tangle with the other proposal's repo).
- Wrap CodeCarbon to run concurrently with hardware counters (paired comparison for RQ1) — **not yet done.**
- Confirm dual-trigger phase-separation boundary matches Bartoli et al.'s design, not a simpler before/after split — **partially done.**
- **Exit test:** uncompressed ResNet-18 ×10 reps, both platforms, all Section 2 checklist items satisfied with real logged values. x86: already at n=3, scale to 10. M1: not started.
- Est. **1–1.5 weeks** of evenings.

### Stage 2 — Compression artifacts
- FP32 baselines: ResNet-18, MobileNetV3-Small, EfficientNet-B0 (data already downloaded).
- INT8: ResNet-18 via `torchvision.models.quantization.resnet18` (confirmed). MobileNetV3-Small via manual PTQ pipeline (§1.6). EfficientNet-B0 INT8 path not yet checked.
- FP16: all three models (§1.6 substitution, already verified working on two of three).
- Structured pruning 30/50/70% via `torch-pruning`, with zero-finetune-vs-brief-recovery policy decided and stated (still open — decide in this stage, not during Stage 4).
- EfficientNet-B0 per-epoch timing check (not covered by the prior sprint).
- Deliverable: 18 artifacts (revised state list) + accuracy table + the §1.6 disclosure sentence.
- Est. 1.5–2 weeks.

### Stage 3 — RQ1 pilot
- Extend the existing 3-pair pilot (one model) to all 6 states, x86 only, ~10 reps/state.
- Deliverable: go/no-go on harness correctness before the full matrix.
- Est. 3–4 days.

### Stage 4 — Full measurement matrix
- Full factorial, x86 then ARM, ≥30 reps/condition, full checklist applied.
- Deliverable: complete raw dataset, logs retained.
- Est. 3–4 weeks, evenings.

### Stage 5 — Statistical analysis
- RQ1/RQ2/RQ3 stats, 5 figures, results summary against Proposal §1's hypotheses.
- Est. 2 weeks.

### Stage 6 — Documentation & release
- Public repo, results write-up, §1.7 repo-layout decision resolved.
- Ongoing.

---

## 4. Tooling: where each coding assistant fits

- **Local coding-assistant CLI — primary driver for Stage 1, 2, 4** (same tool that ran the §1.5 sprint; direct continuity).
- **Antigravity — secondary, for well-isolated non-protocol-sensitive work:** Stage 2 scaffolding, Stage 5 notebooks, refactors on already-reviewed code.
- **Neither tool decides unsupervised:** sampling rates, idle-baseline convention, the §1.6 substitution, nominal-vs-realized distinctions. Verify hardware-behavior claims yourself.
- **This cloud session:** methodology fidelity, statistical interpretation, manuscript writing, periodic protocol-compliance reads of the pushed repo.

---

## 5. Open technical items

- psys vs package RAPL domain — one-line check remaining.
- ~~INT8 GPU dispatch~~ — resolved: segfaults, CPU/fbgemm only. §1.6.
- ~~INT4 feasibility~~ — resolved: infeasible, FP16 substituted. §1.6.
- Pruning fine-tuning policy (zero-finetune vs brief recovery) at 70% — still not pinned down; decide in Stage 2.
- M1 passive sudo rule for `powermetrics` — not set up.
- MobileNetV3-Small INT8 PTQ pipeline — needs building.
- EfficientNet-B0 — INT8 behavior and per-epoch timing both unchecked.

---

## 1.7 Housekeeping: repo and proposal separation

1. **The prior sprint (§1.5) validated a *different* proposal** ("Evaluating Compressed Neural Networks Under Measured Energy and Reliability Constraints" — calibration/reliability angle, TOST-based, Mitra et al. 2024 reproduction), not the Masaryk FLOPs-energy proposal this plan operationalizes. Shared hardware/infrastructure, different RQs. This plan reuses the infrastructure only — the other proposal's RQs and mitra-reproduction results stay with that proposal, not merged into this one's write-up.
2. **Repo sprawl:** `measurement_env`, `research/mitra-reproduction`, `research/energy-pilot`, `research/thesis_arch_results`, `research/push_workspace/energy-reliability-proposal`, plus `ShohanRony/green-ai-proposal` and `ShohanRony/rangamati-landcover-change-1993-2023` — none of the first three pushed to GitHub yet. Decide the final repo layout for *this* proposal's work before Stage 6's public release.
