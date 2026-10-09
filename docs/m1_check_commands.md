# M1 MacBook check commands — for the researcher to run, not executed here

**No assumption of access is made.** This environment has no connection to any Apple hardware or
macOS install — nothing below has been run or verified against a real M1. Every command is given
as exactly what to type; output/behavior is genuinely unknown from here and must be checked by
the researcher. Where I'm not fully confident a flag name is current, I've said so explicitly
rather than guess silently (the project's standing rule).

---

## 1. PyTorch build — MPS availability and version

```bash
python3 -c "
import torch
print('torch version:', torch.__version__)
print('MPS built:', torch.backends.mps.is_built())
print('MPS available:', torch.backends.mps.is_available())
print('platform:', __import__('platform').platform())
"
```
**What this tells you:** whether this is an MPS-enabled build at all (some PyPI wheels ship
without it) and whether MPS can actually initialize on this specific machine/OS version.
**Cannot verify from here:** which PyTorch version is installed, whether it's a conda/pip/source
build, or macOS version compatibility.

## 2. Quantized INT8 — qnnpack backend

```bash
python3 -c "
import torch
print('supported quantized engines:', torch.backends.quantized.supported_engines)
torch.backends.quantized.engine = 'qnnpack'
print('engine set to:', torch.backends.quantized.engine)
"
```
`fbgemm` (this project's x86 INT8 backend, D2) is x86-only and will not appear in the supported
list on Apple Silicon — `qnnpack` is the expected ARM backend. If `qnnpack` is not in
`supported_engines`, INT8 CPU inference is not available on this build at all, and that's a hard
stop for an M1 INT8 condition, not something to work around.

**Smoke test, mirroring what this review ran on x86** (load/run a quantized model, no energy):
```bash
python3 -c "
import torch
torch.backends.quantized.engine = 'qnnpack'
m = torch.jit.load('checkpoints/resnet18_int8.pt', map_location='cpu').eval()
x = torch.randn(4,3,32,32)
with torch.no_grad():
    out = m(x)
print('qnnpack INT8 CPU load/run:', 'OK' if out.shape == (4,10) else 'unexpected output shape')
"
```
**Cannot verify:** whether this project's existing `resnet18_int8.pt` (packed for fbgemm, per D2)
loads at all under qnnpack — the two backends pack INT8 weights differently; this may fail and
require a fresh qnnpack-specific quantization pass, not just a backend switch. Check the error
message if it fails; don't assume compatibility.

## 3. FP16 on CPU

Exact mirror of the x86 smoke test this review ran (`docs/feasibility_x86_cpu_block.md` §a),
run with `device='cpu'` explicitly (not `mps`) to test CPU-only FP16, which is what an x86-CPU-
comparable M1-CPU condition would need:
```bash
python3 -c "
import torch, time
x = torch.randn(1,3,32,32)
m = torch.nn.Conv2d(3,64,3,padding=1)
with torch.no_grad():
    for _ in range(5): m.float()(x.float())
    t0=time.perf_counter()
    for _ in range(50): m.float()(x.float())
    t32=time.perf_counter()-t0
    try:
        for _ in range(5): m.half()(x.half())
        t0=time.perf_counter()
        for _ in range(50): m.half()(x.half())
        t16=time.perf_counter()-t0
        print(f'FP32: {t32*1000:.2f}ms  FP16: {t16*1000:.2f}ms  ratio: {t16/t32:.2f}x')
    except Exception as e:
        print(f'FP16 CPU FAILS: {type(e).__name__}: {e}')
"
```
**Cannot verify:** whether Apple Silicon's CPU (which, unlike this x86 machine, has real ARMv8.2
FP16 hardware support in many configurations) runs this fast or slow — do not assume the x86
result (54-65x slower) transfers; ARM FP16 CPU support is architecturally different and could be
genuinely fast here. This needs to be measured on the real machine, not inferred.

## 4. Thread count and QoS (P-core vs. E-core scheduling hints)

```bash
python3 -c "import torch; print('torch threads:', torch.get_num_threads())"
```
macOS exposes process-level Quality-of-Service scheduling via `taskpolicy` (a standard macOS
command-line tool). To request a process run preferentially on Efficiency cores (QoS "utility" or
"background") vs. the default (which tends toward Performance cores for foreground work):
```bash
taskpolicy -c utility python3 pilot.py ...   # bias toward E-cores
taskpolicy -c default  python3 pilot.py ...  # normal scheduling (baseline)
```
**I'm not fully confident `-c utility` is the exact current flag/value syntax** — check
`man taskpolicy` on the actual machine before relying on it; macOS has changed this tool's options
across versions. Do not treat this as confirmed working.

## 5. Observing P-core vs. E-core utilization directly

```bash
sudo powermetrics --samplers cpu_power -i 1000 -n 5
```
Apple Silicon `powermetrics` output includes per-cluster ("E-Cluster", "P-Cluster") active
residency and frequency when the `cpu_power` sampler is used. This is the standard way to confirm
which core type a workload actually ran on, independent of `taskpolicy`'s scheduling *request*.
**Cannot verify:** exact output field names/format for this specific macOS/powermetrics version —
capture one real sample and check it matches this description before relying on parsed output in
any script.

## 6. `powermetrics` sampler for energy, and a scoped sudoers rule

`powermetrics` requires root. For unattended/scripted use, this project's existing x86 precedent
(`set_cpu_governor.sh`, a single fixed-argument wrapper script with a narrowly scoped NOPASSWD
sudoers rule — `phase1-execution-plan.md`, `stage4-implementation-brief.md`) should be followed
exactly, not a wildcarded sudoers rule (sudoers argument globbing is unreliable/unsafe):

```bash
# 1. Write a fixed-argument wrapper, e.g. /usr/local/bin/run_powermetrics.sh:
#!/bin/bash
exec powermetrics --samplers cpu_power,gpu_power,thermal -i "$1" -n "$2" -o "$3"

# 2. Scope sudo to that exact script only (visudo -f /etc/sudoers.d/green-ai-powermetrics):
researcher_username ALL=(root) NOPASSWD: /usr/local/bin/run_powermetrics.sh

# 3. Validate:
sudo visudo -c
```
**Cannot verify:** exact `powermetrics` sampler names available on this machine/macOS version —
run `powermetrics --help` (or `man powermetrics`) on the real machine first and adjust the sampler
list above to match; don't assume `cpu_power,gpu_power,thermal` are all spelled exactly that way
on every macOS version.

## 7. Thermal logging

Two options, in order of preference:
```bash
# No sudo needed, coarse (nominal/fair/serious/critical):
pmset -g therm

# Via powermetrics (needs the sudoers setup above), fine-grained, continuous:
sudo powermetrics --samplers thermal -i 1000 -n 0 -o thermal_log.txt   # -n 0: unlimited samples
```
**Cannot verify:** whether `-n 0` means "unlimited" on this `powermetrics` version, or whether a
large finite count is required instead — check `man powermetrics` before relying on it for a long
unattended run.

---

## Summary: what I cannot verify at all, stated plainly

Everything above is a command to type, not a confirmed result. I have zero access to any Apple
hardware, macOS install, or remote shell to that machine from this review. Every "cannot verify"
note above should be treated as **unknown, not assumed working**, until the researcher runs each
command and reports the actual output.
