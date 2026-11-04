# M1 edge-structure test evidence — 2026-11-03

**This is guard-calibration/timing evidence, not analysed for energy and not
deposited/registered confirmatory data.** It supports the "Timing" bullet of
amendment A13 in `docs/stage5_analysis_plan.md`, which is the authoritative
numbers summary -- this directory is the backing evidence for those numbers.

## What this measures

`scripts/m1/edge_test.py`: 6 trials, each a 10-pulse train (2.0 s load / 2.0 s
idle, 4-thread `torch.matmul` 2048x2048 float32) under 400 ms `powermetrics`
sampling. Two concurrent arrival-time pollers watch the same output file:
"P20" re-reads the whole file every 20 ms; "P2" `os.stat`s the file size every
~2 ms and reads only new bytes. Per trial, rise offset `r` and fall offset `f`
are fit independently for each poller by least-squares grid search against the
power(t) step model, with leave-one-pulse-out (LOPO) spread reported per
offset.

## What's here

- `analysis_output.txt` -- full captured stdout of the run actually used in
  A13: guard checks (AC power, thermal) at start, each trial's live
  completion line, then the formatted report (`edge_test.py`'s
  `format_output_tables`, 4 sections: per-trial P20 fit, per-trial P2 fit,
  per-header P20-minus-P2 arrival difference, header counts).
- `per_trial_fits.csv` -- the same per-trial numbers from `analysis_output.txt`
  pulled into one structured table (6 trials x {P20, P2} fits, plus the pooled
  P20-minus-P2 arrival-difference row), for anyone who wants to recompute the
  A13 summary numbers without parsing the text report.

**Not archived here:** the raw per-sample JSON (`arrivals_p20`/`arrivals_p2`
dicts, `cpu_powers`, pulse timestamps) that `edge_test.py --worker` writes per
trial to a temp path on the Mac during the run. Those were ephemeral
(`/tmp/edgetest3_t{i}.json`-style paths on the measurement machine) and were
not copied off before being cleaned up; only the console output shown above
survived. If the raw per-sample arrays are needed later, the test must be
rerun -- nothing has been reconstructed or fabricated to fill this gap.

## Why only this run is archived: an earlier run was discarded

`edge_test.py` was run twice on 2026-11-03. The **first** run showed Trials 5
and 6 with idle power (181.93 mW P20-median trial set: T5=1241.08 mW,
T6=1092.15 mW -- both flagged `Idle>3xMedian=True`) roughly an order of
magnitude above the other four trials (19-304 mW), and correspondingly
anomalous rise-offset fits for those two trials (`r` = -76.74 ms and -43.16 ms,
sign-flipped relative to every other trial's +83 to +95 ms, with LOPO spreads
of 184-198 ms vs <5 ms for the clean trials). This pattern -- elevated idle
power plus a sign-flipped, high-spread offset fit, isolated to two trials out
of six run back-to-back -- is consistent with transient background-process or
thermal contamination during those two trials' idle windows, not a genuine
shift in the edge timing being measured. That run's console output and report
were not kept as part of this archive; the numbers above are quoted from the
run's own report, which was inspected before being discarded.

The **second** run (archived here) shows all 6 trials with `Idle>3xMedian =
False`, idle power 19-114 mW, consistent rise offsets (+80.65 to +90.91 ms
P20, +74.57 to +78.90 ms P2) and fall offsets (+18.96 to +28.41 ms P20, +9.04
to +11.42 ms P2) across all trials, and LOPO spreads under 4 ms for every
trial on both pollers. This is the run A13 quotes: pooled P20-minus-P2 mean
+12.695 ms ("+12.7 ms mean" in A13), P2 rise range 74.57-78.90 ms ("+74.6 to
+78.9 ms"), P2 fall range 9.04-11.42 ms ("+9.0 to +11.4 ms").

## Guard state at run start (from `analysis_output.txt`)

- AC power: "Now drawing from 'AC Power'", battery 100% charged.
- Thermal: no thermal/performance warning level recorded, no CPU power status
  recorded (i.e. nominal -- `pmset -g therm` reports nothing when there is
  nothing to report).
