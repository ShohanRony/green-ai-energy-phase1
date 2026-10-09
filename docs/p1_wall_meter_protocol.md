# P1 wall-meter protocol (draft — not executed, no hardware acquired)

**Status:** this protocol has been referenced as a prerequisite (`stage5_analysis_plan.md` A7r1(j),
D1r1) since before it existed. This document is that prerequisite, written now. **Nothing in this
document has been run.** No meter model has been chosen by the researcher — §1 below lists
requirements a meter must satisfy, not a specific product. Every number in this document that is an
assumption rather than a confirmed project fact is marked **"assumption, unverified."**

## 1. Meter requirements (not a product recommendation)

A wall meter used for D1 must satisfy all of the following. These are requirements against which any
candidate product should be checked before purchase, not a specification this document invents from
nothing — each is tied to what D1's comparison actually needs.

- **Logging interval ≤ 1 s.** `pilot.py`'s own sampling interval is sub-second (confirmed from
  `window()`'s sampling loop); a meter that only logs at, say, 1-second or coarser resolution cannot
  resolve the same active-phase windows `pilot.py` already measures (gross energy is the product of
  power and duration — a 5-second active window needs several meter samples within it, not one).
  **Assumption, unverified: whether a consumer-grade smart plug can sustain sub-second logging, or
  whether this requirement forces a lab-grade meter — not checked against any specific product yet.**
- **Power resolution ≤ 0.1 W, or better than 1% of the expected load, whichever is tighter.** The
  project's own measured range spans roughly 35-85 W (battery-affected low end, D25, to active
  compute) down to idle (single-digit watts for the CPU-only idle floor). A meter with 1 W steps
  could not usefully resolve idle-vs-active boundaries at the CPU end.
- **Accuracy class: ±1% of reading or better, with a traceable calibration statement from the
  manufacturer** (or an independent calibration performed before first use, §6). D1's own ±5%
  equivalence margin is not large — a meter with, say, ±3% uncertainty on its own reading would
  consume more than half of D1's margin before any real disagreement is measured, making D1
  uninformative regardless of the actual backend agreement.
- **Logs real power (W), not apparent power (VA).** A laptop charger's switch-mode power supply
  draws a non-unity power factor; a meter that only reports VA (common in the cheapest smart plugs)
  would systematically overstate the true wall-drawn watts relative to what the charger actually
  delivers. Must report true power, or must report power factor alongside VA so real power can be
  computed.
- **Exports raw per-sample data**, not just a session total. D1 needs the same active/idle windowing
  `pilot.py` already does; a meter that only reports an end-of-session cumulative number cannot be
  windowed to match `pilot.py`'s own active-phase boundaries.
- **Does not require closed, uninspectable firmware/cloud round-trip to retrieve data** — needed for
  the same reason the project avoids any instrument whose internals can't be checked (the same
  discipline already applied to NVML/RAPL/CodeCarbon in `stage5_analysis_plan.md` §3).

**Assumption, unverified: no specific meter has been evaluated against this list.** This section is
written so that whichever meter the researcher does acquire can be checked against it before use,
not after.

## 2. Physical connection

- **Power path:** wall outlet → meter → laptop charger (power brick) → laptop DC barrel/USB-C input.
  The meter measures AC power drawn from the wall by the **entire charger + laptop system**, not DC
  power at the laptop's input — this is a different, and larger, quantity than what `pilot.py`'s
  NVML/RAPL readings cover (§3).
- **Machine:** Lenovo LOQ 15IRX9 (confirmed, `docs/phase1-execution-plan.md`), the same machine every
  x86 measurement in this project uses. Battery model `L23M4PK4`, system product name `83DV`
  (confirmed via `/sys/class/power_supply` and `/sys/class/dmi/id/product_name` directly).
- **Charger wattage rating: assumption, unverified** — not confirmed from any project record or
  direct inspection of the charger's label as of this document. Must be read directly off the
  charger's own label (or its original Lenovo spec sheet) before this protocol is run, and recorded
  here once known, not assumed from the laptop's general product class.
- **Nothing else plugged into the same meter/outlet** during a measurement session — any other load
  sharing the meter's circuit would contaminate the wall reading. The researcher's single-machine,
  single-outlet setup (implicit throughout every prior stage's protocol) is assumed to continue; not
  independently re-verified for this document.

## 3. What the wall meter adds, and the adapter-efficiency gap it exposes

`pilot.py`'s existing boundaries (confirmed, `stage5_analysis_plan.md` §3) are **device-internal**:
NVML reads GPU package power; RAPL reads CPU package(+psys) power. Neither includes the laptop's own
other components (display, SSD, RAM beyond what RAPL's psys domain happens to cover, fans) or the
charger's own conversion loss. The wall meter reads **AC input to the charger** — the whole system,
including everything `pilot.py` cannot see, plus the charger's AC→DC conversion loss.

- **Adapter efficiency handling:** `wall-derived DC-equivalent power = wall AC power × charger
  efficiency`. Modern switch-mode laptop chargers are typically quoted in the 85-92% efficiency range
  at moderate-to-high load, falling off sharply near zero load — **assumption, unverified: this
  charger's actual efficiency curve has not been measured or found on a spec sheet; the 85-92% range
  above is a general class figure, not this unit's confirmed number.** Before D1's comparison is
  computed, either (a) the charger's efficiency curve is measured directly (§6's calibration check is
  the natural place to attempt this), or (b) the comparison is reported with the efficiency assumed
  and the resulting uncertainty propagated, explicitly labelled as resting on an unverified
  efficiency figure.
- **This boundary mismatch is why D1 compares *deltas* (active − idle), not absolute wall wattage,
  against `pilot.py`'s own already-net active-phase figures** — subtracting each side's own idle
  baseline (§4) removes most of the "everything else on the machine" component that the wall meter
  sees and `pilot.py` doesn't, leaving the *compute-attributable* delta as the comparable quantity.
  It does not remove the charger-efficiency factor, which is why that factor must be applied, not
  assumed away.

## 4. Idle baseline windows

Mirrors `pilot.py`'s own existing idle-window discipline (`stage5_analysis_plan.md` A7r1(h)), applied
to the wall meter in parallel, not as a separate protocol invented from scratch:

- For each rep being cross-checked, the wall meter's own idle power is the mean of wall-meter samples
  during the **same `idle_before`/`idle_after` phases** `pilot.py` already runs (not a separately
  scheduled idle period) — the two instruments observe the same wall-clock windows, so no new idle
  schedule is introduced.
- **Wall-derived net delta** for an active window = (mean wall power during that active window − mean
  wall idle power from that rep's flanking idle windows) × active window duration, then adjusted by
  charger efficiency (§3) to get a DC-equivalent figure comparable to `pilot.py`'s own net-of-idle
  energy.

## 5. Clock synchronisation

- **Requirement:** the wall meter's internal clock (or its logging host's clock, if it logs via a
  companion app/service rather than onboard storage) must be synchronised to the same clock
  `pilot.py` uses for its own timestamps (`time.perf_counter()` for intervals, system wall-clock for
  `timestamp_utc` in `environment.json`) to within a tolerance much smaller than the shortest active
  window being compared (`pilot.py`'s batch=1 windows are on the order of seconds).
- **Proposed tolerance: ≤ 0.5 s drift** — **assumption, unverified**, chosen as roughly a tenth of a
  typical short active window, not derived from any measured drift rate of a specific meter.
- **Method, to be fixed once a meter is chosen:** if the meter logs via NTP-synced network/companion
  software, rely on NTP and spot-check against the harness machine's own NTP sync status at session
  start. If the meter logs standalone (onboard clock, no network sync), a manual clock-offset
  correction must be computed at the start of each session (e.g., a sharp, identifiable power
  transient — plugging in or starting a heavy load — timestamped on both sides and used to compute
  the offset) and applied before any window-matching is attempted. **Not yet designed in detail
  beyond this outline — depends on which meter is acquired.**

## 6. One calibration check, before any session this protocol is used for

- **Check:** with nothing but the laptop connected (fully idle, screen on, no other load), record
  simultaneously: (a) wall meter's reading, (b) `pilot.py`'s own RAPL package(+psys) idle reading
  (CPU-only, GPU untouched), (c) if feasible, an independent reference meter or a second, different
  measurement method for the same idle draw. Compare (a) against (b) plus a reasonable estimate of
  the rest of the machine's idle draw (display, SSD, RAM, fans) — this won't close exactly (the wall
  meter sees components RAPL doesn't), but a wildly implausible result (e.g., wall reading lower than
  RAPL's own CPU-only reading) would indicate a connection or unit error before any real comparison
  is attempted.
- **Pass/fail, stated now:** wall idle reading must exceed `pilot.py`'s RAPL-only idle reading (the
  wall meter necessarily sees a superset of what RAPL sees) — if it does not, the calibration check
  fails and the setup must be corrected (wrong meter placement, wrong units, meter malfunction)
  before any D1 data collection proceeds. **This is a sanity floor, not a precision calibration** —
  it cannot by itself validate the meter's absolute accuracy, only catch a gross setup error.

## 7. The exact D1 comparison

Restated precisely, consistent with `stage5_analysis_plan.md` D1r1: does
**`delta(NVML power-usage + RAPL package)`** (this project's existing validated backend, summed
across the GPU-NVML and CPU-RAPL-package boundaries for whichever condition is being cross-checked)
agree with the **wall-derived system delta** (§3-§4 above, adapter-efficiency-adjusted) **within
±5%**? Both deltas are net-of-idle (§4), computed over the identical active-phase window, on the
identical rep, synchronised per §5. The comparison is run per architecture/state the researcher
selects for this check — not, by default, every condition in the matrix (a full-matrix wall-meter
cross-check is a separate, larger undertaking not scoped here).

- **YES (within ±5%):** keep the backend, report the agreement as validation (D1r1).
- **NO, cumulative counter agrees instead:** Stage 4b's GPU numbers need a rerun on the cumulative
  interface — only possible for data with both-API logging already in place (D24, data after
  2026-10-09T21:57:54+06:00 only).
- **NO, neither agrees:** wall energy becomes the primary boundary for system-level claims.

## 8. What is logged

Per session this protocol runs:
- Raw per-sample wall-meter readings (timestamp, instantaneous power), for the full session duration
  including all idle and active phases — not just a summary.
- The exact `pilot.py` run(s) being cross-checked (same `environment.json`/`windows.csv` outputs
  already produced, unmodified).
- The clock-offset correction applied (§5) and the method used to derive it.
- The charger-efficiency figure used (§3) and its source (measured vs. assumed class figure) —
  explicitly flagged if assumed.
- The calibration check's outcome (§6), pass/fail, with the actual numbers compared.
- Ambient conditions if available (nothing currently planned beyond what §9's abort rules need).

## 9. Abort rules

- **Abort if the calibration check (§6) fails** — do not proceed to data collection on an
  uncalibrated or misconfigured setup.
- **Abort if clock offset (§5) cannot be established** to the stated tolerance.
- **Abort if AC is not confirmed online** (D26's existing guard) — a wall-meter check inherently
  requires AC power; this is not a new rule, just restating that D26 already covers this protocol's
  own precondition.
- **Abort if any other load shares the meter's circuit** during the session (§2) — checked by visual
  inspection before starting, not instrumented.
- **Abort and discard the session's data if the wall meter's own logging drops samples** for longer
  than one active-window duration at any point — a gap inside an active window makes that window's
  wall-derived delta uncomputable, not just noisy.

## 10. Open, unresolved by this document

- No meter has been selected — §1's requirements are the selection criteria, not a confirmed choice.
- Charger efficiency is unmeasured (§3) — this protocol cannot run a fully quantified D1 comparison
  until either the efficiency is measured or the uncertainty from assuming it is explicitly
  propagated.
- Clock-sync method (§5) is an outline, not a worked procedure — depends on which meter is chosen.
- This document does not propose a schedule or decide which conditions get the wall-meter check —
  that is a researcher decision once a meter exists.
