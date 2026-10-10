# Analysis environment (R, for RQ2's mixed-effects test)

Recorded directly from the installed environment, not asserted from memory. Exact commands used:

```
$ which Rscript
/usr/bin/Rscript

$ Rscript -e 'cat(R.version.string, "\n"); cat("platform:", R.version$platform, "\n")'
R version 4.3.3 (2024-02-29)
platform: x86_64-pc-linux-gnu

$ Rscript -e 'cat("lme4:", as.character(packageVersion("lme4")), "\n")'
lme4: 1.1.35.1

$ Rscript -e 'cat("lmerTest:", as.character(packageVersion("lmerTest")), "\n")'
lmerTest: 3.1.3
```

- **R:** 4.3.3 (2024-02-29), platform `x86_64-pc-linux-gnu`.
- **lme4:** 1.1.35.1.
- **lmerTest:** 3.1.3.

Installed by the researcher directly (no sudo used by the coding assistant, per the standing
instruction). This satisfies the reproducibility-information requirement
`stage5_analysis_plan.md` A7r5(g) registered — the versions above are the required fields for the
freeze manifest once a freeze actually happens.

## Status of the analysis scripts in `analysis/`

**The scripts under `analysis/` are pre-freeze and have been tested only on synthetic data**
(`analysis/simulate_rq2.R` and the unit tests in `analysis/tests/`) — no real Stage 4b data has
been read, opened, or referenced by any script in this directory, consistent with this block's own
scope restriction. **They must run unchanged on real data** — the registered model specification
(`analysis/rq2_model.R`) implements `stage5_analysis_plan.md` A7r3(f) exactly as written, with no
option invented beyond what that entry specifies. **Any change to these scripts after the A7
freeze (`docs/a7_freeze_checklist.md`) requires a dated amendment**, the same append-only
discipline this project applies to the registered plan itself — a script is as much a part of the
registered analysis as the prose describing it, and silently editing it after freezing would defeat
the freeze's purpose.
