# History rewrite map (2026-10-10)

The researcher ordered a cleanup of the repository's commit history: attribution
trailers were removed from every commit message, and references to the coding
assistant's tool name were replaced with neutral wording in commit messages and
in file content. This document records the result so the mapping between the
pre-cleanup and post-cleanup history is traceable. It describes what changed
generically, per the researcher's instruction, and does not name any specific
tool.

## Method

- **Source of the old history**: a full filesystem backup of the repository
  taken immediately before the cleanup, at
  `green-ai-energy-phase1-BACKUP-before-history-rewrite-20261010` (sibling
  directory, not part of this repo).
- **Matching rule**: each old commit was matched to a new commit by exact
  author-date timestamp (ISO-8601, second resolution, including timezone),
  then cross-checked against the commit subject line for agreement.
- **Result**: all 94 pre-cleanup commits matched exactly one post-cleanup
  commit by author date, with zero ambiguous timestamps (no two commits ever
  shared the same author-date in either history) and the subject line
  identical in all 94 matched pairs. No flags were needed.
- **One commit exists only in the new history**: it was authored after the
  cleanup, carries a later author date than any pre-cleanup commit, and is not
  part of this mapping (it is new work, not a rewritten commit).

## Old hash → new hash (94 commits)

| # | Old hash | New hash | Author date | Subject |
|---|---|---|---|---|
| 1 | `14ff16b25fe064b243c5fb2f1c56aa48c76430ba` | `e2922993ed9cd7990eb635269b13c2cc2d5deae6` | 2026-10-03T19:36:29+06:00 | Port energy-pilot harness for Phase 1 (Stage 1, Task 3) |
| 2 | `a573d887e84b6a6dba490c260e31f096faf94262` | `70dba8acf193012916bcfac2cbcb9361f5fce5dc` | 2026-10-03T19:36:53+06:00 | Add Task 2 NVML interval-sweep evidence (2026-10-03 re-run) |
| 3 | `ea89ad1a99ab501eb834241a2b7f4946b3d8b0c2` | `a38aef770eb181d0f9c87127cce5adac386eba04` | 2026-10-03T19:44:43+06:00 | Wrap CodeCarbon to run concurrently with RAPL/NVML reads (Stage 1, Task 4) |
| 4 | `694b7d63c203ec7094f5d06f9e0e3cd81d55d79d` | `d2c01d3419a66f026a16434cecce1d0053c69e38` | 2026-10-03T19:52:11+06:00 | Fix: discard cold first run of each configuration from summary stats |
| 5 | `d2c38b95cdf94912b97f11be57f6439509ed51e5` | `10a9bf5d15a377c670578a674ce4d84cf392a7fe` | 2026-10-03T19:59:18+06:00 | Stage 1 exit test: uncompressed ResNet-18, x10 reps, x86 only (Task 7) |
| 6 | `b23e774107533d6877d0918552361957dc3a8301` | `3d920911601daf2e12ddb5f1cfb77ad68269b570` | 2026-10-03T21:10:38+06:00 | Enforce RAPL sampling-rate ceiling (checklist item 4, CPU half) |
| 7 | `0082612a0c9993557d3e7467e17ffdc2639c3d05` | `e37b62b4d76f5f3fde3f39987f4466c3b8490762` | 2026-10-03T21:11:15+06:00 | Add automatic concurrent-GPU-job guard (checklist item 13) |
| 8 | `2b81df2ba4382b78c179c10a6fff1c791dcb7da7` | `a2226ecb823f4fdc374bf3731f09c1ae0b35fe70` | 2026-10-03T21:24:16+06:00 | Add platform power-profile guard (checklist item 14) |
| 9 | `e2a179847d1c38a5fddfda9cd303cdbaf6883075` | `7faf789421bb917b6e895cea65e9ddcc582ff996` | 2026-10-03T21:31:59+06:00 | Regenerate Stage 1 exit test: all 14 checklist items (was 13) |
| 10 | `b2c7da671dfd8a2f433466fa258177c382e3e1f1` | `d4c63fdefd4a42128566b64f1389482cdebd166f` | 2026-10-03T21:37:00+06:00 | Prove the concurrent-GPU guard's alarm path, not just its silence |
| 11 | `6a4f6170963aa3e8d7116bfe8584f0551ef70e2d` | `ac452713e00de8a74f29051e05823dffe455551e` | 2026-10-03T21:37:10+06:00 | Investigate NVML sampling-rate artifact: not in our aggregation code |
| 12 | `21852c3a7e3380a6086707130d24be50b3735aa0` | `e72ac60ba48118936e2ecb933588857f980fed0c` | 2026-10-03T21:44:16+06:00 | Ignore LibreOffice lock files |
| 13 | `56ae633db77f74b9a97319451e7c147528302b03` | `e963d8f7b013d6a3bd3e43deefcf818fbc9591de` | 2026-10-03T21:50:47+06:00 | NVML investigation follow-up: rule out window-duration drift, confirm it's call-frequency-driven |
| 14 | `f66904ed4cf8fefa9e1444ee868743f7a2eb1e37` | `16a514c291b88e18af29ed556f5751cfe501b89c` | 2026-10-03T22:29:00+06:00 | Investigate recurring idle_after anomaly: not a bug, not a GPU-state transition |
| 15 | `42e5d53cc4ba0bdae0e41c260b3e8fdecd9346b5` | `08b3df9aa6d6bbc966987bbd534989029c593050` | 2026-10-03T22:40:07+06:00 | Add plausibility_flag: GPU power-cap check, threshold verified not assumed |
| 16 | `975b7fdedc29dc6f6b4daf2efc119af43c671309` | `4b7acf90e54f0eb27a50c36c6d1fb30b393ddd0a` | 2026-10-03T22:44:33+06:00 | Resolve the active-power question: counter over-reporting, not real boost |
| 17 | `92fd4c1999630dd8b9096b291dbda16bafa8fe4f` | `a51a853d0d252a90784dfc0abadd495430728728` | 2026-10-03T22:52:38+06:00 | Validate nvmlDeviceGetPowerUsage against live telemetry before any switch |
| 18 | `672ce938f51e96761d844c75f79db2ba81a03fcf` | `23e23e335b1f6ae416bffe2efe2ae2cc9fbc882d` | 2026-10-03T22:58:53+06:00 | Switch default CUDA backend to nvmlDeviceGetPowerUsage (validated) |
| 19 | `3b525cc2266afe87f8c4a81329d21477811eac23` | `982e6c88066ba39abe7edf5105cec6d43286aa26` | 2026-10-03T23:14:08+06:00 | Close out the power-measurement correction: re-run exit test, re-check floor, deprecate old numbers |
| 20 | `bc37ff91a4a3b3fe9d162adaaef6453f45431115` | `86c8f9d91ae51c2e4d87b32696599bf28c5df217` | 2026-10-04T15:31:13+06:00 | Stage 2: compression artifacts for ResNet-18, MobileNetV3-Small, EfficientNet-B0 |
| 21 | `888c58557475eb42ed8d6a2efa2a0ec18047ee09` | `c0a6ecdf14193fa9a7b33d0c29fecc69b438a358` | 2026-10-04T15:50:53+06:00 | Stage 2 closeout: diagnose MobileNetV3-Small pruning, check convergence, materialize all 18 checkpoints |
| 22 | `24a5c944f691a70432095200185900d49ad5a5a2` | `5bb5e06510c676e0fe8df9ffcd1bc4b0589bfee5` | 2026-10-04T20:03:41+06:00 | Retrain MobileNetV3-Small and EfficientNet-B0 at 60 epochs, propagate through all derived artifacts |
| 23 | `a24d5a8fcb06599b13e1c02548fe7f049076102c` | `cab2b592a2b3a412568cde38efedef3ed2a4a196` | 2026-10-04T21:35:40+06:00 | Task H: locate the MobileNetV3-Small pruning cliff precisely, confirm checkpoints load, flag convergence-fragility as observation |
| 24 | `777a0481d6ef1ea73a8ab1d75e5533c1dcfd15b7` | `38bb3696532d1a079499acafd9a81ec5b2acabf8` | 2026-10-05T18:13:00+06:00 | Stage 3: RQ1 full-state harness validation pilot, GO decision |
| 25 | `d1ca7dd88d228edfb46a5913e5bb42c32c9ebe04` | `a63eb2cd6e734b6baa6e75af135e25bbcbc7c12b` | 2026-10-05T18:29:20+06:00 | Add power-loss resilience: watchdog, checkpoint-at-safe-points, --resume |
| 26 | `d3fc046f062c79752dd9c3760fee836ea05a6586` | `00419180e0f487fa8aff7a6bed72ee148536009d` | 2026-10-05T21:30:32+06:00 | Stage 4 pre-flight: batch-size scan reveals GPU power-regime non-determinism |
| 27 | `539ef17eac7460a4af1e6dc392b10f2cdb23d438` | `3a29d9bdc4993665f63fb5000041835bc35b4b19` | 2026-10-05T21:54:01+06:00 | Fix set_cpu_governor.sh: nested braces in usage message broke sudoers invocation |
| 28 | `147cb84cbd0742ddf01e52e57e8e33398089b9de` | `1a6c9bac54447a574bc4bb7294e2227a06b29743` | 2026-10-05T22:48:21+06:00 | Diagnostic sweep: power-regime split correlates with boot session, not per-invocation randomness |
| 29 | `b28328b75597fd58a0c59bd48bd094fc3b7c1a3c` | `e743b2b4557064e949e28e08b1acf164bc2f47a3` | 2026-10-05T23:55:58+06:00 | Rule out CPU governor as a confound for the pin/dip pattern (log analysis, no new GPU time) |
| 30 | `17c16513e2d4640bf66679e84957dc388ddf137a` | `3e4b4859e3e816ef3582ba404360b51841c47ad9` | 2026-10-06T00:44:42+06:00 | Close mechanism investigation (boot-identity falsified); add regime flagging + fresh-boot guard to pilot.py |
| 31 | `c7e13d686654b0b4b3bb8f439d3c6c73d6775f60` | `f62fb5a8749cb072f99357d5348d279c22b6ed34` | 2026-10-06T00:55:37+06:00 | Draft Stage 4 implementation brief: batch=1 locked, x86-only, dip/mixed-triggers-reboot standing rule |
| 32 | `24d9d64432cb4b3832d1868df81a4dc2dd933f37` | `4b5f88279bc5d32075c44890cde828a1a14ca5a7` | 2026-10-06T13:07:57+06:00 | Amend Stage 4 brief §5: sub-saturation dip at batch=1 is expected for heavy pruning, not an escalation trigger |
| 33 | `356e9d4fa3868ecd61ddd1cfbabb4e961cd2bbfa` | `2e30e2ad517b4babab7ea9ec3ecf55afed5da786` | 2026-10-06T14:10:25+06:00 | Generalize Stage 4 §5 carve-out: any sufficiently light model/state at batch=1, not just ResNet-18 pruning |
| 34 | `6a312a3ce93c4110b3f5bc3e6bd4584dda1eb5d1` | `c19508aa3f311ce7867bdc1a94797518daf05491` | 2026-10-06T15:01:15+06:00 | Stage 4 conditions 8-10/18: mobilenet_v3_small int8/fp16/pruned30 (int8 N/A regime, fp16/pruned30 dip per carve-out) |
| 35 | `3e98ee5be1a487836d5adb44f3f6969b6dfab843` | `f12bc63bfdc658827b4d53915a8dc706d948492e` | 2026-10-06T15:35:18+06:00 | Stage 4 conditions 11-12/18: mobilenet_v3_small pruned50/pruned70 (dip, carve-out) -- MobileNetV3-Small complete |
| 36 | `e7aab843f5ed989d5012653944ee876f69dc1242` | `5a230f5369a4be00476cb59c543e9ec8314e47d7` | 2026-10-06T16:22:57+06:00 | Resolve Stage 4 dip/pinned question: confirmed compute-threshold effect, not a per-model mystery -- stop escalating |
| 37 | `2550416f233071a5afc7f29bceb8de1c2b7df91c` | `43c4630417055c5a4219a992ebaa4b381e14b5c7` | 2026-10-06T16:56:46+06:00 | Stage 4 conditions 14-15/18: efficientnet_b0 int8/fp16 |
| 38 | `2ca25cdd6258235c87058a6f3b78d9d3815249c5` | `43dabf831f531e67c01a5c70e94a10214d6cda5f` | 2026-10-06T17:30:35+06:00 | Stage 4 conditions 16-17/18: efficientnet_b0 pruned30/pruned50 |
| 39 | `d0bfb705414b30210fc97416e0db122d1b441d24` | `37b0dc1da8185565cc2a7cb12ea390280806df6b` | 2026-10-06T17:47:51+06:00 | Stage 4 condition 18/18: efficientnet_b0 pruned70 -- full 18-condition x86 matrix complete |
| 40 | `0a358c60a87edb3e575ff1808f92a598f16110d5` | `d9e1fa00428a0842bc132b72fd1fa4445818c2d5` | 2026-10-06T18:11:43+06:00 | Add deviation log and Stage 5 analysis plan, registered before any statistics |
| 41 | `e2df9c114f69b4e50e7876da45c83f99cef32bf3` | `094a7d4c61edce51dd248f7d75ca21a5aa81e906` | 2026-10-06T19:26:22+06:00 | Part B item 3: dip stability check -- 12/12 dips reproduced in a fresh session, 2 with nvidia-smi dmon |
| 42 | `7c5c6d7db9e7c60fc79576c51a6255c30dfd00b2` | `68d59079e0496d4ff398a5b26736e7545979c104` | 2026-10-06T20:16:53+06:00 | D13: add FP32-CPU baselines, one per model (31 reps, batch=1) |
| 43 | `61ebc367e5f613d99680d1d02075455984425615` | `c2a43d33ed0df5913a97944a42fbb28cead1a1cf` | 2026-10-06T21:56:36+06:00 | WIP: 7-item audit -- stats divergence resolution (D14), CodeCarbon absence (D15), D7 timestamp correction, codecarbon_check data |
| 44 | `583de866c30f03fa4d28f7befbad80b2d2c187c6` | `566bf87fb54e08fc2bf49da75e5beddce733e490` | 2026-10-06T22:15:19+06:00 | 7-item follow-up audit: section 6 accuracy rule (MLPerf 99/99.9 proposal), D13/section2 reconciliation, section 4/5 fixes (serial-reps limitation, FLOPs-ratio definition for FP16/INT8, NVML/RAPL confirmation), D15 CodeCarbon component breakdown + corrected overhead argument, D16 RAPL package+psys double-count discovery, D9 OOM-skip closure, D14 left OPEN pending proposal re-verification. Section 6 still pending Shohan's explicit threshold confirmation; no Stage 5 statistics run. |
| 45 | `abed585e66b16fc7a944a47fa6b7d81d38d6ed39` | `65dc51b9b2f52362678afbd2566d411f705c1f3b` | 2026-10-06T22:37:55+06:00 | D16: root cause confirmed (README contradicts code; psys>=package empirically; package-only recompute matches CodeCarbon to 0.1%). Remediation decided: offline recompute + harness patch, no rerun. Plan section 3 rewritten: CPU boundary corrected, D16 marked blocking, component-wise RQ1 comparison (GPU-vs-NVML, CPU-vs-RAPL-package) now primary. Section 6 citation filled (Tschand et al. 2025 HPCA, MLPerf Power), MLPerf Tiny absolute-target and single-seed-variance caveats added. |
| 46 | `8367662d5489e666518884dc9de91ec4cb598a00` | `4f2dfd816985f8ef052e5913eb12d9c9d638600c` | 2026-10-06T22:39:53+06:00 | D16 offline recompute: package-0-only CPU energy from existing raw.jsonl traces, no rerun. |
| 47 | `6d361b5b74b5cbc7dd30dc9a90c930172cae8e37` | `8a2573c59f407cb5a23db092fde4ca327ccbe225` | 2026-10-06T22:43:06+06:00 | Point README/plan/D16 at summary_package.csv as corrected primary CPU energy; log v1/v2 CodeCarbon-check reconciliation in D15; fix section 6 citation provenance. |
| 48 | `70f5d1d28cd3cda3c01e7bf4217be53a52d13f9e` | `4696b98f3233ce82ddef21ff4f3b71dec9b733ec` | 2026-10-06T22:49:33+06:00 | D16 harness patch: log RAPL domains separately, package-0 primary, going forward. |
| 49 | `b990db13f7b7d1844a19f10c817449bba87dc714` | `6f447fb4bb9753f1a5f79104c3efcf1d11bcd38c` | 2026-10-06T23:31:03+06:00 | Accuracy completion: full-test-set eval of all 18 checkpoints via pilot.py's own loading path. |
| 50 | `3a68727e17e03f568264aedb88450dfa76c7af2a` | `60b6347c105388717002f5700b11d4f78a8d556a` | 2026-10-06T23:35:30+06:00 | Pruned-state controls: class histograms + BN-stats-only recalibration (no weight updates). |
| 51 | `b66349d00fd2724d91dab0b1b661799094dff7fc` | `fe08438a7b1c15364c751c77519497ce16491db4` | 2026-10-06T23:45:00+06:00 | Harness patch: concurrent CPU-package RAPL during GPU runs, P-core affinity, images/s. |
| 52 | `c5579d615d8e74d30dbb1247b281f0db3b782ce3` | `0847e5e76648c888d6dc00b1c798b5becb842722` | 2026-10-06T23:57:56+06:00 | Docs mega-update: A2 amendment, stale-status fixes, D7/D8 mechanism correction (dmon-confirmed), D17. |
| 53 | `37bd3974ccbc241c3a3b6102e2919f118118a12a` | `8bd1cb283fb8c150db60b42de8d424ca968f05c2` | 2026-10-06T23:58:53+06:00 | Docs mega-update content (previous commit only captured test_pilot.py's deletion by mistake -- a bad git-add pathspec silently dropped every other file from that commit). |
| 54 | `85d966cd9ed204dabd418bad69b4f29a6bb922f4` | `5d231d53078ec36a1f15d28f97a0517ffa2775fa` | 2026-10-07T00:00:34+06:00 | Draft stage4b-design.md: multi-session replication design, not executed. |
| 55 | `f921cfc25265467ad5a56fd68f7a526975722822` | `da73acf2d8e0bcb5ad9814bebc0e353afeffd992` | 2026-10-07T00:37:11+06:00 | Add overnight instructions, delta, and supervisor review to docs |
| 56 | `6ef0b77dbc73778d82c8470bbdacf679e044546b` | `568cece7b124ca4a7423a6847bb282cf6cff03d4` | 2026-10-07T00:50:12+06:00 | D1: Verify accuracy, add FP32-CPU preds, update harness per Step 5, confirm accuracy rule in plan |
| 57 | `54918f0167ecfdf32233abfdf10a25f63383f4fa` | `efe41d36dbc1bc2c9ef0e25bc455ff050347cc34` | 2026-10-07T02:10:30+06:00 | D2: Add BN-recalibrated and brief-recovery pruning arms, evaluate accuracy, log D18 |
| 58 | `ff0695cb93bf1534db0cc724ae39814da683ab6e` | `34c0f74314d913d17701567ba7120cbbd9edf120` | 2026-10-07T02:17:03+06:00 | D3: Register finalized Stage 4b design (30 conditions with BN-recal arm) |
| 59 | `d0c841ca0b75ce63a931368c57249faaebe6f565` | `7701cbba8f0c83329403d713499723de89355b47` | 2026-10-07T02:20:47+06:00 | Step 7: Add run_stage4b_session.sh runner script (verified with session 0 dry run) |
| 60 | `47e76d765cbdacc27eceb89764f1d6a36eab4a6a` | `53a6054f266ba97093570459a773ad9c9c606322` | 2026-10-07T02:23:22+06:00 | Step 8 & 9: Add overnight morning report, record Stage 4b Session 1 PID 3882 |
| 61 | `d321598ed80228ee9ca1fda70217ffd5a3328e15` | `aaef4b9d918b96f186b4baace1f0acf8426c24cc` | 2026-10-07T04:15:43+06:00 | Results: Stage 4b main session 1 |
| 62 | `f8b9855f00802aa1904b2ecf487e4940e9ac4229` | `ce84d99420dcbcde729564847747dfc15bf21926` | 2026-10-07T04:19:44+06:00 | docs: Update morning report with Stage 4b Session 1 completion details |
| 63 | `7a1619717d96b5c15cc5eb8ed865851ab6e3502b` | `fa344be03ab9183a77a350269749dbec0a962834` | 2026-10-07T10:32:52+06:00 | D19: Patch harness CPU affinity across all runs, move unpinned Session 1, log deviation |
| 64 | `094c3614ffae5adf4136c46d12a9c3f35dd8419c` | `388b099a1b958e3df484bfd35f1b353baf7ca2ad` | 2026-10-07T10:34:53+06:00 | run_stage4b_session: Record boot time in session_log.md |
| 65 | `7df44d08a36dde17c3917ad5e3ed7d54beaf191d` | `9b6153fa7d00382d35f51221b39047cf1f954acf` | 2026-10-07T12:28:31+06:00 | Results: Stage 4b main session 1 |
| 66 | `f574db56be483b2c835db93bbb026329e853af01` | `ea9be507095a074ec02748da443b943bfee8d19d` | 2026-10-07T12:29:12+06:00 | Update: Stage 4b Session 1 (Pinned) |
| 67 | `fbce7c041d9a0bca8ceda0ff4bfaf27a548310b8` | `a9242761614820dee644bedc889e5414c5d43930` | 2026-10-07T12:29:54+06:00 | Update overnight report for pinned Session 1 rerun |
| 68 | `ac3b5937c9e5d5b0b2d0e63891fa9f5a77b5ea1a` | `9fa1a426cbbe1c1c625f8f0d989f9e7f9099a0b6` | 2026-10-07T12:40:54+06:00 | docs: update D19 status to completed |
| 69 | `4c68b64c09ea6eecbfd6a0a508aeaaf70613152c` | `2bb5a30637f20ea385055c3b4de3c28b89cdc6e1` | 2026-10-07T14:38:16+06:00 | Results: Stage 4b main session 2 |
| 70 | `d944ac3ab681f7e1a2c04326425700c3bc316af6` | `b807b171669fc4ebfa07327eb1ef5933f41bfaaf` | 2026-10-07T14:48:42+06:00 | chore: include final runner.log lines for Session 2 |
| 71 | `535a8d8b684088f7a385e3588be30afc1d8dccd2` | `ed752508ed3d9e4f495362ce952554c774c6af9e` | 2026-10-07T16:48:54+06:00 | Results: Stage 4b main session 3 |
| 72 | `558a3217972a1ff6b7bd27d05c6759d573a1e6ce` | `fde2679b88c4c6b960a2f834f52cbea06c7285ba` | 2026-10-07T16:49:08+06:00 | chore: include final runner.log lines for Session 3 |
| 73 | `d68495fdec67e6f2e85e57e0203eea9215b985f5` | `ee0c90d94bd57c1ff97ce60608111c7a8b26ac30` | 2026-10-07T19:03:55+06:00 | Results: Stage 4b main session 4 |
| 74 | `a686cc250f42ad496695bfdd9eba9edb05a9abe1` | `9aa85327f2b2eeba368028ecf5205581350add68` | 2026-10-07T19:04:05+06:00 | chore: include final runner.log lines for Session 4 |
| 75 | `f66a938712bf499d777e9f07adf9fef37aa8bc24` | `317a9723e4d75d60ac8c41c10cd47e9e0631f03b` | 2026-10-07T19:16:27+06:00 | Fix D20: false power stop and row validation in runner |
| 76 | `f88a4e64e607460ff38812de84f880962d1eecae` | `e5be5edab361bc64792c6232febe108b1a202569` | 2026-10-08T01:11:30+06:00 | Draft Amendments A4-A8, harden D20 preflight guard and wait cap |
| 77 | `a405435319a1d7fe2cb28a79075843104a6e5abc` | `c125e8b6198a98fd1dda945496a31891976bccd5` | 2026-10-08T01:12:00+06:00 | Remove accidental scratch dry-run log (session999) |
| 78 | `fc121c2f7952acd2def6349c3c0b4dc1dceebda1` | `9245f81bd942406ba3f51eb273b4a5834834e414` | 2026-10-08T01:12:53+06:00 | Move draft amendments A4-A6 to stage5_analysis_plan.md §12 (append-only); revert misplaced stage4b-design.md section |
| 79 | `da966c4b25907830941959a5d90aed245a344f0a` | `d2c4e770e10a0398e71aacc711662dbb927afe64` | 2026-10-08T01:16:32+06:00 | Add unattended boot orchestrator for Stage 4b sessions 3-6 (A4) |
| 80 | `71404b72798626578fdb81fa0130f5f73a3e4678` | `451a864b0f8b8edb4fa4ad1ddc58801535e060bb` | 2026-10-08T03:15:44+06:00 | Results: Stage 4b main session 3 |
| 81 | `3e51edd049adbb9545368514374c8c82cfbf8720` | `34c842bd48b022faf84508f6ada6bb4809e61b0e` | 2026-10-08T10:11:35+06:00 | Fix trailing runner.log write after git commit and finalize session 3 log |
| 82 | `f2bb473d70798e12083556dcdafbbe46d53f7c4a` | `476855d506c9c71d223f508086b97237300ebffb` | 2026-10-08T10:12:48+06:00 | Allow passwordless systemctl enable in sudoers rule and preserve queue |
| 83 | `67d3057b9752c7805f47c625bf68797e2154ff03` | `5c469413a389f9e57d0f734ff6cc78151ff1372a` | 2026-10-08T20:56:22+06:00 | Results: Stage 4b main session 4 |
| 84 | `3571807757f230d690e23f7cd485af47365964e6` | `8731ddb70f96e0bde9e5cd476dd0a2b646bff276` | 2026-10-08T22:52:26+06:00 | Results: Stage 4b main session 5 |
| 85 | `294949203cfd3d6408350f1250311a2aaaa27539` | `07643d3f72609921217febb3c79daa7be4261c2f` | 2026-10-09T01:09:29+06:00 | Results: Stage 4b main session 6 |
| 86 | `1ac47c04c4f7da851b511c9e43686091f855a500` | `440a829179a2618c2d260d932ed923bfbe7c97b3` | 2026-10-09T21:40:54+06:00 | D21-D23: A4 adopted with corrected calendar spread, P8 untracked-root-run, sudoers exposure. |
| 87 | `b5593f7640388dffb9a9cad88df72abb0325b1f0` | `40c6be54360e9fee169690e77e53d648020ae03c` | 2026-10-09T21:43:48+06:00 | committed after execution; see D-entry |
| 88 | `6e0dc4be659e4dc0b3967a54d67e9e76732a64d8` | `0c89a00e9d0eccb864db4335b2c47261f4405c46` | 2026-10-09T21:44:00+06:00 | docs/p8_spec.md: P8 campaign specification reconstructed from code and logs only. |
| 89 | `d82cd9514401f67197b13f8940886d2dba5e7492` | `2ff4019779eaa413101a476cdb06c6115009f32a` | 2026-10-09T21:57:54+06:00 | D24: log both NVML interfaces plus RAPL package-0/psys as separate columns every GPU window. |
| 90 | `9cd1ee9f27959cbe5e089d1dc3341cf11dc3a6e5` | `415633b2f586805696bb566812accaefeb215595` | 2026-10-09T22:04:17+06:00 | Feasibility smoke tests for an x86-CPU primary block (no measurement). |
| 91 | `0545576c5be2e5e5cd1eac3ef812e7e1e5cb1c82` | `b23cbcc8ef7d81e301e5c650747c29b1e1082569` | 2026-10-09T22:06:03+06:00 | M1 MacBook check commands -- for the researcher to run, nothing executed here. |
| 92 | `d5bbd04bbfcef7a24bb4f98023e2523c72a20f2f` | `3cf11b6a93761285891cd4130767a43216a6e736` | 2026-10-09T22:09:22+06:00 | Draft amendments A7 (analysis freeze) and A8 (runtime-baseline series R). Not executed. |
| 93 | `de8ba5ed860339300fc2fc13d0a20a6d43f5ea04` | `4ed0a3f22a26804ae487aa2a95071dcb3839b801` | 2026-10-10T00:47:46+06:00 | D23 append: researcher's p8-auto/stage4b-auto remediation, with User= facts distinguished. Plus _bnrecal CPU functional-equivalence check (5/9 pass, 4/9 fail by 1 image/10000 under the strict exact-count rule -- magnitude consistent with ordinary CPU/CUDA float kernel differences, not interpreted further here). |
| 94 | `20f2ae1a968c467436ac2843eb7faab11ad3ce99` | `3421250a88c4eb707d16395ef6dd3bc3781f4a5d` | 2026-10-10T00:48:58+06:00 | D25: power outage during the 2026-10-09 evening session -- D24's validation runs were on battery. |

## Files whose content differs between old and new trees

Checked two ways: (a) a direct diff of the old final tree against the matched
new final tree, and (b) a diff of every one of the 94 matched old/new commit
pairs individually, with the changed-file lists unioned across all 94. Both
methods agree on the same 5 files, confirming no file was affected in an
intermediate commit without also differing at the final state:

- `docs/overnight-delta-2026-10-07.md`
- `docs/overnight-instructions-2026-10-07.md`
- `docs/phase1-execution-plan.md`
- `docs/stage1-implementation-brief.md`
- `results_stage2/stage2_task_report.md`

All five are documentation/report files; the differences are prose references
to the coding assistant's tool name, replaced with neutral wording. No source
code, configuration, checkpoint, or results data file differs between the old
and new trees.

## What did not change

- Author and committer name/email on every commit: unchanged.
- Commit subjects: unchanged (verified identical across all 94 matched pairs).
- Commit count, order, and parent relationships: unchanged (94 commits, same
  topology).
- Every tracked file outside the 5 listed above: byte-identical.

## Verification performed after the cleanup (and again when writing this map)

- No attribution-trailer line or tool name found in any commit message, across
  all reachable refs.
- No tool name or attribution line found in any historical diff content,
  across all reachable refs (full `git log --all -p` scan).
- No tool name or attribution line found in the current working tree.
- Only two author/committer identities exist anywhere in the history: the
  researcher's two own name/email pairs. No third identity of any kind.

## Revision, 2026-10-10: local backup tag renamed

The local-only git tag flagged as a residual tool-name hit in the prior review
(its old name contained the coding assistant's name) was renamed to
`backup-pre-history-rewrite-20261010` (`git tag new old; git tag -d old` —
preserves the commit it points to, only the ref name changes). It points to
`3421250a88c4eb707d16395ef6dd3bc3781f4a5d` (the "D25" commit) — same commit as
before the rename, unchanged. That commit's own message and full tree content
were checked directly and contain no attribution line or tool name (consistent
with every other commit in this rewritten history). The tag remains local
only — confirmed not present on `origin` (`git ls-remote --tags origin` lists
only `stage4b-registered`) — and was never pushed, before or after this
rename.
