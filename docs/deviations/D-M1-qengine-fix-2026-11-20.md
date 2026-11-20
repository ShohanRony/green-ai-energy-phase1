# Deviation: M1 code fix after freeze (recorded 2026-11-20, before any M1 session data)
The first session-1 attempt failed at preflight with NoQEngine: INT8
checkpoints were loaded before torch.backends.quantized.engine was set.
Fix: set the engine to qnnpack (the registered M1 engine) before
torch.jit.load in preflight. One line plus one test; no other change.
Commit 2e645f8, tag m1-measurement-v2 supersedes m1-measurement-v1.
The aborted attempt left an empty results/m1/session_1 directory on the Mac
(no files, no measurement data); it was renamed to
results/m1/_aborted_session1_noqengine_2026-11-20 and kept, not deleted.
All 23 conditions were verified to load and run a forward pass on the M1
before any session. No measurement data were collected or discarded.
