# Bound parity child admission

Slice M22-A-S8; attack #445; audit AUD-213.

## Current evidence boundary

The release parity job 133553485486 timed out on K6 at08_recursion_bound.
The original harness could launch all 20 fork rows and five upstream rows,
each running its corpus concurrently. Child deadlines began before process
start. Every child retained GOMAXPROCS=4.

One candidate adds a package-wide channel at runBinaryTimeout. Its capacity
snapshots GOMAXPROCS at package initialization; admission happens before the
child deadline is created, and deferred release covers every return path.
It bounds active helper calls, not all compiler builds or descendant OS
processes. The existing sweep-file semaphore stays separate. APIs, child
budgets, configurations, normalizers, corpus inputs, and stamp rules are
unchanged. The exact implementation patch is `preparation/candidate-001.patch`.

The component tests ran with immutable e96 installed tools and source
overlays. The restored candidate passed under race detection. Fresh
installed combined-toolchain acceptance and the full unfiltered, oracle-on,
non-short release parity/mutant/sweep suite with a clean-tip stamp remain
pending. This packet contains no release stamp and claims no outcome for
the combined make that was running at assembly time.

## Actual component outcomes

| Run | Outcome | Seconds |
|---|---|---:|
| parity-queue-red-001 | Third call ran while both independent rows held slots; error controls passed | 10.929 |
| parity-queue-green-001 | All three frozen scenarios GREEN | 10.788 |
| parity-queue-bypass-mutation-001 | Admission bypass caught by the shared-queue assertion | 7.400 |
| parity-queue-early-deadline-mutation-001 | Early deadline caught after the queued call was released | 11.594 |
| parity-queue-restored-race-001 | All three restored scenarios GREEN under race | 15.145 |

No component run skipped a test or failed during setup. The test source
remained SHA-256
`6c13e3b083cd289ca38aa5c9c9cfe0f8d9eaee9ea6f2654cc7aaa7668b53cc07`.

Each scenario starts a fresh test subprocess with GOMAXPROCS=2 before
package initialization. One helper binary is built through existing helpers.
Two ready holder children run through separate one-program runAll rows. A
third call must remain queued for three seconds despite a two-second child
budget, then return exact ok output after one holder is released. N-1, N,
and N+1 capacity checks also verify successful permit return. Separate
controls perform three missing-executable failures and three timeouts after
actual child readiness, each followed by a healthy child. Tests never read
the limiter variable or use a new production seam. Buffered results,
release markers, joined goroutines, and subprocess watchdogs bound failures.

## Diagnostic evidence, separate from acceptance

`diagnostics/hosted-release-133553485486/` retains the original hosted log
and verified archive review. That run also had independent ledger failures;
its timeout is not inferred from those failures.

The first isolated local probe stopped during VCS discovery and executed no
program. The unchanged probe then ran from a real task-owned directory under
/home/agent. All six K6/U3N runs passed the original 10-second budget with
exact ok output. These isolated observations alone did not prove contention.

The controlled probe imposed two-CPU affinity and ran 24 hardlinked copies
of each same at08 binary. Both unbounded rows retained zero successful
observations and reported a timeout as their first error. The identical 48
binaries then all passed through a diagnostic two-worker schedule. The host
OOM-kill counter remained unchanged and the resource recorder did not abort.
The full raw streams, source, overlays, identities, timings, and sampled
resources are preserved. runAll retains only each row's first error, so the
unbounded record does not classify all 48 individual failures. This is a
controlled stress comparison, not a replay of the complete 775-program
release composition or proof of a full release pass.

## Frozen inputs and integrity

`freeze/` contains all 73 pre-existing package files, including the complete
harness, run/fail corpora, and mutant inputs. `preparation/` retains the
original test freeze, receipt, helper source, and test/implementation patches.
`candidate/` contains the final component source and test. Mutations are
external overlays; their original source and portable patches are retained.
The candidate hashes match `provenance/integration-source-001.json`, whose
source tree is `33752c03819ed5afcb37f87633a3cac668e181a8`.

Every `runs/` directory preserves original input/result JSON and complete
stdout/stderr bytes. The test command is TestRunBinaryExecutionQueue with
-count=1, -parallel=1, -short, -timeout=10m and explicit verbose output;
the restored run also uses -race. `specs/` and `recorders/` retain exact
commands, environment overrides, and recorder code. Original absolute paths
remain unchanged in historical records; `source-map.json` maps copies to
relative packet paths. Regenerate local overlay paths for replay instead
of rewriting these records. Scenario subprocesses do not inherit release
or stamp flags.

`files.sha256` covers every payload file. `manifest.json` also covers that
checksum list. Binaries, archives, caches, and build temporary trees are
omitted; archive verification records remain available.
