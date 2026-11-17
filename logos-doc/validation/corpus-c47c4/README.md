# Completed corpus run before the inline-context repair

[Run 43928753813](https://github.com/gosuda/LoGOs/actions/runs/43928753813)
finished with **failure** on head `c47c4cfc56fc8bfa916d170ed51040195b567dc9`,
source tree `61483affd1671d27e29a1a1aadc5299ba675f789`.
Only the primary corpus acceptance step failed. The raw report contains all
196 cells: 80 pass, 98 findings, 10 clean, 1 candidate-failed, 1 upstream-failed,
2 uncompared, and 4 load-failed. Findings are retained diagnostics, not an
invented clean result.

The `tools/logos` build is the candidate failure. Its stderr contains 44
capability errors: 39 network and 5 process-launch errors. The inline-copy
identity defect and its frozen regression are documented in
[the linker repair](../capability-inline-context/README.md). This archived run
does not validate that later repair. Actual telemetry process launches and
conservative interface-dispatch paths require separate permission review.

The pinned decorder baseline fails because its `golang.org/x/tools@v0.20.0`
dependency uses an incompatible `internal/tokeninternal` layout assertion.
Its stock-Go failure, two uncompared candidate builds, and four incomplete
lint loads are preserved. None counts as a successful analysis.

## Independently completed execution gates

| Profile | Named tests and subtests per driver | Stock Go | Fork Go | LoGOs |
| --- | ---: | --- | --- | --- |
| xxhash assembly | 159 | Pass | Pass | Pass |
| xxhash purego tag | 159 | Pass | Pass | Pass |
| xxhash plugin with cgo | 223 | Pass | Pass | Pass |
| SQLite through cgo | 3 | Pass | Pass | Pass |
| SQLite through modernc | 15 | Pass | Pass | Pass |
| purego FFI | 8 | Pass | Pass | Pass |

The named event multisets match each stock baseline exactly, with no failures,
skips, or omitted names. The two synthetic semantic and C ABI fixtures pass
under all three drivers. Native loGo execution, ejection and upstream-Go
execution of the ejected program agree. Native lint emits its exact three
expected diagnostics on both repetitions, and its runtime controls pass.

The modernc candidate completed its 15 tests in 29.039 seconds in this CI
profile. This does not erase the earlier 601-second timeout under a different
profile. Every source and tool identity checked before and after remained
stable. The reviewed modernc permission file and plugin arena-off setting
were confined to the corresponding private LoGOs copies.

## Permanent evidence

- [Original artifact ZIP](dogfood-43928753813-1.zip), SHA-256
  `8d6f69a36646775185ca3c7292bcfe3c7ce2bb662300cfcdd76e313483b271ed`.
  It includes all raw streams, the 196-cell report, runtime reports, source
  receipts, manifests, policies, and exact resource and tool identities.
- [Complete job log](completed-job.raw.log), [run metadata](completed-run.json),
  [job metadata](completed-jobs.json), and [artifact metadata](completed-artifacts.json).
- [Independent runtime event audit](runtime-audit.json).
- [Unmodified tools failure](tools.logos.stderr), SHA-256
  `d8c559f2ae777b062b7768204b563b45eda9426ca851cb27a1d15526b1be0c70`.
- [SHA-256 manifest](SHA256SUMS).

The ZIP retains its original paths below `_temp/logos-corpus-evidence` and
`_temp/logos-corpus-inputs`. Its complete report is
`_temp/logos-corpus-evidence/corpus/report.json`.

The corpus is 28 pinned configurations of Go/loGo code and C interoperability
fixtures across five source hosts. It is a reproducible sample, not a claim
to cover every repository or every programming language.
