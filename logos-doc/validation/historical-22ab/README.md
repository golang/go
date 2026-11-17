# Historical 22ab validation evidence

This archive preserves observations made with LoGOs head
`22ab6f7a169e905d9fde5e345db101735a321486`, source tree
`8d74f82a97b8509400372ae4818525f19e7105ef`. It does not validate the corrected
environment-contract branch at
`82c3817d309fdbf6092e4a5ad6f2f9e69aadd0dc` or a later integrated source.
Original reports retain their recorded source and executable identities.

The interrupted gglint sweep used the separately recorded executable
`def1b9b9baede40727e79a5ddb2ae09c86aabd61fd27ff82fcdf9cadb5f7c113`;
its version output identifies `v0.0.0-20261116160051-14a66d9d8326` and Go 1.27.4.
Those results are not acceptance of subsequent gglint changes.

## Completed observations

- The installed toolchain build completed. The corrected VCS execution
  context passed the linker, cmd/go/work race and vet gates, and seven
  capability scripts under race. Initial VCS-context failures remain present.
- `corpus-review/installed-acceptance.tar.gz` is the original completed
  evidence bundle. It contains the environment package and script gates,
  nine xxhash commands, and eight actual purego FFI tests under each of
  three drivers. Its original manifest and all 106 regular members were
  verified again. Extract this nested bundle to inspect its xxhash/FFI logs.
- The environment propagation oracle used by those historical checks was
  later found to contradict M18-S9. It is superseded by the correction at
  `82c3817d`; the old successful runs cannot establish the corrected contract.
  Historical matched-O1 plugin runs used that old propagation behavior.
  The corrected plugin profile explicitly supplies the shared experiment.
- Go fixtures, native LoGOs execution, eject plus stock-Go execution, and
  exact native lint expectations have completed reports. The earlier failed
  eject invocation is preserved alongside the successful VCS-context replay.
- The mattn/cgo SQLite replay executed `TestOpen`, `TestClose` and `TestInsert`
  under stock Go, fork Go and LoGOs. The modernc baseline mutex test passed
  under stock/fork Go. Policy refinement attempts and the final explicitly
  granted one-test `TestMutexWakeOnce` LoGOs pass remain separate records.
- The broader modernc SQL campaign has `complete: true`, meaning that the
  campaign recorded all three outcomes. Stock Go passed 15 named test/subtest
  events in 55.518 seconds; fork Go passed the same 15 in 73.233 seconds.
  LoGOs reached its 600-second deadline, recorded 601.308 seconds including
  termination, and produced no started or passed test events. This is not
  completed LoGOs SQL runtime acceptance.

## Interrupted campaigns

| Report | Preserved state | Observed scope |
|---|---|---|
| `installed-audit/upstream/report.json` | `complete: false` | The log exposes `go/types.TestGenerate` failure. There is no completed full upstream result or final source-check record. |
| `corpus-vcs/native-default/report.json` | `complete: false` | Six finished cases, modernc marked running, 24 recorded cells: 16 pass, five findings, two timeouts and one invalid report. The planned campaign contained 28 cases. |
| `corpus-verified-gglint/report.json` | `complete: false` | Seventeen finished cases, sys marked running, 51 recorded mode cells: 43 findings, three clean and five timeouts. The planned campaign contained 28 cases and three modes. |

The counts above describe recorded cells, not complete matrices. A finding
does not establish upstream code quality, and a timeout or invalid report
does not count as a clean analyzer result. Raw partial logs remain included,
including logs without a finalized run record. Their archive hashes identify
the captured bytes; they do not manufacture a completed command or final hash
record in the original report.

## Integrity and scope

`ARCHIVE_INDEX.json` distinguishes campaign completion from pass/failure and
lists report snapshots. `RECORDED_LOG_HASHES.json` lists every stdout/stderr
reference whose original recorded SHA256 was verified against archived bytes,
including completed cells within partial campaigns. `FILES.sha256` covers
the archive payload; the external JSON manifest also covers that hash file.

The existing completed bundle retains SHA256
`844cb749d84f79db09004b1a92d0a4945823946359e714f10656fe9cc08fbad1`.
Original reports, logs, policies, runner sources and provenance files are
preserved without changing completion flags or adding retrospective source
checks. Work directories, source copies used as temporary work, caches,
repository metadata and compiled binaries are excluded.

Archive construction ran no Go workload. The tar contains sorted regular
members with mode 0644 and zero owner/timestamp fields. Gzip has no filename
and a zero timestamp. Repeated in-memory construction produced identical
bytes, and every member was read back and compared with its captured input.
