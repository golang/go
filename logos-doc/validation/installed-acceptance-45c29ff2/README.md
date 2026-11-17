# Installed acceptance at 45c29ff2

The freshly installed shared Go and loGo/s toolchain passed the scoped checks
below. This is an acceptance snapshot, not a complete release certificate.
The main workflow failed on one obsolete CTFE oracle and two lint findings;
its full Go regression never started. The subsequent corrections have their
own [follow-up record](../acceptance-followups/README.md).

## Exact source

| Identity | Value |
|---|---|
| Published PR head | `45c29ff215477f7e403188d7452fc79aef05d791` |
| Hosted event merge | `f2ad58be8742b8ec350db8f06d9dc794ea3302e4` |
| Complete tree, equal for both commits | `32b96f3a7c289c987bc021fe5a5adbf35cfe226b` |
| Compiler source tree | `099367e1031585fa553ef7987b199288df36256a` |

Each hosted raw log identifies the event merge checkout. The local build and
installed checks used the clean published head. Root independently compared
the complete Git trees; `source-identity.json` records that comparison. The
commits differ in ancestry, not tested file contents.

## Completed hosted workflows

All six successful workflows freshly built their exact checked-out source.
Their raw logs, job metadata, fixture identities and acceptance summaries are
under `hosted/`. Expected mutation failures are identified within those
successful workflows; they are followed by restoration checks.

| Workflow | Run | Completed scope |
|---|---|---|
| Generic type arguments | [43906417073](https://github.com/gosuda/LoGOs/actions/runs/43906417073) | Native type arguments and deferred captures; ejection under race; both generic-target and prefix mutations caught; restored syntax, types2 and Lower suites |
| Deferred callees | [43906416743](https://github.com/gosuda/LoGOs/actions/runs/43906416743) | Full Lower; native calls; race ejection; callee-span mutation caught and restored |
| Compiler-focused | [43906416846](https://github.com/gosuda/LoGOs/actions/runs/43906416846) | C23 consumer; affected compiler and command packages; race; native and ejected scripts; native fixtures; vet and actual ledger test |
| JSON profiles | [43906416853](https://github.com/gosuda/LoGOs/actions/runs/43906416853) | Runtime and Lower race suites, native JSON fixtures and ejection/fallback checks under both `jsonv2` and `nojsonv2`; both vet gates |
| SIMD guards | [43906416710](https://github.com/gosuda/LoGOs/actions/runs/43906416710) | Full SSA, host-ISA condition, six named SIMD scripts and README under race; formatting and vet |
| Region GC | [43906417036](https://github.com/gosuda/LoGOs/actions/runs/43906417036) | Native amd64 and arm64, with and without Green Tea; runtime/metrics/profile/trace, race, lifecycle and scan stress, noinline safepoint stress and heap marking; both native ownership mutants caught ten times |

No test-level skips appear in the SIMD result. That observation does not imply
that every instruction path guarded by host hardware was executed. The region
jobs without Green Tea intentionally omit only the ownership mutation and its
evidence step; their normal gates pass. The JSON workflow's quiet test output
does not identify the quoted-C test by name. Its explicit local executions
below establish that separate obligation.

## Main workflow failure

[Run 43906416850](https://github.com/gosuda/LoGOs/actions/runs/43906416850)
tested the same event merge. The unit leg completed its fresh build, smoke,
configured unit suites, formatting and vet. Its complete compiler-fixture
step had exactly one failure: `const_func_unsupported_store.lgo` expected a
diagnostic for a valid fresh local map-field store. Configured lint then
reported unchecked `RemoveAll` cleanup and a constant host parameter in an
adapter used only by tests. The exact diagnostics and logs remain in
`hosted/runtime/run-43906416850-attempt-1-triage.json`.

The scripts leg passed its fresh build, smoke, actual workflow lint, pinned
Node/oracle prerequisites, C23 compile/run probe, ledger test and shared-C
consumer. Matrix fail-fast canceled it during the complete loGo/s script
selection. That selection has no acceptance result. The independent full-Go
step was skipped and never started. Neither cancellation nor an earlier
successful focused run satisfies either gate.

## Fresh local build

`local/fresh-make-45c29ff2/` records a normal `src/make.bash` ending with exit
0 after 1410.125 seconds. Both commands were installed. The compiler used the
existing `default.pgo`; its hash is
`eaa042c52378bb04ee46e03c71c67c8d2b19704aefc07e9048b80c6f3ea59d5a`.
No bootstrap shortcut or cleanup change was used.

The resource bounds were GOMAXPROCS=1, GOGC=20, a 512 MiB soft Go heap limit,
and compiler concurrency one. Dist clears GOFLAGS, so the one-worker bound
comes from GOMAXPROCS, not an assumed inherited `-p` flag. Normal dist cache
cleanup remained intact. The acceptance cache was newly empty after the
build and was not populated by stock Go.

| Installed tool | SHA-256 |
|---|---|
| `bin/go` and `bin/logos` | `987052b593c41cf60390eec80d98b01b89f9536d6d87b4e73672ab6e7a4eb2ef` |
| Compiler | `61749f8f2b2242d26e613993270f654cf47c750739a22b223f94eedd5370bba9` |

The local handoff and build result retain all six tool hashes, command lines,
versions and resource observations. The accepted source stayed clean.

## Local installed gates

| Gate | Actual result |
|---|---|
| Imported local types and lazy fact scopes | All 48 frozen leaf cases passed, with no skips: 20 native-import cases and 28 scope cases; package time 51.768 seconds |
| Native declaration validation | All eleven unchanged script commands passed under race: three accepted declarations and eight required diagnostics; package time 7.320 seconds |
| Quoted C types, `jsonv2` | `TestEjectQuotedCgoTypes` explicitly ran and passed under race with CGO enabled; 8.76 seconds |
| Quoted C types, `nojsonv2` | The same unchanged test explicitly ran and passed under race with CGO enabled; 39.74 seconds |

These commands use the installed compiler, not only direct lowering tests.
The validator covers retained M6-S3a declaration checks; it does not establish
M6-S3b default application or binding. The quoted-C fixture hash is unchanged:
`bc1903980a8fa11ad0724c8eada997d0b5bdbfb5713efbe46d27cffdacac3084`.

The first validator launch failed before its accepted control because a
temporary module inherited unusable workspace VCS ancestry. The evidence
preserves that failure. Running the same source with the same compiler in a
real directory under `/home/agent` succeeded with ordinary VCS behavior.
The comparison is in `local/vcs-environment-recovery/`.

Go 1.27 test temporary directories also use GOTMPDIR. For the quoted-C race
tests, compilation used disk-backed object storage; each exact resulting
test binary then ran with both TMPDIR and GOTMPDIR in a unique real home
directory. No fixture, VCS flag or skip guard was changed. Bootstrap-only
compiler environment flags were absent during acceptance, because they can
make build-dependent tests skip.

## Cache-identity correction

The earlier warning that an unchanged VERSION could conceal compiler changes
was overbroad for this fork. Its existing `forkReportsBuildID` hook reports a
content build ID even for release-version compiler tools. The command's tool-ID
logic includes that identity in compile actions, separating the stock bootstrap
from the rebuilt fork on the audited Linux amd64/arm64 hosts. A separate GOROOT
alone is not that protection. No cache-policy repair is needed.

[The cache audit](hosted/runtime/cache-audit.md) records exact source references
and its boundary: this is source-level exclusion of the proposed collision,
not a measured cache-hit trace. The empty local acceptance cache remains
deliberate isolation. The audit does not invalidate the hosted results.

## Evidence boundaries and reproduction

`evidence.sha256` identifies the copied completed records byte for byte. Raw
logs are unchanged; absolute paths inside them describe the observed session.
Local runner scripts are retained with the records. Downloaded tools, caches,
test binaries and repetitive polling snapshots are not committed here.

The broad scripts, full Go regression, real gopls and QEMU-dependent checks,
six driver/optimization release combinations, stamped full goparity run and
release reproducibility obligations remain separate gates. The retained
M6-S3b attack patch and its three-attempt stop remain unchanged.
