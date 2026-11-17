# Hosted independent release acceptance

This is a file-only proposal for parent review. No candidate repository file,
compiler source, fixture, ref, or cache was changed. No Go command or download
ran while preparing it. Static acceptance does not establish a hosted result.

## Apply these three files

The `candidate/` directory contains their intended repository paths:

- `.github/workflows/release-acceptance.yml`
- `.github/scripts/release-acceptance.sh`
- `logos-doc/validation/release-acceptance-request.json`

Retain this directory's checker, raw static results, mutation patch, extraction
proof, and handoff manifest as validation evidence. The checker is frozen at
SHA-256 `c46d9cc8569ca80659a999ab11a5208de5afecd8c4e337817af0de81bde20b26`.
It was run against unchanged e96 before the workflow or executor existed.

The request binds all jobs to source tree
`e237c399aae8f8065b724e8a0e27585686c7da18`, plus six unchanged fixture/input
hashes. Metadata-only promotion commits can supply the tested clean tip while
retaining this source tree. A later compiler change requires a reviewed request
update and fresh evidence; these jobs will reject it until that update exists.

## One execution direction

Finish the installed evidence for the remaining planned rows, promote only
proved rows in the ledger, then commit this workflow, executor, and request.
Keep that PR head fixed while the jobs run. The new workflow tests the exact PR
head, not the event's temporary merge commit, so the complete parity stamp names
a reviewable clean tip. Both identities are recorded. Manual dispatch also
accepts an exact 40-character commit. The workflow uses read-only permissions,
the existing public shell checkout, and the existing hash-pinned installers.
There are no external actions or policy changes.

The new workflow does not cancel an in-progress release proof when the branch
changes. Existing workflows retain their own cancellation policy, so the
source hold still matters for their broad regression results.

| Job | Required execution and success condition |
|---|---|
| Six `std-cmd` cells | Both `go` and `logos`, O1/O2/O3, complete `go list std cmd` inventory, without short mode. Each cell runs the frozen 31-name exclusion/control script, all packages except `os` at `-p=3`, then `os` at `-parallel=1`. Every command uses `-count=1`. No new explicit skip is added. |
| Complete parity | Reject planned ledger rows, then run the entire unfiltered `cmd/internal/goparity` suite with the upstream oracle, `-goparity.release`, and a real stamp. No short/run/skip/oracle-off selection. Compare the resulting stamp to the unchanged clean checkout. This executes mutants and the upstream sweep. |
| Two repro hosts | Separate Ubuntu 24.04 and current Ubuntu hosted jobs build the same source. Their hostname/boot fingerprints must differ. Each runs unchanged `logos_repro` and `logos_eject_example`, then builds the actual `order_service.lgo` twice with new empty caches for linux/amd64, linux/arm64, and js/wasm. |
| Repro comparison | A third job consumes the two successful job outputs. It compares source/head/tree, target recipe, and all six program/target digests. It rejects reuse of the same host. The current Ubuntu label may resolve to the same image version; the proof requires independent VMs, not different operating systems. |
| Real runtimes | Install pinned gopls with stock Go and a separate cache; verify its module sum, version, and build information. Verify/extract the pinned QEMU archive. Run unchanged LSP, arm64 sum, all Wasm scripts, and plain-Go/logos/upstream Wasm parity under verbose uncached cmd/go race testing. Require all script PASS records, no skips, all eleven LSP output markers, the actual QEMU execution/comparison commands, and stack-switching Node execution. |
| M22 closure | Report the ledger's stricter release scope separately. Only historical blocked rows #216, #217, #218, and #226 are accepted. Additional M6-S3b rows #403–405 still make this job fail. They do not suppress any independent behavioral job, and no blocked binding implementation is retried. |

The existing main workflow still owns the ordinary full `run.bash` regression.
This proposal does not count its normal short run as the six no-short cells or
the parity release tier. Existing JSON-profile, race, compiler-fixture, and
codegen gates remain independent requirements.

## Exact reproducibility scope

`logos_repro.txt` embeds a small `example.com/repro` module. It prints eight
hashes from its three clean-cache comparisons and relocated-source comparison;
the executor verifies that structure before publishing three target digests.
Its unchanged size tests remain the only comparison to the recorded stripped
Wasm size. The larger order-service program is never compared to that record.

M22-S5 names the actual example, so the executor also copies the committed
`order_service.lgo` without renaming it. Its module bytes come from the
`m/go.mod` txtar member in `logos_eject_example.txt`. A file-only check executes
the exact extractor and proves those module bytes and the source hash. Each
actual-example build receives a unique empty cache, with only that completed
owned cache removed afterward. Output binaries remain until the hosted job
ends; their digests and raw commands are retained.

Both hosts use CGO=0, GOAMD64=v3, GOARM64=v8.2, an empty explicit GOEXPERIMENT,
and `-trimpath -buildvcs=false` for these program artifacts. Those are recorded
recipe inputs. This proves the specified program artifacts across hosts; it
does not claim byte-identical compiler bootstraps or arbitrary applications.

## Resource and prerequisite policy

All jobs use existing standard public Ubuntu labels. The official runner
documentation reports four CPUs, 16 GB RAM, and 14 GB SSD nominal for standard
public Ubuntu 24.04. The executor checks actual `MemAvailable` and free disk;
it does not infer them from that nominal specification.

Ordinary heavy jobs require at least 6 GiB available memory and 8 GiB free disk.
Parity requires 10 GiB of each. These measured preflights run both before and
after bootstrap. A failed reserve is an infrastructure failure with its actual
values retained, not a skipped test or a reduced configuration set. The parity
harness launches 20 active fork configurations and supplies child
GOMAXPROCS=8 while dropping outer GOFLAGS/GOMEMLIMIT. Its actual peak on a
standard hosted runner remains unproved. If the unchanged harness exhausts
that host, preserve the concrete failure and provision sufficient resources;
do not delete configurations or assertions.

Each heavy job performs a normal fresh `make.bash` with the pinned stock
Go1.27.4 bootstrap, GOMAXPROCS=2, a 2 GiB soft heap, and `-c=1` for both compiler
phases. Default final PGO remains enabled. Dist may clear GOFLAGS during its
normal bootstrap; the proposal does not claim that `-p=2` bounds every child.
The post-build fork cache starts empty and is distinct from the stock gopls
cache. Compiler bootstrap flags are unset before acceptance tests. The runtime
job verifies native CGO for race compilation without forcing CGO into the
cross-target fixtures.

Temporary source/cache roots are real disk directories under RUNNER_TEMP,
outside any Git ancestor. A tmpfs or Git-ancestor temp root fails before builds.
The executor does not suppress ordinary VCS behavior in general test modules.
Only the reproducibility recipe uses its specified `-buildvcs=false` flag.

All six matrix cells may run concurrently on their independent hosted VMs.
Each retains its own resource preflight. The six matrix jobs have 360-minute
limits, parity 300, repro 180 per host, and runtimes 240. These are ceilings,
not runtime estimates. There are ten independent toolchain builds in total;
the repository action policy precludes the previous external artifact action,
so this proposal does not assume cross-job toolchain caching or transfer.

### Local gopls readiness

The local v0.24.0 main module archive/source and its original hashes are intact.
All 29 dependency versions from its go.mod lack cached metadata, zips, and
extracted source in the three checked local module caches. The missing
telemetry dependency is directly imported by main.go. No gopls executable is
installed, and no dependency download/install has run. Stock Go1.27.4 satisfies
the module's Go1.27.0 requirement, but that alone does not make installation
possible offline. The proposal therefore assigns this real gate to the hosted
network environment and fails if authenticated dependency provisioning fails.
It does not assert that the local proxy endpoint has been probed or rejected.

## Evidence and validation

Every executed command retains its argument vector, start/end timestamps,
exit status, and raw log. Each job retains source identities, versions, binary
hashes, selected settings, preflight values, and all result data. It does not
dump the entire environment. The final always-run step creates a deterministic
gzip/tar bundle of the evidence directory, prints its SHA-256, and writes it
as base64 between named markers in the hosted job log. This preserves raw
non-UTF-8 failures without an external artifact action. Installed binaries,
caches, downloaded tools, and test work trees are excluded from the bundle.

Job logs inherit the repository's retention policy, recorded when supplied by
the runner. Download them promptly and use `recover_log_evidence.py` to recover
the archive by hash. The tool writes an archive only, refuses overwrites, and
does not extract untrusted paths. Verify its internal `files.sha256` manifest
after reviewing/extracting the archive into a dedicated evidence directory.

The frozen contract check records baseline RED for the absent coverage,
candidate static GREEN, a caught one-line addition of `-short`, and restoration.
Actual actionlint1.7.12, ShellCheck0.11.0, and Bash syntax checks also run.
Initial static setup findings and the extractor correction remain recorded.
None of these static results is a hosted runtime PASS. The three M6 binding
blocks remain visible even if every independent behavioral job later passes.

For this environment, rerun the frozen checker with:

```text
PYTHONDONTWRITEBYTECODE=1 uv run --offline --no-project --python /opt/codex/runtimes/codex-primary-runtime/dependencies/python/bin/python3 python check_release_contract.py candidate
```

The current input hashes and exit outcomes are listed in
`arm64-recipe-v2/results.json` and `arm64-recipe-v2/files.sha256`.
The original proposal and its raw static outcomes remain under `history/v1/`. The broader source/evidence assessment is retained separately
at `recovery-evidence/release-review-e96a570d.md`.

Parent integration assigns this coverage slice M22-A-S6 and AUD-209. It adds
no language rule, attack row, or SPEC exception. Commit the frozen checker and
baseline evidence first, then the workflow/executor and static evidence. The
two groups and intended paths are listed in `arm64-recipe-v2/staging-plan.json`.

## Arm64 recipe correction

The initial proposal recorded v8.0 for the direct order-service build. The
cmd/go script harness drops GOARM64, so the unchanged `logos_repro` script
actually uses the deterministic LoGOs default, v8.2. Version 2 changes exactly
the direct export and recorded recipe to v8.2. GOAMD64=v3 already matches
the script default. The explicit empty GOEXPERIMENT input is not a claim
that the driver enables no default experiments; both paths retain the same
LoGOs defaults. No compiler source, fixture, assertion, or frozen checker
changes. The correction patch and fresh static evidence are in
`arm64-recipe-v2/`; V1 outcomes are historical and remain byte-for-byte intact.
