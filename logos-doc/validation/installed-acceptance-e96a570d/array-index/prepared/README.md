# Array-index lint: prepared final acceptance

Status: **prepared only; no Go command has run for this acceptance bundle**.
Root must supply the final source/toolchain binding and release the Go lane
before execution. This directory is external to every source worktree.

## Frozen inputs

The eight production, unit-test, fixture, and script files in `frozen/` are
exact Git blobs from PR58 head
`8afec95580ed24e593307fd9d30d626eff31e22d`, tree
`8f750747af1eed986913e78e56c1106d03aa3d30`.
`frozen-inputs.sha256` identifies their bytes; `frozen-provenance.json`
also records their Git blob IDs. The source and tests are unchanged from
release `7d15ec7671582d0f022013bf36cf8f82a1b04cfc`.

`expected-unit-leaves.txt` names exactly 96 leaf tests: 45 behavior cases in
each language, four diagnostic/span cells, and two suppression cells. The
runner requires every listed leaf to pass; a missing or skipped cell fails
acceptance. Other suite skips are recorded explicitly.

`reference-documents.sha256` preserves the incoming SPEC, IMPLEMENTATION,
REDTEAM, and golangci configuration identities. These are reference hashes,
not final-worktree equality requirements: the approved ledger corrections
and other documentation integration are still pending.

## Final binding and invocation

The runner requires a clean final worktree at the supplied full commit ID,
unchanged frozen inputs, exact installed go/logos and compiler hashes, and
the selected C compiler command. The compiler path is the local
`pkg/tool/linux_amd64/compile`; this runner is the native linux/amd64 gate.
The required hashes must come from root's final build record. Do not supply
the earlier frozen45 binary identities as a substitute.

Once root supplies those values and releases the lane:

```sh
uv run --offline --no-project python \
  /workspace/scratch/9949907ca71e/recovery-evidence/array-index-acceptance/run_acceptance.py \
  --worktree "$ARRAY_FINAL_WORKTREE" \
  --commit "$ARRAY_FINAL_COMMIT" \
  --driver-sha256 "$ARRAY_DRIVER_SHA256" \
  --compiler-sha256 "$ARRAY_COMPILER_SHA256" \
  --cc "$ARRAY_CC" \
  --cache /workspace/scratch/9949907ca71e/postbuild-cache \
  --source-temp /home/agent/logos-array-index-final-9949907ca71e \
  --output /workspace/scratch/9949907ca71e/recovery-evidence/array-index-acceptance/runs/final-001
```

The output directory must not exist. `--phase gates` and `--phase probe`
allow a separately authorized continuation in a new record directory after
an interruption; neither claims the omitted phase. The default runs both.

Compilation is serial (`-p=1`, GOMAXPROCS=1, GOGC=20, 512 MiB soft heap
limit). Cache stays on disk. TMPDIR and GOTMPDIR use a real home directory
so nested source modules do not inherit workspace VCS ancestry. Confirm
adequate home space before execution. The cmd/go script harness constructs
its own child environment, so the wrapper's heap settings are not claimed
to propagate to every nested process.

Every test retains the upstream recorded ten-minute test timeout. Each
command has an eleven-minute outer bound, with process-group termination
and raw evidence on timeout or interruption. A timeout stops subsequent
Go stages. No downloads, compiler rebuild, source edits, or cache cleanup
are performed by this runner.

## Normal gates

From the final worktree's `src/`, using its installed `bin/go`:

```sh
go test -json -p=1 -count=1 -parallel=1 -timeout=10m -short cmd/compile/internal/logos/lint
go test -json -p=1 -count=1 -parallel=1 -timeout=10m -run='^TestEveryLintRuleHasFixture$' cmd/compile/internal/logos
go test -json -p=1 -count=1 -parallel=1 -timeout=10m -run='^Test$/^logos$/^lint$/^(array_index_out_of_bounds\.lgo|array_index_out_of_bounds_go\.go)$' cmd/internal/testdir
go test -json -p=1 -count=1 -parallel=1 -timeout=10m -race -run='^TestScript$/^logos_guarded_enum_index$' cmd/go
go vet -p=1 cmd/compile/internal/logos/lint
```

The runner also requires `gofmt -l` to list no frozen Go input. Both actual
errorcheck leaves must pass, proving the eight expected diagnostics. The
guarded-enum script must execute and pass; its existing assertions require
both native lint findings, three valid executions over all 256 values,
and three specific bounds-panic controls. No script assertion is replaced.

The actual race command also supplies `-o` to retain its exact cmd/go test
binary under the run directory's `binaries/`. Its path, hash, size, source
commit, and installed-tool hashes are recorded in
`guarded-enum-race/retained-binary.json` for tooling and compatible cgo checks.
Keep that executable outside committed evidence; no second cmd/go build is
needed merely to reuse it with another selector.

## G1 mutation and restoration

The sole mutation changes the new reporting condition:

```diff
- if proof.block(fn.Body.List) == arrayIndexReached {
+ if proof.block(fn.Body.List) != arrayIndexReached {
```

`mutation/rule_array_index_out_of_bounds.go` is the complete frozen source
with only that operator changed. At execution the runner writes a Go overlay
mapping the final source's absolute path to this external mutant. It does
not edit the worktree or replace the installed compiler.

The unchanged selected cells are:

- `TestArrayIndexOutOfBounds/guardedReceiver/x.go`
- `TestArrayIndexOutOfBounds/guardedReceiver/x.lgo`

Both safe guards stop witness 2 before indexing a two-element array. The
inverted reporting condition must produce an unwanted deny finding. The
runner requires both leaf tests to fail with the existing assertion's
`correctness::array_index_out_of_bounds deny` and `want []` text. A build
error, timeout, skipped test, or unrelated failure does not count as a probe.

After a completed probe, the same two cells run without the overlay and
must pass. Final hashes prove the source, fixtures, scripts, and installed
binaries stayed unchanged. This proves that the recompiled lint package's
condition matters; it does not claim a mutated installed compiler.

## Records and remaining scope

Each executed command saves argv, selected environment overrides, UTC times,
raw stdout/stderr, exit status, timeout status, and stream hashes. The test
stages add exact leaf checks. Before/after records bind the complete Git
tree, compiler source tree, frozen inputs, and installed tools.

`SOURCE-REVIEW.md` records the independent read-only assessment.
`upstream/` preserves the author's recorded RED provenance and original
test-first input. There is no recovered raw RED stream and no new local RED
claim. The four original core case bodies remain unchanged in the expanded
96-cell suite; preparation verifies that fact.

This is focused acceptance. The complete language fixture gate, broader
scripts, full Go regression, and other final integration gates remain
separate. The bundle does not promote REDTEAM status before execution.
