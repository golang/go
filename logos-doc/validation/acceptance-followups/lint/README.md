# Lint acceptance follow-up

This directory preserves the confirmed evidence for the five-file lint cleanup
based on `45c29ff215477f7e403188d7452fc79aef05d791`.

## Observed failure and bounded change

The [hosted unit job](https://github.com/gosuda/LoGOs/actions/runs/43906416850/job/133436353880)
reported an unchecked `os.RemoveAll` result in `variant_features_test.go` and
an `unparam` warning for the test-only `findVariantContextForFile` adapter in
`eject.go`. The complete job log is retained in
[job-133436353880-attempt-1.log](job-133436353880-attempt-1.log); its original
identity and exact findings are recorded in [hosted-red.json](hosted-red.json).

The cleanup now reports removal errors. The adapter moved into test code as
`findVariantContextFromLinuxAMD64`, with the same Linux/amd64 host supplied by
all eleven original calls. The production `selectVariantContext` implementation,
test inputs, and existing assertions are unchanged. No linter is suppressed.

[transform-proof.json](transform-proof.json), [five-file-current-proof.json](five-file-current-proof.json),
and [lint-followup.patch](lint-followup.patch) record the exact transformation
and old/new hashes. The verifier records the original uncommitted worktree;
its absolute paths are provenance, not a portable installation requirement.
The recovered GitHub tree and blob readback were also checked against every
recorded target hash in [github-reconstruction.json](github-reconstruction.json).

The original compiler-stability and ejection-target-facts frozen manifests
remain unchanged. Their versioned successors record this mechanical adjustment
explicitly. The [static successor checks](../static/summary.json) passed for
all 26 compiler-stability entries and all five target-fact entries, with the
original test paths retained. The active compiler workflow checks its successor.

## Executed component validation

These checks compiled the candidate source with the freshly installed
`45c29ff2` toolchain. They do not establish a fresh build of the follow-up tree.
Tool identities and generated configuration are in [candidate-goroot.json](candidate-goroot.json);
the selected source directory, compiler build ID, and environment are in
[component-environment.log](component-environment.log).

| Gate | Result |
|---|---|
| Configured golangci-lint 2.15.0, full `./go/internal/logos/...` scope | Exit 0; zero issues; 109.337 seconds |
| Existing eject race binary compilation | Exit 0; 52.021 seconds |
| Full existing eject race suite, count 1 and short mode | Exit 0; 45 top-level tests and 100 total cells; no skips; 11.047 seconds |
| Direct eject vet | Inconclusive: interrupted at the ten-minute bound during unrelated filesystem stalls; execution-session exit 130; diagnostic log empty |
| Cleanup-condition mutation probe | Prepared outside the source tree; not compiled or executed |

The checks used one worker and a 512 MiB Go memory limit. Compilation and caches
stayed on disk; race test source directories used a real home-directory path
to preserve ordinary VCS discovery. The initial timing wrapper exited 127
because `/usr/bin/time` was absent, before lint started. The same lint command
then passed using Bash's built-in timing facility. Both records are retained.

[completed-component-results.json](completed-component-results.json) distinguishes
observed shell exit files from the vet execution-session interruption. The
race binary's recorded hash is retained; compiled binaries and incomplete
finalizer outputs are excluded. The transferred tool/include links were
observed unchanged and remained owned by root/language.

Fresh hosted `make.bash` and normal vet remain required acceptance gates.
The prepared mutation is not a passing proof. This evidence makes no claim
that the complete upstream regression suite passed.

## Evidence integrity

`copy-manifest.sha256` identifies the byte-for-byte copies from the confirmed
workspace records. `checksums.sha256` covers this directory, including this
README. Neither replaces a historical frozen test manifest.
