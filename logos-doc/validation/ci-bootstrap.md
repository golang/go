# CI bootstrap under the repository action policy

## Observed failure

On 2026-11-16, [job 133360690933](https://github.com/gosuda/LoGOs/actions/runs/43885015382/job/133360690933)
failed during runner setup, before checkout. Its error states that actions must
come from repositories owned by `gosuda`. The pinned external checkout, Go setup,
and Node setup actions were rejected. Their checksums and versions do not change
that ownership requirement. The same external dependencies appeared in all
seven stabilization workflows, including the artifact upload steps.

This failure is independent of compiler correctness. No compiler test ran in
that job.

## Repair and invariants

The workflows use native shell steps and keep the repository policy unchanged.
Checkout fetches `GITHUB_SHA` from the event's public repository, checks out that
exact commit, and verifies its identity. A pull request therefore continues to
test the event's merge commit. The step does not persist credentials. Its working
directory is the runner workspace because `src` does not exist before checkout.

The shared installer accepts only `go` or `node` on the existing X64 and ARM64
Linux runners. It downloads the selected official archive, checks its pinned
SHA-256 digest before extraction, checks the executable's version, and adds its
`bin` directory to `GITHUB_PATH`. The existing compiler build and upstream
oracle installation then run unchanged. No cache or toolchain version is
silently substituted.

| Tool | Runner architecture | Archive SHA-256 |
|---|---|---|
| Go 1.27.4 | X64 | `7226736c2339719cda28fbf3a54d71aa63bcefe376e98cef00034268dab24840` |
| Go 1.27.4 | ARM64 | `3392582988a273a87ab5043ee1b2c9c17a2c3c00580bf9326c47a2e0aaaeac92` |
| Node 26.13.1 | X64 | `5c063e46bab0fbc1c17cc445d7ff129cbdcd0471be6ed83b6ab864f8143bc864` |
| Node 26.13.1 | ARM64 | `89e1853138db51e8b82d3e98ba9c0d3f2956a28e5e55ea1b7e12d25aae92d5dd` |

Pins were checked against the official [Go download metadata](https://go.dev/dl/?mode=json)
and [Node 26.13.1 release checksums](https://nodejs.org/en/blog/release/v26.13.1#SHASUMS).
Go archives use gzip; Node archives use xz. The installer selects the native
architecture, including the ARM64 GC matrix.

Validation commands still print their output to the job logs. The final evidence
steps also print source identity and file hashes. The workflows no longer depend
on an external artifact action; the reproducible commands and mutation recipes
remain in the checked source.

## Verification boundary

All seven workflow files parse as YAML, all 60 shell blocks and both installer
scripts pass `bash -n`, and no external action reference remains. The existing
suite selections, mutation assertions, race gates, and full upstream regression
command are unchanged. Installer argument and architecture controls reject
unsupported inputs before reaching a download.

A successful hosted checkout, archive installation, fresh `make.bash`, and the
configured acceptance jobs remain required. Static workflow checks do not stand
in for those results.

## CI consolidation follow-up (2026-11-16)

M0-A-S18 merges the release workflow changes while preserving this action-policy
repair. The main workflow now builds exactly twice, for `unit` and `scripts`.
The scripts leg retains Node, all upstream oracles, C23 and ledger prerequisites,
and the independent full Go regression step. Its job budget is 180 minutes;
the full Go step and the unit job each retain a 90-minute budget.

The earlier seven-workflow/60-block result above records M0-A-S16. The combined
validation now covers eight YAML inputs, 67 workflow run blocks, and both
installer scripts. Actual actionlint 1.7.12 with ShellCheck 0.11.0 exits 0;
an isolated masked-export mutation is caught as SC2155, and all ten input
hashes remain unchanged. The bootstrap failure probe preserves exit 37 before
the build marker, and the SIMD array preserves all 17 selected files. See
[the CI merge evidence](ci-merge/README.md) for RED, GREEN, and probe records.
Fresh combined compiler and hosted full-regression acceptance remain pending.
