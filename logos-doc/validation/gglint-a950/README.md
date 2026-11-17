# Current gglint CI evidence

All three jobs passed on commit `a95099aae13ff7ffb4b3825bd63f8510bd184c69` in [run 43923681281](https://github.com/gosuda/gglint/actions/runs/43923681281). The final run status was read at 2026-11-16 18:21:31 UTC.

| Job | Job ID | Result | Observed coverage |
| --- | --- | --- | --- |
| race | 133497329552 | success | 39 tested packages passed; 3 additional packages have no test files; precision corpus included |
| lint | 133497329973 | success | vet, golangci-lint (0 issues), and gglint self-lint |
| vettool | 133497330041 | success | Full internal/vettool package passed with the race detector in 61.286 seconds |

The recorded workflow uses Go 1.27.4 on ubuntu-24.04. It splits the vettool package onto its own runner while retaining the existing nine-worker stress test. The workflow blob is `ec20a287bd5a5a2f0f5280f35060d7b69e844cac`. Test result caching is disabled by `-count=1`; compilation caches remain enabled.

The race command is `go test -race -count=1 -p 1 -parallel 2 -timeout 15m "${test_packages[@]}"`, where the workflow discovers every package and excludes only internal/vettool for its separate job. The vettool command is `go test -race -count=1 -p 1 -parallel 1 -timeout 15m ./internal/vettool`.

## Evidence

- [race.raw.log](race.raw.log), [lint.raw.log](lint.raw.log), and [vettool.raw.log](vettool.raw.log) retain the complete text returned by the GitHub job-log connector.
- [run.json](run.json) and [jobs.json](jobs.json) retain the API response objects.
- [summary.json](summary.json) records the exact revision, toolchain, commands, and observed results.
- [checkout-identities.json](checkout-identities.json) proves that the Actions
  checkout `d67294b7eb975cb3a27dd7dcafab9b00bee267ed` and the PR head have the
  identical complete tree `9fc49c87690544f82703c0210d5259aaf5b89684`.

This is independent CI evidence for the stated commit. Earlier killed local runs
remain failed or incomplete observations; their termination cause was not
established. The earlier run on `077dc30277a1a2431059fd12e20e4028a356ec4a` was
cancelled after the PR advanced and is not used as a passing result. No source,
tests, workflow, or branch was changed while collecting these files.
