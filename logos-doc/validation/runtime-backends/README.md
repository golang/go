# Backend runtime runner validation

This change adds a Python runner for actual CI execution of the pinned backend
profiles. No Go compiler, test, or runtime workload was run locally for this
change. The upstream package test bodies and `run.py` remain unchanged.

Base commit: `53b7c3aaaee74d8b7c5ff5fd0ce3ad7fd62c144d`.
Toolchain source tree: `61483affd1671d27e29a1a1aadc5299ba675f789`.
The backend runtime results themselves remain pending the CI invocation.

## Frozen controls and results

- `red.log` records collection failure because the requested new runner did
  not yet exist. It is a missing-runner witness, not 48 separate regression
  failures.
- The 48 initial controls cover exact cgo and modernc selections; nonempty,
  package-qualified run/pass multisets; multiplicity; missing and unsuccessful
  baselines; cross-profile baseline rejection; matching candidate positives;
  test and package failures; skips; malformed records; immutable policy and
  plugin configuration; and finite deadlines. Hypothesis exercises ordering
  while preserving event multiplicity.
- Independent review then found that non-test JSON records could conceal a
  build failure or unknown action behind passing test events. The separate
  `test_replay_backend_events.py` controls were frozen before that repair.
  `red-package-events.log` records eight failures and six valid-record controls.
- The documented Go build-event shape permits an empty `ImportPath` for a
  global diagnostic when `pkg` is nil. The controls preserve that shape:
  `build-output` can use an empty string, and `build-fail` is a failure with
  either a named or empty import path. See
  [`load/printer.go`](../../../src/cmd/go/internal/load/printer.go#L89-L124).
- `green.log` records 97 passing tests and 27 passing subtests across the
  unchanged shared harness and both new test files. Strict Pyright and Ruff
  passed for the runner and both new test files.
- `mutation-skip.log` records the frozen skip control failing when the
  implementation ignored test skips. `mutation-build-fail.log` records both
  build-failure controls failing when the implementation ignored build failures.
  Both mutations were reverted; the restored suite passed again.
- `frozen-files.sha256` records the unchanged test bytes and shared runner.
  All three matched after the repairs and mutation probes.

The behavioral commit is `304deaace804c0c35d16d2f8086ef521f39708ac`.
A separate follow-up formats the initial test file's import group: Ruff can
classify the new module as local only after that module exists. The frozen
pre-format file is preserved under `frozen/`, and `formatting.json` verifies
unchanged import names, non-import syntax trees, and exact test-function source
bytes. `frozen-files.sha256` now points to that preserved snapshot;
`published-files.sha256` identifies the final files. No assertion was changed.

The final Python command was run from `misc/logos-dogfood`:

```sh
uv run --offline --no-project --with pytest --with hypothesis \
  --python /usr/bin/python3 python -m pytest -q \
  test_run.py test_replay_backends.py test_replay_backend_events.py
```

The plugin profile was corrected before implementation after reviewing the
existing arena-default code. Common inherited `GOEXPERIMENT=none` requires an
explicit LoGOs-only `[build] arena=false` private-copy configuration as well as
`-O=1`. Its bytes and checksum are recorded by the runner. The reviewed modernc
capability policy remains separate and passes the existing validator unchanged.
These profiles do not overwrite the historical default-profile failure or the
historical modernc SQL timeout.
