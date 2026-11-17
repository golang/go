# Corpus integration validation

This evidence validates the preparation, runtime-event, and acceptance helpers.
It does not establish that the compiler or linter passes the external corpus.
That requires the active workflow's recorded execution on its actual source,
tool binaries, reviewed policies, and resource profile.

## Scope and source identity

The integration starts from LoGOs `53b7c3aaaee74d8b7c5ff5fd0ce3ad7fd62c144d`.
The compiler source tree remains
`61483affd1671d27e29a1a1aadc5299ba675f789`. Native CI's subsequent
`bdc5987c526a9e9aa1b43975290f69a2370ae37a` changes only its runner from Ubuntu
26.04 to Ubuntu 24.04 after the former jobs remained unassigned. All native
test commands, fixtures, and timeouts are unchanged. The retry does not claim
Ubuntu 26.04 is unsupported; it had run earlier in this session.

Preparation integrates `946b69bca01a48bed626d91d2c9f92db0665507c`; backend
runtime validation integrates `eb554f609ac3d6167254db163c3aede1356833a2`.
The active workflow replaces the preparer's inactive example. The unchanged
shared `run.py` has SHA-256
`c2f2d0635450cc854c4fc7b4cf4a813d0b2e11a355bc960899bb85618efe9929`.
The corpus retains all original selections and source pins. No local Go build,
test, or corpus execution was run for this integration.

## Completed checks

`final-green.log` records **167 passing tests and 27 passing subtests** across
the original runner, source preparation, corpus acceptance, and both backend
test files. `final-pyright.log` records zero strict type errors or warnings.
`final-ruff.log` and `final-format.log` record successful checks of all seven
new Python files. Repository-wide actionlint 1.7.12 passed, including both
active workflows. `files.sha256` identifies the exact checked inputs and logs.

The original combined Ruff invocation identified import grouping in the
integrated module context. Its output is retained in `ruff.log`. Only import
grouping was corrected. `import-formatting.json` verifies unchanged non-import
syntax trees and exact function source text for all three affected test files.
The earlier successful test/type checks remain separate; the final checks ran
again after formatting. The runtime helper's own historical frozen-file records
refer to its published parent commit, before these integration-only imports.

## Negative controls

`mutations.json` records two temporary implementation changes against the
unchanged acceptance tests:

- Returning immediately from `check_invocation` made the named command/profile
  controls fail. Those controls change the primary and repeated invocations
  together, so matching repeats alone cannot satisfy them.
- Returning immediately from `known_load_failure` made the unrelated-type-error
  control fail. A generic nonzero exit cannot stand in for the pinned decorder
  compatibility error.

Both mutant commands exited one because their named assertions failed. Their
raw stdout/stderr and hashes are retained. The mutations ran in temporary
copies; neither changed the published implementation. The backend parent's
separate [evidence](../runtime-backends/README.md) retains its skip-suppression
and build-failure-suppression controls.

## External execution remains a separate gate

The [CI contract](../../../misc/logos-dogfood/CI.md) specifies 196 primary
compile/report cells, successful lint repeats, the explicit seven-cell failed
load, and independent runtime checks. `accepted_positive_cases` counts accepted
requested tool scopes; it is not a claim that every test file is compiled or
examined by both linters. In particular, the test-only semantic fixture requires
the separate runtime step.

The exact pinned gglint candidate `a95099aae13ff7ffb4b3825bd63f8510bd184c69`
passed its complete hosted Go 1.27.4 package, race, vettool, vet, lint, and
self-lint gates. The [archived CI evidence](../gglint-a950/README.md) preserves
those raw logs separately. Those package tests do not establish external corpus
acceptance. Earlier interrupted campaigns and modernc timeouts remain in the
[historical archive](../historical-22ab/README.md).
