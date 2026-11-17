# Version 2: align the effective arm64 recipe

Use this version's `staging-plan.json` and `files.sha256` for integration. The
original proposal and all of its raw outcomes remain unchanged in scratch;
the plan retains their published copies under `history/v1/`. The already
committed frozen checker, baseline, and freeze records keep their existing
paths. No first-commit rewrite is needed.

## Correction

`cmd/go/script_test.go` constructs the script environment without forwarding
GOAMD64 or GOARM64. The unchanged `logos_repro.txt` therefore uses the LoGOs
deterministic defaults, v3 and v8.2, from `config/targetcpu.go`. Version 1 set
the direct order-service build and its recorded arm64 recipe to v8.0.

`arm64-recipe.patch` changes exactly two executor lines:

- Export GOARM64=v8.2 for the direct example.
- Record `goarm64:"v8.2"` in the cross-host recipe.

The workflow, request JSON, frozen checker, fixtures, assertions, and compiler
source are unchanged. GOAMD64=v3 already matches the script default. Both
paths explicitly use CGO_ENABLED=0 and an empty GOEXPERIMENT input; the normal
LoGOs defaults still add logosarena and Wasm preemptibleloops, with JSONv2 on
by default. Empty input does not mean the driver enables no default experiment.

| Executor | SHA-256 |
|---|---|
| Historical V1 | `2d8ec5d123a2d962ca39fd04952b93412385d43946ee820f875b866da74900f1` |
| Corrected V2 | `a9a5fdd226443ee84ac22bb9ccef3f6c52278d55cb7973b0062a73a2a156a7b3` |

The independent tooling review confirmed that V2 is byte-for-byte V1 plus
these two replacements and found no further concrete execution defect in the
reviewed recipe or executor boundaries.

## Fresh validation

The unchanged coverage checker passes on V2. Adding `-short` to the bulk
matrix command is caught; restoring the exact V2 bytes passes again. Actual
actionlint1.7.12, ShellCheck0.11.0, and Bash syntax checks all pass. These raw
results are new V2 records; no V1 result has been rewritten to describe V2.

`recipe-baseline-red.json`, `recipe-green.json`, and `source-facts.json`
describe the file-only effective-recipe comparison. They are not compiler or
hosted execution results. No Go command or hosted job ran for this correction.

The final manifest maps current production files, current documentation,
unchanged frozen inputs, preserved V1 history, and V2 evidence to explicit
repository paths. Its own path is excluded from its contents to avoid a
self-hash. Historical V1 manifests retain their original mappings as evidence;
the V2 manifest is the verifier for the proposed final tree.
