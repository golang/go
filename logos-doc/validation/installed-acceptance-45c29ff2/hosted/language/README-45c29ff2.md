# Hosted language acceptance at 45c29ff2

All four assigned workflows completed successfully. Their job logs independently
identify the actual checkout as
`f2ad58be8742b8ec350db8f06d9dc794ea3302e4`, the pull-request event merge of
head `45c29ff215477f7e403188d7452fc79aef05d791` into base
`13a968f350564e739bf700df698a9a635dcdddcd`. Each job freshly built and installed
the toolchain. Root independently verified the merge tree equals the published
head tree: full tree `32b96f3a7c289c987bc021fe5a5adbf35cfe226b`, compiler source
tree `099367e1031585fa553ef7987b199288df36256a`.

| Workflow | Run | Job | Result |
| --- | --- | --- | --- |
| Generic type arguments | 43906417073 | 133436353512 | Success |
| Deferred callees | 43906416743 | 133436352713 | Success |
| Compiler-focused | 43906416846 | 133436352976 | Success |
| JSON profiles | 43906416853 | 133436355169 | Success in jsonv2 and nojsonv2 |

`completed-runs.json` indexes the source-bound records. Each job has a JSON
summary and a raw decoded log with a recorded SHA-256 and byte count. Original
UTF-8 BOMs and LF endings were retained. The numbered snapshots preserve API
observations of queued, active, and completed states; these are immutable
observations, not claims that pending gates passed.

## Evidence boundaries

- Generic type arguments passed native execution, race-enabled ejection, both
  caught/restored mutations, and restored syntax/types2/Lower suites. The exact
  mutation diagnostics and fixture restoration checks remain in the raw log.
- Deferred callees passed the dedicated native and race-ejection gates, the
  callee-span mutation, and its restored fixture check. Together these workflows
  supply the requested fresh installed proofs for REDTEAM #386 and #389.
- Compiler-focused passed its C23 consumer check, compiler/frontend/command
  units, race packages, CTFE/default/tuple/inline scripts, Cgo/variant ejection,
  native fixtures, both vet gates, and the actual ledger checker.
- JSONv2 and classic JSON both passed vet, runtime and Lower race tests, native
  JSON fixtures, and race-enabled ejection/fallback scripts.
- The non-verbose JSON logs do not identify TestEjectQuotedCgoTypes or print
  CGO_ENABLED. Keep its explicit local verbose gate in both profiles required.
  Likewise, root's verbose 48-cell imported-symbol gate remains a separate
  acceptance obligation. Broad package success is not a per-test execution log.

Expected FAIL lines in the generic and deferred logs belong to deliberately
caught mutations. No product failure was observed in these four workflows.
This report does not claim that the separate main workflow passed: root is
handling its stale CTFE expectation and other acceptance findings.

## Test-only CTFE correction prepared separately

`const-func-store-review.md` explains why the old unsupported map-field-store
expectation conflicts with the integrated assignment contract. Root accepted a
test-only correction in `logos-acceptance-fixes`. The function body is preserved;
the replacement requires a compile-time constant equal to 4 and output `4`.
Its extra call may fold and is not claimed as forced unfolded runtime coverage.

`const-map-field-oracle/` contains the original negative bytes, frozen positive
hashes, preservation checks, and planned native/ejected commands. No Go process
or commit was performed for that correction. The historical manifest's two
pre-existing adapter mismatches are recorded explicitly and belong to the
tooling lane; the ten frozen CTFE entries and genuine rejection controls match.

No workflow was rerun, no hosted branch was changed, and no compiler source or
repository documentation was edited by this monitoring lane.
