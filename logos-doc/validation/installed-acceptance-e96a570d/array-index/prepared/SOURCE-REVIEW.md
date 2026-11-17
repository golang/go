# Independent source review

Reviewed release `7d15ec7671582d0f022013bf36cf8f82a1b04cfc` and incoming
PR58 head `8afec95580ed24e593307fd9d30d626eff31e22d`, read-only. No Go
command or production edit was used for this review or preparation.

The production rule tries the array length when representable in the
index's concrete target integer type, then signed -1. Checker objects
identify immutable parameters and value receivers. Writes, address
exposure, and captures disqualify a parameter. The path evaluator stops on
unsupported syntax or effects, and checks an entire expression before
claiming the access is reached. One diagnostic is emitted per index node.
These mechanisms match the declared boundary-input scope; no source
correctness blocker was found within it.

The registry marks the rule language-neutral and suppressible at deny.
The new scope and default-policy tests retain the shared dispatch rules.
The contract does not introduce a plain-go compilation lint step.

All eight executable/source inputs frozen here have identical blobs in
the release commit and incoming PR58 head. The new SPEC paragraph and
Table T content survive in that head. Keep that source and normative
contract during integration.

Concrete bookkeeping corrections identified for root:

- The M9 rule row and M9-A-S18 still point to #333; the array-index attack
  is #440. #333 now denotes the Close-result repair.
- Add AUD-204 for M9-A-S18 in both task-ledger copies, preserving root's
  AUD-201 through AUD-203 reservations.
- Retain concurrent GC IDs #430 through #432 and sort active rows
  numerically. Qualify historical GC #333 references as then-#333/now-#430.
- Concurrent #333/#334/#335 describe the same Close/panic/imported-symbol
  repairs as stabilization #416/#417/#418. Preserve evidence and explicit
  historical cross-references; do not count aliases as additional defects.
- The overlay extension lacks its M24-A-S3/AUD-205 record. Its existing
  input names are TestInactiveCensusHonorsOverlay and
  logos_eject_overlay_sources.txt; tooling owns that source review.

PR64 reports installed verification at
`235b67e587d90fa8d58a699202c2f668f2721200` on base `bd318f1c...`:
make, full lint package, fixture completeness, both errorcheck inputs,
race guarded-enum script, vet, and formatting. This is author-recorded
evidence for that earlier source, not fresh acceptance of the merged
candidate. The reviewed body reports no caught/restored mutation; the
external-overlay probe prepared here closes that specific evidence gap.

The original test-first commit is
`326d94605c5ac20a84bdaf74f945e42cd0bb3cee`. Its commit message and PR body
record four missing diagnostics: offByOneReceiver and
signedMissingLowerBound, each in Go and loGo/s. Its valid guardedReceiver
and signedBothBounds controls stayed silent. The original 68-line input
is preserved with Git blob `9b128de39745732fa9c8129ec622528e92c4c285`.
No raw RED execution stream was retrieved, and none was reconstructed.
