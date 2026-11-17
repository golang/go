# Final metadata audit: PASS

Reviewed after the integration-ready signal on 2026-11-16. No concrete
metadata or evidence-integrity defect was found in the frozen inputs.
This is a read-only consistency review, not a new Go test, hosted result,
or release certificate.

## Boundary

- Ready repository HEAD: `0a8eef6bab7d85e89e8e3efd0b933d73e78dff79`.
- HEAD tree: `1b7206610fee4f2ff3f6d7d73e34d4ba7aa0f721`.
- Compiler source: `33752c03819ed5afcb37f87633a3cac668e181a8`.
- The active request and candidate-summary edits, component umbrella README,
  and copied installed packet were intentionally not yet committed at freeze.
  `inputs.json` records their exact hashes and working-tree status.
- All 32 frozen audit inputs remained byte-identical through the final check.
  No source, test, repository metadata, ref, or commit was changed by this audit.

## Results

| Check | Observed result |
|---|---|
| Embedded and file task ledgers | Exactly 306 unique records, byte-equal and ordered; all earlier records unchanged |
| New audit tasks | AUD-210 through AUD-213 match the four owning slice headings and attacks #442 through #445 |
| Attack ledger | Exactly 350 unique rows: 341 tested, two satisfied upstream, no planned rows, seven blocked |
| Proving paths | All 583 occurrences accepted by the actual Go validator's path pattern resolve |
| Blocked set | Exactly #216, #217, #218, #226, #403, #404, #405; all retain Mechanism and Gate |
| Active request | Source33752; all six original frozen input names/hashes and scope preserved |
| Combined source | All twelve repaired-file hashes match the integration manifest; no source diff from HEAD |
| Public projection | Both public `go/types` generated targets retain their frozen hashes |
| Blocked M6 attacks | Original `M6-S3b.patch` remains byte-identical to the pre-repair base |
| Installed packet | All 152 files equal the sealed packet byte-for-byte; 150 checksum entries and 151 manifest payloads pass |
| Component packets | Alias 56, projection 74, environment 113 and parity 147 checksum entries pass |
| Active frozen manifests | All 26 compiler-stability entries and both release-checker entries pass |
| Local documentation links | All 31 links in the new/current summary and packet READMEs resolve |
| Frozen release coverage checker | Exit 0 with unchanged checker and inputs |
| Actual actionlint 1.7.12 | Exit 0 over all eight active workflows |
| Actual ShellCheck 0.11.0 | Exit 0 over all three active shell scripts |
| Integration whitespace | `git diff --check` exits 0; raw evidence remains unmodified |

`metadata-result.json` records the exact counts and component checksum-list
hashes. Each static validator has its full command, executable/input hashes,
exit code and raw streams in its own directory. `tool-versions.json` records
the actual installed validator versions.

## Contract and evidence review

AGENTS.md and G1/G2 require test-first attacks, restored mutation proof,
installed binaries after toolchain changes, uncached gates and race coverage
where applicable. The four slice records and commit bodies preserve those
component/installed distinctions and their one-candidate histories. The
installed receipts bind the combined uncommitted source snapshot at the
earlier base and index tree, not four separate intermediate rebuilds.

The only SPEC amendment in these four commits is the alias/census boundary.
SPEC 0.1 and 1.1 agree with T-0.1 and T-101.1: an unrelated duplicate cannot
abort the census, all active file kinds remain visible, and selecting or
importing the duplicate still fails. Header and mixed-package rejection,
ordinary Go semantics, and alias selection remain intact. The projection,
environment and parity changes add no new language rule or release exception.

The new leading candidate section correctly separates current source33752
from earlier source-e237 acceptance. Historical counts, tool identities,
pending statements and raw outcomes remain historical. The umbrella links the
sealed component proof to the later installed proof without changing either.
The old dogfood production manifest is not referenced by active CI; its
historical hashes have not been rewritten to hide the private-hook extraction.

All nineteen installed commands matched their expected outcomes. The summary
correctly distinguishes eighteen successful exits from the intentional
void-hook failure and subsequent restoration. It retains the eight short
type-checker skips and the three release-only parity skips. Definition checks
for mutants and sweep inputs are not presented as their release execution.

## Remaining acceptance boundaries

The documentation still requires full plain-Go regression, full script and
compiler-fixture suites, all six non-short optimization cells and separate
os legs, actual SWIG/LTO execution, and unfiltered oracle-enabled release
parity with mutant execution, upstream sweep and an unchanged clean-tip stamp.
Fresh runtime and second-host reproducibility jobs must carry their own source
identity. Linux installed evidence is not native Windows acceptance.

The three M6-S3b application rows remain blocked after the correction budget.
The four historical verification exceptions do not cover them. Their
independent release jobs remain runnable, and this audit does not reopen the
binding implementation or certify full language closure.

No Go command was run during this review. The parent's separately executed
compiled ledger test is independent evidence and is not claimed here.
