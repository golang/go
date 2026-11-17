# Final metadata audit checklist

Status: prepared only. The final audit starts after the parent supplies its
integration-ready signal. This checklist does not certify the working tree
while documentation and evidence are being integrated.

Repository: `/workspace/scratch/9949907ca71e/logos-acceptance-fixes`.

Expected compiler source tree:
`33752c03819ed5afcb37f87633a3cac668e181a8`.

Preparation HEAD: `0a8eef6bab7d85e89e8e3efd0b933d73e78dff79`. The installed
receipts instead bind the earlier combined, uncommitted snapshot at base
`2535486fe9596a9212d525a5007963ef9e8682f0`, index tree
`098de80a4ddf1e59b2750c1306ef495414f42d1e`, with the same compiler source tree.
Do not relabel those receipts as builds of the later individual commits.

## 1. Establish the audit boundary

- Record the ready HEAD, full tree or explicitly staged tree, source tree,
  working-tree status, and hashes of each audited input before reading it.
- Account for every integration diff. Only the parent's intended metadata,
  request, and evidence additions may differ from the four completed slice
  commits. Compiler production and frozen test bytes must remain unchanged.
- Re-read identities after the audit. A concurrent change invalidates only
  the affected audit result and requires a fresh bounded check.
- Read `AGENTS.md` and IMPLEMENTATION G1/G2 as the current repository rules.
  Existing user authorization governs publication despite historical planning
  prose. This review performs no Go execution, source edit, commit or ref move.

## 2. Bind the active request and its immutable inputs

- `logos-doc/validation/release-acceptance-request.json` must request source
  tree `33752c03819ed5afcb37f87633a3cac668e181a8` and retain its scope boundary.
- Verify the exact six `frozen_inputs` entries against their actual bytes:
  `logos_redteam_80.txt`, `logos_repro.txt`, `logos_eject_example.txt`,
  `logos_lsp.txt`, `logos_sum_crossarch.txt`, and `order_service.lgo` at the
  paths recorded in the request. The set and historical hashes stay unchanged.
- Confirm that the executor validates the source tree, frozen files and clean
  candidate tip before work. The eventual hosted proof must record the actual
  checkout identity; an old run is not a result for the new request.
- Check active workflow manifests against current files from their documented
  working directories. Inspect called scripts, not only YAML literals.
  Historical manifests remain historical. Never rewrite old evidence to make
  a current source comparison pass.

## 3. Reconcile the task ledgers exactly

- Parse the fenced JSONL under `## Autoplan ledger` in IMPLEMENTATION and
  `tasks/autoplan-ledger.jsonl`. Require the same ordered objects and exactly
  306 unique IDs, with no missing, duplicate or extra entry.
- Match the four new audit IDs to their owning headings and attack rows:

| Slice | Audit task | Attack |
|---|---|---|
| M0-A-S20 Keep duplicate-stem errors package-local | AUD-210 | #442 |
| M0-A-S21 Preserve the Go-only type projection | AUD-211 | #443 |
| M22-A-S7 Isolate acceptance from bootstrap settings | AUD-212 | #444 |
| M22-A-S8 Bound parity child admission | AUD-213 | #445 |

- Preserve the earlier AUD-209/M22-A-S6 task and every older task entry.
- Check each new heading's anchor, red/green/probe account, one-candidate
  boundary, installed evidence links and remaining gates. Check commit bodies
  against G1 without treating intermediate commits as separately rebuilt.

## 4. Audit every attack row using the real parser contract

- Apply the same five-cell split, accepted statuses and proving-path regular
  expression as `src/cmd/internal/goparity/redteam_ledger_test.go`.
- Require 350 unique attacks: 341 `tested`, two `satisfied upstream`, zero
  `planned`, and seven `blocked`.
- Every tested/upstream row must name at least one existing proving path;
  every matched path must exist or satisfy the validator's extension glob.
  Prose such as `test/root` and `test/tool` must not create false path matches.
- Blocked IDs must be exactly #216, #217, #218, #226, #403, #404 and #405.
  Each retains its mechanism and gate. The four historical verification
  exceptions do not authorize the three blocked M6 binding implementations.
- #442 through #445 must state only their observed component/installed scope.
  #444 retains source-e237 runtime/security evidence and does not infer SWIG
  execution from stubs. #445 retains the three release-only skips.
- Record this as a file-only validation. Do not claim that the actual Go
  ledger test was rerun as part of this audit.

## 5. Check SPEC and Table T without weakening contracts

- Match the alias correction in SPEC 0.1/1.1 to T-0.1/T-101.1: an unselected,
  unimported duplicate does not abort the census, all active kinds still count,
  selected/imported duplicates still fail, and header/mixed-package precedence
  remains intact. Plain Go still ignores loGo/s aliases.
- Projection extraction preserves public Go semantics and public `go/types`
  output. It does not introduce a language rule or relax C-type/panic behavior.
- Environment and parity admission repairs implement existing acceptance
  rules. T-Sec17 still requires both drivers, O1/O2/O3, full unmodified suites
  without short mode, runtime race coverage, and reproducibility.
- Preserve SPEC 17, M22 release requirements, and the exact four historical
  blocked exceptions. Do not treat M6 as blocking independent release jobs.
- Preserve T-104.5 and the declaration-validator/default-application distinction.
  Verify `logos-doc/blocked/M6-S3b.patch` remains byte-identical; no fourth
  binding attempt, test weakening, or retrospective SPEC cut is implied.

## 6. Verify the portable installed packet and component packets

- The integrated `logos-doc/validation/installed-acceptance-33752c03` tree must
  equal the sealed portable packet byte-for-byte, with no renamed payloads.
- Expected installed manifest SHA-256:
  `6e4ce2e3a1fe0c13b66fd18b7493989d4c3829d128557b9a24c312dffeeb07aa`.
- Verify all 150 checksum entries and 151 manifest payload entries, 152 files
  total; 147 original copies; 19 gate directories, 19 specs and four review
  checkpoints. Reject additional executable files, cache trees or symlinks.
- Preserve the successful make002 result, first interrupted build without a
  result, restart preparation, tool identities, recorder dependencies, and raw
  streams. Do not normalize text or replace historical absolute paths.
- Preserve all component packet manifests, baseline failures, setup-only
  failures, frozen inputs, caught mutations and restorations unchanged.
- Confirm the installed receipt boundary: 18 successful exits plus one expected
  mutation failure; all 19 outcomes matched; same native panic assertion
  binary; exact two builtin-void failures and seven passing controls; restored
  nine-cell pass; explicit quoted-C execution in both JSON profiles.
- Preserve expected short-suite skips and the native Linux scope. The short
  parity inventory checks are not actual release mutant/sweep execution.

## 7. Check summary claims and remaining gates

- Update only current source/count sections in `stabilization-candidate.md`;
  keep earlier source/build/hosted identities and outcomes historical.
- Current counts must say 306 tasks and 350 attacks with the exact status split.
  Distinguish a passed fresh make and focused installed gates from acceptance
  of the complete published candidate.
- Keep full plain-Go `run.bash`, all six complete optimization cells, unfiltered
  oracle-enabled stamped parity, actual release mutants/upstream sweep, and
  fresh hosted SWIG ordinary/LTO results pending until their real outcomes exist.
- Keep second-host digest comparison and named guarded runtime execution
  separate from static workflow checks. Preserve previously completed results
  at their actual source identities without transplanting them to source33752.
- Preserve the three blocked M6 application rows and their separate reopening
  requirement. No overall release-complete claim is justified by this packet.

## 8. Execute only file/static validators after the ready signal

- Run the unchanged frozen release contract checker through the existing
  offline Python environment; verify its own frozen checksum first.
- Run actual installed actionlint 1.7.12 and ShellCheck 0.11.0 on changed active
  workflow/shell inputs, with exact executable and input hashes recorded.
- Validate packet manifests and copies, local Markdown link targets, JSON
  syntax, ledger equality and path rules. Keep the checks separate from Go
  behavioral evidence; no compilation or test execution is authorized here.
- Use `git diff --check` with the same explicit raw-evidence boundary as the
  parent. Preserve whitespace in raw logs and frozen fixtures rather than
  rewriting evidence to satisfy a formatting check.
- Save commands, exits, input hashes and a concise result in the scratch audit
  directory. Report concrete defects promptly; do not repair main metadata.

## Final handoff criterion

The audit may report metadata/evidence consistency only when every applicable
file check passes on the same ready snapshot and all unresolved runtime/release
boundaries remain explicit. Publication and fresh hosted results remain the
parent's responsibility.
