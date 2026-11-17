# Integrated stabilization candidate

This record describes the reviewed source candidate in
[PR #58](https://github.com/gosuda/LoGOs/pull/58). It is an acceptance record,
not a completed release certificate. The shared Go and loGo/s implementation,
specification changes, frozen attacks, and remaining gates are published
together.

## Current acceptance repair composition

The current `src` tree is `33752c03819ed5afcb37f87633a3cac668e181a8`.
Four independently recorded slices repair failures found by the preceding
hosted acceptance round:

| Slice | Repair | Scoped evidence |
|---|---|---|
| M0-A-S20 / #442 | Keep an unrelated duplicate-stem package in the module census without rejecting the selected package; retain mixed-language and malformed-header rejection | Eight frozen census cases, twelve unchanged scripts, and the complete short loader package suites pass under race detection |
| M0-A-S21 / #443 | Strip the two private C type-printing facts from public `go/types` generation; keep void-panic tracking behind a private compiler hook | Unchanged physical `TestGenerate`, both complete short type-checker suites, native panic controls, and C/ejection scripts pass; removing the hook fails exactly the two builtin-void attacks |
| M22-A-S7 / #444 | End bootstrap resource settings before acceptance, keep Go temporary files under `TMPDIR`, and use a matched GCC 14 C/C++ pair | Four frozen executable bootstrap-contract tests and three caught mutations pass; prior native runtime/security regressions are separately recorded |
| M22-A-S8 / #445 | Limit active parity execution calls across rows and start each child deadline after admission | Three frozen queue scenarios and the complete short parity race suite pass; release-only mutant/sweep execution remains a separate gate |

Each slice used one implementation candidate. Their immutable
[component records](acceptance-repairs-33752c03/README.md) retain RED, GREEN,
caught mutations, restoration, and their original source boundaries. Pending
statements inside those original packets describe their assembly time; the
[installed acceptance packet](installed-acceptance-33752c03/README.md) supplies
the later combined-source results without rewriting historical evidence.

A normal fresh `make.bash` passed on 2026-11-16 from 21:44:15 to 21:52:15 UTC
in 480.027 seconds, with final PGO and dist cleanup enabled. The first make
lost its execution session without a completion receipt; its cause remains
unconfirmed and its partial output is preserved. No implementation or attack
changed between the two build invocations.

All nineteen installed gate commands matched their recorded expectations.
Eighteen exited successfully; the intentional void-hook mutation exited with
the expected two semantic failures, then the restored compiler passed all
nine cells using the same assertion binary. The complete short type-checker
suites retain eight documented upstream/short-mode skips in total. Short
parity retains its three documented release-only skips. No skipped release
test is counted as executed acceptance.

These build and gate receipts bind the combined uncommitted snapshot based
on local HEAD `2535486fe9596a9212d525a5007963ef9e8682f0`, with index tree
`098de80a4ddf1e59b2750c1306ef495414f42d1e`. The subsequent four slice commits
end at local `0a8eef6bab7d85e89e8e3efd0b933d73e78dff79` and preserve the same
complete `src` tree and all twelve repaired-file hashes. The evidence does
not represent a separate rebuild of each intermediate commit.

The synchronized task ledgers contain 306 unique, identical records. The
attack ledger contains 350 unique rows: 341 tested, two satisfied upstream,
no planned rows, and seven blocked rows. The blocked rows remain #216,
#217, #218, #226, #403, #404 and #405. SPEC and Table T now state the selected
package boundary for duplicate-stem errors. No M6-S3b implementation or
frozen attack changed.

### Remaining combined release gates

Fresh hosted acceptance must use this exact source tree. The earlier
published head `e6e3f9bf` has source tree `e237c399`; its results cannot
certify these compiler and harness repairs. The active release request now
names `33752c03` and retains all six original frozen-input hashes.

The remaining gates are the complete plain-Go regression
(`GO_TEST_TIMEOUT_SCALE=8 ./run.bash --no-rebuild`), the full script and
compiler-fixture suites, all six no-short `go`/`logos` O1/O2/O3 cells with
their separate `os` legs, actual SWIG/LTO coverage, and the unfiltered
oracle-on parity/mutant/sweep suite with a clean-tip release stamp. Fresh
focused workflows, runtime jobs, and independent-host reproducibility must
also retain their own source identity. Local installed evidence is Linux
amd64 evidence; it does not establish native Windows acceptance.

M6-S3b remains rolled back after the three-attempt limit. The
[bounded exception plan](m6-binding-exception-plan/README.md) records the
unchanged attacks and next native/export/ejection gates. Completing the
independent acceptance repairs does not authorize another M6 candidate or
establish complete language closure.

## Earlier source composition

The preceding local source freeze for the earlier acceptance round is
`e96a570da271c43a80e23bb1a188aac054acff25`, with complete tree
`0841c3d2a92b7d6091a5e248b82bd820326947a8` and `src` tree
`e237c399aae8f8065b724e8a0e27585686c7da18`. It preserves concurrent PR head
`8afec95580ed24e593307fd9d30d626eff31e22d` as an ordered merge parent and
includes release `7d15ec7671582d0f022013bf36cf8f82a1b04cfc`. The earlier
snapshots below remain historical evidence, with their original identities.

This composition retains the shared linker discovery repair, the array-index
lint rule and its frozen attacks, effective overlay source selection, and
static element-type inference for quoted JSON dispatch. The follow-up removes
the duplicate permissive string decoder in favor of the runtime's existing
owned decoder. The append regression is placed in its required JSONv2 test
file by a separate test-only correction; its assertions are unchanged.

A fresh normal `make.bash` passed from a clean, separate worktree at that
source freeze on 2026-11-16, from 18:23:09 to 18:44:39 UTC (1290.122 seconds).
The source stayed clean and unchanged, and the separate acceptance cache
remained empty. Both installed commands report Go 1.27.4 with their respective
command names. [The new installed record](installed-acceptance-e96a570d/README.md)
preserves the build and binary identities. Component evidence is linked from
[acceptance follow-ups](acceptance-followups/README.md); completed build,
installed behavioral acceptance and hosted gates are separate results.

At that snapshot, the metadata records 302 identical task-ledger entries and 346 unique
attack rows: 337 tested, two satisfied upstream, no planned rows and seven
blocked rows. The four former planned rows #386, #389, #391 and #402 are
promoted against the completed installed source-45 evidence below. Rows #336
and #441 now have fresh installed combined-source proof in the new acceptance
record. The seven blocked rows remain #216, #217, #218, #226 and #403 through
#405. No binding attempt or frozen M6-S3b attack changes in this reconciliation.

## Compiler-source evidence snapshot

These identities record the earlier integrated compiler source and its
published evidence. The subsequent CI merge is recorded separately below.

| Identity | Value |
|---|---|
| Published source commit | `4a77181e3f8a2973fb78e83caf4a934bd2089735` |
| Local source commit | `7b5e1598c3d81c1f628b35e41c4606ff56231298` |
| Complete Git tree | `7de05c6956f71e450e99b4aad1e41928ed6c7ec3` |
| `src` tree | `099367e1031585fa553ef7987b199288df36256a` |
| Included release branch | `bd318f1c558f3aa73ad434929fba7f237db9877f` |

Publication preserved each reviewed commit's complete tree and ordered
parents. At that snapshot the fetched public branch was byte-identical to
the local candidate. The release commit was an ancestor of that branch, and
`git merge-tree` returned the candidate's complete tree without conflicts. GitHub's PR API
nevertheless reported `mergeable: false` and `mergeable_state: dirty` at the
last check on 2026-11-16. The cause of that disagreement is unconfirmed;
no source conflict was found by the exact Git comparison.
A single close/reopen refresh left the PR open at the same head, but did not
clear that reported state or create an Actions run. No branch history was rewritten.

## CI release merge

The reconstructed merge starts at `e82c916ff4d8c2befc6f1891decb0f623401a94b`
and incorporates release `13a968f350564e739bf700df698a9a635dcdddcd`, including
the CI consolidation from `9a628f5b`. The compiler `src` tree remains
`099367e1031585fa553ef7987b199288df36256a`. No compiler repair or frozen
semantic assertion is changed by this merge. The published merge is
`45c29ff215477f7e403188d7452fc79aef05d791`. GitHub subsequently reported the
PR mergeable and created seven workflows. Their event merge
`f2ad58be8742b8ec350db8f06d9dc794ea3302e4` has the same complete tree,
`32b96f3a7c289c987bc021fe5a5adbf35cfe226b`, as the published head.

M0-A-S18 retains the two-build workflow, independent suite results, native-shell
checkout, pinned installers, and the full Go regression with its Node, oracle,
C23 and ledger prerequisites. Actual actionlint 1.7.12/ShellCheck 0.11.0 and
the structural checks pass over eight YAML inputs, 67 run blocks and two
installer scripts. The isolated SC2155 mutation is caught, bootstrap exit 37
is preserved, SIMD selects the same 17 files, and all ten input hashes remain
unchanged. [CI merge evidence](ci-merge/README.md) records this scope. These
results alone do not establish installed compiler or hosted acceptance. The
completed installed checks below now provide that separate evidence.

## Integrated scope

The candidate includes guarded SIMD optimization, indexed and generic
deferred calls, explicit generic type arguments, region chunk ownership,
CTFE assignment and aggregate-storage repairs, tuple names, spread defaults,
and the reviewed inliner change. Direct-C ejection preserves checked source
types and ownership; inactive packages and target-specific variants retain
their actual package and dependency identities. Both JSON profiles retain
their recorded method-dispatch and C-type repairs. Module-language boundaries,
context queries, LSP framing, metadata queries, and the WebAssembly test
environment have their recorded targeted repairs.

The last integrations add three reviewed scopes:

- Assigned `Close` results and builtin `panic` identity retain the original
  source, frozen regression tests, and recorded compiler acceptance from PR #66.
- Imported generic local types no longer force compiler-only lazy symbols
  through source-declaration fact traversal. The integrated source and both
  frozen tests match reviewed candidate `39c62004` exactly.
- The retained PR #63 declaration validator checks unused defaults in their
  lexical scope, assignment compatibility, recursive constant values and map
  keys, and repeated blank-field identity. Its production file and two frozen
  test files match retained PR #63 head `9a656c23` exactly. The new native
  eleven-package script exercises syntax-stored type information independently
  of default application.

SPEC and Table T preserve the matching contracts. M6-S3b's rejected binding
implementation remains rolled back, and its original attack patch remains
unchanged.

## Checks on the integrated tree

The following checks ran against the exact local source commit above:

| Check | Result |
|---|---|
| Compiler stability frozen-test manifest | All entries match |
| Inactive ejection and target-fact manifests | All entries match |
| Panic/Close production and frozen-test manifests | All entries match |
| Imported-local-type source, tests, and evidence | All 76 manifest entries match |
| Retained PR #63 source and two unit tests | Byte-identical to reviewed head |
| Native validator script | SHA-256 `4cd169dc2ab9b2fd0e5c751b75bdd4f68ad55ba72a4df63a446d214372b3aab0` |
| Blocked M6-S3b patch | SHA-256 `3d1bd84ac779294170db7d8e5e3d6403d005be31b94a634f5f9e4a16ae37d012` |
| Formatting of the six latest compiler/test files | No gofmt changes |
| Final integration whitespace check | Pass, except deliberately preserved raw author RED output |
| `go test -count=1 -short cmd/internal/goparity -run '^TestRedteamLedgerRows$'` | Pass, 0.024 seconds |

The ledger test used an existing installed driver to test the current ledger
and checker source. It does not prove that the combined compiler was rebuilt.
The imported-local-type author's raw RED log contains twelve whitespace-only
output lines; they are retained byte for byte and excluded only from the
whitespace check, never from a behavioral assertion.

At the AUD-200 snapshot, the embedded and standalone task ledgers contain
293 identical, unique records. The CI merge adds no REDTEAM row.
That snapshot's attack ledger contains 342 unique rows:

| Status | Count | Remaining rows |
|---|---:|---|
| Tested | 328 | Individual proof boundaries remain as recorded |
| Satisfied upstream | 2 | As recorded |
| Planned | 5 | #386, #389, #391, #402, #418 |
| Blocked | 7 | #216, #217, #218, #226, #403, #404, #405 |

The five planned rows required combined installed-toolchain acceptance for
generic deferred calls, explicit generic arguments, quoted JSON with C types,
native declaration validation, and imported local types respectively. The
scoped installed checks now pass as recorded below. Their final ledger
reconciliation is separate from the failed broad acceptance workflow.

## Behavioral evidence and its limits

The committed component reports retain semantic RED, unchanged assertions,
GREEN, caught mutations, and restored source. They distinguish their tested
source revisions from this final combination. Relevant reports include
[compiler stability](compiler-stability/README.md),
[inactive ejection](ejection-completeness/README.md),
[target facts](ejection-target-facts/README.md),
[panic and Close](dogfood-lint/README.md), and
[imported local types](imported-local-types/README.md).

The independent imported-local-type comparison contains eighteen command
records. Stock Go 1.27.4 and the existing repaired compiler pass all five
frozen programs and the exact upstream `digraph` build. The preceding S3
compiler fails the three captured-generic-local attacks and `digraph`, while
both nearby controls pass. This is concrete compatibility evidence for that
repair, not final merged-toolchain acceptance.

The [native declaration gate](../DEFAULT-VALIDATION-GATE.md) records one
accepted baseline control and ten semantic failures across eleven packages.
Its assertions remain frozen. On the combined installed compiler, all eleven
commands pass: three valid declarations build and eight invalid declarations
produce their required diagnostics.

## Installed acceptance at the CI merge

[The installed acceptance record](installed-acceptance-45c29ff2/README.md)
preserves the exact source, raw hosted logs, local build records, tool hashes
and environment recovery evidence. Six workflows passed after fresh builds:
generic type arguments, deferred callees, compiler-focused checks, both JSON
profiles, SIMD guards and all four native amd64/arm64 region-GC combinations.
Their mutations were caught and restored as recorded in each workflow.

A clean local `make.bash` also passed in 1410.125 seconds with the normal PGO
and cleanup sequence. Both installed commands have SHA-256
`987052b593c41cf60390eec80d98b01b89f9536d6d87b4e73672ab6e7a4eb2ef`;
the compiler has SHA-256
`61749f8f2b2242d26e613993270f654cf47c750739a22b223f94eedd5370bba9`.
That installation passed all 48 imported-symbol cases, the eleven-command
native declaration script under race, and explicit no-skip quoted-C ejection
tests under race in both JSON profiles.

The main workflow passed its configured unit suites, formatting and vet, but
failed on the obsolete `const_func_unsupported_store.lgo` diagnostic oracle
and two ejection-test lint findings. Its scripts leg was canceled by matrix
fail-fast; the complete script selection has no result, and the full-Go step
never started. [Acceptance follow-ups](acceptance-followups/README.md) records
the separate test-oracle correction and test-infrastructure repair. No broad
gate is promoted by the six successful focused workflows.

## Earlier release acceptance setup and results

This section retains the preceding source-e237 acceptance boundary. Its
pending statements are historical; the current requirements are above.

The first hosted follow-up at published head `1a4c7635` found two ambiguous
prose phrases in REDTEAM #336 and #441 at the compiler workflow's final ledger
check. The preceding configured compiler, command, cgo/ejection, testdir and
vet steps passed. The [documentation correction](ledger-prose-correction/README.md)
preserves the validator, every test path and all statuses; its direct compiled
validator replay records RED, GREEN, both caught reversions and exact restored
GREEN. This focused metadata check is not a parity or full Go regression pass.
The corrected metadata needs fresh hosted prerequisites and a clean-tip stamp;
earlier runs remain attached to their actual source and commit identities.

The fresh normal `make.bash` above satisfies the rebuild requirement for source
tree `e237c399aae8f8065b724e8a0e27585686c7da18`. The complete script and
compiler-fixture suites, configured lint and
`GO_TEST_TIMEOUT_SCALE=8 ./run.bash --no-rebuild` still require a successful
combined run. The ordinary regression defaults to short mode; it does not
replace the six no-short `go`/`logos` O1/O2/O3 cells or the complete unfiltered
goparity release suite and clean-tip stamp.

[M22-A-S6 integration](release-acceptance/INTEGRATION.md) adds explicit hosted
jobs for those six cells, stamped parity, second-host reproducibility, and
actual guarded LSP/QEMU/Wasm execution. Its frozen coverage checker records
baseline RED, static GREEN, a caught short-mode mutation and exact restoration;
actual actionlint and ShellCheck also pass. These are workflow checks, not
hosted behavioral results. The V2 recipe uses the same effective arm64 v8.2
setting for the frozen repro script and direct example builds, while preserving
all V1 raw outcomes. The new hosted results remain pending on their exact clean
tip, with all original semantic assertions intact.

The final e96 installed record already proves actual arm64 sum execution under
QEMU and both complete JSON runtime race profiles, including explicit no-skip
quoted-C tests. Real gopls, the selected Wasm runtime legs, the six full
optimization cells, stamped parity and two-host digest comparison remain
independent requirements. Earlier native GC evidence keeps its recorded source
boundary. The seven blocked rows remain unchanged; #403 through #405 are not
among M22's four historical exceptions. Independent job success therefore does
not by itself establish complete language or release closure.

The [CI bootstrap repair](ci-bootstrap.md) and [CI consolidation](ci-merge/README.md)
preserve the repository's action
ownership policy and acceptance commands. It uses exact-revision shell
checkout and pinned, SHA-256-verified Go and Node archives. Its static checks
passed. At the earlier publication check, GitHub confirmed the branch but
returned no Actions runs for that source head. That observation is historical;
the reconstructed merge produced the seven completed runs recorded above.

## Local build preflight

### Historical resource-limited environment

The earlier fresh build was prepared with one worker, a 512 MiB soft Go heap limit,
compiler concurrency one in both bootstrap stages, and the normal final PGO
sequence. Recent resource-limited builds and changing host availability led
to a preflight reserve of 4 GiB workspace disk and 2.5 GiB available RAM.
These are conservative scheduling thresholds, not a measured compiler peak.

Two exact, hashed cleanup batches removed 3,116,597,248 allocated bytes of
reproducible task-owned workspace cache data. Four obsolete archive copies
and fourteen separately verified, exclusive cache-link/target pairs freed
1,328,570,368 allocated tmpfs bytes. Source, frozen tests, raw evidence,
installed reference tools, shared caches, and other workspaces were preserved.
The remaining 38,399 protected cache entries retained their recorded metadata.

Workspace availability continued to fall during that bounded cleanup. The
last snapshot at 2026-11-16 15:42:57 UTC was:

| Resource | Available | Preflight reserve |
|---|---:|---:|
| Workspace disk | 3,965,952,000 bytes, 3.69 GiB | 4 GiB |
| Available RAM | 3,213,959,168 bytes, 2.99 GiB | 2.5 GiB |

Disk remained short by 329,015,296 bytes, so that build was not started.
This 3.69 GiB snapshot describes the former environment, not the replacement
environment's current capacity. The compact
[first](build-preflight/cleanup-first.json),
[second](build-preflight/cleanup-second.json), and
[final](build-preflight/cleanup-final.json) cleanup records retain the measured
values and exact inventory hashes.

### Replacement environment

The fresh replacement environment and pinned Go 1.27.4 bootstrap, Node
26.13.1 runtime, and Go 1.27.2/1.27.3 oracle toolchains are ready. GCC 14.2.0
has passed the C23 compile/run prerequisite. Recovered CI input hashes match
the validated candidates, and fresh workflow checks pass as recorded above.
The combined `make.bash` and the scoped installed semantic gates subsequently
passed. Their command outputs and exact identities are preserved in the
installed acceptance record; restoring the tools was only their prerequisite.

Ordinary dist cleanup must remain intact. It removes `pkg/obj/go-build` at
exit, and `--no-clean` is deprecated and ineffective. The completed installed
acceptance used a newly empty dedicated cache. The earlier warning that an
unchanged `VERSION` could conceal changed compiler code was overbroad for
this fork: its existing release-build-ID hook and command tool-ID logic include
compiler content identity in compilation keys on the audited Linux hosts.
[The source audit](installed-acceptance-45c29ff2/hosted/runtime/cache-audit.md)
records that correction and its limits. Fresh-cache isolation remains useful;
it is not a missing correctness fix. A prior source overlay does not replace
the actual installed build.

## Blocked default application

M6-S3b exhausted the correction budget in G1 item 7 of
[IMPLEMENTATION.md](../IMPLEMENTATION.md). Candidate three failed installed
native/export/ejection checks because default application wrote into nil
`Info.Types` under `StoreTypesInSyntax`; its capture logic also assumed the
map held syntax-stored constants. An empty-map allocation alone would hide
the syntax fallback and would not establish the invariant.

The failed implementation was removed, and its fourteen frozen attack files
remain in [M6-S3b.patch](../blocked/M6-S3b.patch). Constructor-local capture,
private producer defaults, generic declaration origins, blank-field
materialization, nested default composition, and exported anonymous-container
provenance remain unresolved. Rows #403 to #405 remain blocked, with their
mechanisms and gates explicit.

G1 requires an explicit exception before another implementation attempt.
The four older blocked rows (#216 to #218 and #226) are separate verification
exceptions permitted only with their documented release matrix. That
exception does not cover the unresolved default-application contract.
