# Acceptance follow-ups

The first complete hosted acceptance attempt at
`45c29ff215477f7e403188d7452fc79aef05d791` exposed one obsolete compiler-test
oracle and two ejection-test lint findings. These are separate corrections.
The original diagnostics remain in
[the installed acceptance record](../installed-acceptance-45c29ff2/README.md).

## CTFE map-field oracle

The old `const_func_unsupported_store.lgo` expected a diagnostic for a store
through a fresh local struct's map field. Existing M24 assignment support
already evaluates and captures that map-valued destination, as SPEC 2.1 and
T-201 require. Rejecting this valid store would contradict the implemented
assignment contract.

The separate test-only correction keeps the exact `storeField` declaration,
requires the compile-time constant `4` through array type identity, and requires
the output `4` followed by LF. No CTFE implementation or normative rule changes.
All ten frozen M24 entries and the genuine rejection controls remain unchanged.
The preserved original and complete follow-up records are in
[ctfe-map-field](ctfe-map-field/README.md).

The exact output passed natively and after ejection under the installed fork
and pristine Go 1.27.2, 1.27.3 and 1.27.4. The actual testdir harness passed the
new positive leaf, the unchanged map-key rejection leaf and its existing run
control, with no skips. The five selected assignment suites also passed,
including error order, nil-map behavior and both target word widths.

The extra function call in the positive fixture may itself fold. Its guarantee
is the exact compile-time value and required output, not an independently
forced unfolded evaluation.

## Ejection test infrastructure

The tuple adapter for selecting variant contexts was used only by tests, and
every call supplied the same Linux/amd64 host. It now lives in `eject_test.go`
with that host stated in its name. All eleven call inputs and existing
assertions are preserved. The production selector remains unchanged. The
existing target-feature test also reports a failed `RemoveAll` cleanup instead
of discarding its error.

[The lint record](lint/COMPLETED-2026-11-16.md) links the exact transformation proof,
original hosted diagnostics, configured lint with zero findings and all 100
race test cells passing without skips. These component checks use the freshly
installed source-45 compiler. A later bounded vet retry passed without
diagnostics. An external overlay inverted the cleanup condition and all four
existing target cases caught it; a fresh no-overlay race binary then passed
those cases. The earlier interrupted vet attempt retains its original record.

Both original frozen manifests remain unchanged. Their explicitly versioned
successors record the mechanical test changes and new positive fixture. The
compiler successor checks 26 entries; the target-fact successor checks five.
The active focused workflow uses the successor and includes the new native
fixture. It retains every previous test selector and assertion.

[Fresh static validation](static/summary.json) passed actionlint 1.7.12,
ShellCheck 0.11.0 and both successor manifests. All sixteen checked input
hashes remained unchanged. The saved historical CI snapshots remain intact.

## Existing constant-origin contract

The current source already implements the unsafe-derived readonly checks,
aggregate reference provenance, imported aggregate restrictions and returned
alias checks described as planned in older M5 notes. The fault fixtures also
already use deliberately outside-checked-source consumers. The follow-up
executes these existing tests before correcting that status text.

All eight selected testdir leaves passed. Under the actual command race binary,
the alias script passed ten native/ejection rejection pairs and the accepted
control both natively and after ejection under upstream Go. The imported-rodata
and relocation fault scripts also passed.
[Constant-provenance evidence](const-provenance/README.md)
records the exact source, selectors, outputs and coverage. This is a
documentation reconciliation; no M5 source or assertion changes.

## Integration boundary

### Concurrent source repairs

The combined source preserves the concurrent capability-discovery, array-index
lint, overlay ejection and static quoted-JSON type-inference changes. The
ordinary merge retains the incoming PR head as its second parent. Historical
reports keep their original row numbers; the active ledger uses the current
canonical numbers.

[Linker capability evidence](linker-capabilities/README.md) records all 25
selected top-level tests and 75 reported cells passing without skips, plus
normal vet. Its overlay contains the exact incoming source and frozen test.
The graph checks exercise discovery independent of ancestry, shuffled edge
insertion and the existing shortest-path oracle. The original five-second
child limit remains unchanged. Later [installed acceptance](../installed-acceptance-e96a570d/tooling-runtime/README.md)
passes the graph suite, vet and five capability scripts with the final linker;
both native cgo branches execute.

[Overlay ejection evidence](ejection-overlay/README.md) preserves the baseline
unit failure and real-driver script failure. The candidate race test passes;
replacing effective overlay reads with ordinary disk reads is caught; a fresh
restored race binary and normal vet pass. Installed script execution was pending
at the component handoff. The later final-tool race script now passes without
skips and closes row #441, as the installed record documents.

[Quoted JSON evidence](json-text/README.md) records the corrected Unicode RED,
both complete runtime profile race suites, vet, the caught pointer-only
mutation and restored GREEN. The original classic-profile setup failure is
retained separately in [test placement](json-test-placement/README.md). The
final composition preserves the incoming static type inference and reuses
the tested owned decoder. [Fresh installed acceptance](../installed-acceptance-e96a570d/json-runtime/README.md)
now passes both runtime race suites, normal runtime vet and explicit quoted-C
race execution in both profiles. Those final-source results remain separate
from the source-45 overlay component record.

The array-index rule retains its incoming source and 96 frozen unit cases.
[Combined installed acceptance](../installed-acceptance-e96a570d/array-index/README.md)
passes the full lint package, native fixtures, fixture census, guarded-enum race
script, vet and formatting. The reporting-condition mutation is caught in both
source languages, and the unchanged restored cells pass.

### Remaining acceptance

The initial timing-wrapper failure, misselected testdir invocation, missing
ignored timezone data in the temporary GOROOT view and disconnected executor
are preserved as setup or infrastructure observations. None counts as a
semantic RED or a passing gate. The unchanged intended checks ran after the
environment and selectors were corrected.

The combined source passed its fresh build and the scoped installed gates
recorded above. Complete script and compiler-fixture suites and the full Go
regression remain separate acceptance requirements. The first
broad script selection was canceled and its full-Go step never started. The
scoped passes above do not replace those gates. M6-S3b's frozen attacks and
three-attempt stop remain unchanged.
