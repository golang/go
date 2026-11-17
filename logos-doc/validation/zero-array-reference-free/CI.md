# Hosted zero-array proof

The frozen tests and evidence in `FROZEN.sha256` remain unchanged. This follow-up
adds the hosted gate; it does not change the predicate, the real SIMD fixture, or
any expected mutation mask.

`zero_array_ci.py` uses the shared `native_frozen_gate.py` introduced by commit
`e0c50c90984fce048a99d06efd200ec4db2aa5b7`. The shared runner retains the existing
tagged-method event parser and process capture. This wrapper only defines the
frozen inputs, exact test inventory, and four phases:

| Phase | Required result |
| --- | --- |
| Old type predicate | Exit 1; exactly the root, `function` parent, and `function/zero` leaf fail with the original predicate's false classification. |
| Old mutation predicate | Exit 1; exactly the root and `ReturnZero` leaf fail with `ParamMutated="1"` and an unchanged false receiver flag. |
| Candidate type classification | All 67 named tests pass, covering 57 leaves. |
| Candidate mutation facts | All 15 named tests pass, covering 14 leaves. |

Each negative phase overlays only `types2/const_logos.go` with its frozen original.
The candidate phases use the installed compiler and current source. The runner
captures every phase even if an earlier phase fails acceptance. It requires exact
test events, empty stderr, the expected process status, and unchanged input,
source, and tool identities before accepting the suite. The inherited parser
rejects skipped, missing, duplicate, unexpected, or crashing test results.

The unit workflow runs this gate as an independent step after a successful
build and smoke check. The 45-minute outer limit covers four phase limits of
660 seconds each. Each Go test process has a 10-minute test budget. Evidence is
retained with the other native regression artifacts.

## Local checks

Only Python and static checks ran locally. The 26 Python boundary controls pass,
including exact positive and negative transcripts, missing leaves, wrong exit
statuses, diagnostics attached to the wrong test, skipped tests, crashes, and
changed frozen inputs. Strict Pyright, Ruff check and format, actionlint with
ShellCheck, `git diff --check`, and all 21 frozen file hashes pass.

The shared runner was copied byte-for-byte into this checkout only as an
untracked dependency for those Python checks. Its implementation belongs to the
separate shared-runner commit and is not duplicated in this follow-up.

No Go build, test, compiler, analyzer, or vet command ran locally for this gate.
The new focused RED/GREEN runs, installed real SIMD regression replay, and full
native gates remain pending hosted execution. The previously observed hosted
failure is preserved separately in the frozen evidence.

## Compiler-fixture API correction

Hosted run `43967559240` exposed two invalid `.Complete()` calls in the new
classification fixture. Compiler-internal `types2.NewInterfaceType` already
marks its result complete and does not expose that method. Both classification
phases failed package build; their named tests did not run. The mutation negative
control was accepted, and all 15 candidate mutation names passed. The original
helper, fixture,
manifest, complete unit artifact/log, and all phase evidence are frozen under
`api-compatibility/`; that failed result is retained unchanged.

The active fixture removes only those two call suffixes. `FROZEN.sha256` remains
byte-identical and records the historical fixture hash. Its other 20 inputs
still match their original hashes; the historical fixture now also has an
explicit archived copy. The historical `frozen()` check retains its 22-path
contract and verifies every original digest literally, resolving the original
fixture entry to its archived file. The replacement check verifies the active
fixture separately.

Hosted execution additionally requires `REPLACEMENT.sha256`, which binds all
54 replacement inputs: the original 20 unchanged inputs, the corrected active
fixture, the original manifest, and 32 preserved failure files including the
archived original fixture. The suite records that manifest and all its inputs
before and after execution. It verifies that the active fixture equals the
archived original with exactly one removal of each known API suffix. Any other
byte change, including an assertion, expected value, test name, or whitespace,
fails the guard. The 67/15 positive inventories and both exact negative outcomes
are unchanged. No production predicate changes are part of this correction.
