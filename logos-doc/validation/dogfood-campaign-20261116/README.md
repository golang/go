# Dogfood campaign: 16 November 2026

All final hosted runs passed and their retained evidence was independently audited: gglint, native LoGOs, and the corrected corpus/runtime replay. The corpus accepts 27 positive cases and preserves one precisely identified dependency incompatibility as seven incomplete cells. All complete diagnostics and runtime inventories match the preceding accepted replay.

Every result below applies to its recorded source, configuration, and executables. A completed capture can contain failures. Historical failures and incomplete analyses remain visible after later repairs pass.

## Active revisions and publication

| Component | Published head | Root tree | Observed hosted status |
| --- | --- | --- | --- |
| [gglint PR13](https://github.com/gosuda/gglint/pull/13) | `4456e61c4dbdf636fca2b5bd77787873e2d1c1e8` | `dfb46c72af5ce9695844c5d567cbb8aab1568183` | [Run 43963396252](https://github.com/gosuda/gglint/actions/runs/43963396252): all four jobs passed and independently audited |
| [Native LoGOs PR76](https://github.com/gosuda/LoGOs/pull/76) | `6c53eb629a0f572db7ce157372005e0fa2089d1d` | `616b5421bc6912783762783b2ca681cacd50c105` | [Run 43972750963](https://github.com/gosuda/LoGOs/actions/runs/43972750963): unit, scripts, and upstream jobs passed; all three complete upstream partitions and before/after identities independently audited |
| [Corpus PR77](https://github.com/gosuda/LoGOs/pull/77) | `090bac0f2dbb1a1105d4bcb156fc31ed7c025177` | `bef78b17491bafdaf72f6926cc1482a6d7ff814b` | [Run 43972897190](https://github.com/gosuda/LoGOs/actions/runs/43972897190): complete matrix and every runtime gate passed; raw evidence, exact diagnostic equality, and runtime identity comparison independently audited |

Both active LoGOs heads have `src` tree `6a650366141c23a154f6e0b153767527e520a58f` and `test` tree `a64f88184df739e681c29c83e059d44ab6fda7bd`. Their workflows check out the explicit PR heads. The accepted gglint CI merge, `20294f56793727c9408f49bb854ed86f264e42af`, has the exact reviewed root tree above.

The earlier accepted corpus is [run 43967842639](https://github.com/gosuda/LoGOs/actions/runs/43967842639), head `3b2073eb98e488c411f52ba7cd723e8622c8cdfe`, root tree `0cd74a537f57e4c9c4bf983efba09bfe2918b5e2`, and `src` tree `ee5ae95824022485b74d4e3186a1540ef7124f76`. The corrected source changes only two nonexistent `.Complete()` calls in a compiler test fixture. Production code, corpus pins and profiles, runner, acceptance checker, corpus workflow, and gglint pin are byte-identical between these corpus heads. Independent replay and comparison establish exact normalized diagnostic equality between their complete runs. Both runs retain their own raw outputs and source identities.

[Evidence PR75](https://github.com/gosuda/LoGOs/pull/75) consolidates the retained archives. Its final evidence changes are based on the corrected corpus revision and preserve the source/test identities above. Original PR71 and PR73 remain historical records, superseded by the accepted corrected PR76 and PR77. The final evidence publication changes only this campaign directory relative to PR77.

## Archive catalog

The seven archives below retain the accepted candidates and the completed failures that led to their repairs. Each archive passed payload readback and byte-identical deterministic repacking.

| Archive | Recorded result and scope |
| --- | --- |
| [gglint-success-43963396252.tar.gz](gglint-success-43963396252.tar.gz) | All four jobs passed. Raw logs, API metadata, regression streams, source/tool receipts, and both complete Chi copies. The generated CLI is omitted; its hash and original ZIP identity are retained. |
| [native-43951103189-evidence.tar.gz](native-43951103189-evidence.tar.gz) | Unit and upstream failed; scripts passed. Preserves SIMD import/mutation failures and the full-testdir timeout. This older upstream capture has no equivalent final installed-tool receipt. |
| [corpus-43950476822-evidence.tar.gz](corpus-43950476822-evidence.tar.gz) | Complete matrix; acceptance rejected `tools/logos`. Independent runtime gates passed. Original strict-policy failures and diagnostic controls remain intact. |
| [native-43967559240-evidence.tar.gz](native-43967559240-evidence.tar.gz) | Original unit and complement failed only at the two fixture API calls. Scripts, full testdir, and cmd/go passed. All original artifacts and before/after identities are retained. |
| [corpus-43967842639-evidence.tar.gz](corpus-43967842639-evidence.tar.gz) | First accepted complete corpus and runtime replay; all diagnostic deltas independently classified. Includes the original strict failure and unlimited diagnostic probe. |
| [native-43972750963-evidence.tar.gz](native-43972750963-evidence.tar.gz) | Accepted corrected native run: all three jobs, all 18 focused unit phases, exact inline-linker controls, and every complete upstream partition. Includes original ZIPs, raw logs, and unchanged source/tool receipts. |
| [corpus-43972897190-evidence.tar.gz](corpus-43972897190-evidence.tar.gz) | Accepted corrected corpus/runtime replay, with zero diagnostic delta against the prior accepted run and identical named runtime inventories. Includes full raw captures and independent audits. |

The three previously committed archives remain byte-identical. Each of the four newly added native/corpus archives has a same-stem `.receipt.json` sidecar. `ARCHIVES.sha256` covers all seven archives and all four sidecars.

```sh
sha256sum --check --strict ARCHIVES.sha256
```

Each archive retains its payload checksum manifest, original downloaded ZIP identities, raw streams, and source/tool receipts. Readback verifies every member. Repacking normalizes path order, timestamps, ownership, modes, and gzip metadata, then compares bytes. The archived notes describe each bundle's verification and repacking procedure. Receipt sidecars record archive size/hash, member counts, original artifact identities, and readback/repack results; native receipts also record Python/zlib versions. These checks establish the retained bytes, not a successful outcome for a failed run.

## Accepted gglint and corrected native proofs

The accepted gglint race job passed 39 tested packages; three packages without tests are recorded separately. The unchanged nine-worker vettool stress test passed in 51.046 seconds. Vet, pinned golangci-lint with zero issues, and self-lint passed.

Its focused regression requires the archived analyzer to produce exactly two named failures and eight precise diagnostics. The candidate passes seven normal and seven race identities, plus two LoGOs parity identities. All six Chi CLI comparisons pass: basic 3 findings, standard 52, and strict 101. Both audits verified 32 frozen inputs, 32 raw stream hashes, 642 source receipts, six tool receipts, and both 111-file Chi copies. The CI verifier's Go 1.27.4 event-parser repair accepts only documented `OutputType` strings on output events; unknown fields and invalid contexts still fail.

The corrected native unit audit accepts all 18 frozen phases. Positive inventories include tagged-method facts/compiler controls (72/17), tagged-package facts (150), Wasm under Node (7), field-offset controls (50), SIMD work/midway/compiler controls (11/7/19), and zero-array type/mutation controls (67/15). Counts include named parents and subtests. Each has an exact run/pass inventory; deliberate archived-code failures have their own exact named diagnostics and exits. Zero-array negatives retain three type failures and two mutation failures. All 55 zero-array paths verify: the replacement manifest and its 54 payloads. The historical manifest and original failed fixture remain preserved. The complete native unit job also passed the LoGOs compiler-fixture subset (`-run Test/logos/`) in 355.336 seconds, race controls, vet, and lint with zero issues. The full upstream testdir result is recorded separately below.

The corrected script job passed the broad LoGOs scripts in 1,636.836 seconds and the installed-subprocess race command in 30.402 seconds. Its inline-capability proof reproduces exactly two required archived-linker failures, then passes both candidate controls. The source and all four installed tool hashes are restored afterward. The broad commands provide package-level results; the inline proof separately retains exact named event inventories.

The corrected upstream audit verifies all 666 registered dist names: one cmd/go entry, one full testdir entry, and 664 complement entries. Their union is complete and pairwise disjoint. All three driver and console-capture exits are zero, and every partition records `ALL TESTS PASSED`. Full testdir passed in 437.206 seconds and cmd/go in 1,169.836 seconds. The complement Actions step lasted 1,082 seconds; its log provides no single aggregate process duration. Before/after HEAD, source/test trees, clean source status, and all six installed tool receipts match. Broad upstream success uses the recorded Linux/amd64 short-test semantics and does not imply that every internal optional test emitted a named pass.

## Accepted corpus: 196 cells

Both accepted runs, 43967842639 and 43972897190, cover 28 configurations from 26 pinned inputs: 24 external projects and two local fixtures. The selection contains seven popular, 11 reference-practice, and ten stress profiles across GitHub, GitLab, Codeberg, SourceHut, and go.googlesource.com. Those labels guide selection; they do not certify project quality.

| Sampling cohort | Configurations |
| --- | --- |
| Popular: 7 | Chi, xxhash assembly, mattn SQLite cgo, Zap, go-toml/v2, Protobuf, UUID |
| Reference practice: 11 | go-cmp, xxhash purego tag, errorlint, go-scfg, multierr, BurntSushi TOML, x/tools, garif, decorder, modernc mathutil, exhaustive |
| Stress: 10 | purego FFI, modernc SQLite, GOVWA, mattn SQLite without cgo, YAML, x/sys, modernc libc, modernc memory, semantic fixture, C ABI fixture |

The [pinned configuration manifest](../../../misc/logos-dogfood/corpus-ci.json) records each upstream, revision, module digest, package pattern, tag, and cgo setting. Nine distinct inputs use Git revisions, 15 use pinned Go modules, and two are local fixtures. Reusing xxhash and mattn SQLite under different build profiles produces the 28 configurations.

Each case is built with stock Go, the plain fork, and LoGOs, then checked by native lint and gglint basic/standard/strict. Successful lint cells are repeated from fresh private source copies.

| Outcome | Primary cells |
| --- | ---: |
| Successful builds | 81 |
| Valid lint reports with findings | 98 |
| Valid clean lint reports | 10 |
| Explicit incomplete decorder controls | 7 |
| Total | 196 |

Thus 27 positive cases contribute 81 build passes and 108 valid, repeatable lint cells. Decorder retains one stock-Go compilation failure, two uncompared candidate builds, and four incomplete lint loads. Its pinned x/tools v0.20.0 `tokeninternal` layout assertion fails with a negative array length; those seven cells never count as successful analysis.

Each full audit verifies 952 captured streams, 476 process receipts, and 18 native identity receipts across source preparation and replay reports. Each early probe separately verifies 26 raw streams. The first independent lint audit verifies all primary/repeat diagnostic schemas, exact invocation bindings, 608 matrix stream hashes, and six affected source inventories against run 43950476822. It finds only the changes listed below. The corrected run independently reproduces every full normalized diagnostic multiset from the first accepted run, with identical assessments, exits, input receipts, source digests, and case profiles. Its source receipt records the corrected head and source tree; all six native executable receipts match within the same corpus workflow.

### Exact 13 + 2 precision changes

| Cell | Before | After | Removed records |
| --- | ---: | ---: | --- |
| Chi, gglint strict | 102 | 101 | `middleware/throttle.go:95:17` |
| modernc SQLite, gglint strict | 775 | 773 | `stmt.go:101:9`, `:294:9` |
| TOML, gglint strict | 125 | 124 | `lex.go:82:16` |
| x/tools, gglint strict | 3,970 | 3,963 | `go/ssa/task.go:40:7`; `internal/gopathwalk/walk.go:315:10`; `internal/jsonrpc2_v2/conn.go:121:7`, `:270:8`, `:394:7`, `:404:7`; `internal/jsonrpc2_v2/net.go:102:7` |
| modernc libc, gglint strict | 5,634 | 5,632 | `libc_musl.go:357:8`, `:475:14` |
| Protobuf, native lint | 482 | 480 | `internal/impl/presence.go:110:8`, `:137:8` |

All 13 gglint removals are direct communications in a select with a default case. The independent body operations remain checked: Chi's deferred send at line 97 and x/tools' nested deferred receive at `walk.go:321` still warn. Chi basic/standard remain 3/52. Protobuf loses exactly the two same-object field-offset findings and retains every other one of its 480 native records; its builds already passed before this precision correction. Every other full diagnostic multiset is unchanged.

## Observed runtime coverage

Both accepted corpus runs independently produce the results below. Each backend row has the same package-qualified named run/pass multiset under stock Go, plain fork, and LoGOs, with no skipped or failed required named test. Cross-run comparison also verifies identical commands, environments, and event inventories for all 21 backend, FFI, and native-lint runtime invocations. Repeated profiles and drivers are compared separately; their sum is not a count of unique tests.

| Profile | Observed named runs/passes per driver | Identity anchor |
| --- | ---: | --- |
| xxhash assembly core | 159 | Full retained core inventory |
| xxhash purego-tag core | 159 | Same core inventory under the purego tag |
| xxhash cgo plugin | 223 | Includes `dynamic/TestDynamic` |
| mattn SQLite cgo | 3 | `TestOpen`, `TestClose`, `TestInsert` |
| modernc SQLite | 15 | Seven named roots and eight named subtests |
| purego FFI | 8 | Eight exact callback/library-call identities |

Modernc roots are `TestNamedParameters`, `TestMemDB`, `TestBinding`, `TestBeginMode`, `TestMultiStmtQueryStringRoundtrip`, `TestExecReturningMultiRow`, and `TestMutexWakeOnce`. Its subtests are `TestBinding/DB`, `TestBinding/Prepare`, and deferred/immediate/exclusive begin modes under both delete and wal journals.

The FFI identities are `TestCallGoFromSharedLib`, `TestNestedDlopenCall`, `TestNewCallbackFloat64AndIntMix`, `TestNewCallbackInt64ResultWithStackArgs`, `TestRegisterFunc_StackAlignment`, `TestRegisterFunc_StackCallback`, `TestSimpleDlsym`, and `TestSyscallN`.

The plugin profile explicitly supplies `GOEXPERIMENT=none` to all three drivers. Its private LoGOs host uses `-O=1` and `[build] arena = false`; stock and plain-fork source copies remain pristine. Modernc receives its separately reviewed exact-package runtime permissions only in the private LoGOs copy. These passes establish those compatible profiles, not unrestricted interchangeability of plugin objects or default-policy acceptance.

The semantic and C ABI fixtures contribute **six successful package-level cells**, using `test -count=1 -short -timeout=595s ./...`. Their nonverbose logs contain no per-test events. Source defines seven semantic roots and one C ABI root; that static inventory is not counted as observed execution. Native LoGo execution, successful ejection, and stock-Go execution of the ejected program agree on `language semantics ok\n`. Native lint emits exactly two `correctness::unclosed_io_closer` records and one `correctness::infinite_recursion` record on both repetitions. Its runtime controls, `TestExecutedCloseForms` and `TestShadowedPanicReturnsTypedNil`, run and pass under each driver.

## Historical failures remain preserved

| Record | Observed failure and later boundary |
| --- | --- |
| [Initial differential campaigns](../../../misc/logos-dogfood/evidence/2026-11-16/corpus/README.md) | The x/tools implicit-type-parameter compiler error, killed modernc compiler process, capability denials, missing dependencies/vetx imports, and 22 incomplete gglint cells remain in the original reports. Those matrices were builds and lint loads; they executed no upstream test commands. |
| [Historical 22ab evidence](../historical-22ab/README.md) | Includes the superseded environment-propagation oracle, VCS-context failures, `go/types.TestGenerate` failure, interrupted matrices, and a 601.308-second LoGOs modernc deadline with no named test start. Local killed processes and environment outages do not establish a root cause or a pass. |
| [Native run 43927955055](../native-bdc5987/README.md) | C23 consumer-host rejection, the Go Wasm host's fixed 8 KiB argument/environment limit, and the cmd/go script deadline. Two SIGQUIT cases and one no-stderr exit remain limited to their recorded evidence. Host setup and bounded scheduling changed; original Go/JavaScript assertions stayed intact. |
| [Corpus run 43928753813](../corpus-c47c4/README.md) | `tools/logos` was denied after inline-context attribution and permission issues; the full matrix remained rejected while independent backend gates passed. |
| [Native prebuild run 43950355068](https://github.com/gosuda/LoGOs/actions/runs/43950355068) | ShellCheck rejected unquoted `HEAD^{tree}` before any Go build or test. The other two jobs never started and were canceled when the successor advanced. The quote-only correction preserves the source tree; the completed workflow failure remains recorded. |
| gglint run 43960061537 | Regression acceptance rejected the documented Go 1.27.4 `OutputType` field. The original failed report and raw events are frozen in PR13; a typed, action-scoped parser repair precedes the accepted run. |
| Native run 43951103189 | SIMD transitive-import and zero-array mutation-fact defects, followed by a 600.050-second full-testdir timeout. Cmd/go passed in 1,533.587 seconds; scripts also passed. |
| Corpus run 43950476822 | The unchanged strict tools policy remained the sole unexpected matrix blocker. Its runtime gates passed. The original strict denial is retained beside the reviewed build profile. |
| Native run 43967559240 | Two nonexistent `.Complete()` fixture calls caused both rejected classification phases and the sole complement package build failure. The other 16 unit phases were accepted; scripts passed, testdir passed in 465.806 seconds, and cmd/go passed in 1,141.122 seconds. |

The corrected native schedule partitions all 666 registered dist names into a 664-name complement, full testdir shard set, and cmd/go singleton. The sets are complete and disjoint. Complement scale remains 1, testdir uses scale 2 with a 20-minute package bound, and cmd/go uses scale 4 with a 36-minute bound. Before/after receipts cover HEAD, `src`, `test`, clean source status, and six installed tools. All three partitions completed successfully in corrected run 43972750963. Earlier failed invocations retain their original outcomes.

The tools build profile has observed successful builds in both accepted corpus runs. It retains the old three unsafe grants and adds the reviewed two process-launch and 14 network package grants, with no errgroup grant. The original strict profile still fails. Its separate unlimited probe records 48 errors and 78 warnings; the six additional errors beyond the limited trace name already-reviewed package owners. Conservative interface/error-method reachability does not establish observed runtime network activity. Neither diagnostic capture nor a policy edit bypasses full corpus acceptance.

## Scope and limits

This is a finite Linux/amd64 Go/LoGo and C-interoperability campaign, with stock Go 1.27.4 as the comparison toolchain. It covers selected Go-only, assembly, purego-tag, purego FFI, cgo, modernc, plugin, cgo-disabled stub, and generated-code profiles. It does not cover every repository, programming language, operating system, architecture, or backend combination. The build/lint matrix is distinct from the selected runtime tests.

Valid, repeatable findings prove report integrity and determinism. They do not prove every warning precise or a project clean. The documented joined-loop gorace precision limit remains outside the narrow default-select repair. Capability profiles and inherited experiment presence/value are part of the recorded test contract.

The evidence publication preserves the exact corrected source/test trees and all three previously committed archive blobs. Seven verified archives and four receipt sidecars are retained with the complete status history. Publishing this evidence does not merge the code into a release branch.

See the [acceptance contract](../../../misc/logos-dogfood/CI.md), [tools policy review](../../../misc/logos-dogfood/TOOLS_POLICY.md), and the active PRs above. Raw failures, explicit incomplete controls, and original manifests remain available throughout.
