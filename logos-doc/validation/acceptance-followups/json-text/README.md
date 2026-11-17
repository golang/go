# Quoted text Unicode acceptance

This directory records M11-A-S9 / REDTEAM #419 against the freshly installed
toolchain from `45c29ff215477f7e403188d7452fc79aef05d791`. The component repair
passed its frozen tests, both JSON runtime race suites, a caught and restored
mutation, and both vet profiles in one production attempt. Acceptance of the
final merged source remains pending.

## Contract and repair

After a streaming method declines a quoted value, `UnmarshalText` must receive
the same decoded bytes as the active `encoding/json` profile. The JSONv2 text
paths previously used strict `jsontext.AppendUnquote`, which rejected malformed
UTF-8 and unpaired UTF-16 surrogates before the method could receive the
replacement characters used by the standard decoder.

The candidate changes three lines in `path_v2.go`: both `QuotedText` and
`QuotedTextPtr` call the existing `unquoteLiteralBytes(string(tok))`, and the
pointer path declares its method error in the resulting scope. The shared
helper also gives the method independent output storage, so appending to its
input cannot overwrite the source token used for the pointer-null comparison.
The strict scalar decoder, syntax errors, null handling, fallback routing,
method-call counts, and compiler emission remain unchanged by this candidate.

## Frozen tests and profile correction

The two tests originated in incoming commit
`e5b95a2ecb5716814e52fbddddb0d6b491b6e1ce`:

- `TestQuotedTextAppendCannotCorruptToken`
- `TestQuotedTextPermissiveUnquoteMatchesEncodingJSON`

The original append test was placed in untagged `scan_test.go`, but its direct
`"abc"` input uses the JSONv2 method contract. The classic profile first unwraps
the `,string` token and therefore needs an inner quoted text value. The original
placement failed once under `nojsonv2`, as recorded in
`gates/original-classic-setup`. That observation is a test setup failure, not a
product RED or a production attempt.

Before changing production, the exact append helper and test block moved into
the existing JSONv2-tagged `stream_v2_test.go`. All inputs, assertions, and
existing fallback helpers remained unchanged. The corrected tests then exposed
the semantic failure under JSONv2. Both original and corrected proposals are
preserved, including their branchless Git tree identities and full source
snapshots. Their exact block hashes are in `json-regressions-corrected.json`.

## Observed gates

Each `gates/<label>/` contains the complete command and environment in
`inputs.json`, the unchanged raw output in `raw.log`, and the exit status,
duration, test names, and post-run identities in `result.json`.

| Gate | Profile | Observed result |
|---|---|---|
| `original-classic-setup` | `nojsonv2` | Expected setup failure; no production RED claimed. |
| `corrected-jsonv2-red` | `jsonv2` | Permissive-decoding test fails for value and pointer destinations; append control passes. |
| `candidate-jsonv2-focused` | `jsonv2` | Both frozen regressions pass without skips. |
| `candidate-jsonv2-full` | `jsonv2` | Full race suite passes: 141 top-level tests, 4,268 PASS records, no skips. |
| `candidate-nojsonv2-full` | `nojsonv2` | Full race suite passes: 128 top-level tests, 4,255 PASS records, one existing profile skip. |
| `mutant-jsonv2-pointer` | `jsonv2` | Restoring strict decoding only in `QuotedTextPtr` is caught in all three malformed-input pointer cases; append control passes. |
| `restored-jsonv2-focused` | `jsonv2` | Both unchanged regressions pass again without skips. |
| `vet-jsonv2` | `jsonv2` | Normal vet analyses pass. |
| `vet-nojsonv2` | `nojsonv2` | Normal vet analyses pass. |

The classic skip is the pre-existing `TestWalkEnterFrameDepthBound` condition:
`encoding/json` v1 has no depth bound. That test passes in the JSONv2 suite.
The new JSONv2 tests, actual semantic RED, and mutation run have no skips.

The RED and mutation raw logs contain intentional invalid UTF-8 input bytes.
Keep those files as binary data when transferring or publishing them. Their
SHA-256 hashes are recorded in `HANDOFF.json`, the individual results, and
`files.sha256`; decoding and re-encoding them as text changes the evidence.

## Files and reproduction

| File | Purpose |
|---|---|
| `json-regressions.patch` | Original two-file upstream test proposal against source45. |
| `json-regressions-corrected.patch` | Frozen, profile-correct single-file test proposal. |
| `json-regressions-candidate.patch` | Complete tested source45 production and corrected-test patch. |
| `candidate-runtime.patch` | Source45 to the tested production candidate only. |
| `mutant-runtime.patch` | Source45 to the pointer-path mutant only. |
| `pointer-mutation.patch` | Tested candidate to the caught pointer-path mutant. |
| `candidate-runtime/`, `mutant-runtime/` | Exact full production files used by the overlays. |
| `frozen-json-tests/`, `frozen-json-tests-corrected/` | Exact original and corrected test files. |
| `HANDOFF.json` | Consolidated result, source, tool, test, and raw-log hashes. |
| `copied-files.json` | Original evidence paths, byte counts, and verified copy hashes. |
| `provenance/` | Original fresh-build inputs, start record, and successful result. |
| `execution-support/` | Unchanged runner used for these observations. |
| `review/` | Historical integration review and subsequent read-only signature review. |

The historical JSON manifests, overlay paths, and runner retain their recorded
state and absolute paths. For a replay elsewhere, check out the exact source45
revision, rebuild it normally, and create replacement overlay paths pointing
to these archived source snapshots. Keep the frozen file contents unchanged.
Use a separate cache for this fork and for stock Go even when their version
strings match.

The test commands used `-p=1 -count=1 -parallel=1 -race -timeout=20m -v`, with
explicit `GOEXPERIMENT=jsonv2` or `nojsonv2`. The full-suite run pattern was `.`;
the focused pattern selected exactly the two frozen test names. Vet used the
same candidate overlay and explicit profile. The selected environment had
`GOMAXPROCS=1`, `GOGC=20`, `GOMEMLIMIT=512MiB`, `GOTOOLCHAIN=local`, and
`GOFLAGS=-p=1`. Compiler bootstrap flags were unset. Build/cache paths were
task-owned disk directories, and source temporary directories were unique real
directories under `/home/agent`. No VCS suppression was used.

Verify this bundle from its directory with `sha256sum -c files.sha256`.
Historical fields such as `candidate_go_tests_started: false` describe when
their files were created. They have deliberately not been rewritten; later
gate records and `HANDOFF.json` contain the completed observations.

## Integration boundary

Concurrent commit `c986384f43ed5c2b0759580d98814f2494a7bd7c` encodes the text
destination's element type in the helper signature. The read-only review in
`review/signature-review.md` recommends retaining that improvement while
composing these two shared-helper calls. It also identifies the reflection
fallback helpers reintroduced by the later release merge and preserves their
previous removal. This source review does not establish merged execution.

The final combined source still requires a fresh normal toolchain build and
its assigned installed acceptance gates. This directory records component
evidence at source45 and does not promote that evidence to a different source
tree or installed compiler.
