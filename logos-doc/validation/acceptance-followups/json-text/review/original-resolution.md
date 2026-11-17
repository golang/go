# Upstream e5b95a2e integration review

The reviewed candidate is `45c29ff215477f7e403188d7452fc79aef05d791`.
The incoming release is `e5b95a2ecb5716814e52fbddddb0d6b491b6e1ce`.
Their merge base is `13a968f350564e739bf700df698a9a635dcdddcd`.
No candidate worktree, commit, or ref was changed by this review.

## Exact comparison

The local comparison recorded 237 incoming paths: 45 already match the
candidate, 179 are incoming-only, and 13 differ on both sides.
`comparison.json` records all per-path blob identities. Its SHA-256 is
`043ae3ebb2ddba3d9e04cf3fb991b0067e810bc2755a4bb9bc8bf54ce26def3d`.
The raw `merge-tree` output records nine conflicts. Clean textual merging
is insufficient: it reintroduces the deliberately excluded reflection
helpers in `scan.go`.

## Files requiring explicit resolution

| Path | Resolution |
| --- | --- |
| `logos-doc/IMPLEMENTATION.md` | Preserve candidate slices and ledgers; union the new upstream link-discovery records. |
| `logos-doc/REDTEAM.md` | Preserve candidate attacks and proving paths; union the new upstream records. |
| `logos-doc/SPEC.md` | Preserve the candidate and add the incoming shortest-rooted-prefix/cyclic-discovery sentence under T-Sec11. |
| `logos-doc/validation/imported-local-types/README.md` | Preserve the candidate's entire existing evidence directory unchanged. Import the complete incoming directory under `upstream-e5b95a2e/`, with new `INTEGRATION.md` prose outside both frozen manifests. |
| `src/cmd/compile/internal/logos/json_static.go` | Preserve candidate `cgoSource` plumbing. |
| `src/cmd/compile/internal/logos/json_unmarshal.go` | Preserve source-aware quoted type printers, C union spelling, and quoted-null pointer handling. |
| `src/logos/rt/json/path_v1.go` | Preserve candidate quoted-null handling and fallback struct naming. |
| `src/logos/rt/json/path_v2.go` | Preserve candidate syntax-error construction, profile behavior, and quoted-null handling. Test permissive text decoding before a narrow existing-helper repair. |
| `src/logos/rt/json/scan.go` | Preserve the candidate. Do not restore `methodlessType`, `decodePlain`, `FallbackFrom`, or `FallbackFromPtr`. |
| `src/logos/rt/json/scan_test.go` | Preserve candidate tests. The incoming append test's unchanged body belongs in the JSONv2-tagged test file. |
| `src/logos/rt/json/stream_v2_test.go` | Preserve candidate fallback routing and one-call/embedded-type assertions; add the two frozen upstream regression bodies. |
| `test/logos/run/json_marshal_v2_test.lgo` | Preserve candidate formatting; the incoming difference is one trailing blank line. |
| `test/logos/run/json_string_method_wrong_sig_test.lgo` | Preserve candidate pointer-null, error, and alias regression coverage. |

Also exclude these three paths from automatic incoming-blob selection:

- `src/cmd/go/scriptconds_test.go`
- `src/cmd/go/testdata/script/README`
- `src/cmd/go/testdata/script/logos_eject_simd_slice.txt`

They contain the incoming `host-simd` workaround. The initial comparison
classified the other 221 paths as 45 identical and 176 genuinely
incoming-only. Preserve their incoming content, with one provenance
layout refinement: import the complete incoming
`logos-doc/validation/imported-local-types/` snapshot under a new
`upstream-e5b95a2e/` child of that directory. Its `files.sha256` freezes its
own README, while the candidate has independent README/evidence hashes.
Do not overwrite either snapshot or rewrite existing evidence hashes.
This keeps the new linker implementation, its discovery-order tests, and
the dogfood harness, policies, corpus, and historical evidence. Those
harnesses were not executed by this review.

## SIMD guard decision

Keep the unconditional experiment-enabled execution in
`logos_eject_simd_slice.txt`. The fixture uses generated `xs.sum()`
reductions. It does not issue unguarded archsimd instructions.

`simdslice.go` chooses AVX2 for 32-byte forms and AVX512 for byte/halfword
forms. `guardedBody` emits vector and scalar branches. `simdGuardCond`
omits a CPU check only when the selected GOAMD64 level guarantees it.
`ssa/zcse.go` excludes `OpZeroSIMD` from entry-block hoisting, and
`ssa/branchelim.go` rejects speculation of operations using SIMD values.
The frozen nested-guard and zero-dominance scripts exercise normal and
feature-disabled execution. The checked-in SIMD workflow runs the
ejection fixture at GOAMD64=v1.

A blanket v4 host prerequisite would remove the scalar-fallback coverage
that this source contract requires. The recommendation comes from the
source and existing proof inventory, not from a new execution claim.

## New JSON tests

The parent reserved REDTEAM #419 and M11-A-S9 for this regression.

The original exact upstream additions were preserved as branchless tree
`3912153cc5c66fec2783828cecc9651dab39249b` based on the candidate.
Its two-file patch is `json-regressions.patch`, SHA-256
`8657405d111d316e58c754fc8b6aa27281c88ccc5915a45055f30273be2c681f`.

The append test supplies a plain quoted token to `QuotedTextPtr`, which
is JSONv2 behavior. Classic decoding first unwraps the outer `,string`
text and requires the remaining text value to be quoted. The original
untagged setup must therefore be observed as a setup failure, not counted
as a product defect.

The approved correction moves the exact append helper/test block into
the already JSONv2-tagged `stream_v2_test.go`. Assertions, inputs, existing
helpers, and runtime code are unchanged. The corrected branchless tree is
`4f5693091fde49e2849e515de9e9ad94a558a2e4`; its sole changed blob is
`f4ada6bf7f75061701503ed5ec61a404dee9ac4b`.

`json-regressions-corrected.patch` has SHA-256
`8588c088e140b104ebe1ebf08f8f87b9ae748130d8f8e931250531d4483ba251`.
The corrected test file has SHA-256
`08729986d528dd8a0c003893b0069639b5f3d7201b122a4105be162bfa088232`.
The exact test-only mapping is `json-test-overlay-corrected.json`.

The append test needs only its new probe type and the existing scanner
and quoted-pointer helper. The permissive test uses existing stream probe
types and `checkStreamFallback`; the latter remains on the candidate's
corrected generated-code routing. Four inputs cover scalar value and
pointer routes plus the struct fallback control.

`unquoteLiteralBytes` and `appendUnquoted` already replace malformed UTF-8
and unmatched surrogates with the replacement rune, reject invalid
escapes and controls, and return independent output storage. After an
observed semantic RED, only the two quoted text method call sites need a
candidate switch to that existing helper. The separate strict scalar
content decoder must retain its contract. No duplicate decoder or
reflection fallback is justified by this review.

At the time of this report, no JSON regression Go command or production
candidate had run. The assigned execution lane must first be released.
