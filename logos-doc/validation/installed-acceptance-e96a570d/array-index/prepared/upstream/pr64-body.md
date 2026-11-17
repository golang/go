## Why

LoGOs currently accepts both the valid enum guard from [gosuda/gglint#6](https://github.com/gosuda/gglint/issues/6) and the off-by-one variant that reaches an array access with index 2 into a two-element array. Its native lint registry has no array-bounds rule.

[PR #60](https://github.com/gosuda/LoGOs/pull/60) merged the regression script while this implementation was being verified. This follow-up adds the native diagnostic. The separate external gosec correction is [gosuda/gglint#8](https://github.com/gosuda/gglint/pull/8).

## Changes

- Add `correctness::array_index_out_of_bounds` at suppressible `deny`, using the existing directive, severity, and JSON diagnostic pipeline.
- Run it for Go packages under `logos lint` and for loGo/s packages through their existing native lint step.
- Report a concrete failing input, the array length, and the index operand's source span, at most once per index expression.
- Extend the merged command script to require the unsafe guard's native JSON finding in both source languages, then explicitly allow it in the loGo/s negative control to retain the runtime-panic check.
- Add unit and errorcheck coverage, register the rule's scope/default, and document the contract in SPEC, IMPLEMENTATION, and REDTEAM.

| Case | Native result |
| --- | --- |
| Unsigned index guarded by `t >= N` | No finding |
| Unsigned index guarded only by `t > N` | Deny finding with input `N` |
| Signed index with only an upper-bound guard | Deny finding with input `-1` |

## Proof and limits

For a direct fixed-array access indexed by an immutable integer parameter or integer value receiver, try the array length `N` when representable in the target type, and `-1` for signed types. Replay each concrete input through understood pure structured paths. Match variables by checker object and use target integer widths.

Unknown effects or unsupported expressions in the evaluated prefix stop the proof. Mutation, address exposure, and captures disqualify the tracked parameter. Slices, pointer-array bases, local index aliases, arithmetic/conversion indices, loops, and jumps are outside this initial rule. It does not detect every possible bounds failure and offers no autofix.

The guard is checked by concrete input rather than treating an interval that overlaps an unsafe value as proof that the value reaches the access. Whole-expression validation avoids inventing reachability when operand evaluation order is unspecified.

## Test-first evidence

The first commit adds the core regression. Before the implementation, the off-by-one receiver and missing signed lower bound each produced no finding in Go and loGo/s: four missing diagnostics. Valid guards remained silent. The second commit implements the rule and expands adversarial coverage.

The new unit coverage contains 45 behavior cases in each language, four diagnostic/span cases, and two suppression cases: 96 leaf checks. The two errorcheck fixtures expect eight diagnostics in total. The command script checks both lint results, all 256 enum values through three execution paths, and the specific runtime bounds panic through three negative controls.

## Verification

Passed on published head `235b67e587d90fa8d58a699202c2f668f2721200`, based on `bd318f1c558f3aa73ad434929fba7f237db9877f`:

- Clean `src/make.bash` toolchain build.
- Full native lint package: 4.799 s.
- Fixture-completeness check: 1.501 s.
- Both new errorcheck fixtures: 49.902 s.
- Guarded-enum command script under the race detector: 22.247 s.
- Native lint package vet, gofmt, and `git diff --check`.

The targeted commands ran from `src` with this checkout's rebuilt `bin/go`, its `GOROOT`, and `GOTOOLCHAIN=local`:

```sh
go test -count=1 -short -timeout 10m cmd/compile/internal/logos/lint
go test -count=1 -timeout 10m cmd/compile/internal/logos -run '^TestEveryLintRuleHasFixture$'
go test -count=1 -timeout 10m cmd/internal/testdir -run 'Test/logos/lint/array_index_out_of_bounds'
go test -count=1 -race -parallel 1 -timeout 10m cmd/go -run 'Script/^logos_guarded_enum_index$'
go vet cmd/compile/internal/logos/lint
```

The CLI checks confirm no finding for the valid guard and exactly one native deny finding for the faulty guard in each language. The unsafe finding points to the index operand and names input 2 outside bounds `[0, 2)`. Valid runtime executions check every enum value; negative controls require the specific bounds panic.

Validation covers the changed lint package and targeted compiler/command integrations, rather than the full Go toolchain regression suite. Independent source and integration review found no defects. The published tree matches the verified checkout, and the working tree is clean.

Opened as a draft as requested.

