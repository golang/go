# Same-object field offsets

The pinned Protobuf v1.36.12 replay reports two `arena_pointer_escape`
findings in `internal/impl/presence.go`, at lines 110 and 137. The original
JSON lines and exact licensed source are preserved here. Both expressions
subtract immediate field addresses of the same local struct. Subtraction
cancels the common base and leaves a relative offset, so neither value
retains the frame address described by the diagnostic.

## Frozen proof boundary

The new test file is frozen before the production edit. It has 44 rule
cases in Go and loGo/s and four installed compiler cases, with 50 named
runs including both roots. The original Go and loGo/s escape fixtures
remain byte-identical. `frozen.json` binds all of those inputs and the
original rule, real source, license, diagnostic lines, and complete raw
log digest. The full original run is retained separately under
`validation/corpus-c47c4`.

Only subtraction of genuine uintptr conversions immediately enclosing
unsafe.Pointer conversions of direct field addresses on the same checker
variable of struct-value type qualifies. Parentheses and type aliases
preserve the proof. The named variable must be local, each selection must
be direct and non-indirect, and neither operand may contain an index,
call, pointer alias, adjusted address, mask, or scale. Different objects
and addition remain findings. A proven delta combined with another frame
address still reports that separate address.

The rule cases include both exact Protobuf expression shapes, reversed
and identical fields, named conversions, returned and stored offsets,
and adversarial controls for every excluded operand. Installed compiler
cases require empty output and no object for a safe lint-only compile;
the address controls require exit 2, one exact rule diagnostic, and no
object in both languages.

## Execution status

The old hosted real-source findings establish the original precision
failure. New local Go execution is prohibited for this task. No new Go
result is claimed here. Hosted acceptance must run the exact frozen
inventory, reject skips/build failures/timeouts, and run the single
Protobuf-delta unit case with the archived old-rule overlay. That overlay
must produce the named semantic failure. The installed corpus replay
must remove exactly the two original Protobuf findings while retaining
its other diagnostic outcomes.
