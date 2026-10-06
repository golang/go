## How to add an operation to spec

Read `doc.go` for overall guidance.

Any operation we may add is already defined in the `simd` or `simd/archsimd`
packages. Use those as reference. Check their existing documentation using, for
example `GOEXPERIMENT=simd go doc simd ReduceSum` or `GOEXPERIMENT=simd
GOARCH=amd64 go doc archsimd MulAdd`. For each operation, check the archsimd
docs for GOARCH amd64, arm64, and wasm.

If you are writing the function declaration, note that spec is maximalist, so
consider how an operation can be generalized past types that may appear in simd
or archsimd. For example, numerical operations on just ints can often be
generalized to ints or uints. Often, numerical operations on floats can be
generalized to ints or uints, though be wary of introducing new complicated
overflow or rounding behavior in this case.

For the function implementation, look to `simd/simd_emulated.go` and
`simd/archsimd/internal/simd_test` for inspiration. `simd_emulated.go` operates
entirely on concrete types, while spec operates on generic types. Typically it's
easy to generalize across float types or int/uint types, though it may not be
easy to generalize one implementation for both float and int/uint. In that case,
it's okay to split it into two implementations (for an example, see `AbsFloat`
and `AbsInt`).

You can write temporary tests to check your work, but don't keep them. We do not
test every operation in spec because it will ultimately be checked against the
hardware.

Tell the user about any corner-case behavior, how you resolved it, and whether
or not it was consistent across `simd` and `archsimd` across platforms.
